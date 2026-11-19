"""Experiment: call Cellpose 4, Cellpose 3 and StarDist, each in its own pixi
environment, from one host process through Appose. Reports cold and warm timings."""

import os
import sys
import time
from pathlib import Path

import appose
import numpy as np
from skimage import data

# The folder with one installed pixi environment per model.
ENVS = Path(os.environ.get("SEGMENTER_ENVS", Path(__file__).parents[1] / "envs"))

# Per model: parameter -> list of choices, (default, min, max), or "file" / "folder" for
# a path. 0 = the model's default for diameter, prob_thresh and nms_thresh. `own_model`
# is a model you trained yourself; when set it replaces the pretrained one.
# `stitch_threshold` only matters for a 3D volume: planes are segmented one by one and
# masks that overlap more than this (IoU) in neighbouring planes become one object.
PARAMS = {
    "cellpose4": {
        "own_model": "file",
        "diameter": (0.0, 0.0, 500.0),
        "flow_threshold": (0.4, 0.0, 3.0),
        "cellprob_threshold": (0.0, -6.0, 6.0),
        "stitch_threshold": (0.5, 0.0, 1.0),
    },
    "cellpose3": {
        "variant": ["cyto3", "nuclei"],
        "own_model": "file",
        "diameter": (0.0, 0.0, 500.0),
        "flow_threshold": (0.4, 0.0, 3.0),
        "cellprob_threshold": (0.0, -6.0, 6.0),
        "stitch_threshold": (0.5, 0.0, 1.0),
    },
    "stardist": {
        "variant": ["2D_versatile_fluo", "2D_paper_dsb2018"],
        "own_model": "folder",
        "prob_thresh": (0.0, 0.0, 1.0),
        "nms_thresh": (0.0, 0.0, 1.0),
        "pmin": (1.0, 0.0, 100.0),
        "pmax": (99.8, 0.0, 100.0),
    },
}


def defaults(name):
    return {k: "" if isinstance(v, str) else v[0] for k, v in PARAMS[name].items()}


def prefix(name):
    """The environment installed for this machine: AMD GPU, CPU only, or the default (NVIDIA)."""
    envs = ENVS / name / ".pixi" / "envs"
    return next(envs / variant for variant in ("therock", "cpu", "default") if (envs / variant).exists())


# The worker keeps its models between calls (task.export); each variant or own model
# loads on first use.
LOAD = {
    "cellpose4": """
import torch
from cellpose import models
cache = {}
def get(own_model):
    if own_model not in cache:
        chosen = {"pretrained_model": own_model} if own_model else {}
        cache[own_model] = models.CellposeModel(gpu=torch.cuda.is_available(), **chosen)
    return cache[own_model]
task.export(get=get)
task.outputs["device"] = str(get("").device)
""",
    "cellpose3": """
import torch
from cellpose import models
cache = {}
def get(variant, own_model):
    key = own_model or variant
    if key not in cache:
        chosen = {"pretrained_model": own_model} if own_model else {"model_type": variant}
        cache[key] = models.CellposeModel(gpu=torch.cuda.is_available(), **chosen)
    return cache[key]
task.export(get=get)
task.outputs["device"] = str(get("cyto3", "").device)
""",
    "stardist": """
import os
import tensorflow as tf
from csbdeep.utils import normalize
from stardist.models import StarDist2D
cache = {}
def get(variant, own_model):
    key = own_model or variant
    if key not in cache:
        if own_model:  # the folder that holds config.json and the weights
            name, basedir = os.path.basename(own_model), os.path.dirname(own_model)
            cache[key] = StarDist2D(None, name=name, basedir=basedir)
        else:
            cache[key] = StarDist2D.from_pretrained(variant)
    return cache[key]
task.export(get=get, normalize=normalize)
get("2D_versatile_fluo", "")
task.outputs["device"] = str(tf.config.list_physical_devices("GPU") or "cpu")
""",
}

# `image` and `labels` arrive as appose NDArrays on shared memory; the worker writes
# its result straight into `labels`, which the host allocated. `image` is one plane
# (YX) or, for Cellpose, a volume (ZYX).
RUN = {
    "cellpose4": """
pixels = image.ndarray()
volume = {"z_axis": 0, "stitch_threshold": stitch_threshold} if pixels.ndim == 3 else {}
masks = get(own_model).eval(
    pixels, diameter=diameter or None,
    flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, **volume,
)[0]
labels.ndarray()[:] = masks
task.outputs["n"] = int(masks.max())
""",
    "cellpose3": """
pixels = image.ndarray()
volume = {"z_axis": 0, "stitch_threshold": stitch_threshold} if pixels.ndim == 3 else {}
masks = get(variant, own_model).eval(
    pixels, channels=[0, 0], diameter=diameter or None,
    flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, **volume,
)[0]
labels.ndarray()[:] = masks
task.outputs["n"] = int(masks.max())
""",
    "stardist": """
if image.ndarray().ndim != 2:
    raise ValueError("StarDist here is 2D only: choose planes, or Cellpose for a 3D volume")
masks, _ = get(variant, own_model).predict_instances(
    normalize(image.ndarray(), pmin, pmax),
    prob_thresh=prob_thresh or None, nms_thresh=nms_thresh or None,
)
labels.ndarray()[:] = masks
task.outputs["n"] = int(masks.max())
""",
}


def timed(service, script, inputs=None):
    start = time.perf_counter()
    task = service.task(script, inputs).wait_for()
    if task.error:
        raise RuntimeError(task.error)
    return time.perf_counter() - start, task.outputs


def main(names):
    pixels = data.human_mitosis().astype(np.float32)  # 512x512 nuclei
    image = appose.NDArray("float32", list(pixels.shape))
    image.ndarray()[:] = pixels
    labels = appose.NDArray("uint32", list(pixels.shape))
    services = {}
    try:
        for name in names:
            start = time.perf_counter()
            # appose.wrap() would run Appose's own pixi 0.58, which cannot parse our
            # pixi.toml; the environment is already installed, so use its prefix as is.
            service = appose.custom().base(prefix(name)).build().python()
            services[name] = service
            load_s, out = timed(service, LOAD[name])
            cold = time.perf_counter() - start
            print(f"{name}: worker + model ready in {cold:.1f}s on {out['device']}")
            for call in range(3):
                labels.ndarray()[:] = 0
                run_s, out = timed(service, RUN[name], {"image": image, "labels": labels, **defaults(name)})
                got = len(np.unique(labels.ndarray())) - 1
                print(f"  call {call}: {run_s:.2f}s, {out['n']} objects ({got} read back)")
        input_note = "all workers still warm; check nvidia-smi now" if sys.stdin.isatty() else ""
        print(input_note)
    finally:
        for service in services.values():
            service.close()


if __name__ == "__main__":
    main(sys.argv[1:] or list(LOAD))
