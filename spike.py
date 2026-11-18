"""Throwaway spike: call Cellpose 4, Cellpose 3 and StarDist, each in its own pixi
environment, from one host process through Appose. Reports cold and warm timings."""

import sys
import time
from pathlib import Path

import appose
import numpy as np
from skimage import data

ENVS = Path(__file__).parent / "envs"

# Per model: parameter -> list of choices, or (default, min, max). 0 = the model's default
# for diameter, prob_thresh and nms_thresh.
PARAMS = {
    "cellpose4": {
        "diameter": (0.0, 0.0, 500.0),
        "flow_threshold": (0.4, 0.0, 3.0),
        "cellprob_threshold": (0.0, -6.0, 6.0),
    },
    "cellpose3": {
        "variant": ["cyto3", "nuclei"],
        "diameter": (0.0, 0.0, 500.0),
        "flow_threshold": (0.4, 0.0, 3.0),
        "cellprob_threshold": (0.0, -6.0, 6.0),
    },
    "stardist": {
        "variant": ["2D_versatile_fluo", "2D_paper_dsb2018"],
        "prob_thresh": (0.0, 0.0, 1.0),
        "nms_thresh": (0.0, 0.0, 1.0),
        "pmin": (1.0, 0.0, 100.0),
        "pmax": (99.8, 0.0, 100.0),
    },
}


def defaults(name):
    return {k: v[0] for k, v in PARAMS[name].items()}


# The worker keeps its models between calls (task.export); variants load on first use.
LOAD = {
    "cellpose4": """
import torch
from cellpose import models
model = models.CellposeModel(gpu=torch.cuda.is_available())
task.export(model=model)
task.outputs["device"] = str(model.device)
""",
    "cellpose3": """
import torch
from cellpose import models
cache = {}
def get(variant):
    if variant not in cache:
        cache[variant] = models.CellposeModel(gpu=torch.cuda.is_available(), model_type=variant)
    return cache[variant]
task.export(get=get)
task.outputs["device"] = str(get("cyto3").device)
""",
    "stardist": """
import tensorflow as tf
from csbdeep.utils import normalize
from stardist.models import StarDist2D
cache = {}
def get(variant):
    if variant not in cache:
        cache[variant] = StarDist2D.from_pretrained(variant)
    return cache[variant]
task.export(get=get, normalize=normalize)
get("2D_versatile_fluo")
task.outputs["device"] = str(tf.config.list_physical_devices("GPU") or "cpu")
""",
}

# `image` and `labels` arrive as appose NDArrays on shared memory; the worker writes
# its result straight into `labels`, which the host allocated.
RUN = {
    "cellpose4": """
masks = model.eval(
    image.ndarray(), diameter=diameter or None,
    flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
)[0]
labels.ndarray()[:] = masks
task.outputs["n"] = int(masks.max())
""",
    "cellpose3": """
masks = get(variant).eval(
    image.ndarray(), channels=[0, 0], diameter=diameter or None,
    flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold,
)[0]
labels.ndarray()[:] = masks
task.outputs["n"] = int(masks.max())
""",
    "stardist": """
masks, _ = get(variant).predict_instances(
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
            prefix = ENVS / name / ".pixi" / "envs" / "default"
            service = appose.custom().base(prefix).build().python()
            services[name] = service
            load_s, out = timed(service, LOAD[name])
            cold = time.perf_counter() - start
            print(f"{name}: worker + model ready in {cold:.1f}s on {out['device']}")
            for call in range(3):
                labels.ndarray()[:] = 0
                run_s, out = timed(service, RUN[name], {"image": image, "labels": labels, **defaults(name)})
                got = len(np.unique(labels.ndarray())) - 1
                print(f"  call {call}: {run_s:.2f}s, {out['n']} objects ({got} read back)")
            np.save(Path(__file__).parent / f"labels_{name}.npy", labels.ndarray().copy())
        input_note = "all workers still warm; check nvidia-smi now" if sys.stdin.isatty() else ""
        print(input_note)
    finally:
        for service in services.values():
            service.close()


if __name__ == "__main__":
    main(sys.argv[1:] or list(LOAD))
