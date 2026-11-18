"""Experimental demo: micro-SAM's 2D annotator with a "Pre-segment" dock. Cellpose 4,
Cellpose 3 and StarDist run in their own pixi environments (Appose workers); the
result lands in micro-SAM's auto_segmentation layer, to review and commit.

    python napari_demo.py [image.tif]
"""

import os
import sys
import time
from pathlib import Path

import appose
import numpy as np
from skimage import data

from models import LOAD, PARAMS, RUN, defaults, prefix


class Presegmenter:
    """One warm worker per model, started on first use."""

    def __init__(self):
        self._services = {}

    def run(self, name, pixels, params):
        start = time.perf_counter()
        service = self._services.get(name)
        if service is None:
            service = appose.custom().base(prefix(name)).build().python()
            self._check(service.task(LOAD[name]).wait_for())
            self._services[name] = service
        shape = list(pixels.shape)
        with appose.NDArray("float32", shape) as image, appose.NDArray("uint32", shape) as labels:
            image.ndarray()[:] = pixels
            self._check(service.task(RUN[name], {"image": image, "labels": labels, **params}).wait_for())
            return labels.ndarray().copy(), time.perf_counter() - start

    @staticmethod
    def _check(task):
        if task.error:
            raise RuntimeError(task.error)

    def close(self):
        for service in self._services.values():
            service.close()
        self._services = {}


def main():
    if len(sys.argv) > 1:
        import tifffile

        pixels = np.squeeze(tifffile.imread(sys.argv[1])).astype(np.float32)
    else:
        pixels = data.human_mitosis().astype(np.float32)
    pre = Presegmenter()

    if os.environ.get("SELFTEST"):  # no GUI: just prove the calls work from this env
        try:
            for name in LOAD:
                for _ in range(2):
                    labels, seconds = pre.run(name, pixels, defaults(name))
                    print(f"{name}: {int(labels.max())} objects in {seconds:.2f}s")
        finally:
            pre.close()
        return

    import napari
    from magicgui.widgets import ComboBox, Container, FloatSpinBox, Label, PushButton
    from micro_sam.sam_annotator import annotator_2d
    from napari.qt.threading import thread_worker

    viewer = annotator_2d(pixels, model_type="vit_b_lm", return_viewer=True)
    model = ComboBox(label="Model", choices=list(LOAD))
    # One box of parameters per model; only the chosen model's box is shown.
    boxes = {}
    for name, spec in PARAMS.items():
        fields = [
            ComboBox(name=key, label=key, choices=value)
            if isinstance(value, list)
            else FloatSpinBox(
                name=key, label=key, value=value[0], min=value[1], max=value[2],
                step=0.05 if value[2] <= 6 else 1.0,
            )
            for key, value in spec.items()
        ]
        boxes[name] = Container(widgets=fields, labels=True)

    def show_box():
        for name, box in boxes.items():
            box.visible = name == model.value

    model.changed.connect(show_box)
    button = PushButton(text="Pre-segment")
    status = Label(value="Result goes to auto_segmentation.")

    def done(result):
        labels, seconds = result
        # Same place micro-SAM's own automatic segmentation writes to: the user
        # reviews it, then commits it with micro-SAM's Commit (layer: auto_segmentation).
        layer = viewer.layers["auto_segmentation"]
        layer.data = labels
        layer.refresh()
        viewer.layers.selection.active = layer
        status.value = (
            f"{model.value}: {int(labels.max())} objects in {seconds:.1f}s - "
            "commit with layer 'auto_segmentation'"
        )
        button.enabled = True

    def failed(error):
        status.value = f"Failed: {error}"
        button.enabled = True

    def clicked():
        button.enabled = False
        status.value = f"Running {model.value} (first call loads the model)…"
        params = {w.name: w.value for w in boxes[model.value]}
        worker = thread_worker(pre.run)(model.value, pixels, params)
        worker.returned.connect(done)
        worker.errored.connect(failed)
        worker.start()

    button.clicked.connect(clicked)
    viewer.window.add_dock_widget(
        Container(widgets=[model, *boxes.values(), button, status]), name="Pre-segment", area="left"
    )
    if os.environ.get("GUITEST"):  # press the button for each model, no human
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        show_box()
        boxes["stardist"]["prob_thresh"].value = 0.8  # a changed parameter must change the result
        boxes["cellpose3"]["variant"].value = "nuclei"
        for name in LOAD:
            model.value = name
            clicked()
            while not button.enabled:
                app.processEvents()
                time.sleep(0.05)
            layer = viewer.layers["auto_segmentation"]
            print(status.value, "| layer max", int(np.asarray(layer.data).max()), flush=True)
        os.system("nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader")
        pre.close()
        return
    show_box()
    try:
        napari.run()
    finally:
        pre.close()


if __name__ == "__main__":
    main()
