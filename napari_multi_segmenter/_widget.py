"""The "Pre-segment" dock: run Cellpose 4, Cellpose 3 or StarDist on an image layer."""

import atexit
import time
from itertools import product
from pathlib import Path

import appose
import numpy as np
from magicgui.widgets import ComboBox, Container, FileEdit, FloatSpinBox, Label, PushButton
from napari.layers import Image
from napari.qt.threading import thread_worker

from .models import LOAD, PARAMS, RUN, prefix

PLANE, ALL_PLANES, VOLUME = "current plane", "all planes, one by one", "3D volume (Cellpose)"


class Presegmenter:
    """One warm worker per model, started on first use."""

    def __init__(self):
        self._services = {}

    def run(self, name, pixels, params):
        service = self._services.get(name)
        if service is None:
            service = appose.custom().base(prefix(name)).build().python()
            self._check(service.task(LOAD[name]).wait_for())
            self._services[name] = service
        shape = list(pixels.shape)
        with appose.NDArray("float32", shape) as image, appose.NDArray("uint32", shape) as labels:
            image.ndarray()[:] = pixels
            self._check(service.task(RUN[name], {"image": image, "labels": labels, **params}).wait_for())
            return labels.ndarray().copy()

    @staticmethod
    def _check(task):
        if task.error:
            raise RuntimeError(task.error)

    def close(self):
        for service in self._services.values():
            service.close()
        self._services = {}


def block_indices(shape, kept, point, everywhere):
    """Index of each block to segment. A block spans the axes in `kept` (YX or ZYX);
    along the other axes take every position, or only the one at `point`."""
    others = [d for d in range(len(shape)) if d not in kept]
    here = tuple(int(point[d]) for d in others)
    for position in product(*(range(shape[d]) for d in others)) if everywhere else [here]:
        index = [slice(None)] * len(shape)
        for d, i in zip(others, position):
            index[d] = i
        yield tuple(index)


def _field(key, spec):
    if isinstance(spec, str):  # "file" or "folder"
        return FileEdit(name=key, label=key.replace("_", " "), mode="d" if spec == "folder" else "r")
    if isinstance(spec, list):
        return ComboBox(name=key, label=key, choices=spec)
    step = 0.05 if spec[2] <= 6 else 1.0
    return FloatSpinBox(name=key, label=key, value=spec[0], min=spec[1], max=spec[2], step=step)


class PresegmentWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._pre = Presegmenter()
        atexit.register(self._pre.close)

        self.image = ComboBox(label="Image", choices=self._image_layers)
        viewer.layers.events.inserted.connect(self.image.reset_choices)
        viewer.layers.events.removed.connect(self.image.reset_choices)
        self.model = ComboBox(label="Model", choices=list(LOAD))
        # One box of parameters per model; only the chosen model's box is shown.
        self.boxes = {
            name: Container(widgets=[_field(k, v) for k, v in spec.items()], labels=True)
            for name, spec in PARAMS.items()
        }
        self.scope = ComboBox(label="Planes", choices=[PLANE, ALL_PLANES, VOLUME])
        self.button = PushButton(text="Pre-segment")
        self.status = Label(value="")
        self.model.changed.connect(self._show_box)
        self.button.clicked.connect(self._clicked)
        self.extend([self.image, self.model, *self.boxes.values(), self.scope, self.button, self.status])
        self._show_box()

    def _image_layers(self, _=None):
        return [layer for layer in self._viewer.layers if isinstance(layer, Image)]

    def _show_box(self):
        for name, box in self.boxes.items():
            box.visible = name == self.model.value

    def _params(self):
        params = {}
        for field in self.boxes[self.model.value]:
            value = field.value
            if isinstance(field, FileEdit):  # an empty path field reads as "."
                value = "" if str(value) in ("", ".") else str(Path(value).expanduser().resolve())
            params[field.name] = value
        return params

    def _clicked(self):
        layer = self.image.value
        if layer is None:
            self.status.value = "Open an image first."
            return
        data = layer.data[0] if layer.multiscale else layer.data
        shape = tuple(data.shape[:-1] if layer.rgb else data.shape)
        # What the viewer shows (YX, plus the axis before it for a volume), in this
        # layer's own axes and pixel coordinates.
        depth = 3 if self.scope.value == VOLUME else 2
        offset = self._viewer.dims.ndim - len(shape)
        kept = sorted(d - offset for d in self._viewer.dims.order[-depth:])
        if len(shape) < depth or kept[0] < 0:
            self.status.value = "This image has no third axis to use as a volume."
            return
        point = np.clip(np.round(layer.world_to_data(self._viewer.dims.point)), 0, np.array(shape) - 1)
        indices = list(block_indices(shape, kept, point, self.scope.value == ALL_PLANES))
        name, params = self.model.value, self._params()

        @thread_worker
        def work():
            start, results = time.perf_counter(), []
            for done, index in enumerate(indices):
                yield f"Running {name}: {done + 1} of {len(indices)} (first call loads the model)…"
                block = np.asarray(data[index], dtype=np.float32)
                results.append(self._pre.run(name, block.mean(axis=-1) if layer.rgb else block, params))
            return results, time.perf_counter() - start

        def finished(result):
            results, seconds = result
            target = self._target(layer, shape)
            highest = 0
            for index, labels in zip(indices, results):
                if len(indices) > 1:  # objects in different planes get different IDs
                    labels = np.where(labels > 0, labels + highest, 0)
                    highest = max(highest, int(labels.max()))
                target.data[index] = labels.astype(target.data.dtype)
            target.refresh()
            self._viewer.layers.selection.active = target
            objects = sum(len(np.unique(labels)) - 1 for labels in results)
            self.status.value = f"{name}: {objects} objects in {seconds:.1f}s → layer '{target.name}'"
            self.button.enabled = True

        def failed(error):
            self.status.value = f"Failed: {error}"
            self.button.enabled = True

        self.button.enabled = False
        worker = work()
        worker.yielded.connect(lambda text: setattr(self.status, "value", text))
        worker.returned.connect(finished)
        worker.errored.connect(failed)
        worker.start()

    def _target(self, layer, shape):
        """micro-SAM's auto_segmentation when its annotator is open on this image (review
        there, then Commit with layer 'auto_segmentation'); otherwise a labels layer of our own."""
        layers = self._viewer.layers
        if "auto_segmentation" in layers and layers["auto_segmentation"].data.shape == shape:
            return layers["auto_segmentation"]
        name = f"{layer.name} pre-segmentation"
        if name in layers and layers[name].data.shape == shape:
            return layers[name]
        return self._viewer.add_labels(
            np.zeros(shape, np.uint32), name=name, scale=layer.scale, translate=layer.translate
        )
