"""Press the Pre-segment button for every model, without a human. Needs a display.

    python selftest.py [cellpose4=MODEL_FILE] [cellpose3=MODEL_FILE] [stardist=MODEL_FOLDER]
"""

import sys
import time

import napari
import numpy as np
from qtpy.QtWidgets import QApplication
from skimage import data

from napari_multi_segmenter._widget import ALL_PLANES, PLANE, VOLUME
from napari_multi_segmenter.models import LOAD

own_models = dict(arg.split("=", 1) for arg in sys.argv[1:])
nuclei = data.human_mitosis()


def press(widget):
    widget.button.clicked.emit()
    while not widget.button.enabled:
        QApplication.instance().processEvents()
        time.sleep(0.05)
    print("  " + widget.status.value, flush=True)
    assert not widget.status.value.startswith("Failed"), widget.status.value


# 1. Plain napari, a stack of two planes: results go to a labels layer of our own.
viewer = napari.Viewer()
viewer.add_image(np.stack([nuclei, nuclei[::-1]]), name="nuclei")
_, widget = viewer.window.add_plugin_dock_widget("napari-multi-segmenter", "Pre-segment")
for name in LOAD:
    widget.model.value = name
    for scope in (PLANE, ALL_PLANES):
        widget.scope.value = scope
        press(widget)
    labels = viewer.layers["nuclei pre-segmentation"].data
    print(f"  per plane: {[len(np.unique(plane)) - 1 for plane in labels]}")
    if name != "stardist":  # the same plane twice: stitched, the objects span both
        viewer.layers["nuclei"].data = np.stack([nuclei, nuclei])
        widget.scope.value = VOLUME
        press(widget)
        labels = viewer.layers["nuclei pre-segmentation"].data
        print(f"  volume: {len(np.unique(labels)) - 1} objects, {len(np.unique(labels[0])) - 1} in plane 0")
        viewer.layers["nuclei"].data = np.stack([nuclei, nuclei[::-1]])
    if name in own_models:
        widget.boxes[name]["own_model"].value = own_models[name]
        press(widget)
        widget.boxes[name]["own_model"].value = ""
viewer.close()

# 2. micro-SAM's annotator: results go to its auto_segmentation layer.
try:
    from micro_sam.sam_annotator import annotator  # micro-sam 2
except ImportError:
    from micro_sam.sam_annotator import annotator_2d as annotator

viewer = annotator(nuclei, return_viewer=True)
_, widget = viewer.window.add_plugin_dock_widget("napari-multi-segmenter", "Pre-segment")
widget.image.value = viewer.layers["image"]
widget.model.value = "cellpose3"
press(widget)
assert viewer.layers["auto_segmentation"].data.max() > 0
print("micro-SAM annotator: auto_segmentation filled")
widget._pre.close()
viewer.close()
