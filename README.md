# napari-multi-segmenter

Experiment (November 2026): a napari plugin, **Pre-segment**, that runs Cellpose 4,
Cellpose 3 or StarDist on any image layer. With micro-SAM's annotator open, the result
is a proposal that is then reviewed, committed and corrected with micro-SAM.

The models cannot share one Python environment (Cellpose 3 and 4 are the same package;
StarDist needs TensorFlow, the others PyTorch). Each model therefore runs as a warm
worker process in its own [pixi](https://pixi.sh) environment, called through
[Appose](https://github.com/apposed/appose); images and labels travel over shared
memory.

Not a product: no tests, one channel per image layer.

## Run

```bash
./setup.sh                # five environments, roughly 25 GB
./napari.sh [image.tif]   # napari with the Pre-segment dock; needs a display
```

The napari environment also holds micro-SAM 2 and napari-omero, so images can come
from a file or from OMERO (Plugins → OMERO).

With an AMD gfx1103 integrated GPU (Radeon 780M/760M) instead of an NVIDIA GPU:

```bash
SEGMENTER_AMD=1 ./setup.sh
./napari.sh [image.tif]
```

Cellpose and micro-SAM then run on the GPU through AMD's TheRock ROCm wheels
(`therock` environments, pinned nightly), StarDist on the CPU (`cpu` environment).
`models.py` and `napari.sh` use whichever variant is installed.

In the **Pre-segment** dock:

- **Image**: any image layer, with any number of axes.
- **Model** and its parameters. **own model**: a Cellpose model file or a StarDist
  model folder you trained yourself; it replaces the pretrained model.
- **Planes**: the plane on screen, all planes one by one (each plane its own objects),
  or a 3D volume (Cellpose only: planes are stitched into 3D objects by overlap,
  `stitch_threshold`).

With micro-SAM's annotator open on the same image (Plugins → Segment Anything →
Segmentation Annotator), the result goes to its `auto_segmentation` layer: commit it
there, then correct in `committed_objects`. Otherwise it goes to a new labels layer
`<image> pre-segmentation`.

Checks without a human: `SELFTEST=1 ./napari.sh [model=path ...]` presses the button
for every model (on a machine without a screen: `SELFTEST=1 xvfb-run -a ./napari.sh`);
`pixi run --manifest-path envs/host/pixi.toml python napari_multi_segmenter/models.py`
prints cold and warm timings.

## Files

| File | What |
|---|---|
| `envs/<name>/pixi.toml` | one environment per model, plus `host` (timings) and `napari` (micro-SAM + napari) |
| `napari_multi_segmenter/models.py` | model scripts (`LOAD`, `RUN`), parameters (`PARAMS`), timing run |
| `napari_multi_segmenter/_widget.py` | `Presegmenter` (one warm worker per model) and the dock |
| `selftest.py` | presses the button for every model |

## Learned

- Warm calls are fast enough for a per-image annotation loop; the cold start (worker
  plus model load) is paid once per model per session.
- `appose.wrap()` runs Appose's own pixi (0.58), which cannot read the
  `platforms = [{ platform = …, cuda = … }]` syntax. Installing with the normal pixi and
  pointing Appose at the environment prefix (`appose.custom().base(prefix)`) works.
- StarDist runs on current TensorFlow with numpy 2; the old TensorFlow 2.10 / Python
  3.10 pins are not needed. CPU is fast enough in 2D.
- `auto_segmentation` is the right target: micro-SAM's Commit then merges the proposal
  with what is already committed (new IDs, preserve mode).
- Only the layer name ties the dock to micro-SAM; the worker side is plain
  "pixels in, labels out". The name survived micro-SAM 2.
- conda's micro-SAM 2 resolves to the CPU build of PyTorch unless `pytorch-gpu` is
  asked for on the CUDA platform.

## Not looked at

Windows, macOS, other AMD GPUs; micro-SAM 2 on the AMD variant (still 1.9 there); images opened through napari-omero (installed, not tried against a
server); real own-trained models (only tried with pretrained model files passed as a
path); true 3D models (Cellpose `do_3D`, StarDist 3D); what happens when an environment is missing; running all workers and SAM under memory
pressure.

## Acknowledgements

Running each model in its own Python environment, as a separate process, is the idea
behind [Appose](https://github.com/apposed/appose), which this plugin uses for the
workers and shared memory. [scikit-ops](https://github.com/apposed/scikit-ops) and
[napari-ai-lab](https://github.com/True-North-Intelligent-Algorithms/napari-ai-lab)
apply the same idea (ops or models in named environments through Appose) and were the
inspiration for this plugin.

Built on [micro-SAM](https://github.com/computational-cell-analytics/micro-sam),
[Cellpose](https://github.com/MouseLand/cellpose),
[StarDist](https://github.com/stardist/stardist), [napari](https://napari.org) and
[pixi](https://pixi.sh). Please cite the tools you use.

## License

BSD-3-Clause, see `LICENSE`.
