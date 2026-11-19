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
./setup.sh                # five environments, about 25 GB
./napari.sh [image.tif]   # napari with the Pre-segment dock; needs a display
```

The napari environment also holds micro-SAM 2 and napari-omero, so images can come
from a file or from OMERO (Plugins → OMERO).

On a laptop with an AMD Radeon 780M (no NVIDIA GPU):

```bash
SEGMENTER_AMD=1 ./setup.sh   # four environments, about 25 GB
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

## Measured

Linux, NVIDIA RTX PRO 6000 Blackwell, 512×512 nuclei image (`skimage.data.human_mitosis`).

| Model | Environment | Device | Worker + model ready | Warm call |
|---|---|---|---|---|
| Cellpose 4.2.1 | Python 3.11, torch 2.15 | GPU | 10.6 s | 0.15 s |
| Cellpose 3.1.1 | Python 3.11, torch 2.15, numpy 2.0 | GPU | 10.2 s | 0.16 s |
| StarDist 0.9.2 | Python 3.11, TensorFlow 2.21 | CPU | 6.8 s | 0.13 s |

Host then: micro-SAM 1.8.14, napari 0.9.2, Appose 0.12.0 (now micro-SAM 2.0.0). GPU memory with both Cellpose
workers warm: 3.4 GB + 1.4 GB. Each model environment is about 6 GB on disk.

Laptop, Ryzen 7 7840U with Radeon 780M (ROCm 10.3 nightly), same image, called from
the napari environment (micro-SAM 1.9.0):

| Model | Device | First call (worker + model + run) | Warm call |
|---|---|---|---|
| Cellpose 4.2.1 | GPU | 38 s | 7.6 s |
| Cellpose 3.1.1 | GPU | 23 s | 0.8 s |
| StarDist | CPU | 18 s | 0.4 s |

A Nanolive refractive-index volume (96×512×512) as a 3D volume on the same NVIDIA
machine, into micro-SAM 2's annotator: Cellpose 4 18 s, Cellpose 3 13 s (first call,
model load included). Whether the objects are good has not been judged.

## Learned

- Warm calls are fast enough for a per-image annotation loop; the cold start is paid
  once per model per session.
- `appose.wrap()` runs Appose's own pixi (0.58), which cannot read the
  `platforms = [{ platform = …, cuda = … }]` syntax. Installing with the normal pixi and
  pointing Appose at the environment prefix (`appose.custom().base(prefix)`) works.
- StarDist runs on current TensorFlow with numpy 2; the old TensorFlow 2.10 / Python
  3.10 pins are not needed. TensorFlow does not use this GPU; CPU is fast enough in 2D.
- `auto_segmentation` is the right target: micro-SAM's Commit then merges the proposal
  with what is already committed (new IDs, preserve mode).
- Only the layer name ties the dock to micro-SAM; the worker side is plain
  "pixels in, labels out". The name survived micro-SAM 2.
- conda's micro-SAM 2 resolves to the CPU build of PyTorch unless `pytorch-gpu` is
  asked for on the CUDA platform.

## Not looked at

Windows, macOS, AMD GPUs other than the 780M; micro-SAM 2 on the AMD laptop (still
1.9 there); images opened through napari-omero (installed, not tried against a
server); real own-trained models (only tried with pretrained model files passed as a
path); true 3D models (Cellpose `do_3D`, StarDist 3D); what happens when an environment is missing; running all workers and SAM under memory
pressure.

## Related

[scikit-ops](https://github.com/apposed/scikit-ops) and
[napari-ai-lab](https://github.com/True-North-Intelligent-Algorithms/napari-ai-lab)
follow the same idea (ops in named environments through Appose).
