# napari-multi-segmenter

Experiment (November 2026): one napari with micro-SAM's 2D annotator, where
Cellpose 4, Cellpose 3 and StarDist propose a segmentation that is then reviewed,
committed and corrected with micro-SAM.

The models cannot share one Python environment (Cellpose 3 and 4 are the same package;
StarDist needs TensorFlow, the others PyTorch). Each model therefore runs as a warm
worker process in its own [pixi](https://pixi.sh) environment, called through
[Appose](https://github.com/apposed/appose); images and labels travel over shared
memory.

Not a product: no tests, 2D single-channel images only, pretrained models only.

## Run

```bash
./setup.sh                # five environments, about 25 GB
./napari.sh [image.tif]   # needs a display
```

In napari: left dock **Pre-segment** → choose a model, set parameters, press the
button. The result goes to micro-SAM's `auto_segmentation` layer. In micro-SAM's dock
set Commit **layer** to `auto_segmentation`, commit, then correct in
`committed_objects`.

Without a display: `pixi run --manifest-path envs/host/pixi.toml python models.py`
prints cold and warm timings; `SELFTEST=1 ./napari.sh` calls every model from the
napari environment; `GUITEST=1 xvfb-run -a ./napari.sh` presses the button for
every model.

## Files

| File | What |
|---|---|
| `envs/<name>/pixi.toml` | one environment per model, plus `host` (timings) and `napari` (micro-SAM + napari) |
| `models.py` | model scripts (`LOAD`, `RUN`), parameters (`PARAMS`), timing run |
| `napari_demo.py` | `Presegmenter` (one warm worker per model) and the dock |

## Measured

Linux, NVIDIA RTX PRO 6000 Blackwell, 512×512 nuclei image (`skimage.data.human_mitosis`).

| Model | Environment | Device | Worker + model ready | Warm call |
|---|---|---|---|---|
| Cellpose 4.2.1 | Python 3.11, torch 2.15 | GPU | 10.6 s | 0.15 s |
| Cellpose 3.1.1 | Python 3.11, torch 2.15, numpy 2.0 | GPU | 10.2 s | 0.16 s |
| StarDist 0.9.2 | Python 3.11, TensorFlow 2.21 | CPU | 6.8 s | 0.13 s |

Host: micro-SAM 1.8.14, napari 0.9.2, Appose 0.12.0. GPU memory with both Cellpose
workers warm: 3.4 GB + 1.4 GB. Each model environment is about 6 GB on disk.

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
  "pixels in, labels out".

## Not looked at

Windows, macOS, AMD GPUs (ROCm); 3D and multichannel images; own trained models;
what happens when an environment is missing; running all workers and SAM under memory
pressure.

## Related

[scikit-ops](https://github.com/apposed/scikit-ops) and
[napari-ai-lab](https://github.com/True-North-Intelligent-Algorithms/napari-ai-lab)
follow the same idea (ops in named environments through Appose).
