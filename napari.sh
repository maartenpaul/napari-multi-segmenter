#!/usr/bin/env bash
# Run on a machine with a display: ./napari.sh [image.tif]
cd "$(dirname "$0")"
exec pixi run --manifest-path envs/napari/pixi.toml python napari_demo.py "$@"
