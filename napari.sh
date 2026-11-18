#!/usr/bin/env bash
# Run on a machine with a display: ./napari.sh [image.tif]
cd "$(dirname "$0")"
variant=default
[ -d envs/napari/.pixi/envs/therock ] && variant=therock
exec pixi run --manifest-path envs/napari/pixi.toml -e "$variant" python napari_demo.py "$@"
