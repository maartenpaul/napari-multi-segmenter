#!/usr/bin/env bash
# napari with the Pre-segment dock open: ./napari.sh [image ...]
# SELFTEST=1 ./napari.sh [model=path ...] presses the button for every model instead.
here="$(cd "$(dirname "$0")" && pwd)"
variant=default
[ -d "$here/envs/napari/.pixi/envs/therock" ] && variant=therock
run=(pixi run --manifest-path "$here/envs/napari/pixi.toml" -e "$variant")
if [ -n "${SELFTEST:-}" ]; then exec "${run[@]}" python "$here/selftest.py" "$@"; fi
exec "${run[@]}" napari "$@" -w napari-multi-segmenter Pre-segment
