#!/usr/bin/env bash
# Install the environments. Usage: ./setup.sh [env ...]   (default: all)
# SEGMENTER_AMD=1 installs the variants for an AMD Radeon 780M laptop instead
# (Cellpose and micro-SAM on ROCm, StarDist on CPU).
# Set SEGMENTER_SCRATCH=/some/big/disk to keep pixi's cache and temp files off the root disk.
set -euo pipefail
cd "$(dirname "$0")"
if [ -n "${SEGMENTER_SCRATCH:-}" ]; then
  export PIXI_CACHE_DIR="$SEGMENTER_SCRATCH/pixi-cache" TMPDIR="$SEGMENTER_SCRATCH/tmp"
  mkdir -p "$TMPDIR"
fi
if [ -n "${SEGMENTER_AMD:-}" ]; then all="cellpose4 cellpose3 stardist napari"; else all="host cellpose4 cellpose3 stardist napari"; fi
for env in ${@:-$all}; do
  variant=default
  if [ -n "${SEGMENTER_AMD:-}" ]; then
    if [ "$env" = stardist ]; then variant=cpu; else variant=therock; fi
  fi
  echo "== $env ($variant)"
  pixi install --manifest-path "envs/$env/pixi.toml" -e "$variant" 2>&1 | tail -3
  du -sh "envs/$env/.pixi"
done
