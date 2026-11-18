#!/usr/bin/env bash
# Install the environments. Usage: ./setup.sh [env ...]   (default: all)
# Set SPIKE_SCRATCH=/some/big/disk to keep pixi's cache and temp files off the root disk.
set -euo pipefail
cd "$(dirname "$0")"
if [ -n "${SPIKE_SCRATCH:-}" ]; then
  export PIXI_CACHE_DIR="$SPIKE_SCRATCH/pixi-cache" TMPDIR="$SPIKE_SCRATCH/tmp"
  mkdir -p "$TMPDIR"
fi
for env in ${@:-host cellpose4 cellpose3 stardist napari}; do
  echo "== $env"
  pixi install --manifest-path "envs/$env/pixi.toml" 2>&1 | tail -3
  du -sh "envs/$env/.pixi"
done
