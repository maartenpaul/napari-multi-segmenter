#!/usr/bin/env bash
# Activation for `therock` environments: TheRock nightly wheels built natively
# for gfx1103 (Radeon 780M), so no HSA_OVERRIDE_GFX_VERSION and no rocBLAS
# tuning-file symlink. Keeps the APU stability settings from activate.sh.
# Every variable uses ${VAR:-default}; a value exported beforehand wins.

unset HSA_OVERRIDE_GFX_VERSION

# APU stability: avoid SDMA copy engines on integrated GPUs.
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"

# MIOpen: skip exhaustive kernel tuning (looks like a freeze on first use).
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"

# Return freed blocks to shared system RAM instead of caching them.
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}"
