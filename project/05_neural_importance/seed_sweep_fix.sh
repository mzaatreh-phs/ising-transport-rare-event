#!/bin/bash
# Validates the calibrated-crange fix by rerunning the SAME net-seed values
# already used pre-fix (seed1, seed2, seed3), so each pair is a controlled
# before/after comparison: identical trained network weights, only the
# WE[I_theta] bin range differs.
for s in 1 2 3; do
  echo "=== FIX SEED $s START ==="
  ../venv/bin/python3 -u run_milestone.py --model ea --preset custom --beyond 1 --tau 2 \
    --net-seed $s --replicas 10 --jobs 7 --no-cache --out "results_fix_seed${s}.json"
  echo "=== FIX SEED $s DONE ==="
done
