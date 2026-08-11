#!/usr/bin/env bash
# Fast sanity checks for every stage of the project. Total runtime: a few minutes.
# This does NOT reproduce the paper-quality figures (those take longer -- see
# each stage's README/usage banner) -- it only confirms the code runs and the
# validation gates pass on your machine.
set -e
cd "$(dirname "$0")"

echo "=================================================================="
echo "0) Environment check"
echo "=================================================================="
python3 check_env.py

echo
echo "=================================================================="
echo "1) VAN-Ising: tiny run (L=6, few steps) -- confirms training loop works"
echo "=================================================================="
( cd 01_van_ising && python3 run.py --L 6 --steps0 60 --steps 40 --out /tmp/van_smoke )

echo
echo "=================================================================="
echo "2) Phase 1: exact committor / zero-variance demo (seconds)"
echo "=================================================================="
( cd 03_phase1_framing && python3 demo_committor.py )

echo
echo "=================================================================="
echo "3) Phase 2: 1D weight-window unit test (seconds)"
echo "=================================================================="
( cd 04_phase2_core && python3 ww_1d.py )

echo
echo "=================================================================="
echo "4) Phase 2: Weighted Ensemble vs exact enumeration gate at L=4 (~30s)"
echo "=================================================================="
( cd 04_phase2_core && python3 ising_we.py )

echo
echo "=================================================================="
echo "5) Stage 5: learned-importance unbiasedness gate vs exact enumeration"
echo "=================================================================="
( cd 05_neural_importance && python3 neural_importance.py --gate --preset smoke )

echo
echo "=================================================================="
echo "All smoke tests completed. See README.md for the full-scale runs"
echo "(L=20 rare-tail demo, full VAN sweep, stage-5 comparison), or just run:\n  python3 run_all.py --preset laptop"
echo "=================================================================="
