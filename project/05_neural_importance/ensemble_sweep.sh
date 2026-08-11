#!/bin/bash
# Deep-ensemble variance-reduction test for WE[I_theta], matching the canonical
# milestoning config used throughout the n=17 single-net seed sweep (see
# HANDOFF.md 2026-08-06/07): model=ea, preset=custom (L=16), beyond=1, tau=2,
# replicas=10, jobs=7. --ensemble 5 trains 5 independently-initialised nets
# per run (member seeds net_seed*1000+k) and uses their MEAN prediction for
# WE[I_theta], instead of a single trained net.
for s in 20 21 22 23 24; do
  echo "=== ENSEMBLE RUN net-seed=$s START ==="
  ../venv/bin/python3 -u run_milestone.py --model ea --preset custom --beyond 1 --tau 2 \
    --net-seed $s --ensemble 5 --replicas 10 --jobs 7 --out "results_ens${s}.json"
  echo "=== ENSEMBLE RUN net-seed=$s DONE ==="
done
