#!/bin/bash
# Replaces disorder-seed 34567 (killed 2026-08-20 -- its WE-collected training
# pool was 100% boundary states / 0 bulk-anchor states, an outlier disorder
# realization where the self-consistency training's second boundary condition
# structurally cannot fire; confirmed as a real property of that Jx,Jy via a
# direct collect_training_configs() check, not a bug). disorder-seed 45678
# verified healthy first (330k/1.44M bulk-anchor states in a quick diagnostic
# before committing to the full multi-hour run).
set -e
cd "$(dirname "$0")"
source ../venv/bin/activate

echo "=== DISORDER SEED 45678 (replaces 34567) START $(date) ===" >> confirm_disorder_driver.log
python -u neural_importance.py --model ea --preset full --replicas 30 --beyond 4 \
  --disorder-seed 45678 --n-bins 80 --n-per-bin 40 --we-iter 3000 --we-burn 600 --tau 2 \
  --objective selfconsistent --net-seed 0 \
  --net-cache "net_disorder45678_seed0_selfconsistent.pt" \
  --out "results_selfconsistent_deep_disorder45678_r30" \
  > "selfconsistent_beyond4_disorder45678_r30.log" 2>&1
echo "=== DISORDER SEED 45678 DONE $(date) ===" >> confirm_disorder_driver.log

touch CONFIRM_DISORDER_REALIZATIONS_COMPLETE
