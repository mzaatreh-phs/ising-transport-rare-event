#!/bin/bash
# Task A of WHAT_TO_RUN_NEXT.md (2026-08-20): 3 more independent EA disorder
# realizations beyond the 3 already done (12345, 23456, 45678), same
# beyond=4/R=30/one-net-each protocol. Seeds pre-screened healthy via a
# collection-only diagnostic before this run (predeclared feasibility check,
# per the brief's own instruction).
set -e
cd "$(dirname "$0")"
source ../venv/bin/activate

for s in 56789 67890 78901; do
  echo "=== DISORDER SEED $s START $(date) ===" >> confirm_disorder_batch2_driver.log
  python -u neural_importance.py --model ea --preset full --replicas 30 --beyond 4 \
    --disorder-seed $s --n-bins 80 --n-per-bin 40 --we-iter 3000 --we-burn 600 --tau 2 \
    --objective selfconsistent --net-seed 0 \
    --net-cache "net_disorder${s}_seed0_selfconsistent.pt" \
    --out "results_selfconsistent_deep_disorder${s}_r30" \
    > "selfconsistent_beyond4_disorder${s}_r30.log" 2>&1
  echo "=== DISORDER SEED $s DONE $(date) ===" >> confirm_disorder_batch2_driver.log
done

touch CONFIRM_DISORDER_BATCH2_COMPLETE
