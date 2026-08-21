#!/bin/bash
# Retry of Task A's batch2, resuming at 67890 (56789 already done) with the
# bulk-quota fix applied to neural_importance.py.
set -e
cd "$(dirname "$0")"
source ../venv/bin/activate

for s in 67890 78901; do
  echo "=== DISORDER SEED $s (bulk-quota fix) START $(date) ===" >> confirm_disorder_batch2_driver.log
  python -u neural_importance.py --model ea --preset full --replicas 30 --beyond 4 \
    --disorder-seed $s --n-bins 80 --n-per-bin 40 --we-iter 3000 --we-burn 600 --tau 2 \
    --objective selfconsistent --net-seed 0 \
    --net-cache "net_disorder${s}_seed0_selfconsistent.pt" \
    --out "results_selfconsistent_deep_disorder${s}_r30" \
    > "selfconsistent_beyond4_disorder${s}_r30.log" 2>&1
  echo "=== DISORDER SEED $s DONE $(date) ===" >> confirm_disorder_batch2_driver.log
done

touch CONFIRM_DISORDER_BATCH2_COMPLETE
