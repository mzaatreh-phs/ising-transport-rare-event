#!/bin/bash
# Tests whether the deep-rare beyond=4 reliability finding (66.7% vs 48.9%,
# Fisher p=0.023 on the ORIGINAL disorder realization, disorder-seed=12345)
# generalizes to independent EA disorder realizations. Two new realizations,
# one trained network each (not the 3-seed ensemble -- seed_variance_decompose.py
# already showed net-init variance is ~0, so a single net per realization is
# the right allocation of compute here), beyond=4 auto-calibrated fresh for
# each realization's own pilot (not the hardcoded -1.119, which is specific
# to disorder-seed=12345), R=30, matching every other production config.
set -e
cd "$(dirname "$0")"
source ../venv/bin/activate

for s in 23456 34567; do
  echo "=== DISORDER SEED $s START $(date) ===" >> confirm_disorder_driver.log
  python neural_importance.py --model ea --preset full --replicas 30 --beyond 4 \
    --disorder-seed $s --n-bins 80 --n-per-bin 40 --we-iter 3000 --we-burn 600 --tau 2 \
    --objective selfconsistent --net-seed 0 \
    --net-cache "net_disorder${s}_seed0_selfconsistent.pt" \
    --out "results_selfconsistent_deep_disorder${s}_r30" \
    > "selfconsistent_beyond4_disorder${s}_r30.log" 2>&1
  echo "=== DISORDER SEED $s DONE $(date) ===" >> confirm_disorder_driver.log
done

touch CONFIRM_DISORDER_REALIZATIONS_COMPLETE
