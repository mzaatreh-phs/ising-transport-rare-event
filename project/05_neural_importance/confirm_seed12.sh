#!/bin/bash
# Wall-clock-fair confirmatory reruns for deep-rare seed1/seed2 (beyond=4, ea, L=16, r30).
# Companion to the already-completed seed0 confirm (2026-08-14). Reuses cached trained
# nets (net_seed1_selfconsistent.pt, net_seed2_selfconsistent.pt) so only the WE
# evaluation stage (the expensive part) is repaid, with wall_time now a first-class
# field in the output JSON (added since the original pre-confirm seed1/seed2 runs).
set -e
cd "$(dirname "$0")"
source ../venv/bin/activate

echo "=== SEED 1 confirm START $(date) ===" >> confirm_seed12_driver.log
python neural_importance.py --model ea --preset full --replicas 30 \
  --thresh -1.119 --n-bins 80 --n-per-bin 40 --we-iter 3000 --we-burn 600 --tau 2 \
  --objective selfconsistent --net-seed 1 --net-cache net_seed1_selfconsistent.pt \
  --out results_selfconsistent_deep_seed1_r30_confirm \
  > selfconsistent_beyond4_seed1_r30_confirm.log 2>&1
echo "=== SEED 1 confirm DONE $(date) ===" >> confirm_seed12_driver.log

echo "=== SEED 2 confirm START $(date) ===" >> confirm_seed12_driver.log
python neural_importance.py --model ea --preset full --replicas 30 \
  --thresh -1.119 --n-bins 80 --n-per-bin 40 --we-iter 3000 --we-burn 600 --tau 2 \
  --objective selfconsistent --net-seed 2 --net-cache net_seed2_selfconsistent.pt \
  --out results_selfconsistent_deep_seed2_r30_confirm \
  > selfconsistent_beyond4_seed2_r30_confirm.log 2>&1
echo "=== SEED 2 confirm DONE $(date) ===" >> confirm_seed12_driver.log

touch CONFIRM_SEED12_COMPLETE
