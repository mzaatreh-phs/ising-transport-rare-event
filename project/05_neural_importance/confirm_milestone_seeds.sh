#!/bin/bash
# Wall-clock-fair confirmatory reruns for the milestoning 17-seed sweep
# (beyond=1, tau=2, custom preset, R=10). seed1 and seed5 already have
# wall_time-confirmed versions (2026-08-13); this covers the remaining 14
# (seed0 + seed2-4,6-15). "unseeded_run2" (results_milestone_beyond1_boot_tau2.json)
# predates the torch.manual_seed() fix and cannot be legitimately reproduced
# under current (seeded) code, so it is intentionally excluded here and left
# on reconstructed timing.
set -e
cd "$(dirname "$0")"
source ../venv/bin/activate

for s in 0 2 3 4 6 7 8 9 10 11 12 13 14 15; do
  echo "=== MILESTONE SEED $s confirm START $(date) ===" >> confirm_milestone_driver.log
  python run_milestone.py --model ea --preset custom --beyond 1 --tau 2 \
    --net-seed $s --replicas 10 --jobs 7 \
    --out "results_seed${s}_walltime_confirm.json" \
    > "milestone_seed${s}_walltime_confirm.log" 2>&1
  echo "=== MILESTONE SEED $s confirm DONE $(date) ===" >> confirm_milestone_driver.log
done

touch CONFIRM_MILESTONE_SEEDS_COMPLETE
