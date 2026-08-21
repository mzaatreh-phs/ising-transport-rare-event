#!/bin/bash
# Waits for the seed1/seed2 confirm job to finish, then runs the WE[E]
# baseline expansion (30->90 replicas) so it doesn't compete for the same 8
# cores while the neural confirm runs are in progress.
cd "$(dirname "$0")"
source ../venv/bin/activate

while [ ! -f CONFIRM_SEED12_COMPLETE ]; do
  sleep 60
done

echo "=== WE[E] baseline expansion START $(date) ===" >> confirm_seed12_driver.log
python we_baseline_expand.py > we_baseline_expand.log 2>&1
echo "=== WE[E] baseline expansion DONE $(date) ===" >> confirm_seed12_driver.log
touch WE_BASELINE_EXPAND_COMPLETE
