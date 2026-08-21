#!/bin/bash
# Orchestrator: milestoning WE[E] baseline expansion (cheap, ~2.5h) followed
# by the 14 remaining wall-clock-confirm seed reruns (~7h). Sequential to
# avoid CPU contention on this 8-core box.
set -e
cd "$(dirname "$0")"
source ../venv/bin/activate

echo "=== MILESTONE WE[E] baseline expansion START $(date) ===" >> confirm_milestone_driver.log
python we_baseline_expand_milestone.py > we_baseline_expand_milestone.log 2>&1
echo "=== MILESTONE WE[E] baseline expansion DONE $(date) ===" >> confirm_milestone_driver.log
touch MILESTONE_WE_BASELINE_COMPLETE

bash confirm_milestone_seeds.sh
