#!/bin/bash
for s in 1 2 3 4 5; do
  echo "=== SEED $s START ==="
  ../venv/bin/python3 -u run_milestone.py --model ea --preset custom --beyond 1 --tau 2 \
    --net-seed $s --replicas 10 --jobs 7 --no-cache --out "results_seed${s}.json"
  echo "=== SEED $s DONE ==="
done
