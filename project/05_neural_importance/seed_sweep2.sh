#!/bin/bash
for s in 6 7 8 9 10; do
  echo "=== SEED $s START ==="
  ../venv/bin/python3 -u run_milestone.py --model ea --preset custom --beyond 1 --tau 2 \
    --net-seed $s --replicas 10 --jobs 7 --no-cache --out "results_seed${s}.json"
  echo "=== SEED $s DONE ==="
done
