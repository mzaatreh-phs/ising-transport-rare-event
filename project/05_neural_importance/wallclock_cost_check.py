#!/usr/bin/env python3
"""Fairness check flagged 2026-08-08: we_estimate()'s 'cost' field is pure MC
sweep cost (len(S)*tau*L*L) and is BLIND to neural-net inference cost. For a
single net this under-counts a small, roughly constant overhead. For the
--ensemble K run, WE[I_theta] calls K nets per coord() evaluation, and the
FOM reported in results_ens*.json still uses the same MC-only cost -- so the
ensemble's real wall-clock overhead never enters the FOM ratio at all.

This script recovers the REAL per-method wall-clock time from the printed
cumulative timestamps in ensemble_sweep.log (each method line prints time
since that run's total_start) and recomputes a wall-clock-based FOM ratio,
using the same rel_sd already stored in results_ens*.json (unaffected by
which cost definition is used). Compares the MC-cost ratio (as reported) to
the wall-clock ratio, for each of the 5 ensemble runs.
"""
import json
import re
import numpy as np

from neural_importance import fom_from

LOG = 'ensemble_sweep.log'
SEEDS = [20, 21, 22, 23, 24]

line_re = re.compile(
    r'^\s*(naive|WE\[m\]|WE\[E\]|WE\[I_theta\])\s+.*\[(\d+\.\d+)s\]\s*$'
)
start_re = re.compile(r'=== ENSEMBLE RUN net-seed=(\d+) START ===')

blocks = {}
current = None
with open(LOG) as f:
    for line in f:
        m = start_re.search(line)
        if m:
            current = int(m.group(1))
            blocks[current] = {}
            continue
        m = line_re.match(line)
        if m and current is not None:
            method, cum_t = m.group(1), float(m.group(2))
            blocks[current][method] = cum_t

order = ['naive', 'WE[m]', 'WE[E]', 'WE[I_theta]']
print(f"{'seed':6s} {'method':12s} {'cum_t(s)':>10s} {'wall_dt(s)':>11s}")
wall_dt = {}
for s in SEEDS:
    if s not in blocks or not all(m in blocks[s] for m in order):
        print(f"  seed {s}: incomplete in log yet, skipping")
        continue
    wall_dt[s] = {}
    prev = 0.0
    for method in order:
        cum = blocks[s][method]
        dt = cum - prev
        wall_dt[s][method] = dt
        print(f"{s:6d} {method:12s} {cum:10.1f} {dt:11.1f}")
        prev = cum

print("\n--- MC-cost ratio (as reported) vs wall-clock ratio, WE[I_theta]/WE[E] ---")
for s in sorted(wall_dt):
    d = json.load(open(f'results_ens{s}.json'))
    r = d['results']
    est_I = np.array(r['WE[I_theta]']['est'], dtype=float)
    est_E = np.array(r['WE[E]']['est'], dtype=float)
    cost_I, cost_E = r['WE[I_theta]']['cost'], r['WE[E]']['cost']

    fom_I_mc, _, _ = fom_from(est_I, cost_I)
    fom_E_mc, _, _ = fom_from(est_E, cost_E)
    ratio_mc = fom_I_mc / fom_E_mc

    fom_I_wc, _, _ = fom_from(est_I, wall_dt[s]['WE[I_theta]'])
    fom_E_wc, _, _ = fom_from(est_E, wall_dt[s]['WE[E]'])
    ratio_wc = fom_I_wc / fom_E_wc

    print(f"seed{s}: MC-cost ratio={ratio_mc:.4f}  wall-clock ratio={ratio_wc:.4f}  "
          f"(wall-clock WE[I_theta]/WE[E] time ratio="
          f"{wall_dt[s]['WE[I_theta]']/wall_dt[s]['WE[E]']:.2f}x)")
