#!/usr/bin/env python3
"""Extend wallclock_cost_check.py's method (already applied to the K=5
ensemble sweep) to the full 17-seed SINGLE-NET (K=1) milestoning sweep that
produced the paper's headline beyond=1 result (1.19x, CI [0.84,1.75]).

HANDOFF_ensemble_cost.md flagged this exact gap: 'written up as applying to
EVERY WE[I_theta] result... indicts numbers in three other sections that
were never recomputed.' This recomputes them.

Parses each seed_sweep*.log / seedtest_run1.log / milestone_beyond1_boot_tau2.log
for the cumulative-time printout after each method line ('[NNNN.Ns]'),
takes the per-method wall-clock DELTA (cumulative time since previous method
finished) as that method's true cost, and recomputes FOM_ratio = FOM(I)/FOM(E)
on this wall-clock basis instead of the MC-sweep-only 'cost' field already
in results_seed*.json.
"""
import json, re, os, sys
import numpy as np
os.chdir(os.path.expanduser("~/ising_transport_project/project/05_neural_importance"))
sys.path.insert(0, os.getcwd())
from neural_importance import fom_from

LOG_MAP = {
    'seed0':         ('seedtest_run1.log',              'results_seedtest_run1.json'),
    'unseeded_run2':  ('milestone_beyond1_boot_tau2.log', 'results_milestone_beyond1_boot_tau2.json'),
    'seed1': ('seed_sweep.log', 'results_seed1.json'),
    'seed2': ('seed_sweep.log', 'results_seed2.json'),
    'seed3': ('seed_sweep.log', 'results_seed3.json'),
    'seed4': ('seed_sweep.log', 'results_seed4.json'),
    'seed5': ('seed_sweep.log', 'results_seed5.json'),
    'seed6': ('seed_sweep2.log', 'results_seed6.json'),
    'seed7': ('seed_sweep2.log', 'results_seed7.json'),
    'seed8': ('seed_sweep3.log', 'results_seed8.json'),
    'seed9': ('seed_sweep3.log', 'results_seed9.json'),
    'seed10': ('seed_sweep3.log', 'results_seed10.json'),
    'seed11': ('seed_sweep4.log', 'results_seed11.json'),
    'seed12': ('seed_sweep4.log', 'results_seed12.json'),
    'seed13': ('seed_sweep4.log', 'results_seed13.json'),
    'seed14': ('seed_sweep4.log', 'results_seed14.json'),
    'seed15': ('seed_sweep4.log', 'results_seed15.json'),
}
SEED_NUM = {  # net-seed argument used, to locate the right block in a multi-seed log
    'seed0': None, 'unseeded_run2': None,
    'seed1': 1, 'seed2': 2, 'seed3': 3, 'seed4': 4, 'seed5': 5,
    'seed6': 6, 'seed7': 7, 'seed8': 8, 'seed9': 9, 'seed10': 10,
    'seed11': 11, 'seed12': 12, 'seed13': 13, 'seed14': 14, 'seed15': 15,
}

method_re = re.compile(r'^\s*(naive|WE\[m\]|WE\[E\]|WE\[I_theta\])\s+.*\[(\d+\.\d+)s\]\s*$')
start_re = re.compile(r'=== SEED (\d+) START ===')
order = ['naive', 'WE[m]', 'WE[E]', 'WE[I_theta]']

def extract_block(logfile, seed_num):
    lines = open(logfile).read().splitlines()
    if seed_num is None:
        segment = lines
    else:
        start_idx = None
        for i, l in enumerate(lines):
            m = start_re.match(l.strip())
            if m and int(m.group(1)) == seed_num:
                start_idx = i
                break
        assert start_idx is not None, f"seed {seed_num} not found in {logfile}"
        segment = lines[start_idx:start_idx + 40]
    cum = {}
    for l in segment:
        m = method_re.match(l)
        if m:
            cum[m.group(1)] = float(m.group(2))
            if len(cum) == 4:
                break
    assert len(cum) == 4, f"only found {cum} in {logfile} (seed_num={seed_num})"
    deltas = {}
    prev = 0.0
    for meth in order:
        deltas[meth] = cum[meth] - prev
        prev = cum[meth]
    return deltas

rows = []
for name, (logfile, jsonfile) in LOG_MAP.items():
    deltas = extract_block(logfile, SEED_NUM[name])
    d = json.load(open(jsonfile))
    r = d['results']
    est_I = np.array(r['WE[I_theta]']['est'], dtype=float)
    est_E = np.array(r['WE[E]']['est'], dtype=float)
    mc_cost_I, mc_cost_E = r['WE[I_theta]']['cost'], r['WE[E]']['cost']

    fom_I_mc, _, _ = fom_from(est_I, mc_cost_I)
    fom_E_mc, _, _ = fom_from(est_E, mc_cost_E)
    ratio_mc = fom_I_mc / fom_E_mc

    wc_I, wc_E = deltas['WE[I_theta]'], deltas['WE[E]']
    fom_I_wc, _, _ = fom_from(est_I, wc_I)
    fom_E_wc, _, _ = fom_from(est_E, wc_E)
    ratio_wc = fom_I_wc / fom_E_wc

    rows.append(dict(name=name, est_I=est_I, est_E=est_E,
                      wc_I=wc_I, wc_E=wc_E, mc_cost_I=mc_cost_I, mc_cost_E=mc_cost_E,
                      ratio_mc=ratio_mc, ratio_wc=ratio_wc,
                      time_ratio=wc_I / wc_E))
    print(f"{name:16s} wall-clock(E)={wc_E:7.1f}s  wall-clock(I)={wc_I:8.1f}s  "
          f"time_ratio(I/E)={wc_I/wc_E:5.2f}x   "
          f"FOM_ratio: MC-cost={ratio_mc:6.3f}  wall-clock={ratio_wc:6.3f}")

time_ratios = np.array([row['time_ratio'] for row in rows])
print(f"\nmean I/E wall-clock time ratio across 17 seeds: {time_ratios.mean():.2f}x "
      f"(min {time_ratios.min():.2f}x, max {time_ratios.max():.2f}x)")

ratios_mc = np.array([row['ratio_mc'] for row in rows])
ratios_wc = np.array([row['ratio_wc'] for row in rows])
print(f"\nPoint FOM ratios (I/E), MC-cost basis:     mean={ratios_mc.mean():.3f}  "
      f"geo-mean={np.exp(np.log(ratios_mc).mean()):.3f}")
print(f"Point FOM ratios (I/E), wall-clock basis:  mean={ratios_wc.mean():.3f}  "
      f"geo-mean={np.exp(np.log(ratios_wc).mean()):.3f}")

# two-level bootstrap on the wall-clock-fair ratio, same convention as
# pooled_seed_bootstrap.py (outer=seeds, inner=replicas within each seed)
rng = np.random.default_rng(0)
N_BOOT = 10000
S = len(rows)
boot_geomean = []
for _ in range(N_BOOT):
    drawn = rng.integers(0, S, size=S)
    draw_ratios = []
    for i in drawn:
        row = rows[i]
        rI = row['est_I'][rng.integers(0, 10, size=10)]
        rE = row['est_E'][rng.integers(0, 10, size=10)]
        fI, _, _ = fom_from(rI, row['wc_I'])
        fE, _, _ = fom_from(rE, row['wc_E'])
        if np.isfinite(fI) and np.isfinite(fE) and fI > 0 and fE > 0:
            draw_ratios.append(fI / fE)
    if draw_ratios:
        boot_geomean.append(np.exp(np.mean(np.log(draw_ratios))))
boot_geomean = np.array(boot_geomean)
lo, med, hi = np.percentile(boot_geomean, [2.5, 50, 97.5])
print(f"\nTwo-level bootstrap on WALL-CLOCK-FAIR FOM ratio (geometric mean), "
      f"N={len(boot_geomean)} valid draws:")
print(f"  median={med:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
print(f"  excludes 1.0 (no-difference)? {'YES' if lo > 1.0 or hi < 1.0 else 'NO'}")
print(f"\n(for reference, the MC-cost-basis result already published: 1.19x, CI [0.84, 1.75])")
