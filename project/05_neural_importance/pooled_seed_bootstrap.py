#!/usr/bin/env python3
"""Two-level bootstrap over (network-init seed, replica) for WE[I_theta] vs
WE[E] FOM ratio at beyond=1, tau=2. Combines every seed run that used the
correct post-fix config (excludes results_milestone_beyond1_boot.json, which
predates the tau=2 correction and has a visibly different WE[E] FOM/mean).

Outer resample: draw S seeds with replacement from the available seed runs.
Inner resample: within each drawn seed, resample its 10 replica pi estimates
(both methods) with replacement and recompute FOM. Aggregate statistic per
bootstrap draw = mean of the S per-seed ratios. Repeat N_BOOT times.
"""
import json
import numpy as np

from neural_importance import fom_from

FILES = {
    'seed0':   'results_seedtest_run1.json',   # net-seed=0, verified bit-identical twice
    'unseeded_run2': 'results_milestone_beyond1_boot_tau2.json',  # the -17% run
    'seed1':   'results_seed1.json',
    'seed2':   'results_seed2.json',
    'seed3':   'results_seed3.json',
    'seed4':   'results_seed4.json',
    'seed5':   'results_seed5.json',
    'seed6':   'results_seed6.json',
    'seed7':   'results_seed7.json',
    'seed8':   'results_seed8.json',
    'seed9':   'results_seed9.json',
    'seed10':  'results_seed10.json',
    'seed11':  'results_seed11.json',
    'seed12':  'results_seed12.json',
    'seed13':  'results_seed13.json',
    'seed14':  'results_seed14.json',
    'seed15':  'results_seed15.json',
}

seeds = {}
for name, path in FILES.items():
    d = json.load(open(path))
    r = d['results']
    est_I = np.array(r['WE[I_theta]']['est'], dtype=float)
    cost_I = r['WE[I_theta]']['cost']
    est_E = np.array(r['WE[E]']['est'], dtype=float)
    cost_E = r['WE[E]']['cost']
    fom_I, _, _ = fom_from(est_I, cost_I)
    fom_E, _, _ = fom_from(est_E, cost_E)
    seeds[name] = dict(est_I=est_I, cost_I=cost_I, est_E=est_E, cost_E=cost_E,
                        point_ratio=fom_I / fom_E)
    print(f"{name:16s} FOM[I_theta]={fom_I:.3e}  FOM[E]={fom_E:.3e}  "
          f"ratio={fom_I/fom_E:.4f}  ({100*(fom_I/fom_E-1):+.1f}%)")

names = list(seeds.keys())
S = len(names)
point_ratios = np.array([seeds[n]['point_ratio'] for n in names])
print(f"\nn_seeds={S}  mean(point ratios)={point_ratios.mean():.4f}  "
      f"median={np.median(point_ratios):.4f}  sd={point_ratios.std(ddof=1):.4f}")

rng = np.random.default_rng(0)
N_BOOT = 10000
boot_geomean = []
boot_median = []
for _ in range(N_BOOT):
    drawn = rng.integers(0, S, size=S)
    ratios = []
    for i in drawn:
        n = names[i]
        s = seeds[n]
        rI = s['est_I'][rng.integers(0, 10, size=10)]
        rE = s['est_E'][rng.integers(0, 10, size=10)]
        fI, _, _ = fom_from(rI, s['cost_I'])
        fE, _, _ = fom_from(rE, s['cost_E'])
        if np.isfinite(fI) and np.isfinite(fE) and fE > 0 and fI > 0:
            ratios.append(fI / fE)
    if ratios:
        ratios = np.array(ratios)
        boot_geomean.append(np.exp(np.mean(np.log(ratios))))
        boot_median.append(np.median(ratios))

for label, arr in [('geometric-mean-of-ratios', boot_geomean),
                    ('median-of-ratios', boot_median)]:
    arr = np.array(arr)
    lo, med, hi = np.percentile(arr, [2.5, 50, 97.5])
    print(f"\nTwo-level bootstrap, aggregate={label}, N={len(arr)} valid draws:")
    print(f"  median={med:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
    print(f"  excludes 1.0 (no-difference)? {'YES' if lo > 1.0 or hi < 1.0 else 'NO'}")
