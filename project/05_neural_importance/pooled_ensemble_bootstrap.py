#!/usr/bin/env python3
"""Two-level bootstrap over (ensemble-training-run, replica) for the
deep-ensemble WE[I_theta] (K=5 nets, mean prediction) vs WE[E] FOM ratio,
same canonical config as the single-net seed sweep (beyond=1, tau=2, L=16).

Mirrors pooled_seed_bootstrap.py exactly, but pooling the 5 --ensemble 5 runs
(results_ens20..24.json) instead of the 17 single-net runs, so the two CIs
are directly comparable: same aggregation method (geometric-mean-of-ratios
and median-of-ratios), same bootstrap depth, same FOM definition.
"""
import json
import numpy as np

from neural_importance import fom_from

FILES = {
    'ens20': 'results_ens20.json',
    'ens21': 'results_ens21.json',
    'ens22': 'results_ens22.json',
    'ens23': 'results_ens23.json',
    'ens24': 'results_ens24.json',
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
print(f"\nn_ensemble_runs={S}  mean(point ratios)={point_ratios.mean():.4f}  "
      f"median={np.median(point_ratios):.4f}  "
      f"sd={point_ratios.std(ddof=1) if S > 1 else float('nan'):.4f}")

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
