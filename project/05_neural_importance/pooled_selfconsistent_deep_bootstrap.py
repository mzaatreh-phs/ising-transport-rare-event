#!/usr/bin/env python3
"""Bootstrap for the self-consistency-objective WE[I_theta] vs WE[E] comparison
at beyond=4 (deep-rare regime, tau=2, n_bins=80, we_iter=3000 -- see HANDOFF.md
section 12). Three independent training seeds of WE[I_theta] vs the single
fixed WE[E] baseline (WE[E]/WE[m]/naive do not depend on the network and are
bit-identical across all three result files -- there has only ever been ONE
real measurement of WE[E] at this config, not three). Outer resample draws a
training seed for I_theta; inner resample draws with replacement from that
seed's 10 replicas AND independently from WE[E]'s fixed 10 replicas, so the
WE[E] side of the ratio also carries its own real Monte Carlo uncertainty
rather than being treated as a fixed constant.
"""
import json
import numpy as np

from neural_importance import fom_from

FILES = {
    'seed0': 'results_selfconsistent_deep_ea_L16_full_selfconsistent.json',
    'seed1': 'results_selfconsistent_deep_seed1_ea_L16_full_selfconsistent.json',
    'seed2': 'results_selfconsistent_deep_seed2_ea_L16_full_selfconsistent.json',
}

seeds = {}
we_E = None
for name, path in FILES.items():
    d = json.load(open(path))
    rows = {r['method']: r for r in d['rows']}
    est_I = np.array(rows['WE[I_theta]']['est'], dtype=float)
    cost_I = rows['WE[I_theta]']['cost']
    fom_I, mean_I, relsd_I = fom_from(est_I, cost_I)
    seeds[name] = dict(est=est_I, cost=cost_I, fom=fom_I,
                       zeros=int((est_I == 0).sum()))
    print(f"{name:8s} WE[I_theta] zeros={seeds[name]['zeros']}/10  "
          f"FOM={fom_I:.3e}  mean={mean_I:.3e}  rel.sd={relsd_I:.3f}")
    est_E = np.array(rows['WE[E]']['est'], dtype=float)
    cost_E = rows['WE[E]']['cost']
    if we_E is None:
        we_E = dict(est=est_E, cost=cost_E)
    else:
        assert np.array_equal(we_E['est'], est_E), \
            "WE[E] differs across seed files -- assumption broken, investigate"

fom_E, mean_E, relsd_E = fom_from(we_E['est'], we_E['cost'])
print(f"\nWE[E] (fixed, all seeds identical) zeros={int((we_E['est']==0).sum())}/10  "
      f"FOM={fom_E:.3e}  mean={mean_E:.3e}  rel.sd={relsd_E:.3f}")

names = list(seeds.keys())
S = len(names)
point_ratios = np.array([seeds[n]['fom'] / fom_E for n in names])
print(f"\npoint ratios (I_theta/E) by seed: {point_ratios}")
print(f"mean={point_ratios.mean():.4f}  median={np.median(point_ratios):.4f}")

# Reliability comparison (zero-replica counts): I_theta's hit rate across the
# 3 independent networks, pooled, vs WE[E]'s single fixed 10-replica draw.
hits_I = sum(10 - seeds[n]['zeros'] for n in names)
hits_E = 10 - int((we_E['est'] == 0).sum())
print(f"\nreliability: WE[I_theta] pooled hits = {hits_I}/30 across 3 independent "
      f"networks ({[10-seeds[n]['zeros'] for n in names]})")
print(f"             WE[E] hits = {hits_E}/10 (single fixed baseline, not "
      f"independently replicated)")

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
        rI = s['est'][rng.integers(0, 10, size=10)]
        rE = we_E['est'][rng.integers(0, 10, size=10)]
        fI, _, _ = fom_from(rI, s['cost'])
        fE, _, _ = fom_from(rE, we_E['cost'])
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
    print(f"\nTwo-level bootstrap, aggregate={label}, N={len(arr)} valid draws "
          f"(of {N_BOOT} attempted -- some discarded on fE==0 draws):")
    print(f"  median={med:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
    print(f"  excludes 1.0 (no-difference)? {'YES' if lo > 1.0 or hi < 1.0 else 'NO'}")
