#!/usr/bin/env python3
"""Final wall-clock-fair deep-rare (beyond=4) recomputation using FIRST-CLASS
measured wall_time (not log-timestamp reconstruction) from the seed0/1/2
confirmatory reruns (2026-08-18/19), plus the balanced 90-replica WE[E]
baseline from we_baseline_expand.py (also 2026-08-19), replacing the
previously-published fixed 30-replica WE[E] baseline. See
HANDOFF_wallclock_correction.md for the reconstruction-based numbers this
supersedes.
"""
import json
import numpy as np
from neural_importance import fom_from

CONFIRM_FILES = {
    'seed0': 'results_selfconsistent_deep_seed0_r30_confirm_ea_L16_full_selfconsistent.json',
    'seed1': 'results_selfconsistent_deep_seed1_r30_confirm_ea_L16_full_selfconsistent.json',
    'seed2': 'results_selfconsistent_deep_seed2_r30_confirm_ea_L16_full_selfconsistent.json',
}
BASELINE_FILE = 'results_we_E_baseline_r90_pooled.json'

seeds = {}
we_E_30_wall_times = []
for name, path in CONFIRM_FILES.items():
    d = json.load(open(path))
    rows = {r['method']: r for r in d['rows']}
    rI, rE = rows['WE[I_theta]'], rows['WE[E]']
    seeds[name] = dict(est=np.array(rI['est']), wall_time=rI['wall_time'],
                       zeros=rI['zeros'])
    we_E_30_wall_times.append(rE['wall_time'])
    print(f"{name}: WE[I_theta] wall_time={rI['wall_time']:.0f}s  "
          f"hits={30-rI['zeros']}/30   WE[E] wall_time={rE['wall_time']:.0f}s "
          f"(fixed 30-replica baseline, re-measured)")

we_E_30_wall_time_mean = float(np.mean(we_E_30_wall_times))
print(f"\nWE[E] 30-replica wall_time across 3 confirm runs: {we_E_30_wall_times} "
      f"-> mean={we_E_30_wall_time_mean:.0f}s")

baseline = json.load(open(BASELINE_FILE))
est_E_90 = np.array(baseline['est_original'] + baseline['est_new'])
assert len(est_E_90) == 90
wall_time_E_90 = we_E_30_wall_time_mean + baseline['wall_time_new_replicas']
print(f"WE[E] 90-replica pooled wall_time = {we_E_30_wall_time_mean:.0f}s (30, mean) "
      f"+ {baseline['wall_time_new_replicas']:.0f}s (new 60) = {wall_time_E_90:.0f}s")

fom_E_wc, mean_E, relsd_E = fom_from(est_E_90, wall_time_E_90)
print(f"\nWE[E] pooled (n=90): hits={int((est_E_90>0).sum())}/90  rel_sd={relsd_E:.3f}  "
      f"wall-clock FOM={fom_E_wc:.3e}")

print("\nPer-seed point ratios (I_theta / balanced-90 E), wall-clock-fair basis:")
point_ratios = {}
for name, s in seeds.items():
    fom_I_wc, mean_I, relsd_I = fom_from(s['est'], s['wall_time'])
    ratio = fom_I_wc / fom_E_wc
    point_ratios[name] = ratio
    print(f"  {name}: wall_time_ratio(I/E)={s['wall_time']/we_E_30_wall_time_mean:.3f}x  "
          f"FOM_wc(I)={fom_I_wc:.3e}  ratio(I/balanced-E)={ratio:.3f}")

gm = float(np.exp(np.mean(np.log(list(point_ratios.values())))))
print(f"\nGeometric mean of 3 point ratios (vs balanced 90-replica E): {gm:.3f}")

# --- two-level bootstrap: outer resample 3 I_theta seeds, inner resample
# replicas within each (30 for I_theta, 90 for the pooled E baseline), same
# convention as wallclock_kimcai_deeprare.py / pooled_selfconsistent_deep_bootstrap.py
rng = np.random.default_rng(0)
names = list(seeds.keys())
N_BOOT = 10000
boot = []
for _ in range(N_BOOT):
    drawn = rng.integers(0, 3, size=3)
    ratios = []
    for i in drawn:
        n = names[i]
        s = seeds[n]
        rI = s['est'][rng.integers(0, 30, size=30)]
        rE = est_E_90[rng.integers(0, 90, size=90)]
        fI, _, _ = fom_from(rI, s['wall_time'])
        fE, _, _ = fom_from(rE, wall_time_E_90)
        if np.isfinite(fI) and np.isfinite(fE) and fI > 0 and fE > 0:
            ratios.append(fI / fE)
    if ratios:
        boot.append(np.exp(np.mean(np.log(ratios))))
boot = np.array(boot)
lo, med, hi = np.percentile(boot, [2.5, 50, 97.5])
print(f"\nTwo-level bootstrap (measured wall_time, balanced 90-replica E baseline):")
print(f"  median={med:.3f}  95% CI=[{lo:.3f},{hi:.3f}]  N={len(boot)}")
print(f"  excludes 1.0? {'YES' if lo > 1.0 or hi < 1.0 else 'NO'}")

# --- reliability, balanced Fisher exact test (hand-rolled, no scipy in venv)
import math
def fisher_exact_2x2(a, b, c, d):
    n = a + b + c + d
    row1, row2, col1 = a + b, c + d, a + c
    def hyper_p(x):
        return (math.comb(row1, x) * math.comb(row2, col1 - x)) / math.comb(n, col1)
    p_obs = hyper_p(a)
    lo_x, hi_x = max(0, col1 - row2), min(row1, col1)
    return sum(hyper_p(x) for x in range(lo_x, hi_x + 1) if hyper_p(x) <= p_obs * (1 + 1e-9))

hits_I = sum(30 - seeds[n]['zeros'] for n in names)
hits_E = int((est_E_90 > 0).sum())
p_balanced = fisher_exact_2x2(hits_I, 90 - hits_I, hits_E, 90 - hits_E)
p_original = fisher_exact_2x2(hits_I, 90 - hits_I, 12, 18)
print(f"\nReliability: WE[I_theta] pooled {hits_I}/90 ({100*hits_I/90:.1f}%) vs "
      f"WE[E] balanced {hits_E}/90 ({100*hits_E/90:.1f}%)")
print(f"  Fisher exact p (balanced 90 vs 90) = {p_balanced:.4f}")
print(f"  Fisher exact p (original, imbalanced 90 vs 30, for reference) = {p_original:.4f}")
