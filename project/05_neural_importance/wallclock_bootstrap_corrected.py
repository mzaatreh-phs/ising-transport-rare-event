#!/usr/bin/env python3
"""
Corrected two-level bootstrap for both the deep-rare and milestoning
wall-clock-fair FOM comparisons. Supersedes wallclock_confirm_final.py and
wallclock_confirm_final_milestone.py, which both charged the FULL pooled
baseline wall-clock time (e.g. 170 replicas' worth) against the learned
coordinate's NATIVE small-sample cost -- an apples-to-oranges cost charge
that inflated WE[E]'s effective compute cost far beyond what WE[I_theta]
paid, cratering its FOM independent of any real effect (milestoning gave an
implausible 8.3x before this fix).

Correct approach: use the larger pooled WE[E] sample only to get a
better-calibrated estimate of what a NATIVE-SIZED measurement's rel_sd looks
like (bootstrap-resample blocks of the original replica count from the
larger pool), while charging the NATIVE, un-inflated measured wall-clock
cost for that block size. This uses the extra replicas to reduce
lucky/unlucky small-draw risk without violating the fixed-replica-budget
assumption FOM's cost-normalization depends on.
"""
import json
import numpy as np
from neural_importance import fom_from

print("=" * 70)
print("DEEP-RARE beyond=4 (corrected)")
print("=" * 70)

CONFIRM_FILES = {
    'seed0': 'results_selfconsistent_deep_seed0_r30_confirm_ea_L16_full_selfconsistent.json',
    'seed1': 'results_selfconsistent_deep_seed1_r30_confirm_ea_L16_full_selfconsistent.json',
    'seed2': 'results_selfconsistent_deep_seed2_r30_confirm_ea_L16_full_selfconsistent.json',
}
seeds = {}
we_E_native_wall_times = []
for name, path in CONFIRM_FILES.items():
    d = json.load(open(path))
    rows = {r['method']: r for r in d['rows']}
    rI, rE = rows['WE[I_theta]'], rows['WE[E]']
    seeds[name] = dict(est=np.array(rI['est']), wall_time=rI['wall_time'])
    we_E_native_wall_times.append(rE['wall_time'])
we_E_native_wall_time = float(np.mean(we_E_native_wall_times))
print(f"WE[E] native (R=30) wall_time, mean of 3 measurements: {we_E_native_wall_time:.0f}s")

baseline = json.load(open('results_we_E_baseline_r90_pooled.json'))
est_E_pool = np.array(baseline['est_original'] + baseline['est_new'])
NATIVE_R = 30

rng = np.random.default_rng(0)
names = list(seeds.keys())
S = len(names)
N_BOOT = 10000
boot = []
for _ in range(N_BOOT):
    drawn = rng.integers(0, S, size=S)
    ratios = []
    for i in drawn:
        n = names[i]
        s = seeds[n]
        rI = s['est'][rng.integers(0, NATIVE_R, size=NATIVE_R)]
        rE = est_E_pool[rng.integers(0, len(est_E_pool), size=NATIVE_R)]  # native-size draw FROM the larger pool
        fI, _, _ = fom_from(rI, s['wall_time'])
        fE, _, _ = fom_from(rE, we_E_native_wall_time)  # native, un-inflated cost
        if np.isfinite(fI) and np.isfinite(fE) and fI > 0 and fE > 0:
            ratios.append(fI / fE)
    if ratios:
        boot.append(np.exp(np.mean(np.log(ratios))))
boot = np.array(boot)
lo, med, hi = np.percentile(boot, [2.5, 50, 97.5])
print(f"CORRECTED (native-size draws from larger pool, native cost): "
      f"median={med:.3f}  95% CI=[{lo:.3f},{hi:.3f}]  N={len(boot)}")
print(f"excludes 1.0? {'YES' if lo > 1.0 or hi < 1.0 else 'NO'}")

print()
print("=" * 70)
print("MILESTONING beyond=1 (corrected)")
print("=" * 70)

SEED_FILES = {f"seed{s}": f"results_seed{s}_walltime_confirm.json"
              for s in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
mseeds = {}
we_E_native_wall_times_m = []
for name, path in SEED_FILES.items():
    d = json.load(open(path))
    r = d['results']
    rI, rE = r['WE[I_theta]'], r['WE[E]']
    mseeds[name] = dict(est=np.array(rI['est']), wall_time=rI['wall_time'])
    we_E_native_wall_times_m.append(rE['wall_time'])
we_E_native_wall_time_m = float(np.mean(we_E_native_wall_times_m))
print(f"WE[E] native (R=10) wall_time, mean of 16 measurements: {we_E_native_wall_time_m:.0f}s")

mbaseline = json.load(open('results_milestone_we_E_baseline_r170_pooled.json'))
est_E_pool_m = np.array(mbaseline['est_original'] + mbaseline['est_new'])
NATIVE_R_M = 10

mnames = list(mseeds.keys())
SM = len(mnames)
rng2 = np.random.default_rng(0)
boot_m = []
for _ in range(N_BOOT):
    drawn = rng2.integers(0, SM, size=SM)
    ratios = []
    for i in drawn:
        n = mnames[i]
        s = mseeds[n]
        rI = s['est'][rng2.integers(0, NATIVE_R_M, size=NATIVE_R_M)]
        rE = est_E_pool_m[rng2.integers(0, len(est_E_pool_m), size=NATIVE_R_M)]
        fI, _, _ = fom_from(rI, s['wall_time'])
        fE, _, _ = fom_from(rE, we_E_native_wall_time_m)
        if np.isfinite(fI) and np.isfinite(fE) and fI > 0 and fE > 0:
            ratios.append(fI / fE)
    if ratios:
        boot_m.append(np.exp(np.mean(np.log(ratios))))
boot_m = np.array(boot_m)
lo_m, med_m, hi_m = np.percentile(boot_m, [2.5, 50, 97.5])
print(f"CORRECTED (native-size draws from larger pool, native cost): "
      f"median={med_m:.3f}  95% CI=[{lo_m:.3f},{hi_m:.3f}]  N={len(boot_m)}")
print(f"excludes 1.0? {'YES' if lo_m > 1.0 or hi_m < 1.0 else 'NO'}")

# sanity: what does the point estimate look like using just the pooled rel_sd
# (not inflated cost) -- i.e. a better rel_sd estimate at native cost
fE_point, meanE_point, relsdE_point = fom_from(est_E_pool_m[:NATIVE_R_M*17], we_E_native_wall_time_m)
print(f"\n(sanity) pooled-sample rel_sd if we DON'T touch cost: using full 170 array's "
      f"rel_sd={np.std(est_E_pool_m, ddof=1)/est_E_pool_m.mean():.3f} at native cost "
      f"{we_E_native_wall_time_m:.0f}s would give an inconsistent hybrid -- NOT done above, "
      f"noted only to show why block-resampling at native R is the right call instead.")
