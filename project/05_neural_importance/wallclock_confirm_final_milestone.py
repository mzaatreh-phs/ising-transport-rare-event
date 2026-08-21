#!/usr/bin/env python3
"""Final wall-clock-fair milestoning (beyond=1) recomputation using
first-class measured wall_time (not log-timestamp reconstruction) from the
16 seed confirmatory reruns (2026-08-19, seed1/seed5 from 2026-08-13), plus
the balanced 170-replica WE[E] baseline from we_baseline_expand_milestone.py.
'unseeded_run2' is intentionally excluded (predates the torch.manual_seed()
fix, not legitimately reproducible under current code).
"""
import json
import numpy as np
from neural_importance import fom_from

SEED_FILES = {
    f"seed{s}": f"results_seed{s}_walltime_confirm.json"
    for s in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
}
BASELINE_FILE = "results_milestone_we_E_baseline_r170_pooled.json"

seeds = {}
we_E_10_wall_times = []
for name, path in SEED_FILES.items():
    d = json.load(open(path))
    r = d["results"]
    rI, rE = r["WE[I_theta]"], r["WE[E]"]
    seeds[name] = dict(est=np.array(rI["est"]), wall_time=rI["wall_time"],
                       zeros=rI["zeros"])
    we_E_10_wall_times.append(rE["wall_time"])
    print(f"{name:8s}: WE[I_theta] wall_time={rI['wall_time']:7.0f}s  "
          f"hits={10-rI['zeros']}/10   WE[E] wall_time={rE['wall_time']:6.0f}s "
          f"(fixed 10-replica baseline, re-measured)  "
          f"MC-cost ratio(I/E)={rI['fom']/rE['fom']:.3f}")

we_E_10_wall_time_mean = float(np.mean(we_E_10_wall_times))
print(f"\nWE[E] 10-replica wall_time across {len(we_E_10_wall_times)} confirm runs: "
      f"mean={we_E_10_wall_time_mean:.0f}s (range {min(we_E_10_wall_times):.0f}-"
      f"{max(we_E_10_wall_times):.0f}s)")

baseline = json.load(open(BASELINE_FILE))
est_E_170 = np.array(baseline["est_original"] + baseline["est_new"])
assert len(est_E_170) == 170
wall_time_E_170 = we_E_10_wall_time_mean + baseline["wall_time_new_replicas"]
print(f"WE[E] 170-replica pooled wall_time = {we_E_10_wall_time_mean:.0f}s (10, mean) "
      f"+ {baseline['wall_time_new_replicas']:.0f}s (new 160) = {wall_time_E_170:.0f}s")

fom_E_wc, mean_E, relsd_E = fom_from(est_E_170, wall_time_E_170)
print(f"\nWE[E] pooled (n=170): rel_sd={relsd_E:.3f}  mean={mean_E:.3e}  "
      f"wall-clock FOM={fom_E_wc:.3e}")

print("\nPer-seed point ratios (I_theta / balanced-170 E), wall-clock-fair basis:")
point_ratios = {}
for name, s in seeds.items():
    fom_I_wc, mean_I, relsd_I = fom_from(s["est"], s["wall_time"])
    ratio = fom_I_wc / fom_E_wc
    point_ratios[name] = ratio
    print(f"  {name:8s}: wall_time_ratio(I/E, vs 10-rep mean)="
          f"{s['wall_time']/we_E_10_wall_time_mean:.3f}x  "
          f"ratio(I/balanced-E)={ratio:.3f}")

gm = float(np.exp(np.mean(np.log(list(point_ratios.values())))))
print(f"\nGeometric mean of {len(point_ratios)} point ratios (vs balanced 170-replica E): "
      f"{gm:.3f}")

# Two-level bootstrap: outer resample the 16 seeds, inner resample 10
# replicas for WE[I_theta] and 170 for the pooled WE[E], measured wall_time
# as the fixed cost basis throughout (same convention as
# pooled_seed_bootstrap.py / wallclock_confirm_final.py).
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
        rI = s["est"][rng.integers(0, 10, size=10)]
        rE = est_E_170[rng.integers(0, 170, size=170)]
        fI, _, _ = fom_from(rI, s["wall_time"])
        fE, _, _ = fom_from(rE, wall_time_E_170)
        if np.isfinite(fI) and np.isfinite(fE) and fI > 0 and fE > 0:
            ratios.append(fI / fE)
    if ratios:
        boot.append(np.exp(np.mean(np.log(ratios))))
boot = np.array(boot)
lo, med, hi = np.percentile(boot, [2.5, 50, 97.5])
print(f"\nTwo-level bootstrap (measured wall_time, balanced 170-replica E baseline):")
print(f"  median={med:.3f}  95% CI=[{lo:.3f},{hi:.3f}]  N={len(boot)}")
print(f"  excludes 1.0? {'YES' if lo > 1.0 or hi < 1.0 else 'NO'}")

# for reference: what the ORIGINAL (fixed 10-replica E, mean measured wall_time) basis gives
fom_E_wc_10, _, relsd_E_10 = fom_from(est_E_170[:10], we_E_10_wall_time_mean)
boot_orig = []
rng2 = np.random.default_rng(1)
for _ in range(N_BOOT):
    drawn = rng2.integers(0, S, size=S)
    ratios = []
    for i in drawn:
        n = names[i]
        s = seeds[n]
        rI = s["est"][rng2.integers(0, 10, size=10)]
        rE = est_E_170[:10][rng2.integers(0, 10, size=10)]
        fI, _, _ = fom_from(rI, s["wall_time"])
        fE, _, _ = fom_from(rE, we_E_10_wall_time_mean)
        if np.isfinite(fI) and np.isfinite(fE) and fI > 0 and fE > 0:
            ratios.append(fI / fE)
    if ratios:
        boot_orig.append(np.exp(np.mean(np.log(ratios))))
boot_orig = np.array(boot_orig)
lo2, med2, hi2 = np.percentile(boot_orig, [2.5, 50, 97.5])
print(f"\n(For comparison, measured wall_time but UNBALANCED fixed-10 E baseline: "
      f"median={med2:.3f}  95% CI=[{lo2:.3f},{hi2:.3f}])")
