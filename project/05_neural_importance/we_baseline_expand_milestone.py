"""
Expand the milestoning WE[E] baseline (beyond=1, tau=2, custom preset) from a
fixed 10 replicas to a balanced 170 (matching the total pooled I_theta budget
across the 17-seed sweep: 17 seeds x 10 replicas), for the same reason as
we_baseline_expand.py did for the deep-rare section: pooled_seed_bootstrap.py
uses the SAME deterministic 10-replica WE[E] draw (seeds 0..9, base couplings
seed=0) as the denominator for every one of the 17 seeds' FOM ratios, so any
noise in that one small sample propagates identically into all 17 -- a much
larger, independent sample gives a far more trustworthy shared baseline.

Reproduces run_milestone.py's pilot/calibration exactly (make_couplings
seed=0, pilot rng seed=0, beyond=1) to get an identical thresh/bulk_edge,
then runs 160 NEW independent WE[E] replicas (disjoint seed block) and pools
with the original 10 (pulled from results_seedtest_run1.json, which is
bit-identical to every other seedN.json's WE[E] row).
"""
import json
import numpy as np

from neural_importance import (
    make_couplings, sweep_population, energy_per_spin, EnergyCoord,
    run_replicas, fom_from,
)

MODEL = "ea"
L = 16
T = 2.6
BEYOND = 1
CFG = dict(n_bins=50, n_per_bin=40, tau=2, we_iter=1000, we_burn=200)
ORIGINAL_JSON = "results_seedtest_run1.json"
OUT_JSON = "results_milestone_we_E_baseline_r170_pooled.json"
NEW_REPLICAS = 160
NEW_BASE_SEED = 10000
JOBS = 7

Jx, Jy = make_couplings(L, MODEL, seed=0)

rng_pilot = np.random.default_rng(seed=0)
S_pilot = rng_pilot.choice([-1, 1], size=(CFG.get("collect_walkers", 128), L, L))
pilot_energies = []
for _ in range(200):
    S_pilot = sweep_population(S_pilot, T, Jx, Jy, rng_pilot)
    for s in S_pilot:
        pilot_energies.append(energy_per_spin(s[np.newaxis, ...], Jx, Jy)[0])
pilot_energies = np.array(pilot_energies)
bulk_mean = np.mean(pilot_energies)
extreme = np.min(pilot_energies)
bulk_edge = float(pilot_energies.mean() + 2 * pilot_energies.std())
step = (bulk_mean - extreme) / 10
thresh = extreme - BEYOND * step
print(f"bulk_mean={bulk_mean:.6f} extreme={extreme:.6f} bulk_edge={bulk_edge:.6f} "
      f"thresh={thresh:.6f}")

rng_E = (thresh - 4.0 / (L * L), bulk_edge)
static = (L, T, Jx, Jy, MODEL, thresh, CFG)
print(f"rng_E={rng_E}")

print(f"Running {NEW_REPLICAS} new independent WE[E] replicas "
      f"(base_seed={NEW_BASE_SEED}, jobs={JOBS}) ...")
import time
t0 = time.time()
est_new, cost = run_replicas("we", static, EnergyCoord(Jx, Jy), CFG,
                             NEW_REPLICAS, jobs=JOBS,
                             base_seed=NEW_BASE_SEED, crange=rng_E)
wall_time_new = time.time() - t0
print(f"done in {wall_time_new:.0f}s")

with open(ORIGINAL_JSON) as f:
    orig = json.load(f)
est_orig = np.array(orig["results"]["WE[E]"]["est"])
assert len(est_orig) == 10, f"expected 10 original replicas, got {len(est_orig)}"

est_pooled = np.concatenate([est_orig, est_new])
f_pooled, mean_pooled, relsd_pooled = fom_from(est_pooled, cost)
zeros_pooled = int((est_pooled == 0).sum())

out = dict(
    method="WE[E]", model=MODEL, L=L, T=T, thresh=thresh, bulk_edge=bulk_edge,
    n_original=10, n_new=NEW_REPLICAS, n_pooled=len(est_pooled),
    original_base_seed=0, new_base_seed=NEW_BASE_SEED,
    mean=float(mean_pooled), rel_sd=float(relsd_pooled),
    cost=cost, fom=float(f_pooled), zeros=zeros_pooled,
    positive=int(len(est_pooled) - zeros_pooled),
    est_original=est_orig.tolist(), est_new=est_new.tolist(),
    wall_time_new_replicas=wall_time_new,
)
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)

print(f"pooled WE[E]: mean={mean_pooled:.4e}  rel_sd={relsd_pooled:.3f}  "
      f"FOM={f_pooled:.3e}  ({out['positive']}/{out['n_pooled']} positive)")
print(f"saved {OUT_JSON}")
