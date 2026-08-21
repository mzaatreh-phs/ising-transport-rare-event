"""
Expand the WE[E] baseline from 30 to 90 independent replicas at the deep-rare
EA config (L=16, beyond=4, thresh=-1.119), to match the pooled 90
WE[I_theta] replicas (3 net seeds x 30) used in the paper's central 60/90 vs
12/30 comparison. Per chatgpt.md's point 7 (07_paper/), the original 12/30
baseline is a fixed, deterministic 30 replicas (base_seed=1194, identical in
every neural_importance.py run regardless of net-seed) -- reused three times
as the comparison point, never actually pooled to 90 like the learned side.

This script reruns ONLY WE[E] (skips naive/WE[m]/the expensive WE[I_theta])
with a disjoint seed block (base_seed=50000+), producing 60 genuinely new
independent replicas, then pools them with the original 30 (pulled straight
from the seed0 confirm JSON, which stores the full per-replica 'est' array)
into one balanced 90-replica WE[E] result.
"""
import json
import time
import numpy as np

from neural_importance import (
    make_couplings, calibrate_threshold, EnergyCoord, run_replicas,
    fom_from, PRESETS,
)

MODEL = "ea"
L = 16
T = 2.6
THRESH = -1.119
BEYOND = 1
ORIGINAL_JSON = "results_selfconsistent_deep_seed0_r30_confirm_ea_L16_full_selfconsistent.json"
OUT_JSON = "results_we_E_baseline_r90_pooled.json"
NEW_REPLICAS = 60
NEW_BASE_SEED = 50000
JOBS = 7

cfg = dict(PRESETS["full"])
cfg.update(n_bins=80, n_per_bin=40, we_iter=3000, we_burn=600, tau=2)

Jx, Jy = make_couplings(L, MODEL, seed=12345)
_, bulk_edge = calibrate_threshold(L, T, Jx, Jy, MODEL, beyond=BEYOND, seed=99, verbose=False)
rng_E = (THRESH - 4.0 / (L * L), bulk_edge)
print(f"bulk_edge={bulk_edge:.6f}  rng_E={rng_E}")

static = (L, T, Jx, Jy, MODEL, THRESH, cfg)
print(f"Running {NEW_REPLICAS} new independent WE[E] replicas "
      f"(base_seed={NEW_BASE_SEED}, jobs={JOBS}) ...")
t0 = time.time()
est_new, cost = run_replicas("we", static, EnergyCoord(Jx, Jy), cfg,
                             NEW_REPLICAS, jobs=JOBS,
                             base_seed=NEW_BASE_SEED, crange=rng_E)
wall_time_new = time.time() - t0
print(f"done in {wall_time_new:.0f}s")

with open(ORIGINAL_JSON) as f:
    orig = json.load(f)
orig_row = next(r for r in orig["rows"] if r["method"] == "WE[E]")
est_orig = np.array(orig_row["est"])
assert len(est_orig) == 30, f"expected 30 original replicas, got {len(est_orig)}"

est_pooled = np.concatenate([est_orig, est_new])
f_pooled, mean_pooled, relsd_pooled = fom_from(est_pooled, cost)
zeros_pooled = int((est_pooled == 0).sum())

out = dict(
    method="WE[E]",
    model=MODEL, L=L, T=T, thresh=THRESH,
    n_original=30, n_new=NEW_REPLICAS, n_pooled=len(est_pooled),
    original_base_seed=1194, new_base_seed=NEW_BASE_SEED,
    mean=float(mean_pooled), rel_sd=float(relsd_pooled),
    cost=cost, fom=float(f_pooled), zeros=zeros_pooled,
    positive=int(len(est_pooled) - zeros_pooled),
    est_original=est_orig.tolist(), est_new=est_new.tolist(),
    wall_time_new_replicas=wall_time_new,
)
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)

print(f"pooled WE[E]: {out['positive']}/{out['n_pooled']} event-positive "
      f"({100*out['positive']/out['n_pooled']:.1f}%), rel_sd={relsd_pooled:.3f}, "
      f"FOM={f_pooled:.3e}")
print(f"saved {OUT_JSON}")
