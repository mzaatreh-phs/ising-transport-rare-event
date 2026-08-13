#!/usr/bin/env python3
"""Task 3 for HANDOFF_ensemble_cost.md: variance-components decomposition,
zero new compute, uses data already on disk.

Question: of the observed spread in log(FOM_I/FOM_E) across the 17 seeds,
how much is TRUE seed-to-seed (network-init) variance vs R=10 REPLICA
sampling noise within each seed? Determines whether narrowing the CI is
cheaper via more seeds (expensive, ~30min retrain each) or more replicas
on existing cached seeds (near-free per the handoff's own framing).

Method: one-way random-effects decomposition.
  - point_i = log(fom_I/fom_E) computed from seed i's own 10 replicas
    (this is seed_effect_i + replica_noise_i).
  - Var_across(point_i) over the 17 seeds = V_between + V_within (confounded).
  - V_within estimated directly per seed via inner-only bootstrap (resample
    that seed's 10 replicas with replacement, holding the seed fixed),
    then averaged over the 17 seeds.
  - V_between = max(0, Var_across(point_i) - V_within)  [ANOVA subtraction]
"""
import json
import numpy as np
import sys, os
sys.path.insert(0, os.path.expanduser("~/ising_transport_project/project/05_neural_importance"))
os.chdir(os.path.expanduser("~/ising_transport_project/project/05_neural_importance"))

from neural_importance import fom_from

FILES = {
    'seed0':   'results_seedtest_run1.json',
    'unseeded_run2': 'results_milestone_beyond1_boot_tau2.json',
    'seed1':   'results_seed1.json',  'seed2':  'results_seed2.json',
    'seed3':   'results_seed3.json',  'seed4':  'results_seed4.json',
    'seed5':   'results_seed5.json',  'seed6':  'results_seed6.json',
    'seed7':   'results_seed7.json',  'seed8':  'results_seed8.json',
    'seed9':   'results_seed9.json',  'seed10': 'results_seed10.json',
    'seed11':  'results_seed11.json', 'seed12': 'results_seed12.json',
    'seed13':  'results_seed13.json', 'seed14': 'results_seed14.json',
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
    seeds[name] = dict(est_I=est_I, cost_I=cost_I, est_E=est_E, cost_E=cost_E)

names = list(seeds.keys())
S = len(names)
R = 10  # replicas per seed in the existing data

# --- point estimate per seed (full 10 replicas), in log space ---
log_points = []
for n in names:
    s = seeds[n]
    fI, _, _ = fom_from(s['est_I'], s['cost_I'])
    fE, _, _ = fom_from(s['est_E'], s['cost_E'])
    log_points.append(np.log(fI / fE))
log_points = np.array(log_points)
across_seed_var = log_points.var(ddof=1)

print(f"n_seeds={S}, R={R} replicas/seed")
print(f"point ratios (exp of log_points): {np.round(np.exp(log_points), 3)}")
print(f"Var_across(log ratio) [between+within confounded] = {across_seed_var:.5f}  "
      f"(sd={np.sqrt(across_seed_var):.4f})\n")

# --- within-seed variance via inner-only bootstrap, per seed ---
rng = np.random.default_rng(0)
N_INNER = 4000
within_vars = []
for n in names:
    s = seeds[n]
    logs = []
    for _ in range(N_INNER):
        rI = s['est_I'][rng.integers(0, R, size=R)]
        rE = s['est_E'][rng.integers(0, R, size=R)]
        fI, _, _ = fom_from(rI, s['cost_I'])
        fE, _, _ = fom_from(rE, s['cost_E'])
        if np.isfinite(fI) and np.isfinite(fE) and fI > 0 and fE > 0:
            logs.append(np.log(fI / fE))
    logs = np.array(logs)
    within_vars.append(logs.var(ddof=1))
within_vars = np.array(within_vars)
mean_within_var = within_vars.mean()

print("per-seed within-seed (replica-noise) sd of log ratio:")
for n, v in zip(names, within_vars):
    print(f"  {n:16s} sd={np.sqrt(v):.4f}")
print(f"\nmean within-seed variance V_within = {mean_within_var:.5f} (sd={np.sqrt(mean_within_var):.4f})")

V_between = max(0.0, across_seed_var - mean_within_var)
print(f"V_between (ANOVA subtraction, floored at 0) = {V_between:.5f} (sd={np.sqrt(V_between):.4f})")

frac_inner = mean_within_var / across_seed_var
frac_between = V_between / across_seed_var
print(f"\nFraction of observed across-seed spread attributable to:")
print(f"  within-seed replica noise (R=10): {100*frac_inner:.1f}%")
print(f"  true between-seed (init) variance: {100*frac_between:.1f}%")

# --- sanity check against the actual two-level bootstrap SE (mean-of-log-ratio) ---
# Var(mean of S seeds, each with its own inner resample) ~= V_between/S + V_within/S
predicted_se_mean = np.sqrt(V_between / S + mean_within_var / S)
print(f"\nPredicted SE of pooled mean(log ratio) over these {S} seeds (ANOVA model): "
      f"{predicted_se_mean:.4f}")
empirical_se_mean = log_points.std(ddof=1) / np.sqrt(S)
print(f"Naive empirical SE (std(log_points)/sqrt(S), ignores decomposition): "
      f"{empirical_se_mean:.4f}  [should be close, cross-check]")

# --- cost-optimal (n_seeds, R) to exclude 1.0 ---
print("\n--- cost-optimal (n_seeds, R) allocation ---")
mean_log_ratio = log_points.mean()  # current point estimate, ~log(1.19)
print(f"current pooled point estimate: ratio={np.exp(mean_log_ratio):.3f}, "
      f"|log ratio|={abs(mean_log_ratio):.4f}")

# within-seed variance scales ~ 1/R relative to the R=10 baseline measured above
# (replica noise averages down with more replicas at fixed seed).
def se_mean(n_seeds, R_new):
    v_within_R = mean_within_var * (R / R_new)
    return np.sqrt(V_between / n_seeds + v_within_R / n_seeds)

# find minimum n_seeds needed to exclude 1 at fixed R=10 (current allocation)
for n_seeds in [17, 20, 30, 50, 80, 100, 150, 200]:
    se = se_mean(n_seeds, R)
    excludes = abs(mean_log_ratio) > 1.96 * se
    print(f"  n_seeds={n_seeds:4d}, R=10 (new seeds, current replica count): "
          f"SE={se:.4f}  95%->excludes1? {excludes}")

print()
# find minimum R needed to exclude 1 at fixed n_seeds=17 (just add replicas, no new training)
for R_new in [10, 20, 40, 80, 160, 320, 640, 1280]:
    se = se_mean(S, R_new)
    excludes = abs(mean_log_ratio) > 1.96 * se
    print(f"  n_seeds=17 (fixed), R={R_new:5d} (replicas only, cached nets): "
          f"SE={se:.4f}  95%->excludes1? {excludes}")
