#!/usr/bin/env python3
"""Cheap depth scan (naive Monte Carlo only, no network, no WE) to find the
beyond-step at which naive starts genuinely failing at the custom preset's
budget (naive_chains=150, naive_sweeps=3500) -- the regime transport theory
actually needs, since every milestoning result so far (beyond=1, beyond=2)
has naive still winning outright with 0/10 zero-replicas. Reuses the exact
pilot-calibration procedure from run_milestone.py so thresholds line up with
every existing result.
"""
import time
import numpy as np

from neural_importance import (
    make_couplings, sweep_population, energy_per_spin,
    naive_estimate, fom_from, PRESETS
)

cfg = PRESETS['custom']
L, T, model = cfg['L'], cfg['T'], 'ea'
REPLICAS = 10
BEYONDS = [1, 2, 3, 4, 5, 6, 7, 8]

Jx, Jy = make_couplings(L, model, seed=0)

print("Pilot (identical to run_milestone.py)...")
rng_pilot = np.random.default_rng(seed=0)
S_pilot = rng_pilot.choice([-1, 1], size=(cfg['collect_walkers'], L, L))
pilot_energies = []
for _ in range(cfg['collect_iter']):
    S_pilot = sweep_population(S_pilot, T, Jx, Jy, rng_pilot)
    for s in S_pilot:
        pilot_energies.append(energy_per_spin(s[np.newaxis, ...], Jx, Jy)[0])
pilot_energies = np.array(pilot_energies)
bulk_mean = np.mean(pilot_energies)
extreme = np.min(pilot_energies)
step = (bulk_mean - extreme) / 10
print(f"bulk_mean={bulk_mean:.4f}  extreme={extreme:.4f}  step={step:.4f}")

print(f"\n{'beyond':>6s} {'thresh':>9s} {'pi_hat':>10s} {'rel.sd':>8s} "
      f"{'FOM':>10s} {'zeros':>7s} {'time(s)':>8s}")
for beyond in BEYONDS:
    thresh = extreme - beyond * step
    t0 = time.time()
    pi_hats, zeros = [], 0
    for r in range(REPLICAS):
        pi, cost = naive_estimate(L, T, Jx, Jy, model, thresh,
                                   cfg['naive_chains'], cfg['naive_sweeps'], seed=r)
        pi_hats.append(pi)
        if pi == 0:
            zeros += 1
    pi_arr = np.array(pi_hats, dtype=float)
    fom, mean_pi, rel_sd = fom_from(pi_arr, cost)
    dt = time.time() - t0
    rel_sd_s = f"{rel_sd:.3f}" if np.isfinite(rel_sd) else str(rel_sd)
    fom_s = f"{fom:.3e}" if np.isfinite(fom) else str(fom)
    print(f"{beyond:6d} {thresh:9.4f} {mean_pi:10.3e} {rel_sd_s:>8s} "
          f"{fom_s:>10s} {zeros:3d}/{REPLICAS:<3d} {dt:8.1f}")
