#!/usr/bin/env python3
"""Task 1 profiling script for HANDOFF_ensemble_cost.md.

Instruments the WE loop (mirroring we_estimate() in neural_importance.py)
with separate timers for MC propagation, coordinate evaluation, and
binning/resampling bookkeeping. Runs coord_E (baseline, no torch) against
coord_net at K=1 and K=5 (mirroring run_milestone.py's ensemble mean),
at the CANONICAL config used throughout the project (L=16, tau=2,
n_bins=50, n_per_bin=40) but with fewer WE iterations (cost is linear in
n_iter, so per-iteration timing generalizes).

Networks are randomly initialised (untrained) -- irrelevant for a cost
profile, only the forward-pass shape/cost matters.
"""
import os, sys, time
sys.path.insert(0, os.path.expanduser("~/ising_transport_project/project/05_neural_importance"))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
torch.set_num_threads(1)

from neural_importance import make_couplings, sweep_population, energy_per_spin, in_target, ImportanceNet

L, T, model = 16, 2.6, "ea"
Jx, Jy = make_couplings(L, model, seed=0)
thresh = -1.0625  # matches tab:prefix EA threshold, representative depth
n_bins, n_per_bin, tau = 50, 40, 2
n_iter, burn = 100, 20   # scaled down from we_iter=1000/we_burn=200; linear in n_iter

def coord_E(S):
    return energy_per_spin(S, Jx, Jy)

def make_coord_net(K, seed=0):
    torch.manual_seed(seed)
    nets = [ImportanceNet(ch=24) for _ in range(K)]
    for net in nets:
        net.eval()
    def coord_net(S):
        with torch.no_grad():
            Xb = torch.tensor(S, dtype=torch.float32).unsqueeze(1)
            member_outs = np.stack(
                [net_k(Xb).numpy().reshape(-1) for net_k in nets], axis=0
            )
            return member_outs.mean(axis=0)
    return coord_net

def profiled_we(coord, seed, n_iter=n_iter, burn=burn):
    rng = np.random.default_rng(seed)
    S = rng.choice([-1.0, 1.0], size=(n_per_bin, L, L))
    w = np.full(len(S), 1.0 / len(S))

    t_mc = t_coord = t_bin = 0.0
    c = coord(S)
    lo, hi = float(c.min()), float(c.max())
    acc = 0.0; n_acc = 0

    for it in range(n_iter):
        t0 = time.perf_counter()
        for _ in range(tau):
            S = sweep_population(S, T, Jx, Jy, rng)
        t1 = time.perf_counter(); t_mc += t1 - t0

        c = coord(S)
        t2 = time.perf_counter(); t_coord += t2 - t1

        lo = min(lo, float(c.min())); hi = max(hi, float(c.max()))
        edges = np.linspace(lo, hi + 1e-12, n_bins + 1)
        b = np.clip(np.digitize(c, edges) - 1, 0, n_bins - 1)
        tgt = in_target(S, Jx, Jy, model, thresh)
        if it >= burn:
            acc += w[tgt].sum() / w.sum(); n_acc += 1
        newS, neww = [], []
        for bi in np.unique(b):
            sel = np.where(b == bi)[0]
            Wb = w[sel].sum()
            if Wb <= 0:
                continue
            probs = np.clip(w[sel] / Wb, 0, None); probs /= probs.sum()
            pick = rng.choice(sel, size=n_per_bin, p=probs)
            newS.append(S[pick]); neww.append(np.full(n_per_bin, Wb / n_per_bin))
        S = np.concatenate(newS); w = np.concatenate(neww); w /= w.sum()
        t3 = time.perf_counter(); t_bin += t3 - t2

    total = t_mc + t_coord + t_bin
    return dict(total=total, mc=t_mc, coord=t_coord, bin=t_bin,
                mc_pct=100*t_mc/total, coord_pct=100*t_coord/total, bin_pct=100*t_bin/total)

print(f"config: L={L}, tau={tau}, n_bins={n_bins}, n_per_bin={n_per_bin} "
      f"(pop={n_bins*n_per_bin} max), n_iter={n_iter} (scaled from 1000)\n")

results = {}
results['WE[E]'] = profiled_we(coord_E, seed=0)
results['WE[I_theta] K=1'] = profiled_we(make_coord_net(1, seed=0), seed=0)
results['WE[I_theta] K=5'] = profiled_we(make_coord_net(5, seed=0), seed=0)

for name, r in results.items():
    print(f"{name:20s} total={r['total']:.3f}s  "
          f"mc={r['mc']:.3f}s({r['mc_pct']:.1f}%)  "
          f"coord={r['coord']:.3f}s({r['coord_pct']:.1f}%)  "
          f"bin={r['bin']:.3f}s({r['bin_pct']:.1f}%)")

base = results['WE[E]']['total']
print()
for name, r in results.items():
    print(f"{name:20s} wall-clock ratio vs WE[E]: {r['total']/base:.2f}x")
