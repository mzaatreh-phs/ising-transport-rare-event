#!/usr/bin/env python3
"""Step 1 of the deep-rare scoping (see HANDOFF.md): does WE[E] (hand-picked
energy coordinate) actually beat naive once naive genuinely fails? Cheap
naive-only scan (scope_deep_rare.py) found naive breaks down at beyond=3
(7/10 zero-replicas) and is fully blind at beyond=4 (10/10 zero-replicas,
pi_hat=0.0 exactly) at the custom preset's fixed budget. No network here --
this is the load-bearing checkpoint before bothering to train one.
"""
import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from neural_importance import (
    make_couplings, sweep_population, energy_per_spin,
    naive_estimate, we_estimate, fom_from, PRESETS
)

_WORKER = {}


def _run_one(seed):
    w = _WORKER
    return we_estimate(
        w['L'], w['T'], w['Jx'], w['Jy'], w['model'], w['thresh'],
        w['coord'], w['n_bins'], w['n_per_bin'], w['tau'],
        w['we_iter'], w['we_burn'], seed, crange=w['crange']
    )


def fmt(x, spec):
    return format(x, spec) if np.isfinite(x) else f"{x}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--beyond', type=int, nargs='+', default=[3, 4])
    ap.add_argument('--replicas', type=int, default=10)
    ap.add_argument('--jobs', type=int, default=7)
    ap.add_argument('--n-bins', type=int, default=None,
                     help='override cfg n_bins (WE population/bin resolution)')
    ap.add_argument('--we-iter', type=int, default=None,
                     help='override cfg we_iter (resampling depth budget)')
    ap.add_argument('--we-burn', type=int, default=None,
                     help='override cfg we_burn (default: same fraction of we_iter as preset)')
    ap.add_argument('--tau', type=int, default=2,
                     help="resampling interval. custom preset's OWN default is tau=3, "
                          "but every canonical WE[E]/WE[I_theta] result in this project "
                          "uses --tau 2 (HANDOFF.md 2026-08-06/07): tau=3 exceeds the "
                          "~1-sweep correlation time at T=2.6, so split walkers "
                          "decorrelate before the next resampling and WE degenerates "
                          "toward naive. Defaulting here to 2 to match, NOT cfg['tau'].")
    ap.add_argument('--out', type=str, default=None,
                     help='save raw per-replica estimates + costs to this JSON path '
                          '(one file per beyond value, suffixed _beyond{N}.json) for '
                          'later bootstrapping.')
    args = ap.parse_args()

    cfg = PRESETS['custom'].copy()
    cfg['tau'] = args.tau
    if args.n_bins is not None:
        cfg['n_bins'] = args.n_bins
    if args.we_iter is not None:
        burn_frac = cfg['we_burn'] / cfg['we_iter']
        cfg['we_iter'] = args.we_iter
        cfg['we_burn'] = args.we_burn if args.we_burn is not None else round(args.we_iter * burn_frac)
    elif args.we_burn is not None:
        cfg['we_burn'] = args.we_burn
    L, T, model = cfg['L'], cfg['T'], 'ea'
    print(f"WE budget: n_bins={cfg['n_bins']}  n_per_bin={cfg['n_per_bin']}  "
          f"we_iter={cfg['we_iter']}  we_burn={cfg['we_burn']}  tau={cfg['tau']}")
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
    bulk_edge = float(pilot_energies.mean() + 2 * pilot_energies.std())
    step = (bulk_mean - extreme) / 10
    print(f"bulk_mean={bulk_mean:.4f}  extreme={extreme:.4f}  "
          f"bulk_edge={bulk_edge:.4f}  step={step:.4f}")

    def _as_batch(s):
        s = np.asarray(s)
        if s.ndim == 2:
            return s[np.newaxis, ...], True
        if s.ndim == 3:
            return s, False
        raise ValueError(f"unexpected shape {s.shape}")

    def _unwrap(out, single):
        return float(out[0]) if single else np.asarray(out, dtype=float)

    def coord_m(s):
        S, single = _as_batch(s)
        return _unwrap(np.abs(S.mean(axis=(1, 2))), single)

    def coord_E(s):
        S, single = _as_batch(s)
        return _unwrap(energy_per_spin(S, Jx, Jy), single)

    ctx = mp.get_context('fork')

    for beyond in args.beyond:
        thresh = extreme - beyond * step
        print(f"\n=== beyond={beyond}  thresh={thresh:.4f} ===")
        run_start = time.time()
        results = {}

        # naive
        pi_hats, zeros = [], 0
        for r in range(args.replicas):
            pi, cost_n = naive_estimate(L, T, Jx, Jy, model, thresh,
                                         cfg['naive_chains'], cfg['naive_sweeps'], seed=r)
            pi_hats.append(pi)
            if pi == 0:
                zeros += 1
        pi_arr = np.array(pi_hats, dtype=float)
        fom, mean_pi, rel_sd = fom_from(pi_arr, cost_n)
        results['naive'] = {'mean': mean_pi, 'rel_sd': rel_sd, 'cost': cost_n,
                            'fom': fom, 'zeros': zeros, 'est': pi_arr.tolist()}
        print(f"   naive        pi={fmt(mean_pi,'.3e')}  rel.sd={fmt(rel_sd,'.3f')}  "
              f"FOM={fmt(fom,'.3e')}  zeros={zeros}/{args.replicas}  "
              f"[{time.time()-run_start:.1f}s]")

        coord_map = {
            'WE[m]': (coord_m, (0.0, 1.0)),
            'WE[E]': (coord_E, (thresh - 4.0 / (L * L), bulk_edge)),
        }
        for method, (coord, crange) in coord_map.items():
            _WORKER.clear()
            _WORKER.update(dict(
                L=L, T=T, Jx=Jx, Jy=Jy, model=model, thresh=thresh,
                coord=coord, crange=crange,
                n_bins=cfg['n_bins'], n_per_bin=cfg['n_per_bin'], tau=cfg['tau'],
                we_iter=cfg['we_iter'], we_burn=cfg['we_burn'],
            ))
            with ctx.Pool(processes=args.jobs) as pool:
                outputs = pool.map(_run_one, range(args.replicas))
            pi_arr = np.array([o[0] for o in outputs], dtype=float)
            cost_arr = np.array([o[1] for o in outputs], dtype=float)
            mean_cost = float(np.mean(cost_arr))
            zeros = int(np.sum(pi_arr == 0))
            fom, mean_pi, rel_sd = fom_from(pi_arr, mean_cost)
            results[method] = {'mean': mean_pi, 'rel_sd': rel_sd, 'cost': mean_cost,
                               'fom': fom, 'zeros': zeros, 'est': pi_arr.tolist()}
            print(f"   {method:10s} pi={fmt(mean_pi,'.3e')}  rel.sd={fmt(rel_sd,'.3f')}  "
                  f"FOM={fmt(fom,'.3e')}  zeros={zeros}/{args.replicas}  "
                  f"[{time.time()-run_start:.1f}s]")

        print(f"   beyond={beyond} total: {time.time()-run_start:.1f}s")

        if args.out:
            out_path = f"{args.out}_beyond{beyond}.json"
            with open(out_path, 'w') as f:
                json.dump({'beyond': beyond, 'thresh': thresh, 'cfg': cfg,
                          'results': results}, f, indent=2)
            print(f"   saved: {out_path}")


if __name__ == '__main__':
    main()
