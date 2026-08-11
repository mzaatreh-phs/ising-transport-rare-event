#!/usr/bin/env python3
"""
run_milestone.py -- Milestoning version

Fixes relative to the original:
  1. fom_from() is called with a SCALAR cost (np.mean(cost_arr)), not the
     per-replica cost array. Passing the array made FOM broadcast into a
     4-element array, which is what raised the TypeError. fom_from itself
     was never broken and is used unmodified; its (FOM, mean, rel_sd)
     return is unpacked directly, including its NaN semantics for
     degenerate cases.
  2. rel_sd is now taken from fom_from (ddof=1) instead of being recomputed
     with np.std (ddof=0), so the printed rel.sd and the FOM are consistent.
  3. Pool worker moved to module level; local-closure pickling removed.
  4. torch thread count pinned to 1 to avoid oversubscription across workers.
  5. Milestone labels cached to disk (they cost ~200 s to regenerate).
  6. Pilot / training lattices use a seeded RNG so the cache is valid.
"""

import argparse
import json
import os
import time
import multiprocessing as mp

import numpy as np
import torch

from neural_importance import (
    make_couplings, sweep_population, energy_per_spin, in_target,
    ImportanceNet, PRESETS,
    we_estimate, naive_estimate, fom_from
)

# Each worker process would otherwise spawn its own BLAS/OMP thread pool,
# oversubscribing the CPU and adding noise to the cost measurements.
torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# --------------------------------------------------------------------------
# Parallel worker (module level so it pickles by reference; the payload is
# inherited by the child via fork, not pickled).
# --------------------------------------------------------------------------
_WORKER = {}


def _run_one(seed):
    w = _WORKER
    return we_estimate(
        w['L'], w['T'], w['Jx'], w['Jy'], w['model'], w['thresh'],
        w['coord'], w['n_bins'], w['n_per_bin'], w['tau'],
        w['we_iter'], w['we_burn'], seed, crange=w['crange']
    )


def _json_safe(o):
    """Convert NaN/inf to null so the results file is strict-JSON parseable
    (jq, JS, and json.loads(strict=True) all reject bare NaN)."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (int, np.integer)):
        return int(o)
    return o


# --------------------------------------------------------------------------
def generate_milestone_labels(S, T, Jx, Jy, model, milestones, n_roll, roll_K, rng):
    """label = 0.0 at the bulk, 1.0 at the target.

    The previous version binned with np.searchsorted on np.sort(milestones).
    The target is the MOST NEGATIVE milestone, so sorting ascending places it
    at index 0: a walker that rolled all the way down to the target received
    label 0.0, and one that stayed in the bulk received 0.857. The net was
    trained as an ANTI-importance function.

    neural_importance.progress_value() states the required convention:
    "oriented so LARGER = closer to the target".

    The continuous form also preserves the within-bin gradient that the bin
    index discarded. `milestones` is kept only for reporting bin occupancy.
    """
    n = S.shape[0]
    labels = np.zeros(n)
    bulk, target = float(milestones[0]), float(milestones[-1])
    for i in range(n):
        s = S[i:i + 1]
        best_energy = energy_per_spin(s, Jx, Jy)[0]
        for _ in range(n_roll):
            for _ in range(roll_K):
                s = sweep_population(s, T, Jx, Jy, rng)
            e = energy_per_spin(s, Jx, Jy)[0]
            if e < best_energy:
                best_energy = e
        labels[i] = np.clip((best_energy - bulk) / (target - bulk), 0.0, 1.0)
    return labels


def fmt(x, spec):
    """Format a value that may legitimately be NaN or inf."""
    return format(x, spec) if np.isfinite(x) else f"{x}"


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['ferro', 'ea'], default='ea')
    parser.add_argument('--preset', choices=['smoke', 'laptop', 'full', 'custom'], default='custom')
    parser.add_argument('--beyond', type=int, default=2)
    parser.add_argument('--replicas', type=int, default=10)
    parser.add_argument('--jobs', type=int, default=7)
    parser.add_argument('--out', type=str, default='results_milestone.json')
    parser.add_argument('--cache-dir', type=str, default='.label_cache')
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore and overwrite any cached milestone labels.')
    parser.add_argument('--tau', type=int, default=None,
                        help='Sweeps between resamplings. Must be comparable to the '
                             'correlation time (~1 sweep at T=2.6) or splitting is '
                             'undone by relaxation and WE degenerates to naive.')
    parser.add_argument('--n-bins', type=int, default=None,
                        help='WE bin count. Deeper --beyond widens the calibrated '
                             'bulk-to-target range, so a fixed bin count coarsens '
                             'resolution near the tail unless raised to compensate.')
    parser.add_argument('--we-iter', type=int, default=None,
                        help='WE resampling iterations. n_bins tuning alone was found '
                             'not to fix zero-replica failure at deeper --beyond (more '
                             'bins raised WE[I_theta]\'s own walker-population cost '
                             'without helping); this raises the actual depth budget -- '
                             'more chances to ratchet into the tail -- instead.')
    parser.add_argument('--net-seed', type=int, default=0,
                        help='torch.manual_seed for the ImportanceNet weight init. '
                             'Found 2026-08-06 that this dominates the WE[I_theta] vs '
                             'WE[E] comparison at beyond=1 (three seeds gave -17%%, '
                             '+14%%, +112%%) -- sweep this to characterise the spread, '
                             'not --replicas. With --ensemble > 1 this is the BASE seed '
                             'the member seeds are derived from, not a single net seed.')
    parser.add_argument('--ensemble', type=int, default=1,
                        help='Number of independently-initialised ImportanceNets to '
                             'train on the same labels; WE[I_theta] uses their MEAN '
                             'prediction. K=1 reproduces the original single-net '
                             'behaviour at --net-seed exactly. K>1 is a deep-ensemble '
                             'variance-reduction attempt aimed at the diagnosed root '
                             'cause of the WE[I_theta] vs WE[E] seed-to-seed sign flips '
                             '(untrained-weight-init variance, characterised but not '
                             'fixed by the n=17 seed sweep in HANDOFF.md 2026-08-06/07). '
                             'Member seeds are net_seed*1000+k for k in range(K), kept '
                             'disjoint from the plain single-seed sweep\'s seeds 0-15.')
    args = parser.parse_args()

    cfg = PRESETS[args.preset].copy()
    cfg['beyond'] = args.beyond
    cfg['replicas'] = args.replicas
    cfg['jobs'] = args.jobs
    if args.tau is not None:
        cfg['tau'] = args.tau
    if args.n_bins is not None:
        cfg['n_bins'] = args.n_bins
    if args.we_iter is not None:
        cfg['we_iter'] = args.we_iter
    L, T = cfg['L'], cfg['T']
    if cfg['tau'] > 4:
        print(f"   WARNING: tau={cfg['tau']} exceeds the ~1-sweep correlation time "
              f"at T={T}. Split walkers fully decorrelate between resamplings, so "
              f"WE cannot ratchet into the tail. Built-in presets use tau=3-4; "
              f"tau=1-2 penetrates deepest. Override with --tau.")
    model = args.model

    if args.replicas < 2:
        print("   NOTE: --replicas < 2 gives no variance estimate; FOM will be NaN.")

    print("=== Milestoning version ===")
    print(f"model={model} L={L} T={T} preset={args.preset} "
          f"beyond={args.beyond} replicas={args.replicas} jobs={args.jobs}")

    Jx, Jy = make_couplings(L, model, seed=0)

    # ---------------------------------------------------------------- pilot
    print("[0/3] Running pilot and calibrating target...")
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
    # bulk_edge as in calibrate_threshold(): the ladder must span
    # [bulk -> target] and nothing beyond, or bins are spent on phase space
    # the walkers never occupy.
    bulk_edge = float(pilot_energies.mean() + 2 * pilot_energies.std())
    step = (bulk_mean - extreme) / 10
    thresh = extreme - args.beyond * step
    print(f"   bulk mean = {bulk_mean:.4f}, extreme = {extreme:.4f}, "
          f"bulk_edge = {bulk_edge:.4f}, target = {thresh:.4f}")
    assert step > 0 and thresh < extreme, "target is not beyond the pilot extreme"

    n_milestones = 8
    milestones = np.linspace(bulk_mean, thresh, n_milestones)
    print(f"   milestones: {milestones}")

    # --------------------------------------------------- training + labels
    print("[1/3] Collecting training data and computing milestone labels...")
    n_train = cfg['n_train']

    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(
        args.cache_dir,
        f"labels_{model}_L{L}_T{T}_b{args.beyond}_n{n_train}.npz"
    )

    if os.path.exists(cache_path) and not args.no_cache:
        d = np.load(cache_path)
        S_train = d['S']
        labels = d['labels']
        print(f"   loaded cached labels from {cache_path}")
    else:
        rng_train = np.random.default_rng(seed=1)
        S_train = rng_train.choice([-1, 1], size=(n_train, L, L))
        for _ in range(100):
            S_train = sweep_population(S_train, T, Jx, Jy, rng_train)

        t0 = time.time()
        labels = generate_milestone_labels(
            S_train, T, Jx, Jy, model, milestones,
            cfg['n_roll'], cfg['roll_K'], rng_train
        )
        np.savez(cache_path, S=S_train, labels=labels, milestones=milestones)
        print(f"   computed in {time.time() - t0:.1f}s, cached to {cache_path}")

    print(f"   labels: mean={labels.mean():.4f} sd={labels.std():.4f} "
          f"range=[{labels.min():.4f},{labels.max():.4f}]")

    # Diagnostic: how much of the milestone ladder the training set reaches.
    occupied = np.unique(np.round(labels * (n_milestones - 1)).astype(int))
    print(f"   milestone bins occupied: {occupied.tolist()} of {list(range(n_milestones))}")
    if labels.max() < 0.8:
        print(f"   WARNING: deepest training rollout only reached label "
              f"{labels.max():.3f} (1.0 = target); the net must extrapolate.")

    # ------------------------------------------------------------- training
    mean_lab, std_lab = labels.mean(), labels.std()
    if std_lab <= 0:
        raise RuntimeError("Milestone labels have zero spread; cannot standardise. "
                           "Increase n_roll/roll_K or lower --beyond.")
    labels_std = (labels - mean_lab) / std_lab

    X = torch.tensor(S_train, dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor(labels_std, dtype=torch.float32)
    n_batches = max(1, n_train // cfg['batch'])

    def train_one_net(seed, verbose):
        # Unlike every other rng in this file, the network's weight init and
        # any torch-internal randomness were previously UNSEEDED -- two
        # independent runs of this exact config produced different trained
        # networks, and the resulting WE[I_theta] comparison against WE[E]
        # disagreed even in SIGN (found 2026-08-05/06: +14% one run, -17%
        # the next). Seeding makes a single run reproducible; the ensemble
        # (below) is what actually attacks that init-variance, rather than
        # just reporting it.
        torch.manual_seed(seed)
        net_k = ImportanceNet(ch=cfg['channels'])
        optimizer = torch.optim.Adam(net_k.parameters(), lr=cfg['lr'])
        loss_fn = torch.nn.MSELoss()
        net_k.train()
        t0 = time.time()
        for epoch in range(cfg['epochs']):
            perm = torch.randperm(n_train)
            X_shuf, Y_shuf = X[perm], Y[perm]
            loss = 0.0
            for i in range(0, n_train, cfg['batch']):
                batch_X = X_shuf[i:i + cfg['batch']]
                batch_Y = Y_shuf[i:i + cfg['batch']]
                optimizer.zero_grad()
                pred = net_k(batch_X)
                l = loss_fn(pred, batch_Y)
                l.backward()
                optimizer.step()
                loss += l.item()
            if verbose and (epoch + 1) % 5 == 0:
                print(f"      epoch {epoch + 1}/{cfg['epochs']}  "
                      f"MSE={loss / n_batches:.4f}  [{time.time() - t0:.1f}s]")
        net_k.eval()
        return net_k, time.time() - t0

    ensemble_seeds = ([args.net_seed] if args.ensemble == 1
                       else [args.net_seed * 1000 + k for k in range(args.ensemble)])
    print(f"[2/3] Training ImportanceNet ensemble (K={args.ensemble}, "
          f"seeds={ensemble_seeds}) on milestone labels...")
    nets = []
    t_ens0 = time.time()
    for k, seed in enumerate(ensemble_seeds):
        print(f"   member {k + 1}/{len(ensemble_seeds)} (seed={seed})")
        net_k, dt = train_one_net(seed, verbose=(args.ensemble == 1))
        nets.append(net_k)
        print(f"      done [{dt:.1f}s]")
    print(f"   ensemble training done [{time.time() - t_ens0:.1f}s total]")

    # we_estimate calls coord(S) on the ENTIRE walker population, shape
    # (N, L, L) -- not one configuration at a time. Every coord function must
    # therefore be vectorised and return an array of length N. _as_batch
    # normalises the input so a single (L, L) lattice still works.
    def _as_batch(s):
        s = np.asarray(s)
        if s.ndim == 2:                      # (L, L)      -> (1, L, L)
            return s[np.newaxis, ...], True
        if s.ndim == 3:                      # (N, L, L)   -> unchanged
            return s, False
        if s.ndim == 4 and s.shape[1] == 1:  # (N, 1, L, L) -> (N, L, L)
            return s[:, 0], False
        # Flat buffer of whole lattices, e.g. (N*L*L,)
        if s.size % (L * L) == 0:
            return s.reshape(-1, L, L), False
        raise ValueError(f"coord: cannot interpret array of shape {s.shape} "
                         f"as a population of {L}x{L} lattices")

    def _unwrap(out, was_single):
        return float(out[0]) if was_single else np.asarray(out, dtype=float)

    def coord_m(s):
        S, single = _as_batch(s)
        return _unwrap(np.abs(S.mean(axis=(1, 2))), single)

    def coord_E(s):
        S, single = _as_batch(s)
        return _unwrap(energy_per_spin(S, Jx, Jy), single)

    def coord_net(s):
        S, single = _as_batch(s)
        with torch.no_grad():
            Xb = torch.tensor(S, dtype=torch.float32).unsqueeze(1)
            # Deep-ensemble mean: averaging K independently-initialised nets'
            # predictions is the variance-reduction step (K=1 is a no-op, so
            # this is exactly the original single-net behaviour there).
            member_outs = np.stack(
                [net_k(Xb).numpy().reshape(-1) for net_k in nets], axis=0
            )
            out = member_outs.mean(axis=0)
        return _unwrap(out, single)

    coord_fns = {'m': coord_m, 'E': coord_E, 'I_theta': coord_net}

    # Fail fast on a dummy population rather than inside a worker process.
    _probe = np.random.choice([-1, 1], size=(3, L, L))
    for _name, _fn in coord_fns.items():
        _out = np.asarray(_fn(_probe))
        assert _out.shape == (3,), f"coord '{_name}' returned shape {_out.shape}, expected (3,)"

    # WE[I_theta]'s bin range is adaptive (crange=None, auto-ranging from
    # whatever the population actually explores), unlike WE[E]/WE[m] whose
    # ranges are calibrated to the coordinate's own physical scale.
    #
    # TESTED AND REJECTED, 2026-08-06: a calibrated fixed crange was tried
    # here, on the hypothesis (from diagnose_ea_full.log's verdict, "if
    # I_theta still does not beat WE[E], the bottleneck is the weight-window
    # schedule, not the map" -- I_theta has real signal, incremental
    # R^2=+0.25 beyond energy) that an uncalibrated adaptive range was
    # wasting bin resolution. Two calibrations were tried:
    #   1. Theoretical, from the label z-score anchors (label=0 at bulk,
    #      label=1 at target by construction). WRONG: the network's actual
    #      output never approaches those extremes (training MSE never
    #      reaches 0), so this made the ladder ~3.5x wider than the real
    #      data span (verify_crange_calibration.py).
    #   2. Empirical, from what the trained network actually outputs on the
    #      pilot population and training set, with a 25% margin.
    # Calibration 2 was run head-to-head against the untouched adaptive
    # baseline on matched net-seeds (results_fix_seed{1,2,3}.json vs
    # results_seed{1,2,3}.json -- same trained network each pair, only the
    # bin range differs): mean FOM ratio vs WE[E] WORSENED (0.79 -> 0.61),
    # and rel.sd WORSENED (0.214 -> 0.286), in 2 of 3 matched pairs. A fixed
    # range clips any walker that exceeds it into the boundary bin, while
    # adaptive tracking never does -- for a coordinate whose true dynamic
    # range during an actual multi-thousand-iteration WE trajectory is not
    # well known from a static pre-run snapshot, that clipping cost more
    # resolution than the calibration gained. Reverted to crange=None.
    # Net effect: this rules out bin-range calibration as the seed-variance
    # driver, strengthening the case that it is intrinsic to what each
    # trained network learns, not an infrastructure artifact.
    crange_I = None

    # --------------------------------------------------------- head-to-head
    print("[3/3] Running head-to-head...")
    results = {}
    total_start = time.time()

    # wall_time per method, alongside the existing MC-sweep 'cost'. Added
    # 2026-08-08: 'cost' = len(S)*tau*L*L counts MC sweeps only and is BLIND
    # to neural-net inference, so WE[I_theta] with --ensemble K>1 was getting
    # its Kx forward-pass overhead entirely uncharged in the FOM. Measured
    # directly (ensemble_sweep.log, 5 runs): WE[I_theta] at K=5 took ~10.8x
    # WE[E]'s wall-clock, not the ~1.3x its MC-cost implied. Recording real
    # wall_time here lets FOM be recomputed on a compute-fair basis without
    # log-scraping (see wallclock_cost_check.py / pooled_ensemble_bootstrap.py
    # for the post-hoc version used on the runs that predate this field).
    stage_start = time.time()

    # ---- naive (serial)
    pi_hats, costs, zeros = [], [], 0
    for r in range(cfg['replicas']):
        pi, cost = naive_estimate(L, T, Jx, Jy, model, thresh,
                                  cfg['naive_chains'], cfg['naive_sweeps'], seed=r)
        pi_hats.append(pi)
        costs.append(cost)
        if pi == 0:
            zeros += 1
    pi_arr = np.array(pi_hats, dtype=float)
    cost_arr = np.array(costs, dtype=float)
    mean_cost = float(np.mean(cost_arr))
    wall_time = time.time() - stage_start
    stage_start = time.time()

    # fom_from expects a SCALAR cost; passing cost_arr is what broke this.
    fom, mean_pi, rel_sd = fom_from(pi_arr, mean_cost)

    results['naive'] = {'mean': mean_pi, 'rel_sd': rel_sd,
                        'cost': mean_cost, 'fom': fom, 'zeros': zeros,
                        'est': pi_arr.tolist(), 'wall_time': wall_time}
    print(f"   naive        pi={fmt(mean_pi, '.3e')}  rel.sd={fmt(rel_sd, '.3f')}  "
          f"cost={mean_cost:.2e}  FOM={fmt(fom, '.3e')}  "
          f"zeros={zeros}/{cfg['replicas']}  [{time.time() - total_start:.1f}s]")

    # ---- weighted-ensemble variants (parallel)
    # crange: bins must reach the target FROM THE START or walkers are never
    # split into the tail (neural_importance.py:906). Without it lo/hi seed
    # from the random initial configuration (E/N ~ 0) and over half the ladder
    # covers states never visited at this temperature.
    coord_map = {
        'WE[m]': (coord_fns['m'], (0.0, 1.0)),
        'WE[E]': (coord_fns['E'], (thresh - 4.0 / (L * L), bulk_edge)),
        'WE[I_theta]': (coord_fns['I_theta'], crange_I),
    }
    ctx = mp.get_context('fork')  # fork lets children inherit net / Jx / Jy

    for method in ['WE[m]', 'WE[E]', 'WE[I_theta]']:
        _coord, _crange = coord_map[method]
        _WORKER.clear()
        _WORKER.update(dict(
            L=L, T=T, Jx=Jx, Jy=Jy, model=model, thresh=thresh,
            coord=_coord, crange=_crange,
            n_bins=cfg['n_bins'], n_per_bin=cfg['n_per_bin'], tau=cfg['tau'],
            we_iter=cfg['we_iter'], we_burn=cfg['we_burn'],
        ))

        with ctx.Pool(processes=cfg['jobs']) as pool:
            outputs = pool.map(_run_one, range(cfg['replicas']))
        wall_time = time.time() - stage_start
        stage_start = time.time()

        pi_arr = np.array([o[0] for o in outputs], dtype=float)
        cost_arr = np.array([o[1] for o in outputs], dtype=float)
        mean_cost = float(np.mean(cost_arr))
        zeros = int(np.sum(pi_arr == 0))

        fom, mean_pi, rel_sd = fom_from(pi_arr, mean_cost)

        results[method] = {'mean': mean_pi, 'rel_sd': rel_sd,
                           'cost': mean_cost, 'fom': fom, 'zeros': zeros,
                           'est': pi_arr.tolist(), 'wall_time': wall_time}
        print(f"   {method:10s} pi={fmt(mean_pi, '.3e')}  rel.sd={fmt(rel_sd, '.3f')}  "
              f"cost={mean_cost:.2e}  FOM={fmt(fom, '.3e')}  "
              f"zeros={zeros}/{cfg['replicas']}  [{time.time() - total_start:.1f}s]")

    # ---------------------------------------------------------------- save
    out_data = {
        'config': cfg,
        'model': model,
        'L': L,
        'T': T,
        'thresh': float(thresh),
        'milestones': milestones.tolist(),
        'label_stats': {
            'mean': float(labels.mean()),
            'sd': float(labels.std()),
            'min': float(labels.min()),
            'max': float(labels.max()),
            'bins_occupied': occupied.tolist(),
        },
        'I_theta_crange': {'value': crange_I, 'net_seed': args.net_seed,
                           'ensemble_size': args.ensemble,
                           'ensemble_seeds': ensemble_seeds},
        'fom_definition': 'fom_from(): 1/(rel_var * mean_cost), rel_sd uses ddof=1; '
                          'NaN = no variance estimate, 0.0 = event never observed',
        'results': results,
    }
    with open(args.out, 'w') as f:
        json.dump(_json_safe(out_data), f, indent=2)
    print(f"\nResults saved to {args.out}")
    print(f"Total wall time: {time.time() - total_start:.1f}s")


if __name__ == '__main__':
    main()
