#!/usr/bin/env python3

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
        w['we_iter'], w['we_burn'], seed
    )


def _json_safe(o):
    """Convert NaN/inf to null so the results file is strict-JSON parseable."""
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
    n = S.shape[0]
    labels = np.zeros(n)
    ms = np.sort(milestones)  # ascending
    n_ms = len(ms)
    for i in range(n):
        s = S[i:i + 1]
        best_energy = energy_per_spin(s, Jx, Jy)[0]
        for _ in range(n_roll):
            for _ in range(roll_K):
                s = sweep_population(s, T, Jx, Jy, rng)
            e = energy_per_spin(s, Jx, Jy)[0]
            if e < best_energy:
                best_energy = e
        idx = np.searchsorted(ms, best_energy, side='right') - 1
        idx = max(0, idx)
        labels[i] = idx / (n_ms - 1)
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
    args = parser.parse_args()

    cfg = PRESETS[args.preset].copy()
    cfg['beyond'] = args.beyond
    cfg['replicas'] = args.replicas
    cfg['jobs'] = args.jobs
    L, T = cfg['L'], cfg['T']
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
    # step is a POSITIVE magnitude (bulk_mean > extreme, both negative), so
    # subtracting it drives the threshold DEEPER than the pilot minimum.
    step = (bulk_mean - extreme) / 10.0
    thresh = extreme - args.beyond * step
    print(f"   bulk mean = {bulk_mean:.4f}, extreme = {extreme:.4f}, "
          f"step = {step:.4f}, target = {thresh:.4f}")
    assert step > 0, "step must be positive; check bulk_mean/extreme ordering"
    assert thresh < extreme, "target is shallower than the pilot minimum -- not a rare event"

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
    if occupied.max() < n_milestones - 2:
        print("   WARNING: no training sample reaches the upper milestones; "
              "the net must extrapolate into the target region.")

    # ------------------------------------------------------------- training
    print("[2/3] Training ImportanceNet on milestone labels...")
    net = ImportanceNet(ch=cfg['channels'])
    mean_lab, std_lab = labels.mean(), labels.std()
    if std_lab <= 0:
        raise RuntimeError("Milestone labels have zero spread; cannot standardise. "
                           "Increase n_roll/roll_K, or the target is too deep for "
                           "equilibrium-sampled training configurations to reach.")
    if std_lab < 0.02:
        print(f"   WARNING: label spread is very small (sd={std_lab:.4f}); "
              f"standardisation will amplify noise and I_theta may be near-constant.")
    labels_std = (labels - mean_lab) / std_lab

    X = torch.tensor(S_train, dtype=torch.float32).unsqueeze(1)
    Y = torch.tensor(labels_std, dtype=torch.float32)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
    loss_fn = torch.nn.MSELoss()
    net.train()
    t0 = time.time()
    n_batches = max(1, n_train // cfg['batch'])
    for epoch in range(cfg['epochs']):
        perm = torch.randperm(n_train)
        X_shuf, Y_shuf = X[perm], Y[perm]
        loss = 0.0
        for i in range(0, n_train, cfg['batch']):
            batch_X = X_shuf[i:i + cfg['batch']]
            batch_Y = Y_shuf[i:i + cfg['batch']]
            optimizer.zero_grad()
            pred = net(batch_X)
            l = loss_fn(pred, batch_Y)
            l.backward()
            optimizer.step()
            loss += l.item()
        if (epoch + 1) % 5 == 0:
            print(f"   epoch {epoch + 1}/{cfg['epochs']}  "
                  f"MSE={loss / n_batches:.4f}  [{time.time() - t0:.1f}s]")
    net.eval()
    print(f"   training done [{time.time() - t0:.1f}s]")

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
            out = net(Xb).numpy().reshape(-1)
        return _unwrap(out, single)

    coord_fns = {'m': coord_m, 'E': coord_E, 'I_theta': coord_net}

    # Fail fast on a dummy population rather than inside a worker process.
    _probe = np.random.choice([-1, 1], size=(3, L, L))
    for _name, _fn in coord_fns.items():
        _out = np.asarray(_fn(_probe))
        assert _out.shape == (3,), f"coord '{_name}' returned shape {_out.shape}, expected (3,)"

    # --------------------------------------------------------- head-to-head
    print("[3/3] Running head-to-head...")
    results = {}
    total_start = time.time()

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

    # fom_from expects a SCALAR cost; passing cost_arr is what broke this.
    fom, mean_pi, rel_sd = fom_from(pi_arr, mean_cost)

    results['naive'] = {'mean': mean_pi, 'rel_sd': rel_sd,
                        'cost': mean_cost, 'fom': fom, 'zeros': zeros}
    print(f"   naive        pi={fmt(mean_pi, '.3e')}  rel.sd={fmt(rel_sd, '.3f')}  "
          f"cost={mean_cost:.2e}  FOM={fmt(fom, '.3e')}  "
          f"zeros={zeros}/{cfg['replicas']}  [{time.time() - total_start:.1f}s]")

    # ---- weighted-ensemble variants (parallel)
    coord_map = {
        'WE[m]': coord_fns['m'],
        'WE[E]': coord_fns['E'],
        'WE[I_theta]': coord_fns['I_theta'],
    }
    ctx = mp.get_context('fork')  # fork lets children inherit net / Jx / Jy

    for method in ['WE[m]', 'WE[E]', 'WE[I_theta]']:
        _WORKER.clear()
        _WORKER.update(dict(
            L=L, T=T, Jx=Jx, Jy=Jy, model=model, thresh=thresh,
            coord=coord_map[method],
            n_bins=cfg['n_bins'], n_per_bin=cfg['n_per_bin'], tau=cfg['tau'],
            we_iter=cfg['we_iter'], we_burn=cfg['we_burn'],
        ))

        with ctx.Pool(processes=cfg['jobs']) as pool:
            outputs = pool.map(_run_one, range(cfg['replicas']))

        pi_arr = np.array([o[0] for o in outputs], dtype=float)
        cost_arr = np.array([o[1] for o in outputs], dtype=float)
        mean_cost = float(np.mean(cost_arr))
        zeros = int(np.sum(pi_arr == 0))

        fom, mean_pi, rel_sd = fom_from(pi_arr, mean_cost)

        results[method] = {'mean': mean_pi, 'rel_sd': rel_sd,
                           'cost': mean_cost, 'fom': fom, 'zeros': zeros}
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
