"""
neural_importance.py -- Stage 5: a LEARNED importance function in full
configuration space, benchmarked head-to-head against reaction-coordinate
baselines by figure of merit.

This is the component the Phase-1 memo argued for and Phase 2 deliberately did
NOT do: Phase 2 used weight windows on a hand-picked 1-D coordinate (the
magnetization m). Here we learn I_theta(s) on the whole configuration and use
IT as the binning coordinate for the same weight-window machinery.

Rare event:   ferro  --  pi = P(|m| >= m*)      (m is a GOOD coordinate)
              ea     --  pi = P(E/N <= e*)      (m is a USELESS coordinate:
                                                 a spin glass has m ~ 0)
The spin glass is the point of the exercise: a learned importance map should win
precisely where the hand-picked coordinate is a poor descriptor.

Methods compared (all estimate the same pi, at matched cost):
  naive        analog Monte Carlo
  WE[m]        weight windows binned on magnetization        (Phase-2 baseline)
  WE[E]        weight windows binned on energy               (better baseline for ea)
  WE[I_theta]  weight windows binned on the LEARNED importance
Figure of merit  FOM = 1 / (rel_var * cost), rel_var from independent replicas.

How I_theta is learned (committor regression, no labels needed from theory):
  1. collect configurations along WE trajectories (they cover the tail),
  2. from each, run n_roll short rollouts of K sweeps; label = fraction that
     reach the target set  -> a Monte Carlo estimate of the finite-time committor,
  3. fit a small CNN to the log-odds of that label.
By the Phase-1 identity this committor IS the adjoint importance function, so
binning on it is the learned analogue of a CADIS weight-window map.

VALIDATION GATE: at L=4 (ferro) the answer is available by exact enumeration.
Every method must reproduce it -- weight windows are unbiased for ANY binning
coordinate, good or bad, so this checks the implementation, while the FOM checks
the coordinate's quality.

Two training objectives for I_theta (--objective):
  surrogate        (default) regress onto "best value reached in a rollout" --
                   dense and label-collapse-free, but a monotone SURROGATE for
                   the committor, not the committor itself.
  selfconsistent   Kim & Cai-style (arXiv:2602.12294): no labels at all -- train
                   by minimizing violation of the tilted kernel's own
                   self-consistency condition over single-spin-flip moves. See
                   train_importance_selfconsistent() for the derivation.

Usage:
    python3 neural_importance.py --preset laptop            # ferro, quick
    python3 neural_importance.py --model ea --preset laptop  # spin glass
    python3 neural_importance.py --model ea --preset full --jobs 8
    python3 neural_importance.py --gate --objective selfconsistent --preset laptop
"""
from __future__ import annotations
import argparse, json, os, time
from multiprocessing import Pool

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")      # keep torch from fighting Pool
import torch
import torch.nn as nn

TC = 2.0 / np.log(1 + np.sqrt(2))


# --------------------------------------------------------------------------- #
#  model definition: couplings, dynamics, observables
# --------------------------------------------------------------------------- #
def make_couplings(L: int, model: str, seed: int = 0):
    """Jx[i,j] couples (i,j)-(i+1,j); Jy[i,j] couples (i,j)-(i,j+1)."""
    if model == "ferro":
        return np.ones((L, L)), np.ones((L, L))
    rng = np.random.default_rng(seed)                      # fixed disorder sample
    Jx = rng.choice([-1.0, 1.0], size=(L, L))
    Jy = rng.choice([-1.0, 1.0], size=(L, L))
    return Jx, Jy


def neighbour_field(S, Jx, Jy):
    """sum_j J_ij s_j for every site, for a stack S of shape (W,L,L)."""
    return (Jx * np.roll(S, -1, 1) + np.roll(Jx, 1, 0) * np.roll(S, 1, 1)
            + Jy * np.roll(S, -1, 2) + np.roll(Jy, 1, 1) * np.roll(S, 1, 2))


def sweep_population(S, T, Jx, Jy, rng):
    """One checkerboard Metropolis sweep on a stack S of shape (W,L,L)."""
    W, L, _ = S.shape
    ii, jj = np.indices((L, L))
    beta = 1.0 / T
    for color in (0, 1):
        mask = ((ii + jj) % 2 == color)
        dE = 2.0 * S * neighbour_field(S, Jx, Jy)
        acc = (rng.random((W, L, L)) < np.exp(-beta * dE)) & mask[None]
        S[acc] *= -1
    return S


def magnetization(S):
    return S.sum((1, 2)) / (S.shape[1] * S.shape[2])


def energy_per_spin(S, Jx, Jy):
    e = -((Jx * S * np.roll(S, -1, 1)).sum((1, 2))
          + (Jy * S * np.roll(S, -1, 2)).sum((1, 2)))
    return e / (S.shape[1] * S.shape[2])


def in_target(S, Jx, Jy, model, thresh):
    if model == "ferro":
        return np.abs(magnetization(S)) >= thresh
    return energy_per_spin(S, Jx, Jy) <= thresh


# --------------------------------------------------------------------------- #
#  exact enumeration (L <= 4) for the validation gate
# --------------------------------------------------------------------------- #
def exact_pi(L, T, Jx, Jy, model, thresh):
    N = L * L
    idx = np.arange(2 ** N, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(N)) & 1).astype(np.int8)
    S = (2 * bits - 1).reshape(-1, L, L).astype(np.float64)
    E = energy_per_spin(S, Jx, Jy) * N
    w = np.exp(-(E - E.min()) / T)
    tgt = in_target(S, Jx, Jy, model, thresh)
    return w[tgt].sum() / w.sum()


# --------------------------------------------------------------------------- #
#  the learned importance function
# --------------------------------------------------------------------------- #
class ImportanceNet(nn.Module):
    """Small periodic-padding CNN mapping a configuration to a scalar logit.
    Output is interpreted as log-odds of reaching the target set (the committor),
    which by the Phase-1 identity is the adjoint importance function."""

    def __init__(self, ch=16):
        super().__init__()
        self.c1 = nn.Conv2d(1, ch, 3, padding=0)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=0)
        self.head = nn.Linear(2 * ch, 1)

    @staticmethod
    def _pad(x):
        return torch.nn.functional.pad(x, (1, 1, 1, 1), mode="circular")

    def forward(self, x):                      # x: (B,1,L,L)
        h = torch.relu(self.c1(self._pad(x)))
        h = torch.relu(self.c2(self._pad(h)))
        feat = torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)
        return self.head(feat).squeeze(-1)     # logit


def progress_value(S, Jx, Jy, model):
    """The quantity the rare event is defined on, oriented so LARGER = closer to
    the target: |m| for ferro, -E/N for the spin glass (low energy is the target)."""
    return np.abs(magnetization(S)) if model == "ferro" \
        else -energy_per_spin(S, Jx, Jy)


def rollout_labels(S, T, Jx, Jy, model, K, n_roll, rng):
    """DENSE labels: the mean over n_roll rollouts of the BEST (most extreme)
    progress value reached within K sweeps.

    Why not binary committor labels: for a genuinely rare target almost no short
    rollout ever reaches it, so binary labels are ~all zero and the network
    collapses to 'never' (we measured frac>0 = 0.003 -- useless). The best-value-
    reached is a dense, monotone surrogate for the committor: configurations from
    which the dynamics can get further are exactly the more important ones. It is
    a SURROGATE, not the committor itself -- see the README's honesty notes."""
    tot = np.zeros(S.shape[0])
    for _ in range(n_roll):
        X = S.copy()
        b = progress_value(X, Jx, Jy, model)
        for _ in range(K):
            X = sweep_population(X, T, Jx, Jy, rng)
            b = np.maximum(b, progress_value(X, Jx, Jy, model))
        tot += b
    return tot / n_roll


def collect_training_configs(L, T, Jx, Jy, model, thresh, n_iter, n_walk, rng):
    """Sample configurations along a weight-window (WE) run on a crude
    coordinate so the training set covers the tail, not just the bulk."""
    coord = ProgressCoord(Jx, Jy, model)
    S = rng.choice([-1.0, 1.0], size=(n_walk, L, L))
    lo, hi = coord(S).min(), coord(S).max()
    keep = []
    for it in range(n_iter):
        S = sweep_population(S, T, Jx, Jy, rng)
        c = coord(S)
        lo, hi = min(lo, c.min()), max(hi, c.max())
        edges = np.linspace(lo, hi + 1e-9, 13)
        b = np.clip(np.digitize(c, edges) - 1, 0, 11)
        # crude split/merge: equalise occupancy across bins to reach the tail
        newS = []
        for bi in np.unique(b):
            sel = np.where(b == bi)[0]
            pick = rng.choice(sel, size=max(1, n_walk // 12))
            newS.append(S[pick])
        S = np.concatenate(newS)[:n_walk]
        if it % 3 == 0:
            keep.append(S.copy())
    return np.concatenate(keep)


def train_importance(L, T, Jx, Jy, model, thresh, cfg, seed=0, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    X = collect_training_configs(L, T, Jx, Jy, model, thresh,
                                 cfg["collect_iter"], cfg["collect_walkers"], rng)
    if len(X) > cfg["n_train"]:
        X = X[rng.choice(len(X), cfg["n_train"], replace=False)]
    y = rollout_labels(X, T, Jx, Jy, model,
                       cfg["roll_K"], cfg["n_roll"], rng)
    mu, sd = float(y.mean()), float(y.std() + 1e-12)
    if verbose:
        print(f"   training set: {len(X)} configs; label (best reached) "
              f"mean={mu:.4f} sd={sd:.4f} range=[{y.min():.3f},{y.max():.3f}]  "
              f"[{time.time()-t0:.0f}s]")
    if sd < 1e-6:
        print("   WARNING: labels have no spread -- the learned coordinate will be "
              "flat and useless.\n   Raise --roll-K / collect_iter, or pick a less "
              "extreme target.")

    net = ImportanceNet(cfg["channels"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    yt = torch.tensor((y - mu) / sd, dtype=torch.float32)      # standardized
    lossf = nn.MSELoss()
    n = len(Xt)
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if verbose and (ep + 1) % max(1, cfg["epochs"] // 4) == 0:
            print(f"   epoch {ep+1}/{cfg['epochs']}  MSE={tot/n:.4f} "
                  f"(1.0 = predicting the mean)  [{time.time()-t0:.0f}s]")
    net.eval()
    return net


# --------------------------------------------------------------------------- #
#  alternative training objective: self-consistency (Kim & Cai, arXiv:2602.12294)
#
#  train_importance() above regresses I_theta onto a SURROGATE label (best
#  progress value reached in a rollout) because binary committor labels
#  collapse to all-zero for a genuinely rare target. Kim & Cai train their
#  learned importance function a different way that needs no labels at all:
#  minimize how badly the tilted kernel violates its own self-consistency
#  condition. This is worth trying as an alternative fix for the surrogate-
#  label problem, instead of (or before) building milestoning.
# --------------------------------------------------------------------------- #
def single_spin_flip_weights(S, T, Jx, Jy):
    """K(s -> flip site n) for a single-spin-flip elementary Metropolis kernel:
    propose one of the N = L*L sites uniformly at random, accept with the
    standard Metropolis probability. Returns shape (W, N).

    This elementary kernel is used ONLY for the self-consistency objective
    below, which needs to enumerate every neighbour of a state exactly --
    intractable for this project's checkerboard sweep kernel (used everywhere
    else), whose per-step neighbour set is combinatorially large. Single-spin
    Metropolis is the standard elementary move for which this is tractable:
    every ΔE_n is known in closed form via the existing neighbour_field()."""
    W, L, _ = S.shape
    N = L * L
    dE = 2.0 * S * neighbour_field(S, Jx, Jy)          # (W, L, L)
    p = np.minimum(1.0, np.exp(-dE / T))                # (W, L, L)
    return (p / N).reshape(W, N)


def flip_each_site(S):
    """Return an (W, N, L, L) stack where entry [:, n] is S with site n
    flipped (N = L*L) -- every single-spin-flip neighbour of every
    configuration in the batch. Only used by the self-consistency objective."""
    W, L, _ = S.shape
    N = L * L
    out = np.repeat(S[:, None, :, :], N, axis=1).copy()
    for n in range(N):
        a, b = divmod(n, L)
        out[:, n, a, b] *= -1
    return out


def _find_boundary_states(X_pool, Jx, Jy, model, thresh, L):
    """Which states in X_pool are boundary anchors for the self-consistency
    loss: already in the target, or one single-spin-flip away from it.

    Checking every neighbour of every pooled state would be the same memory
    blow-up the O(N) cost note warns about, repeated at pool scale instead of
    batch scale -- infeasible for the `full` preset. Cheaply narrow to
    CANDIDATES first using the known analytic step size (progress changes by
    at most 2/N per spin for ferro, 4/N for ea), then only pay the expensive
    flip-and-check on that (small) candidate subset."""
    N = L * L
    step = (2.0 if model == "ferro" else 4.0) / N
    prog = progress_value(X_pool, Jx, Jy, model)
    thresh_prog = thresh if model == "ferro" else -thresh
    in_tgt = in_target(X_pool, Jx, Jy, model, thresh)
    cand = (prog >= thresh_prog - step) & ~in_tgt
    near = np.zeros(len(X_pool), dtype=bool)
    if cand.any():
        Xc = X_pool[cand]
        hits = in_target(flip_each_site(Xc).reshape(-1, L, L), Jx, Jy,
                         model, thresh).reshape(len(Xc), N).any(axis=1)
        near[np.where(cand)[0]] = hits
    return in_tgt | near


BULK_LOG_ANCHOR = -6.0     # ln I(bulk) reference value, i.e. I(bulk) = e^-6 ~ 0.0025


def train_importance_selfconsistent(L, T, Jx, Jy, model, thresh, cfg, seed=0,
                                    verbose=True, bulk_edge=None):
    """Alternative to train_importance(): learn ln I(s) directly by enforcing
    the tilted kernel's SELF-CONSISTENCY condition instead of regressing onto
    a surrogate committor label -- no rollouts, no labels, no rarity-dependent
    label collapse.

    This is the training objective from Kim & Cai (arXiv:2602.12294), adapted
    to this project's single-absorbing-target setup and grounded in the exact
    committor identity this project already proved in 03_phase1_framing: the
    committor h is HARMONIC for the untilted kernel away from the boundary,
    (I-K)h = 0, i.e.

        sum_j K(s -> j) * I(j) = I(s)                                   (*)

    with the boundary condition I(s) = 1 for s already in the target set,
    exactly as in demo_committor.py. Working in log-space (the network output
    IS ln I(s), unconstrained) avoids the underflow problem Kim & Cai report
    for a raw I(s) parameterisation, and matches their stated motivation.

    SECOND BOUNDARY CONDITION -- found necessary by direct instrumentation,
    not anticipated up front. Equation (*) plus I=1 on the target is NOT
    enough to pin down a unique solution: the constant function I(s)=1
    EVERYWHERE satisfies both (*) and the target condition exactly, for any
    amount of boundary data, so training can converge to a perfect-looking
    zero loss while learning a completely flat, useless coordinate (measured:
    trained ln I had sd < 0.01 across the training set even with boundary
    states forced into it). A committor is only well-posed with boundary
    conditions on ALL absorbing states of the process; demo_committor.py's
    1D walk has two (a target and a "returned to start" failure boundary),
    and Kim & Cai's own setup has two by construction (their two metastable
    basins F and S). This project's one-sided rare-EXCEEDANCE setup only has
    one natural absorbing boundary (the target), so a second, reference one
    is added here as a modelling choice: states in the typical/bulk
    equilibrium region (on the far side of `bulk_edge` from the target, the
    same quantity calibrate_threshold() computes for the weight-window
    ladder) are anchored to I(bulk) = e^BULK_LOG_ANCHOR -- clearly distinct
    from the target's I=1, breaking the degenerate constant solution. The
    exact anchor VALUE is arbitrary (WE binning only needs the right
    ordering, not an exact scale) but its existence is not.

    Trained over the single-spin-flip elementary kernel (see
    single_spin_flip_weights), not the checkerboard sweep used for sampling
    everywhere else in this file -- see that function's docstring for why.

    Cost note: this needs N=L*L extra network forward passes per training
    configuration (one per neighbour), vs. O(1) for train_importance(). Cheap
    at the L=4 gate; noticeably slower at the `full` preset (L=16, N=256) --
    try `smoke`/`laptop` first."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # The self-consistency equation (*) is satisfied trivially by ANY constant
    # I(s) = c -- sum_j K(s->j)*c = c*1 = c = I(s) for a properly normalized
    # kernel. The only thing that rules out this useless flat solution is the
    # boundary condition I=1 on the target. If NEITHER the collected states
    # NOR any of their single-spin-flip neighbours ever touch the target, that
    # anchor never fires and training silently collapses to the trivial
    # constant while reporting a deceptively small loss (found by direct
    # instrumentation -- see BUGFIXES.md-style honesty notes elsewhere in this
    # project). Fix: explicitly find boundary-anchor states in the collected
    # pool and FORCE their inclusion in the training subsample rather than
    # risk losing them to a uniform random subsample.
    #
    # collect_training_configs's iter/walker counts were originally left at
    # cfg["collect_iter"]/cfg["collect_walkers"] -- tuned for train_importance's
    # rollout-based labels, which only need broad tail COVERAGE, not an actual
    # target hit. That undershoots the real WE depth badly: measured to reach
    # 0 boundary states even after a 3x retry, at both L=12 and L=16, beyond=0
    # and beyond=1 -- while the real head-to-head's we_estimate(), run with
    # cfg["we_iter"]/cfg["n_bins"]*cfg["n_per_bin"], reliably DOES observe the
    # target (see the naive/WE[E] results in the same run). Fix: collect at
    # that same depth instead of the cheaper rollout-tuned one.
    collect_walk = cfg["n_bins"] * cfg["n_per_bin"]
    X_pool = collect_training_configs(L, T, Jx, Jy, model, thresh,
                                      cfg["we_iter"], collect_walk, rng)
    boundary_mask = _find_boundary_states(X_pool, Jx, Jy, model, thresh, L)
    n_boundary = int(boundary_mask.sum())
    retried = False
    if n_boundary == 0:
        # try harder once: deeper still and more walkers
        retried = True
        X_pool2 = collect_training_configs(L, T, Jx, Jy, model, thresh,
                                           int(cfg["we_iter"] * 1.5),
                                           collect_walk * 2, rng)
        boundary_mask2 = _find_boundary_states(X_pool2, Jx, Jy, model, thresh, L)
        if boundary_mask2.any():
            X_pool, boundary_mask = X_pool2, boundary_mask2
            n_boundary = int(boundary_mask.sum())

    if len(X_pool) > cfg["n_train"] and n_boundary > 0:
        b_idx = np.where(boundary_mask)[0]
        o_idx = np.where(~boundary_mask)[0]
        n_other = max(0, cfg["n_train"] - len(b_idx))
        pick_other = rng.choice(o_idx, size=min(n_other, len(o_idx)), replace=False)
        X = X_pool[np.concatenate([b_idx, pick_other])]
    elif len(X_pool) > cfg["n_train"]:
        X = X_pool[rng.choice(len(X_pool), cfg["n_train"], replace=False)]
    else:
        X = X_pool
    in_tgt = in_target(X, Jx, Jy, model, thresh)

    # Second boundary: anchor typical/bulk-equilibrium states to a reference
    # value far from the target's I=1, so the trivial constant solution is no
    # longer available (see docstring). bulk_edge is in raw (|m| or E/N) units
    # from calibrate_threshold(); convert to the same "progress" orientation
    # used everywhere else (larger = closer to target).
    prog = progress_value(X, Jx, Jy, model)
    if bulk_edge is not None:
        bulk_prog = bulk_edge if model == "ferro" else -bulk_edge
    else:
        bulk_prog = float(np.quantile(prog, 0.2))     # fallback: bottom 20%
    bulk_mask = (prog <= bulk_prog) & ~in_tgt
    if verbose:
        print(f"   training set: {len(X)} configs, {int(in_tgt.sum())} already in "
              f"the target, {n_boundary} boundary-anchor states force-included"
              f"{' (needed a longer collection pass)' if retried else ''}, "
              f"{int(bulk_mask.sum())} bulk-anchor states  [{time.time()-t0:.0f}s]")
        if n_boundary == 0:
            print("   WARNING: no training state and no single-spin-flip neighbour "
                  "of one ever reaches the target,\n            even after a longer "
                  "collection pass. The boundary condition I=1 never fires, so the\n"
                  "            self-consistency loss cannot rule out the trivial "
                  "constant solution I(s)=c.\n            Try a smaller --beyond or "
                  "larger --L before trusting this run.")
        if int(bulk_mask.sum()) == 0:
            print("   WARNING: no bulk-anchor states found either -- the second "
                  "boundary condition cannot fire.\n            The degenerate "
                  "constant solution is still available; do not trust this run.")

    net = ImportanceNet(cfg["channels"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    N = L * L
    n = len(X)

    for ep in range(cfg["epochs"]):
        perm = rng.permutation(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, cfg["batch"]):
            bidx = perm[i:i + cfg["batch"]]
            Sb = X[bidx]                                        # (B, L, L)
            tgt_b = in_tgt[bidx]
            bulk_b = bulk_mask[bidx]
            interior = ~tgt_b
            if not interior.any():
                continue
            B = Sb.shape[0]

            w = single_spin_flip_weights(Sb, T, Jx, Jy)                 # (B, N)
            Sflip = flip_each_site(Sb)                                  # (B, N, L, L)
            flip_in_tgt = in_target(Sflip.reshape(-1, L, L), Jx, Jy,
                                    model, thresh).reshape(B, N)

            Xt = torch.tensor(Sb, dtype=torch.float32).unsqueeze(1)
            Xf = torch.tensor(Sflip.reshape(-1, L, L),
                              dtype=torch.float32).unsqueeze(1)
            opt.zero_grad()
            logI_s = net(Xt)                                            # (B,)
            logI_j = net(Xf).reshape(B, N)                              # (B, N)
            flip_mask = torch.tensor(flip_in_tgt)
            logI_j = torch.where(flip_mask, torch.zeros_like(logI_j), logI_j)

            wt = torch.tensor(w, dtype=torch.float32)                    # (B, N)
            k_stay = (1.0 - wt.sum(1)).clamp(min=1e-12)                   # (B,)
            logw = torch.log(wt.clamp(min=1e-30))                        # (B, N)
            logk_stay = torch.log(k_stay)                                # (B,)

            terms = torch.cat([logw + logI_j,
                               (logk_stay + logI_s).unsqueeze(1)], dim=1)
            logsum = torch.logsumexp(terms, dim=1)     # ln sum_j K(s->j) I(j)
            interior_t = torch.tensor(interior)
            resid = (logsum - logI_s)[interior_t]
            loss = (resid ** 2).mean()

            if tgt_b.any():
                bnd_t = torch.tensor(tgt_b)
                loss = loss + (logI_s[bnd_t] ** 2).mean()

            if bulk_b.any():
                # second boundary condition -- see docstring. Without this,
                # I(s)=1 everywhere is a perfect solution of everything above.
                bulk_t = torch.tensor(bulk_b)
                loss = loss + ((logI_s[bulk_t] - BULK_LOG_ANCHOR) ** 2).mean()

            loss.backward(); opt.step()
            tot += float(loss.item()) * B; cnt += B
        if verbose and (ep + 1) % max(1, cfg["epochs"] // 4) == 0:
            print(f"   epoch {ep+1}/{cfg['epochs']}  mean self-consistency "
                  f"residual^2={tot/max(cnt,1):.6f}  [{time.time()-t0:.0f}s]")
    net.eval()
    if verbose:
        with torch.no_grad():
            out = net(torch.tensor(X, dtype=torch.float32).unsqueeze(1)).numpy()
        print(f"   trained ln I(s) over the training set: mean={out.mean():.4f} "
              f"sd={out.std():.4f} range=[{out.min():.3f},{out.max():.3f}]")
        if out.std() < 1e-3:
            print("   WARNING: ln I(s) is essentially CONSTANT across the training "
                  "set -- the network has\n            collapsed to the trivial "
                  "solution the self-consistency equation always allows. This\n"
                  "            coordinate carries no information; do not trust "
                  "the head-to-head result below.")
    return net


# --------------------------------------------------------------------------- #
#  binning coordinates
#
#  These MUST be picklable module-level classes, not lambdas or closures.
#  multiprocessing.Pool pickles the work items to send them to worker processes;
#  a lambda raises PicklingError, which an earlier version swallowed in a
#  try/except and silently fell back to serial execution -- so `--jobs N` did
#  nothing at all for the WE methods. Callable classes pickle fine.
# --------------------------------------------------------------------------- #
class AbsMagCoord:
    """|m| -- the Phase-2 hand-picked coordinate."""
    def __call__(self, S):
        return np.abs(magnetization(S))


class EnergyCoord:
    """E/N -- the natural coordinate for a spin-glass low-energy target."""
    def __init__(self, Jx, Jy):
        self.Jx, self.Jy = Jx, Jy

    def __call__(self, S):
        return energy_per_spin(S, self.Jx, self.Jy)


class NetCoord:
    """The learned importance function used as a binning coordinate.
    torch modules pickle correctly, so this parallelises like the others."""
    def __init__(self, net):
        self.net = net

    def __call__(self, S):
        with torch.no_grad():
            x = torch.tensor(S, dtype=torch.float32).unsqueeze(1)
            return self.net(x).numpy()


class ProgressCoord:
    """Oriented so LARGER = closer to the target (used when collecting configs)."""
    def __init__(self, Jx, Jy, model):
        self.Jx, self.Jy, self.model = Jx, Jy, model

    def __call__(self, S):
        return progress_value(S, self.Jx, self.Jy, self.model)


# --------------------------------------------------------------------------- #
#  estimators
# --------------------------------------------------------------------------- #
def calibrate_threshold(L, T, Jx, Jy, model, beyond=1, n_chains=64,
                        n_sweeps=500, seed=0, verbose=True):
    """Pick a rare-but-REACHABLE target from a short pilot run.

    Hard-coding a threshold is fragile: for the 2D +-J spin glass the ground
    state sits near E/N ~ -1.35, so a target of E/N <= -1.55 has probability
    EXACTLY zero and every method silently returns 0. Instead we set the target
    `beyond` discrete lattice steps past the most extreme value the pilot ever
    observed. That makes the event rarer than ~1/n_pilot_samples (so a naive run
    of comparable size should struggle) while keeping it physically attainable at
    this L and T. Step sizes are set by the lattice: 2/N for m, 4/N for E."""
    rng = np.random.default_rng(seed)
    S = rng.choice([-1.0, 1.0], size=(n_chains, L, L))
    vals = []
    burn = n_sweeps // 5
    for it in range(n_sweeps):
        S = sweep_population(S, T, Jx, Jy, rng)
        if it >= burn:
            vals.append(np.abs(magnetization(S)) if model == "ferro"
                        else energy_per_spin(S, Jx, Jy))
    v = np.concatenate(vals)
    N = L * L
    # Push `beyond` discrete steps PAST the most extreme value the pilot ever saw,
    # so the event is rarer than ~1/len(v) -- beyond the reach of a naive run of
    # comparable size. Steps are set by the lattice: 2/N for m, 4/N for E.
    extreme = float(v.max()) if model == "ferro" else float(v.min())
    saturated = False
    if model == "ferro":
        raw = extreme + beyond * 2.0 / N
        thresh = min(1.0, raw)
        # |m| is bounded above by 1. On a small lattice the pilot often ALREADY
        # reaches |m| = 1, in which case clamping produces a target the pilot has
        # already hit -- i.e. NOT beyond it, and not rare. Earlier versions still
        # printed "rarer than ~1e-5" here, which was simply false.
        saturated = raw > 1.0 and extreme >= 1.0 - 1e-12
        desc = f"|m| >= {thresh:.4f}"
    else:
        thresh = extreme - beyond * 4.0 / N
        desc = f"E/N <= {thresh:.4f}"
    # The BULK end of the weight-window ladder. Bins must span [bulk -> target];
    # anything beyond the bulk is phase space the walkers essentially never
    # occupy in equilibrium, and spending bins there throws away resolution
    # exactly where the ladder needs to be finest. An earlier version used the
    # energy of RANDOM (infinite-temperature) configurations as the upper edge,
    # which at T=2.6 sits ~0 while equilibrium sits near -0.72 -- so over half
    # the bins covered states that are never visited.
    bulk_edge = (max(0.0, float(v.mean() - 2 * v.std())) if model == "ferro"
                 else float(v.mean() + 2 * v.std()))
    if verbose:
        print(f"   pilot saw {len(v):,} samples, extreme = {extreme:.4f}, "
              f"bulk mean = {v.mean():.4f}")
        print(f"   ladder spans {'|m|' if model == 'ferro' else 'E/N'} "
              f"{bulk_edge:.4f} -> {thresh:.4f}")
        print(f"   calibrated target ({beyond} step(s) beyond pilot extreme): {desc}")
        if saturated:
            print("   WARNING: |m| is capped at 1.0 and the pilot already reached it, "
                  "so this target\n            is NOT beyond the pilot and is probably "
                  "NOT rare. On this lattice the\n            magnetization target "
                  "cannot be made rarer by raising the threshold.\n            Use a "
                  "larger --L, or a higher --T, or switch to --model ea.")
        else:
            print(f"   -> rarer than ~{1/len(v):.1e}, so naive MC of this size should "
                  f"struggle or fail")
    return thresh, bulk_edge


def naive_estimate(L, T, Jx, Jy, model, thresh, n_chains, n_sweeps, seed):
    rng = np.random.default_rng(seed)
    S = rng.choice([-1.0, 1.0], size=(n_chains, L, L))
    burn = n_sweeps // 5
    hits = 0.0; n = 0
    for it in range(n_sweeps):
        S = sweep_population(S, T, Jx, Jy, rng)
        if it >= burn:
            hits += in_target(S, Jx, Jy, model, thresh).sum()
            n += n_chains
    return hits / max(n, 1), n_chains * n_sweeps * L * L


def we_estimate(L, T, Jx, Jy, model, thresh, coord, n_bins, n_per_bin,
                tau, n_iter, burn, seed, crange=None):
    """Weight windows on an arbitrary coordinate; returns (pi_hat, cost).
    Bins are adaptive (running min/max of the coordinate) so this works for a
    learned coordinate whose range is unknown a priori. Split/merge to a fixed
    occupancy per bin is weight-conserving -> unbiased for ANY coordinate."""
    rng = np.random.default_rng(seed)
    S = rng.choice([-1.0, 1.0], size=(n_per_bin, L, L))
    w = np.full(len(S), 1.0 / len(S))
    c = coord(S)
    fixed = crange is not None
    lo, hi = crange if fixed else (float(c.min()), float(c.max()))
    acc = 0.0; n_acc = 0; cost = 0

    for it in range(n_iter):
        for _ in range(tau):
            S = sweep_population(S, T, Jx, Jy, rng)
        cost += len(S) * tau * L * L
        c = coord(S)
        if not fixed:      # adaptive bins (needed when the coordinate's range
            lo = min(lo, float(c.min()))   # is unknown, e.g. a learned map)
            hi = max(hi, float(c.max()))
        edges = np.linspace(lo, hi + 1e-12, n_bins + 1)
        b = np.clip(np.digitize(c, edges) - 1, 0, n_bins - 1)

        # tally target weight BEFORE resampling
        tgt = in_target(S, Jx, Jy, model, thresh)
        if it >= burn:
            acc += w[tgt].sum() / w.sum()
            n_acc += 1

        newS, neww = [], []
        for bi in np.unique(b):
            sel = np.where(b == bi)[0]
            Wb = w[sel].sum()
            if Wb <= 0:
                continue
            probs = np.clip(w[sel] / Wb, 0, None)
            probs /= probs.sum()          # guard: rng.choice demands an exact sum
            pick = rng.choice(sel, size=n_per_bin, p=probs)
            newS.append(S[pick])
            neww.append(np.full(n_per_bin, Wb / n_per_bin))
        S = np.concatenate(newS)
        w = np.concatenate(neww); w /= w.sum()

    return (acc / max(n_acc, 1)), cost


# --------------------------------------------------------------------------- #
#  replica drivers (parallel over cores)
# --------------------------------------------------------------------------- #
def _job(args):
    kind, payload, seed = args
    L, T, Jx, Jy, model, thresh, cfg = payload["static"]
    if kind == "naive":
        return naive_estimate(L, T, Jx, Jy, model, thresh,
                              cfg["naive_chains"], cfg["naive_sweeps"], seed)
    coord = payload["coord"]
    return we_estimate(L, T, Jx, Jy, model, thresh, coord,
                       cfg["n_bins"], cfg["n_per_bin"], cfg["tau"],
                       cfg["we_iter"], cfg["we_burn"], seed,
                       crange=payload.get("crange"))


def run_replicas(kind, static, coord, cfg, R, jobs, base_seed, crange=None):
    """Run R independent replicas, in parallel when jobs > 1.

    All coordinates are picklable classes (see above), so every method -- naive,
    fixed-coordinate and learned -- parallelises. Pickling failures are no longer
    swallowed: if one occurs it is reported rather than silently degrading to
    serial execution, because a silent fallback made `--jobs` look effective when
    it was not."""
    payload = {"static": static, "coord": coord, "crange": crange}
    args = [(kind, payload, base_seed + r) for r in range(R)]
    if jobs > 1 and R > 1:
        try:
            with Pool(min(jobs, R)) as pool:
                out = pool.map(_job, args)
        except Exception as exc:
            # Fall back to serial, but LOUDLY. An earlier version swallowed this
            # silently, so `--jobs` appeared to work while running on one core
            # (see BUGFIXES.md #1). macOS and Windows use the 'spawn' start
            # method rather than fork, which is the most likely way this trips.
            print(f"   WARNING: parallel execution failed ({type(exc).__name__}: "
                  f"{exc});\n            falling back to SERIAL for this method. "
                  f"Results are unaffected, only speed.", flush=True)
            out = [_job(a) for a in args]
    else:
        out = [_job(a) for a in args]
    est = np.array([o[0] for o in out])
    cost = float(np.mean([o[1] for o in out]))
    return est, cost


def fom_from(est, cost):
    """Return (FOM, mean, relative sd). FOM = 1/(rel_var * cost).

    Degenerate cases are reported as NaN rather than a misleading infinity:
    a single replica gives no variance estimate, and an exactly-zero spread is
    a sampling artefact of too few replicas, not genuine zero variance."""
    m = float(est.mean())
    if m <= 0:                       # event never observed -> FOM is 0, not inf
        return 0.0, m, float("inf")
    if len(est) < 2:
        return float("nan"), m, float("nan")
    rel_sd = float(est.std(ddof=1) / m)
    rel_var = rel_sd ** 2
    if rel_var <= 0 or not np.isfinite(rel_var):
        return float("nan"), m, rel_sd
    return 1.0 / (rel_var * cost), m, rel_sd


def bootstrap_fom_ratio(est_a, cost_a, est_b, cost_b, n_boot=2000, seed=0):
    """Bootstrap CI for FOM(b)/FOM(a), resampling each method's replica axis
    independently (they were run with different, unrelated seeds -- no shared
    randomness to pair on, unlike fw_cadis.py's CRN attempt). Returns
    (median_ratio, lo95, hi95); ratio > 1 means b is more efficient than a.
    A bootstrap draw where either side's FOM is non-finite is skipped, not
    counted as a zero -- matches fw_cadis.py's bootstrap_spread convention."""
    rng = np.random.default_rng(seed)
    Ra, Rb = len(est_a), len(est_b)
    ratios = []
    for _ in range(n_boot):
        fa, _, _ = fom_from(est_a[rng.integers(0, Ra, size=Ra)], cost_a)
        fb, _, _ = fom_from(est_b[rng.integers(0, Rb, size=Rb)], cost_b)
        if np.isfinite(fa) and np.isfinite(fb) and fa > 0:
            ratios.append(fb / fa)
    if len(ratios) < n_boot // 2:
        return float("nan"), float("nan"), float("nan")
    ratios = np.array(ratios)
    return (float(np.median(ratios)), float(np.percentile(ratios, 2.5)),
            float(np.percentile(ratios, 97.5)))


# --------------------------------------------------------------------------- #
PRESETS = {
    "smoke":  dict(L=8,  T=2.6, R=3,  n_bins=14, n_per_bin=8,  tau=3, we_iter=300,
                   we_burn=60,  naive_chains=60,  naive_sweeps=900,
                   collect_iter=60,  collect_walkers=64, n_train=400, n_roll=4,
                   roll_K=10, epochs=8,  batch=64, lr=3e-3, channels=8),
    "laptop": dict(L=12, T=2.6, R=6,  n_bins=16, n_per_bin=8,  tau=4, we_iter=400,
                   we_burn=80,  naive_chains=80,  naive_sweeps=1400,
                   collect_iter=90,  collect_walkers=96, n_train=900, n_roll=6,
                   roll_K=8,  epochs=12, batch=128, lr=2e-3, channels=16),
    "full":   dict(L=16, T=2.6, R=10, n_bins=22, n_per_bin=10, tau=4, we_iter=1000,
                   we_burn=200, naive_chains=150, naive_sweeps=3500,
                   collect_iter=200, collect_walkers=128, n_train=2500, n_roll=8,
                   roll_K=10, epochs=20, batch=128, lr=2e-3, channels=24),
    'custom': {
        'L': 16,
        'T': 2.6,
        'beyond': 1,
        'n_bins': 50,
        'n_per_bin': 40,      # was n_walkers
        'tau': 3,              # was resample_tau; 20 let split walkers fully
                                # relax between resamplings at T=2.6 (~1-sweep
                                # correlation time), degenerating WE to naive
        'n_train': 2500,
        'epochs': 20,
        'channels': 24,
        'n_roll': 40,
        'roll_K': 10,
        'collect_iter': 200,
        'collect_walkers': 128,
        'we_iter': 1000,
        'we_burn': 200,
        'naive_chains': 150,
        'naive_sweeps': 3500,
        'batch': 128,
        'lr': 2e-3,
        'replicas': 10,
        'jobs': 7
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["ferro", "ea"], default="ferro")
    ap.add_argument("--preset", choices=list(PRESETS), default="laptop")
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--thresh", type=float, default=None,
                    help="target threshold; default = auto-calibrated from a pilot run")
    ap.add_argument("--beyond", type=int, default=1,
                    help="how many discrete steps beyond the pilot extreme to set the target")
    ap.add_argument("--replicas", type=int, default=None)
    # Training is ~10% of a run and the loss is typically still falling when the
    # preset stops, so these are cheap knobs with real effect on map quality.
    ap.add_argument("--n-train", type=int, default=None,
                    help="training-set size (preset default; cost is ~linear)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="training epochs (preset default; cost is ~linear)")
    ap.add_argument("--channels", type=int, default=None,
                    help="CNN width (preset default)")
    ap.add_argument("--n-roll", type=int, default=None,
                    help="rollouts per config when labelling (less label noise)")
    ap.add_argument("--tau", type=int, default=None,
                    help="resampling interval (preset default otherwise). The "
                         "'custom' preset's own default is tau=3 and 'full'/'laptop' "
                         "default to tau=4; the milestoning canonical runs found "
                         "tau=2 necessary for WE to penetrate deep targets "
                         "(HANDOFF.md 2026-08-06/07) -- pass --tau 2 explicitly to "
                         "match, this is NOT applied automatically.")
    ap.add_argument("--n-bins", type=int, default=None,
                    help="WE bin count (preset default otherwise); deeper --beyond "
                         "may need more bins to keep resolution near the tail.")
    ap.add_argument("--n-per-bin", type=int, default=None,
                    help="WE walkers per bin (preset default otherwise).")
    ap.add_argument("--we-iter", type=int, default=None,
                    help="WE resampling iterations (preset default otherwise); "
                         "--we-burn scales with it proportionally unless also given.")
    ap.add_argument("--we-burn", type=int, default=None,
                    help="WE burn-in iterations (default: same fraction of --we-iter "
                         "as the preset, or preset default if --we-iter not given).")
    ap.add_argument("--objective", choices=["surrogate", "selfconsistent"],
                    default="surrogate",
                    help="surrogate = regress onto best-value-reached label (default); "
                         "selfconsistent = Kim & Cai-style label-free training, see "
                         "train_importance_selfconsistent()")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    ap.add_argument("--net-seed", type=int, default=0,
                    help="seed passed to train_importance()/train_importance_"
                         "selfconsistent() (torch.manual_seed + numpy RNG for "
                         "training-set collection). Both objectives were hardcoded "
                         "to seed=0 previously; this project's milestoning surrogate "
                         "objective already found training-seed variance to be the "
                         "dominant source of noise in WE[I_theta] results "
                         "(HANDOFF.md 2026-08-06/07) -- sweep this, not --replicas, "
                         "to characterise whether a single selfconsistent result "
                         "reproduces.")
    ap.add_argument("--net-cache", default=None,
                    help="path to save/load the trained I_theta net's state_dict. "
                         "Training is deterministic (fixed seed=0), so re-running the "
                         "same config only to change --replicas otherwise repays the "
                         "full training cost for a bit-identical network -- expensive "
                         "for --objective selfconsistent (~76min at full preset). If "
                         "the path exists, loads it and skips training entirely; "
                         "otherwise trains normally and saves to this path.")
    ap.add_argument("--gate", action="store_true",
                    help="run the L=4 exact-enumeration validation gate and exit")
    ap.add_argument("--out", default="results_neural")
    a = ap.parse_args()

    cfg = dict(PRESETS[a.preset])
    if a.L: cfg["L"] = a.L
    if a.T: cfg["T"] = a.T
    if a.replicas: cfg["R"] = a.replicas
    if a.n_train: cfg["n_train"] = a.n_train
    if a.epochs: cfg["epochs"] = a.epochs
    if a.channels: cfg["channels"] = a.channels
    if a.n_roll: cfg["n_roll"] = a.n_roll
    if a.tau is not None: cfg["tau"] = a.tau
    if a.n_bins is not None: cfg["n_bins"] = a.n_bins
    if a.n_per_bin is not None: cfg["n_per_bin"] = a.n_per_bin
    if a.we_iter is not None:
        burn_frac = cfg["we_burn"] / cfg["we_iter"]
        cfg["we_iter"] = a.we_iter
        cfg["we_burn"] = a.we_burn if a.we_burn is not None else round(a.we_iter * burn_frac)
    elif a.we_burn is not None:
        cfg["we_burn"] = a.we_burn
    L, T, R = cfg["L"], cfg["T"], cfg["R"]
    print(f"WE budget: n_bins={cfg['n_bins']}  n_per_bin={cfg['n_per_bin']}  "
          f"we_iter={cfg['we_iter']}  we_burn={cfg['we_burn']}  tau={cfg['tau']}")
    Jx, Jy = make_couplings(L, a.model, seed=12345)
    # Single-threaded torch is only needed once the multiprocessing.Pool of
    # replica workers spawns (each worker's own torch must not also try to
    # grab every core, or they fight each other). Training happens BEFORE
    # that, in this one process, so let it use every core -- the
    # selfconsistent objective in particular is forward-pass-heavy (N=L*L
    # extra evaluations per config) and was needlessly single-threaded here.
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    # ------------------------------------------------------------------ gate
    if a.gate:
        L4 = 4
        Jx4, Jy4 = make_couplings(L4, a.model, seed=12345)
        th4 = 1.0 if a.model == "ferro" else -1.25   # rarest reachable at L=4
        pi_ex = exact_pi(L4, T, Jx4, Jy4, a.model, th4)
        print(f"[GATE] L=4 {a.model} T={T} thresh={th4}: exact pi = {pi_ex:.6e}")
        gc = dict(cfg, n_bins=8, n_per_bin=12, tau=3, we_iter=1200, we_burn=200)
        for name, coord in [("WE[m]", AbsMagCoord()),
                            ("WE[E]", EnergyCoord(Jx4, Jy4))]:
            e, _ = we_estimate(L4, T, Jx4, Jy4, a.model, th4, coord,
                               gc["n_bins"], gc["n_per_bin"], gc["tau"],
                               gc["we_iter"], gc["we_burn"], seed=7)
            print(f"       {name:12s} pi = {e:.6e}   ratio to exact = {e/pi_ex:.3f}")

        print(f"       training I_theta at L=4 (objective={a.objective}) to gate "
              f"the learned coordinate too ...")
        trainer4 = (train_importance_selfconsistent if a.objective == "selfconsistent"
                   else train_importance)
        net4 = trainer4(L4, T, Jx4, Jy4, a.model, th4, cfg, seed=0, verbose=False)
        e, _ = we_estimate(L4, T, Jx4, Jy4, a.model, th4, NetCoord(net4),
                           gc["n_bins"], gc["n_per_bin"], gc["tau"],
                           gc["we_iter"], gc["we_burn"], seed=7)
        print(f"       {'WE[I_theta]':12s} pi = {e:.6e}   ratio to exact = {e/pi_ex:.3f}")

        print("       (weight windows are unbiased for ANY coordinate; ratios ~1 "
              "confirm the implementation)")
        print("       NOTE: a 16-spin system has no truly rare events (pi cannot go "
              "far below ~0.1),\n       so this gate validates UNBIASEDNESS only -- "
              "efficiency is measured at larger L.")
        return

    print("=== Stage 5: learned importance vs coordinate baselines ===")
    print(f"model={a.model}  L={L}  T={T} (Tc={TC:.3f})  preset={a.preset}  "
          f"replicas={R}  jobs={a.jobs}")
    t0 = time.time()
    print("[0/2] calibrating a rare-but-reachable target ...")
    if a.thresh is not None:
        thresh = a.thresh
        # still need the bulk edge for the ladder, so run the pilot anyway
        _, bulk_edge = calibrate_threshold(L, T, Jx, Jy, a.model,
                                           beyond=a.beyond, seed=99, verbose=False)
        print(f"   using user-supplied threshold {thresh} "
              f"(ladder bulk edge {bulk_edge:.4f} from pilot)")
    else:
        thresh, bulk_edge = calibrate_threshold(L, T, Jx, Jy, a.model,
                                                beyond=a.beyond, seed=99)
    print()

    # ------------------------------------------------- learn the importance map
    print(f"[1/2] learning I_theta(s) (objective={a.objective}) ...")
    if a.net_cache and os.path.exists(a.net_cache):
        net = ImportanceNet(cfg["channels"])
        net.load_state_dict(torch.load(a.net_cache, map_location="cpu"))
        net.eval()
        print(f"   loaded cached net from {a.net_cache} -- skipped training "
              f"[{time.time()-t0:.0f}s]\n")
    else:
        if a.objective == "selfconsistent":
            net = train_importance_selfconsistent(L, T, Jx, Jy, a.model, thresh, cfg,
                                                  seed=a.net_seed, bulk_edge=bulk_edge)
        else:
            net = train_importance(L, T, Jx, Jy, a.model, thresh, cfg, seed=a.net_seed)
        if a.net_cache:
            torch.save(net.state_dict(), a.net_cache)
            print(f"   saved trained net to {a.net_cache} for reuse")
        print(f"      done [{time.time()-t0:.0f}s]\n")
    net_coord = NetCoord(net)

    torch.set_num_threads(1)      # now restore this -- the Pool workers start next

    # ------------------------------------------------------------- comparison
    print("[2/2] head-to-head (FOM divides out cost; costs are NOT equalised) ...")
    static = (L, T, Jx, Jy, a.model, thresh, cfg)
    # bins must reach the target from the start, or walkers never get split into
    # the tail. For the hand-picked coordinates we know where the target is.
    if a.model == "ferro":
        rng_m = (bulk_edge, min(1.0, thresh + 2.0 / (L * L)))
        rng_E = None                      # energy is the mismatched coordinate here
    else:
        rng_m = (0.0, 1.0)                # magnetization is the mismatched one
        rng_E = (thresh - 4.0 / (L * L), bulk_edge)
    methods = [
        ("naive", None, None),
        ("WE[m]", AbsMagCoord(), rng_m),
        ("WE[E]", EnergyCoord(Jx, Jy), rng_E),
        ("WE[I_theta]", net_coord, None),      # learned range unknown -> adaptive
    ]
    rows = []
    # wall_time per method, alongside the existing MC-sweep-only 'cost' --
    # mirrors run_milestone.py's fix (2026-08-08, see HANDOFF.md/
    # HANDOFF_wallclock_correction.md). 'cost' is blind to network inference;
    # every WE[I_theta] result from this script predating this field was
    # reconstructed post-hoc from log timestamps (wallclock_kimcai_deeprare.py)
    # for exactly that reason -- recording it directly here makes that
    # reconstruction unnecessary for future runs.
    stage_start = time.time()
    for name, coord, crange in methods:
        kind = "naive" if name == "naive" else "we"
        est, cost = run_replicas(kind, static, coord, cfg, R,
                                 jobs=a.jobs,
                                 base_seed=1000 + 97 * len(rows), crange=crange)
        wall_time = time.time() - stage_start
        stage_start = time.time()
        f, mean, relsd = fom_from(est, cost)
        rows.append(dict(method=name, mean=float(mean), rel_sd=float(relsd),
                         cost=cost, fom=float(f), zeros=int((est == 0).sum()),
                         est=est.tolist(), wall_time=wall_time))
        print(f"   {name:12s} pi={mean:.3e}  rel.sd={relsd:.3f}  "
              f"cost={cost:.2e}  FOM={f:.3e}  zero-replicas={rows[-1]['zeros']}/{R}"
              f"  [{time.time()-t0:.0f}s]")

    base = next((r["fom"] for r in rows if r["method"] == "WE[m]"), None)
    print("\n--- summary (FOM relative to the Phase-2 baseline WE[m]) ---")
    for r in rows:
        rel = (r["fom"] / base) if base else float("nan")
        print(f"   {r['method']:12s} FOM={r['fom']:.3e}   x{rel:8.2f} vs WE[m]")
    print("\nInterpretation: for 'ferro', m is already a good coordinate, so the "
          "learned map need only match it.\nFor 'ea' (spin glass) m is useless -- "
          "that is where a learned importance map should win.")
    if any(r["zeros"] > 0 for r in rows):
        print("NOTE: methods with zero-replicas failed to observe the event at all "
              "in some replicas;\ntheir FOM is not meaningful -- that IS the failure "
              "mode being demonstrated.")

    obj_suffix = "" if a.objective == "surrogate" else f"_{a.objective}"
    out = f"{a.out}_{a.model}_L{L}_{a.preset}{obj_suffix}.json"
    with open(out, "w") as fh:
        json.dump(dict(config={k: v for k, v in cfg.items()}, objective=a.objective,
                       model=a.model, L=L, T=T, thresh=thresh, rows=rows), fh, indent=2)
    print(f"\nsaved {out}   total wall time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
