"""
ising_we.py -- Phase 2 core: rare magnetization tails of the 2D Ising model by
Weighted Ensemble, i.e. WEIGHT WINDOWS ON THE MAGNETIZATION COORDINATE.

Why this is the FW-CADIS idea. We keep a fixed number of walkers per bin of the
reaction coordinate m = (1/N) sum_i s_i. Bins in the rare tail are kept as
populated as bins in the bulk by SPLITTING walkers that reach them and MERGING
(resampling) walkers where they are abundant -- the split/Russian-roulette of a
weight window, applied along a coordinate. The consequence is FW-CADIS's
signature: roughly UNIFORM RELATIVE ERROR across the whole distribution P(m),
including the deep tail that naive Monte Carlo never resolves.

Correctness gate: at L=4 the exact P(m) is available by enumeration; the
Weighted-Ensemble estimate must match it (unbiasedness) before we trust larger L.
"""
import numpy as np

TC = 2.0 / np.log(1 + np.sqrt(2))


# --------------------------------------------------------------- exact (L<=4)
def exact_Pm(L, T):
    N = L * L
    idx = np.arange(2 ** N, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(N)) & 1).astype(np.int8)
    spins = (2 * bits - 1).reshape(-1, L, L).astype(np.float64)
    E = -((spins * np.roll(spins, -1, 1)).sum((1, 2))
          + (spins * np.roll(spins, -1, 2)).sum((1, 2)))
    M = spins.sum((1, 2))
    w = np.exp(-(E - E.min()) / T)
    Z = w.sum()
    mags = np.arange(-N, N + 1, 2)
    P = np.array([w[M == mm].sum() for mm in mags]) / Z
    return mags / N, P


# ---------------------------------------------- vectorized population dynamics
def sweep_population(S, T, rng):
    """One checkerboard Metropolis sweep applied to a stack S of shape (W,L,L)."""
    W, L, _ = S.shape
    ii, jj = np.indices((L, L))
    beta = 1.0 / T
    for color in (0, 1):
        mask = ((ii + jj) % 2 == color)
        nb = (np.roll(S, 1, 1) + np.roll(S, -1, 1)
              + np.roll(S, 1, 2) + np.roll(S, -1, 2))
        dE = 2.0 * S * nb
        acc = (rng.random((W, L, L)) < np.exp(-beta * dE)) & mask[None]
        S[acc] *= -1
    return S


def magnetization(S):
    L = S.shape[1]
    return S.sum((1, 2)) / (L * L)


# --------------------------------------------------------- Weighted Ensemble
def weighted_ensemble(L, T, n_bins=40, n_per_bin=12, tau=4, n_iter=1500,
                      burn=300, seed=0, coord=None, crange=(-1.0, 1.0)):
    """Return (centres, P) of a reaction coordinate, uniform relative error.

    coord(S)->array is the reaction coordinate (default: signed magnetization).
    Walkers carry weights; total weight is conserved. Each iteration: evolve tau
    sweeps, bin by the coordinate, then resample each occupied bin to n_per_bin
    walkers of equal weight (bin_total/n_per_bin) chosen in proportion to their
    weight. This is systematic split/merge -- weight-conserving and unbiased."""
    rng = np.random.default_rng(seed)
    if coord is None:
        coord = magnetization
    edges = np.linspace(crange[0], crange[1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # start from high-T random configs (bulk, m ~ 0), equal weights summing to 1
    W0 = n_per_bin
    S = rng.choice([-1.0, 1.0], size=(W0, L, L))
    w = np.full(W0, 1.0 / W0)
    accum = np.zeros(n_bins)
    n_acc = 0

    for it in range(n_iter):
        for _ in range(tau):
            S = sweep_population(S, T, rng)
        m = coord(S)
        b = np.clip(np.digitize(m, edges) - 1, 0, n_bins - 1)

        newS, neww = [], []
        binw = np.zeros(n_bins)
        for bi in np.unique(b):
            sel = np.where(b == bi)[0]
            Wb = w[sel].sum()
            binw[bi] = Wb
            if Wb <= 0:
                continue
            probs = w[sel] / Wb
            probs = np.clip(probs, 0, None)
            probs /= probs.sum()          # guard: rng.choice demands an exact sum
            pick = rng.choice(sel, size=n_per_bin, p=probs)   # resample ∝ weight
            for pk in pick:
                newS.append(S[pk].copy())
            neww.extend([Wb / n_per_bin] * n_per_bin)
        S = np.array(newS)
        w = np.array(neww)
        w /= w.sum()                                          # guard drift; total weight = 1

        if it >= burn:
            accum += binw / binw.sum()
            n_acc += 1

    P = accum / n_acc
    return centers, P, S.shape[0]


# --------------------------------------------------------------- naive MC P(m)
def naive_Pm(L, T, n_sweeps, n_bins=40, seed=0):
    rng = np.random.default_rng(seed)
    edges = np.linspace(-1, 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    S = rng.choice([-1.0, 1.0], size=(1, L, L))
    hist = np.zeros(n_bins)
    for _ in range(n_sweeps):
        S = sweep_population(S, T, rng)
        m = magnetization(S)[0]
        hist[min(int(np.digitize([m], edges)[0]) - 1, n_bins - 1)] += 1
    return centers, hist / hist.sum()


if __name__ == "__main__":
    # ---------- GATE: unbiasedness at L=4 vs exact enumeration ----------
    L, T = 4, 2.6
    me, Pe = exact_Pm(L, T)
    mw, Pw, nwalk = weighted_ensemble(L, T, n_bins=17, n_per_bin=20, tau=3,
                                      n_iter=2500, burn=500, seed=1)
    # map exact onto bin centers for comparison (nearest)
    print(f"L={L}, T={T}  (Tc={TC:.3f}); exact vs Weighted Ensemble on P(m):")
    print(f"{'m':>7} {'exact P':>12} {'WE P':>12} {'ratio':>8}")
    tail = 0
    for mm, pe in zip(me, Pe):
        j = int(np.argmin(np.abs(mw - mm)))
        pw = Pw[j]
        if pe > 1e-9:
            flag = "  <-- tail" if pe < 1e-3 else ""
            print(f"{mm:7.3f} {pe:12.3e} {pw:12.3e} {pw/pe:8.2f}{flag}")
    # summary error metric on log P over populated bins
    good = Pe > 1e-8
    logerr = np.array([abs(np.log(Pw[int(np.argmin(np.abs(mw-mm)))]+1e-30)
                           - np.log(pe)) for mm, pe in zip(me[good], Pe[good])])
    print(f"\nmean |Δ ln P| over populated m = {logerr.mean():.3f}  "
          f"(walkers≈{nwalk}) -- should be small if unbiased")
