"""
demo_tail_r20.py -- re-run of demo_tail.py's L=20 production comparison with
R=20 replicas instead of 4, per WHAT_TO_RUN_NEXT.md Task B (2026-08-20).
Same protocol throughout (n_bins=22, n_per_bin=10, tau=4, n_iter=1200,
burn=350 for WE; matched-cost naive with n_chains=200). Seeds 10..29 for WE
and 50..69 for naive are a superset of the original run's seeds (10-13,
50-53), so this strengthens rather than replaces the original 4-replica
result. Adds explicit error bars (mean +/- 1 std across replicas) to the
tail figure, not just the separate relative-error panel.
"""
import time, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ising_we import weighted_ensemble, sweep_population, magnetization, TC

absm = lambda S: np.abs(magnetization(S))


def naive_absm(L, T, n_chains, n_sweeps, edges, seed):
    rng = np.random.default_rng(seed)
    S = rng.choice([-1.0, 1.0], size=(n_chains, L, L))
    hist = np.zeros(len(edges) - 1); burn = n_sweeps // 5
    for it in range(n_sweeps):
        S = sweep_population(S, T, rng)
        if it >= burn:
            b = np.clip(np.digitize(absm(S), edges) - 1, 0, len(edges) - 2)
            np.add.at(hist, b, 1.0)
    return hist / hist.sum(), n_chains * n_sweeps * L * L


def main():
    L, T, n_bins, R = 20, 2.60, 22, 20
    edges = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    t0 = time.time()

    Pw = []
    for r in range(R):
        _, P, nw = weighted_ensemble(L, T, n_bins=n_bins, n_per_bin=10, tau=4,
                                     n_iter=1200, burn=350, seed=10 + r,
                                     coord=absm, crange=(0, 1))
        Pw.append(P); print(f"WE {r+1}/{R} [{time.time()-t0:.0f}s]")
    Pw = np.array(Pw)
    we_cost = int(n_bins * 0.8 * 10 * 4 * 1200 * L * L)

    n_chains = 200; n_sweeps = max(1, we_cost // (n_chains * L * L))
    Pn = []
    for r in range(R):
        P, cost = naive_absm(L, T, n_chains, n_sweeps, edges, seed=50 + r)
        Pn.append(P); print(f"naive {r+1}/{R} [{time.time()-t0:.0f}s]")
    Pn = np.array(Pn)

    def stats(P):
        mean = P.mean(0); sd = P.std(0, ddof=1)
        rel = np.divide(sd, mean, out=np.full_like(mean, np.nan), where=mean > 0)
        return mean, sd, rel
    mW, sdW, relW = stats(Pw); mN, sdN, relN = stats(Pn)
    N = L * L

    print(f"\ncost WE~{we_cost:.1e}  naive~{n_chains*n_sweeps*L*L:.1e}")
    print(f"{'|m|':>6} {'P(WE)':>11} {'relW':>7} {'P(naive)':>11} {'relN':>7}")
    for i in range(len(centers)):
        pn = f"{mN[i]:11.2e}" if mN[i] > 0 else f"{'0':>11}"
        rn = f"{relN[i]:7.2f}" if mN[i] > 0 else f"{'inf':>7}"
        print(f"{centers[i]:6.3f} {mW[i]:11.2e} {relW[i]:7.2f} {pn} {rn}")

    np.savez("tail_data_r20.npz", centers=centers, mW=mW, sdW=sdW, relW=relW,
             mN=mN, sdN=sdN, relN=relN, L=L, T=T, N=N, we_cost=we_cost, R=R)

    okW, okN = mW > 0, mN > 0
    naive_floor = centers[okN].max()
    we_floor = centers[okW].max()

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.1))
    RED, BLUE = "#c0392b", "#2c5aa0"

    # rate function with an explicit +/- 1 std band from the R=20 replicas
    y_hi = -np.log(np.clip(mW[okW] - sdW[okW], 1e-300, None)) / N
    y_lo = -np.log(mW[okW] + sdW[okW]) / N
    y_mid = -np.log(mW[okW]) / N
    ax[0].fill_between(centers[okW], y_lo, y_hi, color=BLUE, alpha=0.18, zorder=1)
    ax[0].plot(centers[okW], y_mid, "o-", color=BLUE, ms=4, zorder=3,
               label=f"Weighted Ensemble ($R={R}$)")
    ax[0].plot(centers[okN], -np.log(mN[okN]) / N, "s", color=RED, ms=6,
               label=f"naive MC, matched cost ($R={R}$)")
    ax[0].axvline(naive_floor, color=RED, ls=":", lw=1)
    ax[0].text(naive_floor - 0.01, ax[0].get_ylim()[1] * 0.55,
               "naive sampling\nfloor", color=RED, ha="right", fontsize=8)
    ax[0].set_xlabel(r"$|m|$")
    ax[0].set_ylabel(r"rate function $-\frac{1}{N}\ln P(|m|)$")
    ax[0].set_title(f"(a) WE reaches beyond naive's hard support\nfloor "
                     f"(shaded: $\\pm1$ s.d.\\ over {R} replicas)")
    ax[0].legend(frameon=False, fontsize=9); ax[0].grid(alpha=0.3)

    ax[1].semilogy(centers[okW], relW[okW], "o-", color=BLUE, ms=4,
                   label=f"Weighted Ensemble ($R={R}$)")
    ax[1].semilogy(centers[okN], relN[okN], "s-", color=RED, ms=6,
                   label=f"naive MC ($R={R}$)")
    ax[1].axvline(naive_floor, color=RED, ls=":", lw=1)
    ax[1].set_xlabel(r"$|m|$")
    ax[1].set_ylabel("relative error of $P$ (across replicas)")
    ax[1].set_title("(b) naive: low error then a hard cutoff;\nWE: finite, graceful degradation")
    ax[1].legend(frameon=False, fontsize=9); ax[1].grid(alpha=0.3, which="both")
    fig.suptitle(f"2D Ising rare tail $P(|m|)$  (L={L}, T={T}, "
                 f"$T_c$={TC:.2f}); naive floor $|m|\\!=\\!{naive_floor:.2f}$, "
                 f"WE reaches ${we_floor:.2f}$", fontsize=11)
    fig.tight_layout(); fig.savefig("figs/ising_tail_r20.pdf", bbox_inches="tight")
    print(f"\nnaive floor |m|={naive_floor:.3f} (P~{mN[okN][-1]:.1e}); "
          f"WE reaches |m|={we_floor:.3f} (P~{mW[okW][-1]:.1e})")
    print(f"relative s.d. at deepest WE point: {relW[okW][-1]:.3f} (was ~2.0 at R=4)")
    print(f"saved figs/ising_tail_r20.pdf  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
