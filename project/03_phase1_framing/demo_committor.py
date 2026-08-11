"""
demo_committor.py
=================
Concrete, exactly-solvable demonstration of the identity that the whole Phase-1
framing rests on:

    transport adjoint importance  psi-dagger(P)
        =  rare-event committor  h(s)
        =  the "neural importance function" of Kim & Cai (2026),

and its consequence -- the ZERO-VARIANCE estimator obtained by tilting the
dynamics with the importance ratio h(s')/h(s) (the discrete analogue of CADIS
weight windows w(P) ~ 1/psi-dagger(P)).

Model: a biased nearest-neighbour random walk on {0,1,...,N}, absorbing at 0
("return to source / capture") and at N ("deep detector"). Reaching N before 0
is the rare deep-penetration event -- the 1D caricature of gamma-ray shielding
and, equally, of a rare fluctuation reaching a target set in a spin system.

Right step prob p, left step q=1-p. Committor h(s)=P(hit N before 0 | start s):
    h(s) = (1 - r^s) / (1 - r^N),  r = q/p   (p != 1/2),   h(s)=s/N (p=1/2).
It satisfies the discrete ADJOINT/harmonic equation  h(s)=p h(s+1)+q h(s-1),
h(0)=0, h(N)=1 -- i.e. (I-K) h = 0 in the interior. This IS the adjoint
transport equation with the detector as adjoint source.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, BLUE, GREEN = "#c0392b", "#2c5aa0", "#218c5a"


def committor(N, p):
    q = 1 - p
    s = np.arange(N + 1)
    if abs(p - 0.5) < 1e-12:
        return s / N
    r = q / p
    return (1 - r ** s) / (1 - r ** N)


# ---------------------------------------------------------------- naive MC
def naive_mc(N, p, s0, M, rng):
    """Analog Monte Carlo: run M walks, score 1 if the walk reaches N first."""
    hits = 0
    steps = 0
    for _ in range(M):
        s = s0
        while 0 < s < N:
            s += 1 if rng.random() < p else -1
            steps += 1
        hits += (s == N)
    pi_hat = hits / M
    var = pi_hat * (1 - pi_hat) / M
    return pi_hat, var, steps


# --------------------------------------------- importance-sampled MC (tilt)
def tilted_mc(N, p, s0, M, h, rng):
    """Tilt the kernel by the importance ratio: p~(s)=p h(s+1)/h(s).
    Accumulate the likelihood-ratio weight. If h is the EXACT committor the
    estimator has zero variance; if h is approximate the variance is small."""
    q = 1 - p
    est = np.empty(M)
    steps = 0
    for m in range(M):
        s = s0
        w = 1.0
        while 0 < s < N:
            pr = p * h[s + 1] / h[s]                    # tilted right prob
            pl = q * h[s - 1] / h[s]                    # tilted left prob
            Z = pr + pl                                 # =1 exactly iff h exact
            pr, pl = pr / Z, pl / Z
            if rng.random() < pr:
                w *= p / (pr)                           # LR = K/Ktilde
                s += 1
            else:
                w *= q / (pl)
                s -= 1
            steps += 1
        est[m] = w if s == N else 0.0
    return est.mean(), est.var(ddof=1), steps


# NOTE: an earlier draft defined a weight-window sampler here. It was never
# called (dead code) and, worse, it applied windows to an UNBIASED analog walk
# without also tilting the kernel -- violating the CADIS consistency condition,
# which is why it underperformed. The correct, validated implementation lives in
# ../04_phase2_core/ww_1d.py (`tilt_plus_ww`), where it is checked for
# unbiasedness against the exact answer. Phase 1 deliberately shows only the
# zero-variance ideal and the fragility of pure tilting.


def fom(rel_var, total_steps):
    """Transport figure of merit FOM = 1/(sigma_rel^2 * cost)."""
    return 1.0 / (rel_var * total_steps) if rel_var > 0 else np.inf


def main():
    rng = np.random.default_rng(0)
    N, p, s0 = 15, 0.40, 1
    h = committor(N, p)
    pi = h[s0]
    print(f"Model: N={N}, p={p} (drift toward 0), start s0={s0}")
    print(f"Exact committor / importance  h(s0) = pi = {pi:.6e}  (deep-penetration prob)\n")

    # ---- naive MC ----
    M_naive = 400_000
    pihat, var, steps = naive_mc(N, p, s0, M_naive, rng)
    rel = var / pi ** 2
    print(f"[naive analog MC]      M={M_naive:,}")
    print(f"   pi_hat = {pihat:.6e}   rel.var = {rel:.4e}   steps = {steps:,}   "
          f"FOM = {fom(rel, steps):.3e}")

    # ---- zero-variance IS with EXACT importance ----
    M_zv = 2_000
    e, v, steps_zv = tilted_mc(N, p, s0, M_zv, h, rng)
    rel_zv = v / e ** 2 if e > 0 else 0.0
    print(f"\n[exact-importance IS]  M={M_zv:,}  (CADIS with the true adjoint)")
    print(f"   estimate = {e:.6e}   rel.var = {rel_zv:.3e}   steps = {steps_zv:,}   "
          f"FOM = {fom(rel_zv, steps_zv):.3e}")
    print("   -> every successful path carries the identical weight h(s0); variance ~ 0")

    # ---- approximate importance (cheap/wrong adjoint), realistic CADIS ----
    h_approx = committor(N, 0.43)          # cheap solve assumed slightly wrong drift
    h_approx[0], h_approx[-1] = 0.0, 1.0    # boundaries are exact by construction
    e2, v2, steps2 = tilted_mc(N, p, s0, 20_000, h_approx, rng)
    rel2 = v2 / e2 ** 2
    print("\n[approx-importance PURE TILT] M=20,000  (biasing with an INEXACT adjoint)")
    print(f"   estimate = {e2:.6e}   rel.var = {rel2:.4e}   steps = {steps2:,}   "
          f"FOM = {fom(rel2, steps2):.3e}   <-- FRAGILE: worse than naive")
    print("   (pure tilting is exquisitely sensitive to importance error; weight")
    print("    windows -- split/roulette -- are the transport fix, quantified in Phase 2.)")

    # ---- analytic deep-penetration scaling table (why naive collapses) ----
    print("\n[analytic] FOM per unit cost vs rarity (fixed statistics):")
    print("   naive rel.var ~ 1/(pi*M): FOM_naive ∝ pi ; exact-IS FOM independent of pi")
    rows = []
    for Ni in (10, 20, 30, 40):
        hi = committor(Ni, p)
        rows.append((Ni, hi[s0]))
        print(f"   N={Ni:2d}  pi={hi[s0]:.3e}   naive relative cost to 1% error ∝ 1/pi = {1/hi[s0]:.2e}")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    s = np.arange(N + 1)
    ax[0].plot(s, h, "o-", color=BLUE, lw=1.8, ms=5)
    ax[0].set_xlabel("state $s$ (depth into shield / distance to target set)")
    ax[0].set_ylabel(r"importance $h(s)=\psi^\dagger(s)$")
    ax[0].set_title("(a) Adjoint importance $=$ committor\n" r"$h(s)=p\,h(s{+}1)+q\,h(s{-}1)$")
    ax[0].set_yscale("log"); ax[0].grid(alpha=0.3, which="both")

    naive_fom = fom(rel, steps)
    fom_exact_display = naive_fom * 5e5                    # tall bar, labelled
    methods = ["naive\nanalog MC", "approx. imp.\npure tilt", "exact imp.\ntilt"]
    foms = [naive_fom, fom(rel2, steps2), fom_exact_display]
    colors = [RED, "#e08e0b", BLUE]
    ax[1].bar(methods, foms, color=colors)
    ax[1].axhline(naive_fom, color="k", ls=":", lw=0.8)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("figure of merit  $1/(\\sigma_{rel}^2\\,T)$")
    ax[1].set_title("(b) Exact importance $\\Rightarrow$ zero variance;\napprox. pure tilt is fragile")
    ax[1].text(2, fom_exact_display, r"$\to\infty$" "\n(zero var.)", ha="center",
               va="top", fontsize=8, color="white")
    ax[1].tick_params(axis="x", labelsize=9)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig("figs/committor_demo.pdf", bbox_inches="tight")
    print("\nsaved figs/committor_demo.pdf")


if __name__ == "__main__":
    main()
