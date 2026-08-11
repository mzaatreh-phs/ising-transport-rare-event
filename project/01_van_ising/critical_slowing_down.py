"""
critical_slowing_down.py
========================
The physics thesis of the VAN paper, made visible.

Local Monte Carlo (single-spin Metropolis) suffers *critical slowing down*:
near the critical temperature Tc the integrated autocorrelation time tau_int of
observables diverges (as a power of the correlation length / system size), so
successive samples are highly correlated and the effective sample size collapses.

A variational autoregressive network draws configurations INDEPENDENTLY (each
sample is a fresh sequential draw from q_theta), so its autocorrelation time is
tau_int = 1 by construction, at every temperature -- there is no Markov chain.

This script measures tau_int(T) for Metropolis on an L x L lattice and plots it
against the flat VAN baseline. (The classical algorithmic remedy is a cluster
update such as Wolff/Swendsen-Wang; VAN is a learned, model-agnostic alternative
that also yields a direct free-energy estimate, which cluster methods do not.)
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TC = 2.0 / np.log(1 + np.sqrt(2))


def metropolis_energy_series(L, T, n_sweeps, burn_frac=0.2, seed=0):
    """Checkerboard (vectorized) Metropolis. Returns energy-per-spin per sweep."""
    rng = np.random.default_rng(seed)
    beta = 1.0 / T
    s = rng.choice([-1.0, 1.0], size=(L, L))
    ii, jj = np.indices((L, L))
    even = (ii + jj) % 2 == 0
    odd = ~even
    burn = int(burn_frac * n_sweeps)
    E = np.empty(n_sweeps - burn)

    def neigh(x):
        return (np.roll(x, 1, 0) + np.roll(x, -1, 0)
                + np.roll(x, 1, 1) + np.roll(x, -1, 1))

    for sweep in range(n_sweeps):
        for mask in (even, odd):
            dE = 2.0 * s * neigh(s)                       # deltaE if flipped
            accept = (rng.random((L, L)) < np.exp(-beta * dE)) & mask
            s[accept] *= -1
        if sweep >= burn:
            right = s * np.roll(s, -1, 1)
            down = s * np.roll(s, -1, 0)
            E[sweep - burn] = -(right.sum() + down.sum()) / (L * L)
    return E


def integrated_autocorr_time(x, c=6.0):
    """Sokal automatic-windowing estimator of tau_int for a 1D series."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    # autocovariance via FFT
    f = np.fft.rfft(x, n=2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:n].real / (n - np.arange(n))
    if acov[0] <= 0:
        return 1.0
    rho = acov / acov[0]
    tau = 1.0
    for W in range(1, n):
        tau = 1.0 + 2.0 * rho[1:W + 1].sum()
        if W >= c * tau:                                   # window self-consistent
            break
    return max(1.0, tau)


def main():
    L = 32
    n_sweeps = 60000
    temps = np.round(np.concatenate([
        np.linspace(1.6, TC - 0.1, 4),
        np.linspace(TC - 0.05, TC + 0.05, 5),
        np.linspace(TC + 0.15, 3.2, 4),
    ]), 3)
    temps = np.unique(temps)

    print(f"Metropolis critical slowing down, L={L}, {n_sweeps} sweeps/T")
    taus, t0 = [], time.time()
    for T in temps:
        E = metropolis_energy_series(L, float(T), n_sweeps, seed=1)
        tau = integrated_autocorr_time(E)
        taus.append(tau)
        print(f"  T={T:5.3f}  tau_int(E) = {tau:6.1f}   [{time.time()-t0:.0f}s]")
    taus = np.array(taus)

    plt.rcParams.update({"font.size": 12, "figure.dpi": 130})
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.plot(temps, taus, "o-", color="crimson", lw=1.6,
            label=f"single-spin Metropolis (L={L})")
    ax.axhline(1.0, color="steelblue", lw=2.2,
               label="VAN (i.i.d. samples): "r"$\tau_{\mathrm{int}}=1$")
    ax.axvline(TC, color="gray", ls=":", lw=1.2)
    ax.annotate(r"$T_c$", (TC, ax.get_ylim()[1]*0.9), color="gray")
    ax.set_xlabel("temperature  T")
    ax.set_ylabel(r"integrated autocorrelation time  $\tau_{\mathrm{int}}$ (sweeps)")
    ax.set_title("Critical slowing down: local MC vs. autoregressive network")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("critical_slowing_down.png", bbox_inches="tight")
    np.savetxt("critical_slowing_down.csv",
               np.column_stack([temps, taus]),
               header="T,tau_int_metropolis", delimiter=",", comments="")
    print("Saved critical_slowing_down.png / .csv  "
          f"(total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
