"""
run.py -- train a VAN across a temperature sweep (annealing) and produce the
thermodynamics figure benchmarked against the exact Onsager solution.

Usage:
    python run.py                 # default L=8 demo (~6 min on 1 CPU)
    python run.py --L 16 --steps0 1000 --steps 400   # bigger, slower, better
"""
import argparse, csv, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ising_van import (VAN, train_one_temperature, measure,
                       onsager_free_energy_per_spin, exact_enumeration)

TC = 2.0 / np.log(1 + np.sqrt(2))          # 2.269185...


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--steps0", type=int, default=400, help="steps at first (hottest) T")
    ap.add_argument("--steps", type=int, default=220, help="steps per subsequent T (warm-started)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    # sanity check against exact enumeration on a tiny lattice
    ex = exact_enumeration(4, 2.0)
    print(f"[check] exact L=4 @ T=2.0: F/N={ex['F_per_spin']:.4f}  "
          f"E/N={ex['E_per_spin']:.4f}  |m|={ex['m_abs']:.4f}")

    # temperature grid, hot -> cold (annealing / warm start)
    T_grid = np.concatenate([
        np.linspace(3.5, TC + 0.15, 5),
        np.linspace(TC + 0.05, TC - 0.05, 5),      # dense near Tc
        np.linspace(TC - 0.15, 1.0, 5),
    ])
    T_grid = np.round(np.unique(T_grid)[::-1], 4)  # descending, unique

    van = VAN(args.L, hidden=tuple(args.hidden))
    rows = []
    t_start = time.time()
    for i, T in enumerate(T_grid):
        steps = args.steps0 if i == 0 else args.steps
        train_one_temperature(van, float(T), steps=steps, batch=args.batch,
                               lr=args.lr, beta_anneal=(i == 0))
        obs = measure(van, float(T), n=20000)
        obs["F_exact"] = onsager_free_energy_per_spin(float(T))
        rows.append(obs)
        print(f"[{i+1:2d}/{len(T_grid)}] T={T:6.3f}  "
              f"F/N={obs['F_per_spin']:+.4f} (Onsager {obs['F_exact']:+.4f})  "
              f"E/N={obs['E_per_spin']:+.4f}  C={obs['C_per_spin']:.3f}  "
              f"|m|={obs['m_abs']:.3f}  [{time.time()-t_start:.0f}s]")

    # ---- save CSV ----
    keys = ["T", "F_per_spin", "F_exact", "E_per_spin", "C_per_spin", "m_abs", "S_per_spin"]
    with open(f"{args.out}_L{args.L}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})

    # ---- figure ----
    R = {k: np.array([r[k] for r in rows]) for k in keys}
    Tfine = np.linspace(T_grid.min(), T_grid.max(), 200)
    Fons = np.array([onsager_free_energy_per_spin(t) for t in Tfine])

    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 130})
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))

    ax[0, 0].plot(Tfine, Fons, "-", color="k", lw=1.4, label="Onsager (exact, L→∞)")
    ax[0, 0].plot(R["T"], R["F_per_spin"], "o", ms=5, color="crimson",
                  label=f"VAN (L={args.L})")
    ax[0, 0].set_ylabel(r"free energy  $F/N$")

    ax[0, 1].plot(R["T"], R["E_per_spin"], "s-", ms=4, color="steelblue")
    ax[0, 1].set_ylabel(r"energy  $E/N$")

    ax[1, 0].plot(R["T"], R["C_per_spin"], "^-", ms=4, color="darkgreen")
    ax[1, 0].set_ylabel(r"specific heat  $C/N$")

    ax[1, 1].plot(R["T"], R["m_abs"], "d-", ms=4, color="darkorange")
    ax[1, 1].set_ylabel(r"magnetization  $\langle|m|\rangle$")

    for a in ax.ravel():
        a.axvline(TC, color="gray", ls=":", lw=1)
        a.set_xlabel("temperature  T")
    ax[0, 0].legend(frameon=False, fontsize=9)
    ax[0, 0].annotate(r"$T_c$", (TC, ax[0, 0].get_ylim()[0]),
                      xytext=(3, 3), textcoords="offset points", color="gray")
    fig.suptitle(f"2D Ising via Variational Autoregressive Network  "
                 f"(L={args.L}, MADE {args.hidden})", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{args.out}_L{args.L}_thermo.png", bbox_inches="tight")
    print(f"\nSaved {args.out}_L{args.L}.csv and {args.out}_L{args.L}_thermo.png")
    print(f"Total wall time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
