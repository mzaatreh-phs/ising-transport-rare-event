"""
diagnose_coordinate.py -- is the learned importance map actually learning anything
that the energy does not already tell us?

MOTIVATION. On the L=16 spin-glass run the network trained well (MSE 1.0 -> 0.27)
yet WE[I_theta] performed like WE[E] (pi 3.51e-6 vs 3.91e-6; 7/10 vs 6/10 dead
replicas). The simplest explanation is that I_theta(s) is a noisy re-derivation of
E/N: the training label is "best progress value reached in K sweeps", and progress
IS energy for the `ea` model, so a network can score well on that label by simply
reading off the current energy -- learning nothing about WHICH configurations at a
given energy are on their way down.

If that is what happened, no amount of extra training will help; the LABEL has to
change (milestoning / committor between intermediate thresholds), not the network.

This script measures it directly:
  * Pearson and Spearman correlation between I_theta(s) and E/N(s)
  * the residual structure: how much variance in I_theta survives after
    regressing out E/N (this is the only part that could beat WE[E])
  * whether I_theta discriminates *within* a narrow energy shell -- the decisive
    test, since inside a shell E/N is constant by construction

Usage:
    python3 diagnose_coordinate.py --model ea --preset full
    python3 diagnose_coordinate.py --model ea --preset laptop   # faster
"""
from __future__ import annotations
import argparse

import numpy as np
import torch

from neural_importance import (PRESETS, make_couplings, calibrate_threshold,
                               collect_training_configs, train_importance,
                               rollout_labels, energy_per_spin, magnetization,
                               NetCoord)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ferro", "ea"], default="ea")
    ap.add_argument("--preset", choices=list(PRESETS), default="laptop")
    ap.add_argument("--beyond", type=int, default=2)
    ap.add_argument("--shell-width", type=float, default=0.02,
                    help="half-width in E/N of the narrow energy shell")
    a = ap.parse_args()

    cfg = dict(PRESETS[a.preset])
    L, T = cfg["L"], cfg["T"]
    Jx, Jy = make_couplings(L, a.model, seed=12345)
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)

    print(f"model={a.model} L={L} T={T} preset={a.preset}\n")
    thresh, _bulk = calibrate_threshold(L, T, Jx, Jy, a.model,
                                        beyond=a.beyond, seed=99)

    print("\ntraining I_theta (same procedure as the main script) ...")
    net = train_importance(L, T, Jx, Jy, a.model, thresh, cfg, seed=0)
    coord = NetCoord(net)

    print("\ncollecting an independent evaluation set ...")
    X = collect_training_configs(L, T, Jx, Jy, a.model, thresh,
                                 cfg["collect_iter"], cfg["collect_walkers"], rng)
    if len(X) > 4000:
        X = X[rng.choice(len(X), 4000, replace=False)]
    I = coord(X)
    E = energy_per_spin(X, Jx, Jy)
    M = np.abs(magnetization(X))

    print(f"\nevaluation set: {len(X)} configs, "
          f"E/N in [{E.min():.3f}, {E.max():.3f}]")

    # ---------------------------------------------------------------- global
    pear_E = float(np.corrcoef(I, E)[0, 1])
    spear_E = spearman(I, E)
    pear_M = float(np.corrcoef(I, M)[0, 1])
    print("\n--- Is I_theta just energy? ---")
    print(f"  Pearson  corr(I_theta, E/N) = {pear_E:+.4f}")
    print(f"  Spearman corr(I_theta, E/N) = {spear_E:+.4f}")
    print(f"  Pearson  corr(I_theta, |m|) = {pear_M:+.4f}   (sanity: should be weak for ea)")

    # variance of I_theta not explained by a linear function of E
    slope, intercept = np.polyfit(E, I, 1)
    resid = I - (slope * E + intercept)
    frac_resid = float(resid.var() / I.var())
    print(f"  fraction of Var(I_theta) NOT explained by E/N = {frac_resid:.4f}")

    # ------------------------------------------------- within an energy shell
    # The decisive test. Inside a narrow shell E/N is ~constant, so ANY useful
    # ordering I_theta provides here is information energy does not have.
    centre = float(np.quantile(E, 0.05))          # deep-ish shell, still populated
    width = a.shell_width
    sel = np.abs(E - centre) <= width
    while sel.sum() < 80 and width < 0.25:        # widen rather than give up
        width *= 1.5
        sel = np.abs(E - centre) <= width
    print(f"\n--- Within a narrow energy shell (E/N = {centre:.3f} "
          f"+/- {width:.3f}) ---")
    print(f"  shell population: {sel.sum()} configs")
    if sel.sum() < 50:
        print("  too few configs even after widening; use a larger preset.")
    else:
        Ish, Esh = I[sel], E[sel]
        print(f"  spread of I_theta inside the shell: sd = {Ish.std():.4f} "
              f"(global sd = {I.std():.4f}, ratio = {Ish.std()/I.std():.3f})")
        print(f"  residual corr(I_theta, E/N) inside shell = "
              f"{float(np.corrcoef(Ish, Esh)[0, 1]):+.4f}")

    # ------------------------------------------- decisive: does I add SIGNAL?
    # Being uncorrelated with E/N is necessary but NOT sufficient. An
    # undertrained network outputs near-noise, and noise is uncorrelated with
    # energy by construction -- so "independent variance" alone cannot
    # distinguish signal from noise. The honest test is whether I_theta improves
    # prediction of the LABEL beyond what E/N already achieves.
    print("\n--- Does I_theta add PREDICTIVE power beyond E/N? ---")
    print("   (computing rollout labels on the evaluation set ...)")
    n_eval = min(len(X), 600)                 # rollouts are the expensive part
    idx = rng.choice(len(X), n_eval, replace=False)
    Xe, Ie, Ee = X[idx], I[idx], E[idx]
    ylab = rollout_labels(Xe, T, Jx, Jy, a.model,
                          cfg["roll_K"], cfg["n_roll"], rng)

    def r2(preds, y):
        preds = np.atleast_2d(preds).T if preds.ndim == 1 else preds
        A = np.hstack([preds, np.ones((len(y), 1))])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        return float(1 - resid.var() / y.var())

    r2_E = r2(Ee, ylab)
    r2_I = r2(Ie, ylab)
    r2_EI = r2(np.column_stack([Ee, Ie]), ylab)
    incr = r2_EI - r2_E
    print(f"  R^2(label ~ E/N)          = {r2_E:.4f}")
    print(f"  R^2(label ~ I_theta)      = {r2_I:.4f}")
    print(f"  R^2(label ~ E/N + I_theta)= {r2_EI:.4f}")
    print(f"  INCREMENTAL R^2 from I_theta beyond E/N = {incr:+.4f}")

    # ------------------------------------------------------------- verdict
    print("\n--- verdict ---")
    undertrained = r2_I < 0.15
    if undertrained:
        print(f"  WARNING: I_theta predicts the label poorly (R^2 = {r2_I:.3f}).")
        print("  The network is UNDERTRAINED, so its variance is largely noise and")
        print("  the correlation numbers above cannot be interpreted. Rerun with")
        print("  --preset full (more data, more epochs, wider net) before drawing")
        print("  any conclusion; the laptop preset trains a much weaker model than")
        print("  the one the main comparison actually deploys.")
    elif incr < 0.02:
        print("  I_theta adds essentially NOTHING beyond energy for predicting the")
        print("  label. Whatever variance it has that energy lacks is noise, not")
        print("  signal. Binning on it cannot beat WE[E].")
        print("  Fix the LABEL, not the network: use milestoning (committor between")
        print("  consecutive intermediate thresholds) so the target distinguishes")
        print("  configurations at the SAME energy.")
    else:
        print(f"  I_theta adds real predictive power beyond energy "
              f"(incremental R^2 = {incr:+.3f}).")
        print("  If it still does not beat WE[E] in sampling, the bottleneck is the")
        print("  weight-window schedule (bin range, walkers per bin, tau), not the map.")


if __name__ == "__main__":
    main()
