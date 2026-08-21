"""
replot_tail_r20.py -- regenerate figs/ising_tail_r20.pdf from the already-saved
tail_data_r20.npz (no re-simulation), with the panel (a) title corrected to
match the data: both methods reach the identical deepest bin (|m|=0.932), so
the earlier "WE reaches beyond naive's hard support floor" title was wrong.
"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

TC = 2.269

d = np.load("tail_data_r20.npz")
centers, mW, sdW, relW = d["centers"], d["mW"], d["sdW"], d["relW"]
mN, sdN, relN = d["mN"], d["sdN"], d["relN"]
L, T, N, R = int(d["L"]), float(d["T"]), int(d["N"]), int(d["R"])

okW, okN = mW > 0, mN > 0
naive_floor = centers[okN].max()
we_floor = centers[okW].max()

fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.1))
RED, BLUE = "#c0392b", "#2c5aa0"

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
ax[0].set_title(f"(a) WE and naive resolve the tail to matched depth\n"
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
print(f"naive floor |m|={naive_floor:.3f}; WE reaches |m|={we_floor:.3f}")
print("saved figs/ising_tail_r20.pdf")
