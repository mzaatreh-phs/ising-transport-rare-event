"""
Generates the 5 figures for paper.tex from verified source data.

All numbers here were cross-checked against the raw result JSON files
(see comments per figure) or, where raw per-replica arrays no longer
exist (older R=10 Kim & Cai runs, predating raw-array logging), taken
verbatim from the already-published, audited paper.tex tables.

Run: python3 make_figures.py   (writes PDFs into this directory)
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent           # ~/ising_transport_project/project
MYTEST = Path.home() / "ising_transport_project_mytest" / "project"
NI = PROJ / "05_neural_importance"
FW = PROJ / "06_fw_cadis"

# ---- palette (dataviz skill, validated categorical order) ----
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
VIOLET = "#4a3aa7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

COLOR = {
    "naive": BLUE,
    "WE[m]": ORANGE,
    r"WE[$E$]": AQUA,
    r"WE[$I_\theta$]": YELLOW,
    r"WE[$I_\theta$] self-consistent": VIOLET,
    "uniform": BLUE,
    "FW-CADIS": ORANGE,
}

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.8,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.frameon": False,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
})


def style_axes(ax, logy=False):
    if logy:
        ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, color=BASELINE)


# ============================================================
# Figure 1: L-scaling (Table lscaling, sec:lscaling)
# Source: ~/ising_transport_project_mytest/project/05_neural_importance/
#         results_L{8,24,32}_fast_milestone.json (as-is) and
#         results_L16_correct_milestone.json (the fixed L=16 rerun —
#         the plain results_L16_milestone.json was overwritten by an
#         unrelated later run and does not match the paper's numbers).
# Verified 2026-08-07 against paper.tex Table~\ref{tab:lscaling}.
# ============================================================
def fig_lscaling():
    files = {
        8: "results_L8_milestone.json",
        16: "results_L16_correct_milestone.json",
        24: "results_L24_fast_milestone.json",
        32: "results_L32_fast_milestone.json",
    }
    Ls, naive_fom, best_fom, best_method = [], [], [], []
    for L, fn in files.items():
        d = json.load(open(MYTEST / "05_neural_importance" / fn))
        r = d["results"]
        Ls.append(L)
        naive_fom.append(r["naive"]["fom"])
        we = {k: v["fom"] for k, v in r.items() if k != "naive"}
        best_k = max(we, key=we.get)
        best_fom.append(we[best_k])
        best_method.append(best_k)

    ratio = [n / b for n, b in zip(naive_fom, best_fom)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7),
                                    gridspec_kw={"width_ratios": [1.3, 1]})

    x = np.arange(len(Ls))
    w = 0.34
    ax1.bar(x - w / 2, naive_fom, width=w, color=COLOR["naive"], label="naive", zorder=3)
    ax1.bar(x + w / 2, best_fom, width=w, color=YELLOW, label="best WE variant", zorder=3)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$L={L}$" for L in Ls])
    ax1.set_ylabel("figure of merit")
    style_axes(ax1, logy=True)
    ax1.legend(loc="upper right", handlelength=1.2)
    label_map = {"WE[I_theta]": r"WE[$I_\theta$]", "WE[E]": r"WE[$E$]", "WE[m]": "WE[$m$]"}
    for xi, m, h in zip(x, best_method, best_fom):
        ax1.text(xi + w / 2, h * 1.5, label_map.get(m, m),
                  ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax1.set_ylim(top=ax1.get_ylim()[1] * 3)

    ax2.plot(Ls, ratio, "-o", color=INK, markersize=5, zorder=3)
    ax2.axhline(1.0, color=BASELINE, lw=1, ls="--", zorder=1)
    ax2.set_xlabel("$L$")
    ax2.set_ylabel(r"naive FOM / best WE FOM")
    ax2.set_xticks(Ls)
    ax2.set_ylim(0, 4.2)
    style_axes(ax2)
    for xi, yi in zip(Ls, ratio):
        ax2.annotate(f"{yi:.1f}$\\times$", (xi, yi), textcoords="offset points",
                      xytext=(0, 7), ha="center", fontsize=7.5, color=INK2)

    fig.suptitle("Naive vs. best weight-window variant across lattice size, "
                  r"$\mathrm{beyond}=1$, $\tau=2$", fontsize=9, y=1.03)
    fig.tight_layout()
    fig.savefig(HERE / "fig_lscaling.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Figure 2: Milestoning seed sweep (Table milestone, sec:milestone)
# Source: 05_neural_importance/pooled_seed_bootstrap.log (17 point
# ratios + n=17 bootstrap CI, verified by re-reading the log directly)
# and the n=7/9/12 checkpoint CIs stated in paper.tex (already audited
# there; the raw per-checkpoint bootstrap draws were not persisted
# separately from the final n=17 run).
# ============================================================
def fig_seed_sweep():
    seeds = ["0", "unseeded2", "1", "2", "3", "4", "5", "6", "7",
             "8", "9", "10", "11", "12", "13", "14", "15"]
    ratios = [2.1213, 0.8267, 0.7335, 1.1043, 0.5435, 2.0913, 1.8160,
              1.5564, 0.4811, 1.8639, 1.3418, 1.3880, 0.8262, 0.6549,
              1.5099, 1.6044, 0.6299]
    order = np.argsort(ratios)
    ratios_sorted = np.array(ratios)[order]

    checkpoints_n = [7, 9, 12, 17]
    checkpoints_med = [1.28, 1.17, 1.28, 1.19]
    checkpoints_lo = [0.72, 0.70, 0.83, 0.84]
    checkpoints_hi = [2.45, 2.06, 2.01, 1.75]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7),
                                    gridspec_kw={"width_ratios": [1.15, 1]})

    y = np.arange(len(ratios_sorted))
    colors = [ORANGE if r < 1 else AQUA for r in ratios_sorted]
    ax1.scatter(ratios_sorted, y, s=26, color=colors, zorder=3)
    ax1.axvline(1.0, color=BASELINE, lw=1, ls="--", zorder=1)
    med17 = 1.1949
    ax1.axvline(med17, color=INK, lw=1.2, zorder=2)
    ax1.fill_betweenx([-1, len(y)], 0.8388, 1.7501, color=INK, alpha=0.08, zorder=1)
    ax1.set_yticks([])
    ax1.set_ylim(-1, len(y))
    ax1.set_xlabel(r"per-seed FOM ratio, WE[$I_\theta$] / WE[$E$]")
    ax1.set_ylabel("17 independent training seeds\n(sorted)")
    style_axes(ax1)
    ax1.text(med17, len(y) - 0.5, "  pooled\n  median 1.19$\\times$",
              fontsize=7, color=INK, va="top")

    ax2.errorbar(checkpoints_n, checkpoints_med,
                 yerr=[np.array(checkpoints_med) - np.array(checkpoints_lo),
                       np.array(checkpoints_hi) - np.array(checkpoints_med)],
                 fmt="-o", color=INK, ecolor=MUTED, elinewidth=1.2,
                 capsize=3, markersize=5, zorder=3)
    ax2.axhline(1.0, color=BASELINE, lw=1, ls="--", zorder=1)
    ax2.set_xlabel("seeds pooled ($n$)")
    ax2.set_ylabel("geometric-mean ratio, 95% CI")
    ax2.set_xticks(checkpoints_n)
    style_axes(ax2)

    fig.suptitle(r"Milestoning: WE[$I_\theta$] vs. WE[$E$], $L=16$, $\mathrm{beyond}=1$",
                 fontsize=9, y=1.03)
    fig.tight_layout()
    fig.savefig(HERE / "fig_seed_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Figure 3: FW-CADIS flatness vs threshold (Tables fwcadis, fwferro)
# Source: 06_fw_cadis/results_fwcadis_beyond0_r90_ea_L16_full.json
#         06_fw_cadis/results_fwcadis_ferro_b0_r30_ferro_L16_full.json
# Uses the per-threshold "rel" (relative std) arrays directly —
# richer than the single flatness (max/min) number in the table.
# ============================================================
def fig_fwcadis():
    ea = json.load(open(FW / "results_fwcadis_beyond0_r90_ea_L16_full.json"))
    ferro = json.load(open(FW / "results_fwcadis_ferro_b0_r30_ferro_L16_full.json"))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, d, title in zip(axes, [ea, ferro],
                             ["EA spin glass ($R=90$)", "ferromagnet ($R=30$)"]):
        thr = np.arange(1, len(d["results"]["uniform(a=0)"]["rel"]) + 1)
        u = d["results"]["uniform(a=0)"]
        f = d["results"]["FW-CADIS(a=1.0)"]
        ax.plot(thr, u["rel"], "-o", color=COLOR["uniform"], label="uniform",
                 markersize=4.5, zorder=3)
        ax.plot(thr, f["rel"], "-s", color=COLOR["FW-CADIS"], label="FW-CADIS",
                 markersize=4.5, zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel("threshold (shallow $\\to$ deep)")
        ax.set_xticks(thr)
        ax.set_title(
            f"{title}\nflatness {u['spread']:.0f} vs. {f['spread']:.0f}",
            fontsize=8)
        style_axes(ax, logy=True)
    axes[0].set_ylabel("relative error per threshold")
    axes[0].legend(loc="upper left", handlelength=1.4)

    fig.suptitle("Per-threshold relative error: uniform vs. FW-CADIS allocation",
                 fontsize=9, y=1.05)
    fig.tight_layout()
    fig.savefig(HERE / "fig_fwcadis.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Figure 4: Kim & Cai forest plot (Tables kimcaiboot, kimcaideep)
# R=30 rows: recomputed directly here from per-replica "est" arrays
# in 05_neural_importance/results_kimcai_{sc,surrogate}_b0_r30_*.json,
# using the same resample-the-replica-axis bootstrap described in
# paper.tex sec:methods. Independently reproduced the published
# medians/CIs to within Monte-Carlo noise (median 1.41 vs. paper's
# 1.39; 1.07 vs. 1.07) before use here.
# R=10 rows: those runs predate raw-array logging (only summary
# stats were saved), so the paper.tex Table~kimcaiboot/kimcaideep
# values are used verbatim.
# ============================================================
def _load_rows(fn):
    d = json.load(open(NI / fn))
    return {r["method"]: r for r in d["rows"]}


def _boot_ratio(rowA, rowB, n_draws=20000, seed=0):
    rng = np.random.default_rng(seed)

    def fom_draw():
        idxA = rng.integers(0, len(rowA["est"]), len(rowA["est"]))
        idxB = rng.integers(0, len(rowB["est"]), len(rowB["est"]))
        sA = np.array(rowA["est"])[idxA]
        sB = np.array(rowB["est"])[idxB]
        mA, mB = sA.mean(), sB.mean()
        if mA == 0 or mB == 0:
            return None
        rsdA, rsdB = sA.std(ddof=1) / mA, sB.std(ddof=1) / mB
        if rsdA == 0 or rsdB == 0:
            return None
        fomA = 1.0 / (rsdA ** 2 * rowA["cost"])
        fomB = 1.0 / (rsdB ** 2 * rowB["cost"])
        return fomA / fomB

    ratios = [v for v in (fom_draw() for _ in range(n_draws)) if v is not None]
    ratios = np.array(ratios)
    return np.median(ratios), np.percentile(ratios, 2.5), np.percentile(ratios, 97.5)


def fig_kimcai():
    sc30 = _load_rows("results_kimcai_sc_b0_r30_ea_L16_full_selfconsistent.json")
    surr30 = _load_rows("results_kimcai_surrogate_b0_r30_ea_L16_full.json")
    med_ss30, lo_ss30, hi_ss30 = _boot_ratio(sc30["WE[I_theta]"], surr30["WE[I_theta]"])
    med_se30, lo_se30, hi_se30 = _boot_ratio(sc30["WE[I_theta]"], sc30["WE[E]"])

    rows = [
        ("self-consistent / surrogate, $R{=}10$, beyond=0", 1.83, 0.49, 9.23),
        ("self-consistent / surrogate, $R{=}30$, beyond=0", med_ss30, lo_ss30, hi_ss30),
        ("self-consistent / surrogate, $R{=}10$, beyond=1", 1.14, 0.27, 4.81),
        (r"self-consistent / WE[$E$], $R{=}10$, beyond=0", 1.19, 0.33, 4.95),
        (r"self-consistent / WE[$E$], $R{=}30$, beyond=0", med_se30, lo_se30, hi_se30),
        (r"self-consistent / WE[$E$], $R{=}10$, beyond=1", 1.33, 0.34, 5.74),
    ]

    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    y = np.arange(len(rows))[::-1]
    for yi, (label, med, lo, hi) in zip(y, rows):
        c = VIOLET if "surrogate" in label else BLUE
        ax.errorbar(med, yi, xerr=[[med - lo], [hi - med]], fmt="o",
                     color=c, ecolor=c, elinewidth=1.4, capsize=3,
                     markersize=5, zorder=3)
    ax.axvline(1.0, color=BASELINE, lw=1, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlabel("FOM ratio (median, 95% bootstrap CI)")
    ax.set_xlim(0, 10)
    style_axes(ax)
    ax.set_title("Kim & Cai self-consistency objective vs. this project's "
                  "surrogate and $\\mathrm{WE}[E]$", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "fig_kimcai.pdf", bbox_inches="tight")
    plt.close(fig)
    return med_ss30, lo_ss30, hi_ss30, med_se30, lo_se30, hi_se30


# ============================================================
# Figure 5: learned vs. hand-picked coordinates (Table prefix,
# sec:res-learned). Values taken verbatim from paper.tex Table~1
# (tab:prefix) — the underlying raw-result file for this specific
# run was overwritten by a later, differently-configured run sharing
# the same default output filename (confirmed while sourcing figure
# data: naive/WE[E] match a same-named file but WE[I_theta] does not,
# consistent with milestoning's known unseeded run-to-run spread), so
# the published table is the only recoverable record of this exact run.
# ============================================================
def fig_prefix():
    ferro = {
        "naive": (3.83e-8, 0),
        "WE[m]": (1.31e-8, 0),
        r"WE[$E$]": (5.65e-9, 2),
        r"WE[$I_\theta$]": (6.90e-9, 0),
    }
    ea = {
        "naive": (1.86e-8, 1),
        "WE[m]": (7.80e-9, 7),
        r"WE[$E$]": (5.00e-9, 2),
        r"WE[$I_\theta$]": (6.36e-9, 2),
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharey=True)
    for ax, data, title in zip(axes, [ferro, ea], ["ferromagnet", "EA spin glass"]):
        methods = list(data.keys())
        fom = [data[m][0] for m in methods]
        zeros = [data[m][1] for m in methods]
        colors = [COLOR[m] for m in methods]
        bars = ax.bar(methods, fom, color=colors, zorder=3, width=0.6)
        ax.set_yscale("log")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=0, fontsize=7.5)
        ax.set_title(f"{title}", fontsize=8.5, pad=10)
        style_axes(ax, logy=True)
        ax.set_ylim(top=max(fom) * 6)
        for b, z in zip(bars, zeros):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.35,
                     f"{z}/10\nzero-rep.", ha="center", va="bottom",
                     fontsize=6, color=INK2)
    axes[0].set_ylabel("figure of merit")

    fig.suptitle(r"Full scale ($L=16$, 10 replicas), target one step beyond "
                 "the pilot's observed extreme", fontsize=9, y=1.06)
    fig.tight_layout()
    fig.savefig(HERE / "fig_prefix.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_lscaling()
    print("fig_lscaling.pdf done")
    fig_seed_sweep()
    print("fig_seed_sweep.pdf done")
    fig_fwcadis()
    print("fig_fwcadis.pdf done")
    m1, l1, h1, m2, l2, h2 = fig_kimcai()
    print(f"fig_kimcai.pdf done  (R=30 recheck: sc/surr {m1:.2f} [{l1:.2f},{h1:.2f}], "
          f"sc/WE[E] {m2:.2f} [{l2:.2f},{h2:.2f}])")
    fig_prefix()
    print("fig_prefix.pdf done")
