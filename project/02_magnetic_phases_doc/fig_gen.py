"""Generate all physics illustrations for the magnetic-phases teaching document.
Outputs vector PDFs into ./figs/. Physics figures are computed, not cartoons."""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

os.makedirs("figs", exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150,
                     "axes.linewidth": 0.8, "savefig.bbox": "tight"})
TC = 2.0 / np.log(1 + np.sqrt(2))
RED, BLUE = "#c0392b", "#2c5aa0"


# ----------------------------------------------------------------------
def draw_ising(ax, config, title):
    """config: L x L of +/-1. Up = red arrow, down = blue arrow."""
    L = config.shape[0]
    Y, X = np.mgrid[0:L, 0:L]
    U = np.zeros_like(config, float)
    V = config.astype(float)
    col = np.where(config > 0, 0, 1)
    ax.quiver(X, Y, U, V, col, cmap=matplotlib.colors.ListedColormap([RED, BLUE]),
              pivot="mid", scale=L * 1.25, width=0.012, headwidth=4, headlength=5)
    ax.set_xlim(-0.7, L - 0.3); ax.set_ylim(-0.7, L - 0.3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)


def fig_phases():
    rng = np.random.default_rng(3)
    L = 12
    fm = np.ones((L, L), int)
    ii, jj = np.indices((L, L))
    afm = np.where((ii + jj) % 2 == 0, 1, -1)
    pm = rng.choice([-1, 1], (L, L))
    sg = rng.choice([-1, 1], (L, L))       # frozen random
    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
    draw_ising(ax[0], pm, "Paramagnet\n(disordered, fluctuating)")
    draw_ising(ax[1], fm, "Ferromagnet\n" r"$m\neq0$, uniform")
    draw_ising(ax[2], afm, "Antiferromagnet\n" r"staggered $m_s\neq0$")
    draw_ising(ax[3], sg, "Spin glass\n(frozen, random)")
    fig.tight_layout(); fig.savefig("figs/phases.pdf"); plt.close(fig)


# ----------------------------------------------------------------------
def fig_skyrmion():
    L = 21
    x = np.linspace(-1, 1, L); X, Y = np.meshgrid(x, x)
    r = np.sqrt(X**2 + Y**2); phi = np.arctan2(Y, X)
    theta = np.pi * np.clip(1 - r, 0, 1)          # pi at center -> 0 at edge
    # Neel skyrmion: in-plane points radially
    mx = np.sin(theta) * np.cos(phi)
    my = np.sin(theta) * np.sin(phi)
    mz = np.cos(theta)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    q = ax.quiver(X, Y, mx, my, mz, cmap="coolwarm",
                  norm=TwoSlopeNorm(0, -1, 1), pivot="mid",
                  scale=22, width=0.006, headwidth=4)
    cb = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"out-of-plane spin $S_z$")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Magnetic skyrmion\n(topological winding number $Q=1$)")
    fig.tight_layout(); fig.savefig("figs/skyrmion.pdf"); plt.close(fig)


# ----------------------------------------------------------------------
def fig_frustration():
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    # (a) Ising triangle frustration
    P = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    ax[0].add_patch(plt.Polygon(P, fill=False, ec="gray", lw=1.5))
    lab = ["$\\uparrow$", "$\\downarrow$", "?"]
    col = [RED, BLUE, "green"]
    for (px, py), t, c in zip(P, lab, col):
        ax[0].plot(px, py, "o", ms=26, mfc="white", mec=c, mew=2)
        ax[0].text(px, py, t, ha="center", va="center", fontsize=16, color=c)
    ax[0].text(0.5, -0.28, "no assignment satisfies all\nthree antiferro bonds",
               ha="center", fontsize=9)
    ax[0].set_xlim(-0.4, 1.4); ax[0].set_ylim(-0.5, 1.15)
    ax[0].set_aspect("equal"); ax[0].axis("off")
    ax[0].set_title("(a) Geometric frustration", fontsize=11)
    # (b) 120-degree compromise order
    ang = {0: 90, 1: 210, 2: 330}
    for (px, py), k in zip(P, ang):
        a = np.deg2rad(ang[k])
        ax[1].plot(px, py, "o", ms=10, color="k")
        ax[1].arrow(px, py, 0.28 * np.cos(a), 0.28 * np.sin(a),
                    head_width=0.06, color="purple", lw=2, length_includes_head=True)
    ax[1].add_patch(plt.Polygon(P, fill=False, ec="gray", lw=1.5))
    ax[1].set_xlim(-0.4, 1.4); ax[1].set_ylim(-0.5, 1.15)
    ax[1].set_aspect("equal"); ax[1].axis("off")
    ax[1].set_title(r"(b) $120^\circ$ compromise order", fontsize=11)
    fig.tight_layout(); fig.savefig("figs/frustration.pdf"); plt.close(fig)


# ----------------------------------------------------------------------
def fig_vortex():
    L = 15
    x = np.arange(L); X, Y = np.meshgrid(x, x)
    def angle(cx, cy, charge):
        return charge * np.arctan2(Y - cy, X - cx)
    th = angle(4, 7, +1) + angle(10, 7, -1)      # vortex + antivortex
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.quiver(X, Y, np.cos(th), np.sin(th), pivot="mid",
              scale=26, width=0.005, color="#333")
    ax.plot(4, 7, "o", ms=9, mfc=RED, mec="k"); ax.text(4, 8.4, "vortex\n$+1$", ha="center", fontsize=9, color=RED)
    ax.plot(10, 7, "o", ms=9, mfc=BLUE, mec="k"); ax.text(10, 8.4, "antivortex\n$-1$", ha="center", fontsize=9, color=BLUE)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(r"XY model: a vortex--antivortex pair (BKT physics)")
    fig.tight_layout(); fig.savefig("figs/vortex.pdf"); plt.close(fig)


# ----------------------------------------------------------------------
def fig_orderparam():
    T = np.linspace(0.5, 3.5, 600)
    beta = 1.0 / T
    s = np.sinh(2 * beta)
    m = np.where(T < TC, np.power(np.clip(1 - s**(-4), 0, None), 1 / 8), 0.0)
    chi = 1.0 / np.abs(T - TC)**1.75            # schematic gamma=7/4 divergence
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax[0].plot(T, m, color=RED, lw=2)
    ax[0].axvline(TC, color="gray", ls=":"); ax[0].set_xlabel("temperature $T$")
    ax[0].set_ylabel(r"spontaneous magnetization $m$")
    ax[0].text(TC + 0.05, 0.8, "$T_c$", color="gray")
    ax[0].text(1.0, 0.35, r"$m\sim(T_c-T)^{\beta}$" "\n" r"$\beta=1/8$", fontsize=9)
    ax[0].set_title("(a) Order parameter (exact, 2D Ising)")
    ax[1].plot(T, np.clip(chi, 0, 20), color=BLUE, lw=2)
    ax[1].axvline(TC, color="gray", ls=":"); ax[1].set_xlabel("temperature $T$")
    ax[1].set_ylabel(r"susceptibility $\chi$ (schematic)")
    ax[1].text(TC + 0.05, 15, "$T_c$", color="gray")
    ax[1].text(2.7, 12, r"$\chi\sim|T-T_c|^{-\gamma}$" "\n" r"$\gamma=7/4$", fontsize=9)
    ax[1].set_title("(b) Response function diverges at $T_c$")
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figs/orderparam.pdf"); plt.close(fig)


# ----------------------------------------------------------------------
def fig_phasediagram():
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    # schematic B-T diagram with a skyrmion-lattice pocket
    ax.fill_between([0, 1], 0, 1, color="#eef3fa")                 # background PM at high T handled below
    ax.fill_betweenx([0, 0.28], 0, 0.62, color="#f6d6c8", label="helical / spiral")
    ax.fill([0.18, 0.5, 0.5, 0.18], [0.28, 0.28, 0.55, 0.55],
            color="#c9a0dc", alpha=0.9)
    ax.text(0.34, 0.42, "skyrmion\nlattice", ha="center", fontsize=9)
    ax.fill_between([0, 0.62], 0.55, 1.0, color="#cfe8cf")
    ax.text(0.3, 0.8, "field-polarized\nferromagnet", ha="center", fontsize=9)
    ax.text(0.13, 0.13, "helical", ha="center", fontsize=9)
    ax.fill([0.62, 1.0, 1.0, 0.62], [0, 0, 1, 1], color="#eaeaea")
    ax.text(0.82, 0.5, "paramagnet", ha="center", fontsize=9)
    ax.plot([0.62, 0.62], [0, 1], color="k", lw=1, ls="--")
    ax.set_xlabel("temperature $T$"); ax.set_ylabel("magnetic field $B$")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Schematic field--temperature phase diagram\n(chiral magnet)")
    fig.tight_layout(); fig.savefig("figs/phasediagram.pdf"); plt.close(fig)


# ----------------------------------------------------------------------
def metropolis_configs(L, T, n_cfg, seed, therm=400, gap=8):
    rng = np.random.default_rng(seed)
    beta = 1.0 / T
    s = rng.choice([-1.0, 1.0], (L, L))
    ii, jj = np.indices((L, L)); even = (ii + jj) % 2 == 0
    def neigh(x): return (np.roll(x,1,0)+np.roll(x,-1,0)+np.roll(x,1,1)+np.roll(x,-1,1))
    def sweep():
        for m in (even, ~even):
            dE = 2*s*neigh(s)
            acc = (rng.random((L,L)) < np.exp(-beta*dE)) & m
            s[acc] *= -1
    for _ in range(therm): sweep()
    out = []
    for _ in range(n_cfg):
        for _ in range(gap): sweep()
        out.append(s.copy())
    return np.array(out)


def fig_pca():
    """Real unsupervised learning: PCA on Ising configs recovers the order
    parameter as PC1 and separates ordered from disordered phase."""
    L = 16
    temps = np.round(np.linspace(1.2, 3.4, 20), 3)
    X, Tlab = [], []
    for k, T in enumerate(temps):
        cfg = metropolis_configs(L, float(T), n_cfg=25, seed=100 + k)
        # fix global Z2 sign so PC1 is meaningful (align to positive magnetization)
        signs = np.sign(cfg.reshape(len(cfg), -1).sum(1)); signs[signs == 0] = 1
        cfg = cfg * signs[:, None, None]
        X.append(cfg.reshape(len(cfg), -1)); Tlab += [T] * len(cfg)
    X = np.vstack(X).astype(float); Tlab = np.array(Tlab)
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pc = Xc @ Vt[:2].T
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
    sc = ax[0].scatter(pc[:, 0], pc[:, 1], c=Tlab, cmap="viridis", s=14)
    cb = fig.colorbar(sc, ax=ax[0]); cb.set_label("temperature $T$")
    ax[0].set_xlabel("principal component 1"); ax[0].set_ylabel("principal component 2")
    ax[0].set_title("(a) Unsupervised: PCA of raw spin configs")
    # PC1 vs T next to true |m|
    ax[1].scatter(Tlab, np.abs(pc[:, 0]), s=10, color=BLUE, label="|PC1| (learned)")
    truem = np.abs(X.reshape(len(X), -1).sum(1)) / (L * L)
    ax[1].scatter(Tlab, truem * np.abs(pc[:, 0]).max() / truem.max(),
                  s=10, color=RED, alpha=0.5, label=r"$|m|$ (rescaled)")
    ax[1].axvline(TC, color="gray", ls=":"); ax[1].set_xlabel("temperature $T$")
    ax[1].set_ylabel("|PC1|"); ax[1].legend(fontsize=8, frameon=False)
    ax[1].set_title("(b) PC1 tracks the magnetization")
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("figs/pca.pdf"); plt.close(fig)


if __name__ == "__main__":
    print("phases");        fig_phases()
    print("skyrmion");      fig_skyrmion()
    print("frustration");   fig_frustration()
    print("vortex");        fig_vortex()
    print("orderparam");    fig_orderparam()
    print("phasediagram");  fig_phasediagram()
    print("pca (Monte Carlo, ~1 min)"); fig_pca()
    print("done ->", os.listdir("figs"))
