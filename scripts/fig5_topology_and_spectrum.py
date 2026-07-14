"""Fig. 5 (rewired_spectra): three panels with_examples_1/2/3 for a 10x10 lattice at
increasing disorder (p=0.01, 0.1, 0.3). Each panel = the graph topology (top, rewired edges
highlighted) + the Liouvillian eigenvalue spectrum in the complex plane (bottom). Only the
plotting is styled; the spectrum computation (dense eig) is unchanged and cached.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, MaxNLocator

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.liouvillian import build_liouvillian_dense
from qdyn_research.spectral import analyze_liouvillian_modes_dense
from qdyn_research.topology import generate_grid_tau, generate_rewired_grid_tau
from qdyn_research.plot_style import apply_style, save_pdf, WIDTH_THIRD

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
PRECALC = Path(__file__).resolve().parent / "precalc"
CACHE = PRECALC / "fig5_spectra.npz"

H = W = 10
N = H * W
J, GAMMA = 1.0, 0.1
P_VALUES = [0.01, 0.1, 0.3]  # weak/medium/strong disorder; p=0.01 (not 0) so a few rewired
#                              edges are visible, resolving the original "p=0 with rewired edges"
N_PLOT = 200


def compute():
    """One dense Liouvillian eig per disorder level; keep the 200 slowest eigenvalues + tau."""
    taus, lams = [], []
    for p in P_VALUES:
        tau = generate_grid_tau(H, W) if p == 0.0 else generate_rewired_grid_tau(H, W, p)
        lambdas, _ = analyze_liouvillian_modes_dense(build_liouvillian_dense(tau, J, GAMMA))
        taus.append(tau)
        lams.append(lambdas[:N_PLOT])
    return np.array(taus), np.array(lams)


def draw_topology(ax, tau):
    pos = {i: (i % W, (H - 1) - (i // W)) for i in range(N)}
    base = generate_grid_tau(H, W)
    rewired_nodes = set()
    for i in range(N):
        for j in range(i + 1, N):
            if tau[i, j] == 1 and base[i, j] == 1:  # surviving grid edge
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="0.8", lw=0.4, zorder=0)
            elif tau[i, j] == 1 and base[i, j] == 0:  # rewired (long-range) edge
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], color="#D55E00", lw=0.8, zorder=1)
                rewired_nodes |= {i, j}
    colors = ["#D55E00" if i in rewired_nodes else "0.55" for i in range(N)]
    xs = [pos[i][0] for i in range(N)]
    ys = [pos[i][1] for i in range(N)]
    ax.scatter(xs, ys, s=7, c=colors, edgecolors="black", linewidths=0.2, zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_spectrum(ax, lambdas):
    # framed axes + a thin reference cross at Re=0 / Im=0 (imaginary axis marks the slow modes);
    # standard left/bottom labels avoid the right-edge clipping of the spine-at-zero style.
    ax.axvline(0, color="0.7", lw=0.5, zorder=0)
    ax.axhline(0, color="0.7", lw=0.5, zorder=0)
    ax.scatter(lambdas.real, lambdas.imag, c="#0072B2", s=3.5, edgecolors="none", alpha=0.75, zorder=2)
    ax.set_xlim(-0.1, 0.008)   # common axes across the three panels for a fair comparison
    ax.set_ylim(-4.3, 4.3)
    ax.set_xlabel(r"$\mathrm{Re}\,\lambda$", fontsize=8, labelpad=1)
    ax.set_ylabel(r"$\mathrm{Im}\,\lambda$", fontsize=8, labelpad=1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(labelsize=8, length=2, width=0.5)
    for s in ax.spines.values():
        s.set_linewidth(0.5)


def main():
    apply_style()
    ps = np.array(P_VALUES, dtype=float)
    z = np.load(CACHE) if CACHE.exists() else None
    if z is not None and "ps" in z.files and np.allclose(z["ps"], ps):
        taus, lams = z["taus"], z["lams"]
    else:
        taus, lams = compute()
        PRECALC.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, taus=taus, lams=lams, ps=ps)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(len(P_VALUES)):
        fig = plt.figure(figsize=(WIDTH_THIRD, WIDTH_THIRD * 2.05))
        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1.15],
                               left=0.2, right=0.97, top=0.99, bottom=0.09, hspace=0.16)
        draw_topology(fig.add_subplot(gs[0]), taus[i])
        draw_spectrum(fig.add_subplot(gs[1]), lams[i])
        save_pdf(fig, str(FIG_DIR / f"with_examples_{i + 1}.pdf"))


if __name__ == "__main__":
    main()
