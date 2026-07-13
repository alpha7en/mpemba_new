"""Fig. 4 (average path length): mean shortest-path length <L> vs rewiring probability p
on a 10x10 grid, log-p axis. Two panels from the same ensemble: full scale and a zoom on
the crossover ("crossovering gap"). Only the plotting is styled; the ensemble experiment
(run_average_path_experiment) is unchanged.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research import generate_rewired_grid_tau, run_average_path_experiment
from qdyn_research.plot_style import apply_style, save_pdf, WIDTH_FULL, LINE_COLORS

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
PRECALC = Path(__file__).resolve().parent / "precalc"
CACHE = PRECALC / "fig4_pathlength.npz"


def main():
    apply_style()

    # ---- ensemble (unchanged), cached so restyling doesn't re-run the ~7 min sweep ----
    if CACHE.exists():
        z = np.load(CACHE)
        p, L = z["p"], z["L"]
    else:
        height, width = 10, 10
        runs_per_p = 1000
        p_values = np.concatenate(([0.0], np.logspace(-4, 0, num=100)))
        results = run_average_path_experiment(height, width, runs_per_p, p_values, generate_rewired_grid_tau)
        p = np.array(list(results.keys()))
        L = np.array(list(results.values()))
        PRECALC.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, p=p, L=L)

    pos = p > 0  # log axis: drop the p=0 reference point

    def panel(xlim, out_name):
        w = 0.49 * WIDTH_FULL  # 0.49\linewidth (2-up filling the text width)
        fig, ax = plt.subplots(figsize=(w, w * 0.72), layout="constrained")
        ax.plot(p[pos], L[pos], "-o", color=LINE_COLORS[0], markersize=2.2)
        ax.set_xscale("log")
        ax.set_xlabel("$p$")
        ax.set_ylabel(r"$L_{avg}$")
        ax.grid(True, which="both", ls="--")
        if xlim is not None:
            ax.set_xlim(*xlim)
            win = (p >= xlim[0]) & (p <= xlim[1])
            lo, hi = L[win].min(), L[win].max()
            pad = 0.05 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        save_pdf(fig, str(FIG_DIR / out_name))

    panel(None, "log10x10_p_L.pdf")                 # full scale
    panel((1e-3, 1e-1), "log10x10_p_L_SCALED.pdf")  # crossover zoom


if __name__ == "__main__":
    main()
