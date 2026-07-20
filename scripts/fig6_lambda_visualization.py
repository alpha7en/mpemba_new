"""Fig. 6 (v2, referee fix CF-2/CF-5a): statistics of the TRUE slowest mode vs rewiring p.

Reads the corrected sweep archive (sweep_rightmost_10x10.npz, modes selected by max Re via the
verified exp-transform solver; see scripts/verify_rightmost_solver.py) and draws four panels:
  (a) mean Re(lambda_1) -- the corrected relaxation rate of the slowest non-trivial mode;
  (b) fraction of realizations whose slowest mode oscillates (|Im| > 1e-6): the
      oscillating->real crossover, ~90% at p->1e-4, 50% at p_c ~ 4e-3, 0% for p >~ 2e-2;
  (c) mean |Im(lambda_1)| over the oscillating realizations (the quadruplet frequency);
  (d) separation between the two slowest DISTINCT relaxation rates (Re-groups, tol 1e-6) --
      individual-mode differences are ill-defined inside degenerate multiplets.

Error bars: (a), (c), (d) show +-1 std over the 30 disorder realizations; (b) shows the
binomial standard error sqrt(f(1-f)/n). Only lambdas are read (fast); reduction cached.
"""
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.plot_style import apply_style, save_pdf, WIDTH_COL, LINE_COLORS

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
PRECALC = Path(__file__).resolve().parent / "precalc"
ARCHIVE = PRECALC / "sweep_rightmost_10x10.npz"
CACHE = PRECALC / "fig6_rightmost.npz"

P_VALUES = np.logspace(-4, 0, num=40)
RUNS = 30
OSC_TOL = 1e-6
GROUP_TOL = 1e-6


def reduce_archive():
    """Per-(p, run) scalars from the archive lambdas: Re l1, osc flag, |Im l1|, group separation."""
    z = np.load(ARCHIVE)
    shape = (len(P_VALUES), RUNS)
    re1 = np.full(shape, np.nan)
    osc = np.full(shape, np.nan)
    im1 = np.full(shape, np.nan)
    sep = np.full(shape, np.nan)
    for pi, p in enumerate(P_VALUES):
        for r in range(RUNS):
            key = f"p_{p}_run_{r}_lambdas"
            if key not in z.files:
                continue
            lam = z[key]
            lam = lam[np.abs(lam) > 1e-10]            # drop the stationary mode
            l1 = lam[0]
            re1[pi, r] = l1.real
            osc[pi, r] = float(abs(l1.imag) > OSC_TOL)
            im1[pi, r] = abs(l1.imag)
            groups = []
            for v in lam:
                if not any(abs(v.real - g) < GROUP_TOL for g in groups):
                    groups.append(v.real)
            if len(groups) > 1:
                sep[pi, r] = groups[0] - groups[1]
    return re1, osc, im1, sep


def _panel(x, y, yerr, ylabel, color, out_name, ylim=None):
    fig, ax = plt.subplots(figsize=(WIDTH_COL, WIDTH_COL * 0.78), layout="constrained")
    ax.errorbar(x, y, yerr=yerr, fmt="-o", capsize=2, markersize=3,
                color=color, ecolor=color, alpha=0.9, lw=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("$p$")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, which="both", ls="--")
    ax.margins(x=0.03)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_pdf(fig, str(FIG_DIR / out_name))


def main():
    apply_style()

    if CACHE.exists():
        c = np.load(CACHE)
        re1, osc, im1, sep = c["re1"], c["osc"], c["im1"], c["sep"]
    else:
        re1, osc, im1, sep = reduce_archive()
        PRECALC.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, re1=re1, osc=osc, im1=im1, sep=sep, p=P_VALUES)

    p = P_VALUES

    # (a) corrected relaxation rate of the slowest mode
    _panel(p, np.nanmean(re1, axis=1), np.nanstd(re1, axis=1),
           r"$\mathrm{Re}(\lambda_1)$", LINE_COLORS[0], "re1_p_log.pdf")

    # (b) oscillating fraction with binomial standard error
    n = np.sum(~np.isnan(osc), axis=1)
    f = np.nanmean(osc, axis=1)
    ferr = np.sqrt(np.clip(f * (1 - f), 0, None) / np.maximum(n, 1))
    _panel(p, 100 * f, 100 * ferr, r"oscillating fraction (\%)", LINE_COLORS[1],
           "oscfrac_p_log.pdf", ylim=(-4, 104))

    # (c) |Im lambda_1| over the oscillating subensemble (needs >=2 oscillating realizations)
    im_mean = np.full(len(p), np.nan)
    im_std = np.full(len(p), np.nan)
    for pi in range(len(p)):
        sel = (osc[pi] == 1)
        if np.sum(sel) >= 2:
            im_mean[pi] = np.nanmean(im1[pi][sel])
            im_std[pi] = np.nanstd(im1[pi][sel])
    keep = ~np.isnan(im_mean)
    _panel(p[keep], im_mean[keep], im_std[keep],
           r"$|\mathrm{Im}(\lambda_1)|$ (osc.)", LINE_COLORS[2], "im1_p_log.pdf")

    # (d) separation of the two slowest distinct relaxation rates
    _panel(p, np.nanmean(sep, axis=1), np.nanstd(sep, axis=1),
           r"$\mathrm{Re}(\lambda_1) - \mathrm{Re}(\lambda_2)$", LINE_COLORS[3], "sep_p_log.pdf")

    print("fig6 v2: re1_p_log / oscfrac_p_log / im1_p_log / sep_p_log rebuilt")


if __name__ == "__main__":
    main()
