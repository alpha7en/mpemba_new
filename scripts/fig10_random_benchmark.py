import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.plot_style import apply_style, save_pdf, WIDTH_FULL

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
# Definitive honest Monte-Carlo benchmark, produced with the TRUE Dirichlet(1,...,1) sampling now
# in MpembaValidator.generate_random_diag_density_matrix. Each admissible random pair (initial
# relative-entropy gap >= 10% of the engineered gap) is propagated on a fine time grid [0,200] and
# its crossing time t* recorded (NaN if no crossing). Physics identical to the repo spectral kernel
# (worker_get_decomposition/entropic_distance) to machine precision.
DATA = Path(__file__).resolve().parent / "precalc" / "dirichlet_benchmark_30k.npz"

# Crop the near-equilibrium noise tail: with tau ~ 13.5 the genuine crossing mode sits at t~85 and
# real crossings extend to ~150 (D still ~1e-10). Beyond t~160 (~12 tau) both trajectories have
# relaxed to within numerical resolution of equilibrium (D~1e-13) and the "crossings" there are
# floating-point sign flips, not physics -> excluded from the plotted range.
X_MAX = 160.0


def main():
    apply_style()

    d = np.load(DATA)
    tstars = d["tstars"]            # t* for every admissible pair (NaN if no crossing in [0,200])
    t_algo = float(d["t_algo"])     # engineered pair, same run/grid (converged ~44.7)
    n_adm = int(d["n_adm"])
    tc = tstars[np.isfinite(tstars)]
    frac_faster = 100.0 * np.nansum(tstars < t_algo) / n_adm   # robust metric (see caption/text)

    fig, ax = plt.subplots(figsize=(WIDTH_FULL * 0.9, WIDTH_FULL * 0.9 * 0.58), layout="constrained")
    bins = np.linspace(0, X_MAX, 41)          # 40 bins, width 4.0 -> smooth (was 80/2.0, too jagged)
    counts, _, _ = ax.hist(tc[tc < X_MAX], bins=bins, color="0.62", edgecolor="0.35", linewidth=0.3,
                           label="Random pairs (Dirichlet)")
    ax.axvline(t_algo, color="#D55E00", linestyle="--", linewidth=1.6,
               label=f"Engineered pair ($t^*={t_algo:.1f}$)")
    ax.set_xlabel(r"crossing time $t^*$")
    ax.set_ylabel("number of pairs")
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, counts.max() * 1.22)       # headroom so the legend clears the bars
    # upper-right is clear of the engineered-pair line (t*=44.7); light white backing guarantees
    # no visual clash with the descending shoulder of the histogram.
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none")
    ax.grid(True, axis="y", alpha=0.4)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_pdf(fig, str(FIG_DIR / "alg_compare.pdf"))
    print(f"alg_compare.pdf: n_adm={n_adm}, t_algo={t_algo:.3f}, "
          f"faster-than-algo={frac_faster:.2f}% of admissible pairs")


if __name__ == "__main__":
    main()
