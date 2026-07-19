"""Fig. 2 (9states -> 8 states): modal content of the candidate initial states on a 10x10
lattice. Each panel = a schematic of the initial photon distribution + a log-scale bar chart
of the mode-contribution weights.

PROJECTION METHOD (v2, referee fix CF-1): coefficients are the EXACT solution of
V c = vec(rho) (LU of the right-eigenvector matrix), not the legacy pairwise biorthogonal
division <w_k|rho>/<w_k|v_k>, which is invalid inside the degenerate multiplets of the regular
lattice (reconstruction error up to ~1e5 % for the "Coherent diagonal" state). Reconstruction
residuals are stored in the cache as a hard control. The cache additionally stores the full
eigenvalues, raw coefficients, degenerate-cluster ids and basis-invariant per-multiplet weights
||V_mu c_mu||^2 (individual |c_k|^2 inside a degenerate multiplet are basis-dependent; the
projected-component norm is not). Legacy pairwise cache kept as fig2_projections.npz.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, MaxNLocator, FuncFormatter

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research import (
    build_liouvillian_dense,
    calculate_distance_metric_logm,
    generate_grid_tau,
    create_localized_state,
    create_opposite_corners_state,
    create_four_corners_state,
    create_mixed_diagonal_state,
    create_entangled_diagonal_state,
    create_inner_corners_state,
    create_top_bottom_edges_state,
    create_checkerboard_state,
)
from qdyn_research.metrics import factor_mode_basis, project_rho_on_modes_exact
from qdyn_research.plot_style import apply_style, save_pdf, WIDTH_FULL

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
PRECALC = Path(__file__).resolve().parent / "precalc"
CACHE = PRECALC / "fig2_projections_exact.npz"   # v2 (exact solve); legacy pairwise cache kept
N = 10
J, GAMMA = 1.0, 0.1
CLUSTER_TOL = 1e-6   # eigenvalues closer than this are one degenerate multiplet (true splittings ~1e-10)

# 8 states (the article caption says "eight"; the legacy "Boundary" state is dropped).
STATES = [
    ("Center", create_localized_state),
    ("Opp. corners", create_opposite_corners_state),
    ("Four corners", create_four_corners_state),
    ("Mixed diag.", create_mixed_diagonal_state),
    ("Coherent diag.", create_entangled_diagonal_state),
    ("Inner corners", create_inner_corners_state),
    ("Edges (t/b)", create_top_bottom_edges_state),
    ("Checkerboard", create_checkerboard_state),
]


def analyze_cluster_weights(inv_w):
    """Normalized per-multiplet weights W_{k'} (stationary cluster 0 excluded) + top-8 x-limit.

    Individual |c_k|^2 are meaningless inside (nearly) degenerate multiplets: the coefficients
    are basis-dependent and, in the nearly defective quadruplets of the regular lattice, grow to
    ~1e8 with mutual cancellation. The projected-component norm ||V_mu c_mu||^2 per multiplet is
    basis-invariant and numerically stable — that is what we plot (x-axis = multiplet k')."""
    contributions = inv_w[1:]
    total = np.sum(contributions)
    if total < 1e-12:
        return np.zeros_like(contributions), 8
    normalized = contributions / total
    idxs = np.arange(1, len(normalized) + 1)
    top_8 = idxs[np.argsort(normalized)[::-1]][:8]
    return normalized, int(np.max(top_8))


def compute():
    """Dense 10x10 eig (right vectors only) -> EXACT modal projections + diagnostics.

    Returns everything worth caching: legacy-format normalized weights (drawing parity),
    raw coefficients, reconstruction residuals, full spectrum, degenerate-cluster structure
    and basis-invariant per-cluster weights ||V_mu c_mu||^2 for every state.
    """
    import time
    t0 = time.time()
    tau = generate_grid_tau(N, N)
    liouvillian = build_liouvillian_dense(tau, J, GAMMA)
    print(f"dense eig {liouvillian.shape[0]}x{liouvillian.shape[0]} ...", flush=True)
    evals, right = np.linalg.eig(liouvillian)
    del liouvillian
    order = np.argsort(evals.real)[::-1]
    evals = evals[order]
    right = np.ascontiguousarray(right[:, order])
    print(f"  eig done ({time.time()-t0:.0f}s); LU ...", flush=True)
    lu = factor_mode_basis(right)
    print(f"  LU done ({time.time()-t0:.0f}s)", flush=True)

    # degenerate-multiplet clustering of eigenvalues
    m = len(evals)
    cluster_id = np.full(m, -1, dtype=np.int64)
    cid = 0
    for i in range(m):
        if cluster_id[i] >= 0:
            continue
        members = np.where((cluster_id < 0) & (np.abs(evals - evals[i]) < CLUSTER_TOL))[0]
        cluster_id[members] = cid
        cid += 1
    n_clusters = cid
    print(f"  {n_clusters} eigenvalue clusters (of {m} modes)", flush=True)

    contribs, kmax, coeffs_all, residuals = [], [], [], []
    inv_weights = np.zeros((len(STATES), n_clusters))
    for s, (_name, gen) in enumerate(STATES):
        rho, _idx = gen(N, N)
        c, res = project_rho_on_modes_exact(rho, lu, right, order="F")
        coeffs_all.append(c)
        residuals.append(res)
        for cl in range(n_clusters):
            idx = np.where(cluster_id == cl)[0]
            inv_weights[s, cl] = float(np.linalg.norm(right[:, idx] @ c[idx]) ** 2)
        norm, km = analyze_cluster_weights(inv_weights[s])
        contribs.append(norm)
        kmax.append(km)
        print(f"  {_name:<15} residual = {res:.2e}   max|c| = {np.abs(c).max():.2e}", flush=True)
    return (np.array(contribs), np.array(kmax), np.array(coeffs_all), np.array(residuals),
            evals, cluster_id, inv_weights)


def draw_bar(ax, contribs, k_max, y_top, show_xlabel, show_ylabel):
    ax.set_yscale("log")
    ymin = 0.01
    modes = np.arange(1, k_max + 1)
    pct = contribs[:k_max] * 100.0
    keep = pct > ymin
    if k_max <= 40:
        ax.bar(modes[keep], pct[keep], color="#0072B2", width=0.85)
    else:
        ax.vlines(modes[keep], ymin, pct[keep], color="#0072B2", linewidth=0.7)
    ax.set_xlim(0.5, k_max + 0.5)
    ax.set_ylim(ymin, y_top)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", which="major", ls="--", lw=0.4, alpha=0.6)
    if show_xlabel:
        ax.set_xlabel("multiplet $k'$", fontsize=8)
    if show_ylabel:
        ax.set_ylabel(r"$W_{k'}$ (%)", fontsize=8)


def main():
    apply_style()

    if CACHE.exists():
        z = np.load(CACHE)
        contribs_all, kmax_all = z["contribs"], z["kmax"]
    else:
        (contribs_all, kmax_all, coeffs_all, residuals,
         evals, cluster_id, inv_weights) = compute()
        # Hard control: residual r perturbs the normalized weights by ~2r; the plotted floor is
        # 1e-4 (0.01%), so r < 1e-6 keeps weight errors two orders below anything visible.
        # (7 of 8 states reach ~1e-13; "Coherent diag." is limited by the nearly defective
        # quadruplet, where ||c||~1e8 sets the attainable floor even after refinement.)
        if residuals.max() > 1e-6:
            raise RuntimeError(f"reconstruction control FAILED: max residual {residuals.max():.2e}")
        PRECALC.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, contribs=contribs_all, kmax=kmax_all,
                            coeffs=coeffs_all, residuals=residuals, evals=evals,
                            cluster_id=cluster_id, inv_weights=inv_weights,
                            state_names=np.array([n for n, _ in STATES]))

    # cheap per-state extras (schematic indices + distance metric), then sort by D
    panels = []
    for i, (name, gen) in enumerate(STATES):
        rho, idx = gen(N, N)
        panels.append({
            "name": name,
            "idx": idx,
            "D": calculate_distance_metric_logm(rho),
            "contribs": contribs_all[i],
            "kmax": int(kmax_all[i]),
        })
    panels.sort(key=lambda p: p["D"], reverse=True)

    y_top = max(p["contribs"].max() for p in panels) * 100 * 1.6

    fig = plt.figure(figsize=(WIDTH_FULL, WIDTH_FULL * 1.2))
    outer = gridspec.GridSpec(4, 2, figure=fig, left=0.10, right=0.985, top=0.96,
                              bottom=0.085, hspace=0.7, wspace=0.30)

    for i, p in enumerate(panels):
        r, c = divmod(i, 2)
        # bar chart on the LEFT (its y-axis has room); schematic on the RIGHT (no collisions)
        cell = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[r, c],
                                                width_ratios=[3.0, 1], wspace=0.15)
        ax_b = fig.add_subplot(cell[0])
        draw_bar(ax_b, p["contribs"], p["kmax"], y_top,
                 show_xlabel=(r == 3), show_ylabel=(c == 0))
        ax_b.set_title(f"{p['name']}  ($D_0$={p['D']:.2f})", fontsize=8, pad=3)

        ax_s = fig.add_subplot(cell[1])
        sch = np.zeros((N, N))
        for j in p["idx"]:
            sch[divmod(j, N)] = 1.0
        ax_s.imshow(sch, cmap="viridis", vmin=0, vmax=1, aspect="equal")
        ax_s.set_xticks([]); ax_s.set_yticks([])

    save_pdf(fig, str(FIG_DIR / "new_tabled_10x10_bar_chart_log_ENG_new_8STATES.pdf"))


if __name__ == "__main__":
    main()
