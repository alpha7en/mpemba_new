"""Fig. 3 (simulation_lattice_10x10): relaxation dynamics D(rho(t)) for the candidate initial
states on a 10x10 lattice. Trajectory crossings (bold curves) are the Mpemba effect. Rendered
at gamma = 0.1 (as stated in the text) with the window extended to t=100; the crossings then
occur at their true gamma=0.1 timescale (t ~ 40-60) rather than the published t ~ 0.85 and 4.5.
Only the plotting is styled; the ODE integration is unchanged and cached.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research import (
    QuantumSimulatorCore,
    build_liouvillian_dense,
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
from qdyn_research.plot_style import apply_style, save_pdf, WIDTH_FULL

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
PRECALC = Path(__file__).resolve().parent / "precalc"
CACHE = PRECALC / "fig3_dynamics.npz"

N = 10
J, GAMMA = 1.0, 0.1     # gamma as stated in the text (weakly dissipative)
T_MAX = 4.0             # window covering the gamma=0.1 Mpemba crossing of pair (2,5) at t ~ 2.2

# numbered so the caption's "pairs (3 and 5) and (4 and 5)" keeps meaning; Boundary (old #9) dropped
STATES = [
    ("1. Center", create_localized_state),
    ("2. Opp. corners", create_opposite_corners_state),
    ("3. Four corners", create_four_corners_state),
    ("4. Mixed diag.", create_mixed_diagonal_state),
    ("5. Coherent diag.", create_entangled_diagonal_state),
    ("6. Inner corners", create_inner_corners_state),
    ("7. Edges (t/b)", create_top_bottom_edges_state),
    ("8. Checkerboard", create_checkerboard_state),
]
HIGHLIGHT = {1, 4}  # indices (0-based) of states 2 and 5 -> bold (the gamma=0.1 crossing pair)


def compute():
    tau = generate_grid_tau(N, N)
    liouvillian = build_liouvillian_dense(tau, J, GAMMA)
    sim = QuantumSimulatorCore()
    curves = []
    for _name, gen in STATES:
        rho, _idx = gen(N, N)
        rho = rho / np.trace(rho)
        # tiny atol => the plateau event never fires, so we integrate the full [0, T_MAX] window
        t, d, _ = sim.run_simulation(liouvillian, rho, t_span=(0, T_MAX), atol=1e-12)
        curves.append(d)
    return np.asarray(t), np.asarray(curves)


def find_intersection(t, d1, d2):
    for i in range(1, len(t)):
        if (d1[i - 1] - d2[i - 1]) * (d1[i] - d2[i]) < 0:
            f = abs(d1[i - 1] - d2[i - 1]) / abs((d1[i] - d2[i]) - (d1[i - 1] - d2[i - 1]))
            xt = t[i - 1] + (t[i] - t[i - 1]) * f
            xd = d1[i - 1] + (d1[i] - d1[i - 1]) * (xt - t[i - 1]) / (t[i] - t[i - 1])
            return xt, xd
    return None


def main():
    apply_style()

    cached = None
    if CACHE.exists():
        z = np.load(CACHE)
        if np.isclose(z["gamma"], GAMMA) and "t_max" in z.files and np.isclose(z["t_max"], T_MAX):
            cached = (z["t"], z["curves"])
    if cached is not None:
        t, curves = cached
    else:
        t, curves = compute()
        PRECALC.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, t=t, curves=curves, gamma=GAMMA, t_max=T_MAX)

    palette = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(WIDTH_FULL, WIDTH_FULL * 0.6), layout="constrained")

    for i, (name, _gen) in enumerate(STATES):
        bold = i in HIGHLIGHT
        ax.plot(t, curves[i], color=palette(i), lw=2.0 if bold else 1.0,
                alpha=1.0 if bold else 0.55, label=name, zorder=3 if bold else 2)

    for a, b in [(1, 4)]:  # crossing for pair (2,5)
        hit = find_intersection(t, curves[a], curves[b])
        if hit:
            ax.plot(*hit, "o", ms=4, color="crimson", zorder=5)

    ax.set_xlim(0, T_MAX)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"distance to equilibrium $D(\rho(t))$")
    ax.grid(True, ls="--", lw=0.4, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8, ncol=1, handlelength=1.6, labelspacing=0.3)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_pdf(fig, str(FIG_DIR / "9states_modes_simulation_line_10x10_with_selection_ENG.pdf"))


if __name__ == "__main__":
    main()
