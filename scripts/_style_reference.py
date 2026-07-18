"""C1 style reference: shows the three figure archetypes in the new unified style
(line plot / diverging mode map / viridis excitability map) on a cheap 5x5 lattice.
Purpose: approve the look before rolling it out to all figN scripts. Not a paper figure.
"""
from _bootstrap import ensure_src_on_path
ensure_src_on_path()

import numpy as np

from qdyn_research.plot_style import apply_style, save_pdf, WIDTH_FULL, SEQ_CMAP
from qdyn_research.liouvillian import build_liouvillian_dense
from qdyn_research.spectral import analyze_liouvillian_modes_dense_strict
from qdyn_research.topology import (generate_grid_tau,
                                    generate_rewired_grid_tau_guaranteed_connectivity)
from qdyn_research.metrics import calculate_excitability_map
from qdyn_research.visualization import draw_population_mode_on_axis

import matplotlib.pyplot as plt
import networkx as nx

apply_style()
n, J, gamma = 5, 1.0, 0.1

# ---- data (cheap, dense 5x5) ----
# (A) Re(lambda_k) vs p with error bars
p_vals = np.logspace(-3, 0, 7)
means = {1: [], 2: [], 3: []}
stds = {1: [], 2: [], 3: []}
for p in p_vals:
    samp = {1: [], 2: [], 3: []}
    for r in range(8):
        tau = generate_rewired_grid_tau_guaranteed_connectivity(n, n, float(p), seed=r)
        ev, _, _ = analyze_liouvillian_modes_dense_strict(build_liouvillian_dense(tau, J, gamma))
        for k in (1, 2, 3):
            samp[k].append(ev[k].real)
    for k in (1, 2, 3):
        means[k].append(np.mean(samp[k]))
        stds[k].append(np.std(samp[k]))

# (B) regular-lattice slow mode populations (signed)
tau_reg = generate_grid_tau(n, n)
_, _, RV = analyze_liouvillian_modes_dense_strict(build_liouvillian_dense(tau_reg, J, gamma))
pop = np.diag(RV[:, 1].reshape((n * n, n * n), order="F")).real.reshape((n, n))
k_scaler = 1.0 / (np.max(np.abs(pop)) + 1e-9)

# (C) excitability B(1,i) on a rewired lattice
tau_rw = generate_rewired_grid_tau_guaranteed_connectivity(n, n, 0.15, seed=3)
L = build_liouvillian_dense(tau_rw, J, gamma)
_, LVr, RVr = analyze_liouvillian_modes_dense_strict(L)
bmap = calculate_excitability_map(LVr, RVr, 1, n * n)

# ---- figure: full width, 3 panels ----
fig, axes = plt.subplots(1, 3, figsize=(WIDTH_FULL, WIDTH_FULL * 0.34))
axA, axB, axC = axes

# panel A: line plot
for k, lab in zip((1, 2, 3), (r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$")):
    axA.errorbar(p_vals, means[k], yerr=stds[k], fmt="-o", capsize=2, label=lab)
axA.set_xscale("log")
axA.set_xlabel("rewiring probability $p$")
axA.set_ylabel(r"$\mathrm{Re}(\lambda_k)$")
axA.grid(True, which="both", ls="--")
axA.legend(title="mode")
axA.set_title("(a) line plot")

# panel B: diverging red/blue mode map (white = 0)
draw_population_mode_on_axis(axB, pop, n, n, k_scaler, "(b) mode map (signed)",
                             radius=0.42, title_fontsize=9, title_y=None)

# panel C: viridis excitability network
axC.set_title("(c) excitability map")
G = nx.from_numpy_array(tau_rw)
posn = {i: (i % n, (n - 1) - i // n) for i in range(n * n)}
nx.draw_networkx_edges(G, posn, ax=axC, edge_color="0.6", width=0.6)
nodes = nx.draw_networkx_nodes(G, posn, ax=axC, node_color=bmap, cmap=SEQ_CMAP,
                               node_size=90, edgecolors="black", linewidths=0.4)
axC.set_aspect("equal"); axC.axis("off")
cb = fig.colorbar(nodes, ax=axC, fraction=0.046, pad=0.04)
cb.set_label(r"$B(1,i)$")

save_pdf(fig, "_style_reference.pdf")
print("wrote _style_reference.pdf")
