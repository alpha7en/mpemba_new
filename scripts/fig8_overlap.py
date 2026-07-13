"""Fig. 8 (community_overlap): mean maximum community-overlap of the three slowest
non-trivial modes vs rewiring probability p (log axis). Communities are found with the
Louvain method on each realization's graph (tau read straight from the npz, so it matches
the stored eigenvectors). Only the plotting is styled; calculate_max_overlap is unchanged.
"""
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
import networkx as nx
import community as community_louvain  # python-louvain

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.plot_style import apply_style, save_pdf, WIDTH_THIRD, LINE_COLORS

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
PRECALC = Path(__file__).resolve().parent / "precalc"


def calculate_max_overlap(eigenvector: np.ndarray, communities: dict, N: int) -> float:
    """Maximum fraction of a mode's population mass sitting in a single community.
    Mass at node i is |(rho_k)_ii|^2 (mode matrix via Fortran-order reshape)."""
    rho_k = eigenvector.reshape((N, N), order="F")
    population_distribution = np.abs(np.diag(rho_k)) ** 2
    total_mass = np.sum(population_distribution)
    if np.isclose(total_mass, 0):
        return 0.0
    num_communities = max(communities.values()) + 1
    community_masses = np.zeros(num_communities)
    for node_idx, community_id in communities.items():
        community_masses[community_id] += population_distribution[node_idx]
    return float(np.max(community_masses / total_mass))


def find_npz():
    cands = glob.glob(str(PRECALC / "rewiring_spectrum_data_*.npz")) + glob.glob("rewiring_spectrum_data_*.npz")
    if not cands:
        raise FileNotFoundError("no rewiring_spectrum_data_*.npz in scripts/precalc or cwd")
    return max(cands, key=os.path.getctime)


def main():
    apply_style()
    data = np.load(find_npz())

    # group (vectors, tau) by p; the fresh archive stores tau alongside the eigenvectors
    grouped = defaultdict(list)
    for key in data.files:
        if key.endswith("_vectors"):
            base = key[: -len("_vectors")]
            p = float(base.split("_")[1])
            tau_key = base + "_tau"
            if tau_key not in data.files:
                raise RuntimeError("npz has no tau; regenerate with run_multicore_npz.py")
            grouped[p].append((data[key], data[tau_key]))

    p_pos = sorted(x for x in grouped if x > 0)
    N = grouped[p_pos[0]][0][0].shape[0]           # density-matrix dim (10000 -> reshape 100x100)
    sites = int(np.sqrt(N))                          # 100
    side = int(np.sqrt(sites))                       # 10

    for mode, color in zip((1, 2, 3), LINE_COLORS):
        avg, std = [], []
        for p in p_pos:
            overlaps = []
            for vectors, tau in grouped[p]:
                graph = nx.from_numpy_array(tau)
                communities = community_louvain.best_partition(graph)
                overlaps.append(calculate_max_overlap(vectors[:, mode], communities, sites))
            avg.append(np.mean(overlaps))
            std.append(np.std(overlaps))

        fig, ax = plt.subplots(figsize=(WIDTH_THIRD, WIDTH_THIRD * 0.95), layout="constrained")
        ax.errorbar(p_pos, avg, yerr=std, fmt="-o", capsize=2, color=color, ecolor=color, alpha=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("$p$")
        ax.set_ylabel("average max overlap")
        # compact exponent ticks every 2 decades (decimal labels collide at 42 mm width)
        ax.set_xticks([1e-4, 1e-2, 1e0])
        ax.xaxis.set_major_formatter(LogFormatterMathtext())
        ax.grid(True, which="both", ls="--")
        ax.margins(x=0.05)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        save_pdf(fig, str(FIG_DIR / f"{side}x{side}_community_analyse_mode_k_{mode}_ENG.pdf"))


if __name__ == "__main__":
    main()
