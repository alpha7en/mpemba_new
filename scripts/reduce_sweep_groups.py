"""Reduce the corrected sweep archive to per-group localization metrics (Figs. 7-8 input).

For every disorder realization the retained rightmost modes are clustered into DISTINCT
relaxation-rate groups (Re lambda, tolerance 1e-6): individual eigenvectors inside a (nearly)
degenerate multiplet are basis-arbitrary, so per-mode IPR/overlap are ill-defined there (the
same pathology as CF-1). The basis-invariant object is the diagonal weight of the group's
right-eigenspace: with {q_k} an ORTHONORMAL basis (QR) of the span of the group's eigenvectors,

    d_i = sum_k |(Mat q_k)_{ii}|^2          (Mat = F-order unvec, paper convention)

d depends only on the eigenspace span (any other orthonormal basis is Q U with U unitary and
leaves d invariant). For a non-degenerate group {v} this reduces exactly to the legacy
definitions: IPR = sum d^2/(sum d)^2 == calculate_ipr(v), and the community overlap mass
d_i == |(Mat v)_{ii}|^2 of fig8. Conjugate partners have conjugate diagonals (identical d),
so their possible aggregation only rescales d uniformly and changes neither metric.

Louvain community detection is SEEDED (random_state=42) for reproducibility; one partition
per graph, shared by all groups. Output: precalc/fig78_groups.npz with
ipr/overlap/grouplam/groupsize arrays of shape (40 p-values, 30 runs, 3 groups).
"""
import time
from pathlib import Path

import numpy as np
import networkx as nx
import community as community_louvain  # python-louvain

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

PRECALC = Path(__file__).resolve().parent / "precalc"
ARCHIVE = PRECALC / "sweep_rightmost_10x10.npz"
OUT = PRECALC / "fig78_groups.npz"

P_VALUES = np.logspace(-4, 0, num=40)
RUNS = 30
N_GROUPS = 3
SITES = 100
GROUP_TOL = 1e-6
LOUVAIN_SEED = 42


def group_diag_weight(vecs):
    """Basis-invariant diagonal weight d_i of the group eigenspace (orthonormalized)."""
    q, _r = np.linalg.qr(vecs)
    d = np.zeros(SITES)
    for k in range(q.shape[1]):
        d += np.abs(np.diag(q[:, k].reshape(SITES, SITES, order="F"))) ** 2
    return d


def main():
    z = np.load(ARCHIVE)
    shape = (len(P_VALUES), RUNS, N_GROUPS)
    ipr = np.full(shape, np.nan)
    overlap = np.full(shape, np.nan)
    grouplam = np.full(shape, np.nan, dtype=complex)
    groupsize = np.full(shape, 0, dtype=np.int64)

    t0 = time.time()
    for pi, p in enumerate(P_VALUES):
        for r in range(RUNS):
            base = f"p_{p}_run_{r}"
            if f"{base}_lambdas" not in z.files:
                continue
            lam = z[f"{base}_lambdas"]
            vec = z[f"{base}_vectors"]
            nz = np.abs(lam) > 1e-10                     # drop the stationary mode
            lam, vec = lam[nz], vec[:, nz]

            # distinct relaxation-rate groups (Re lambda, encounter order = Re-descending)
            reps, members = [], []
            for k, v in enumerate(lam):
                for gi, g in enumerate(reps):
                    if abs(v.real - g) < GROUP_TOL:
                        members[gi].append(k)
                        break
                else:
                    reps.append(v.real)
                    members.append([k])

            graph = nx.from_numpy_array(z[f"{base}_tau"])
            communities = community_louvain.best_partition(graph, random_state=LOUVAIN_SEED)
            n_comm = max(communities.values()) + 1

            for gi in range(min(N_GROUPS, len(reps))):
                idx = members[gi]
                d = group_diag_weight(vec[:, idx])
                total = d.sum()
                if total < 1e-12:
                    continue
                ipr[pi, r, gi] = float(np.sum((d / total) ** 2))
                masses = np.zeros(n_comm)
                for node, cid in communities.items():
                    masses[cid] += d[node]
                overlap[pi, r, gi] = float(masses.max() / total)
                grouplam[pi, r, gi] = lam[idx[0]]
                groupsize[pi, r, gi] = len(idx)
        print(f"[{pi+1:02d}/40] p={p:.5f} reduced ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(OUT, p=P_VALUES, ipr=ipr, overlap=overlap,
                        grouplam=grouplam, groupsize=groupsize)
    print(f"saved {OUT.name}")


if __name__ == "__main__":
    main()
