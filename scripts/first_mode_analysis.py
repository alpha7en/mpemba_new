from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research.liouvillian import build_liouvillian_dense
from qdyn_research.metrics import calculate_excitability_map
from qdyn_research.mpemba import find_guaranteed_mpemba_dense
from qdyn_research.spectral import analyze_liouvillian_modes_dense_strict
from qdyn_research.topology import generate_rewired_grid_tau_guaranteed_connectivity

def main(height=5, width=5, p=0.15, j=1.0, gamma=0.5):
    # ---------------------------
    # System parameters
    # ---------------------------
    n = height * width
    tau = generate_rewired_grid_tau_guaranteed_connectivity(height, width, p)
    liouvillian = build_liouvillian_dense(tau, j, gamma)
    evals, left_vecs, right_vecs = analyze_liouvillian_modes_dense_strict(liouvillian)

    # Analyze the first non-stationary mode (k=1, largest negative Re(lambda)).
    slowest_idx = 1
    b_map = calculate_excitability_map(left_vecs, right_vecs, slowest_idx, n)
    vec_hot, vec_cold_ratio, vec_cold_score = find_guaranteed_mpemba_dense(
        left_vecs,
        right_vecs,
        n,
        excitability_map=b_map,
        mode_idx=slowest_idx,
        distance_order="C",
    )

    print(f"Slow mode lambda_1: {evals[slowest_idx]}")
    print(f"Hot/C ratio candidate norm: {abs(vec_cold_ratio).sum():.4f}")
    print(f"Hot/C score candidate norm: {abs(vec_cold_score).sum():.4f}")

    return tau, liouvillian, vec_hot, vec_cold_score, height, width


if __name__ == "__main__":
    main()


