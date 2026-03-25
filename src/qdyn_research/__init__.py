"""Scientific kernels for dissipative single-photon dynamics on rewired grids.

The package provides topology generators, Liouvillian builders, spectral solvers,
state metrics, and analysis utilities used for figure production.
"""

from .topology import (
    check_connectivity,
    generate_grid_tau,
    generate_grid_with_manual_links_tau,
    generate_rewired_grid_tau,
    generate_rewired_grid_tau_guaranteed_connectivity,
)
from .liouvillian import build_liouvillian_dense, build_liouvillian_sparse
from .spectral import (
    analyze_liouvillian_modes_dense,
    analyze_liouvillian_modes_dense_strict,
    analyze_liouvillian_modes_sparse_robust,
    get_biorthogonal_modes_sparse_strict,
)
from .states import (
    create_localized_state,
    create_opposite_corners_state,
    create_four_corners_state,
    create_mixed_diagonal_state,
    create_entangled_diagonal_state,
    create_inner_corners_state,
    create_top_bottom_edges_state,
    create_checkerboard_state,
    create_boundary_state,
)
from .metrics import (
    get_projection_coefficient,
    project_rho_on_modes,
    calculate_excitability_map,
    calculate_ipr,
    entropic_distance,
    calculate_distance_metric_logm,
)
from .simulation import QuantumSimulatorCore
from .npz_io import (
    run_single_sparse_rewiring_job,
    save_npz_bundle,
    parse_grouped_lambdas,
    parse_grouped_vectors,
)
from .network_metrics import calculate_average_shortest_path_length, run_average_path_experiment
from .visualization import draw_population_mode_on_axis, generate_relative_amplitude_colorbar
from .mpemba import find_guaranteed_mpemba_dense
from .mpemba_validation import (
    MpembaValidator,
    worker_get_decomposition,
    simulation_worker_spectral,
    fast_diag_entropy,
)

__all__ = [name for name in globals() if not name.startswith("_")]
