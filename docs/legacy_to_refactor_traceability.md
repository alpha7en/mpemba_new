# Legacy to Refactor Traceability

This document tracks how validated legacy scientific kernels were moved into `research_refactor/`.
It is intended for scientific audit, not for software-product architecture.

## 1) Topology and graph generation

- `на проверку/rewiring_spectrum_data_npz_gemeration_for_figs/multicore_main_simulation.py:check_connectivity`
  -> `src/qdyn_research/topology.py:check_connectivity`
- `на проверку/fig_1/diagrams_2_without_numbers.py:generate_grid_tau`
  -> `src/qdyn_research/topology.py:generate_grid_tau`
- `на проверку/fig_4/main_visualisation.py:generate_rewired_grid_tau` (single-pass)
  -> `src/qdyn_research/topology.py:generate_rewired_grid_tau`
- `на проверку/rewiring_spectrum_data_npz_gemeration_for_figs/multicore_main_simulation.py:generate_rewired_grid_tau_GUARANTEED_CONNECTIVITY`
  -> `src/qdyn_research/topology.py:generate_rewired_grid_tau_guaranteed_connectivity`

## 2) Liouvillian assembly and eigensolvers

- `на проверку/fig_2/core.py:build_liouvillian`
  -> `src/qdyn_research/liouvillian.py:build_liouvillian_dense`
- `на проверку/rewiring_spectrum_data_npz_gemeration_for_figs/multicore_main_simulation.py:build_liouvillian_sparse`
  -> `src/qdyn_research/liouvillian.py:build_liouvillian_sparse`
- `на проверку/fig_2/core.py:analyze_liouvillian_modes`
  -> `src/qdyn_research/spectral.py:analyze_liouvillian_modes_dense_strict`
- `на проверку/fig_5/first_with_visulisation_help_old.py:analyze_liouvillian_modes_dense`
  -> `src/qdyn_research/spectral.py:analyze_liouvillian_modes_dense`
- `на проверку/rewiring_spectrum_data_npz_gemeration_for_figs/multicore_main_simulation.py:analyze_liouvillian_modes_sparse_ROBUST`
  -> `src/qdyn_research/spectral.py:analyze_liouvillian_modes_sparse_robust`

## 3) Metrics and mode analysis

- `на проверку/fig_2/core.py:project_rho_on_modes`
  -> `src/qdyn_research/metrics.py:project_rho_on_modes`
- `на проверку/fig_7/visual_IPR_ENG_captions.py:calculate_ipr`
  -> `src/qdyn_research/metrics.py:calculate_ipr`
- `на проверку/fig_2/core.py:calculate_distance_metric`
  -> `src/qdyn_research/metrics.py:calculate_distance_metric_logm`
- `на проверку/fig_9_10_11/first_mode_analys.py:get_projection_coefficient`
  -> `src/qdyn_research/metrics.py:get_projection_coefficient`
- `на проверку/fig_9_10_11/first_mode_analys.py:entropic_distance` (C-order)
  -> `src/qdyn_research/metrics.py:entropic_distance`

## 4) ODE simulation kernels

- `на проверку/fig_2/fully_simulation_core.py:QuantumSimulatorCore`
  -> `src/qdyn_research/simulation.py:QuantumSimulatorCore`
- `на проверку/fig_9_10_11/random_check.py:MpembaValidator`
  -> `src/qdyn_research/mpemba_validation.py:MpembaValidator`

## 5) Figure scripts and produced artifacts

- `на проверку/fig_1/diagrams_2_without_numbers.py`
  -> `scripts/fig1_grouped_modes.py`
- `на проверку/fig_2/main.py` + `на проверку/fig_2/core.py`
  -> `scripts/fig2_projection_bars_and_modes.py`
- `на проверку/fig_3/fig_3_main.py`
  -> `scripts/fig3_relaxation_dynamics.py`
- `на проверку/fig_4/main_visualisation.py`
  -> `scripts/fig4_average_path_length.py`
- `на проверку/fig_5/first_with_visulisation_help_old.py`
  -> `scripts/fig5_topology_and_spectrum.py`
- `на проверку/fig_6/visualisation_ENG_captions.py`
  -> `scripts/fig6_lambda_visualization.py`
- `на проверку/fig_7/visual_IPR_ENG_captions.py`
  -> `scripts/fig7_ipr_visualization.py`
- `на проверку/fig_9_10_11/9_10_script_task_B_heatmap_ENG.py`
  -> `scripts/fig9_task_b_heatmap.py`
- `на проверку/fig_9_10_11/11_algoritm_visual.py`
  -> `scripts/fig11_algorithm_visual.py`
- `на проверку/fig_9_10_11/random_check.py`
  -> `scripts/fig10_random_benchmark.py`

## 6) Intentional strategy split (kept by design)

- Rewiring topology:
  - single-pass (`generate_rewired_grid_tau`) for topological diagnostics,
  - connectivity-guaranteed (`generate_rewired_grid_tau_guaranteed_connectivity`) for spectral statistics.
- Vectorization convention:
  - `order='F'` for Liouvillian/mode-based kernels,
  - legacy-compatible C-order entropy path for `first_mode_analys.py` workflows.

