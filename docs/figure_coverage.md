# Figure Coverage Map

This map tracks which refactored script generates each final research block.

## Coverage status

- `fig_1` -> `scripts/fig1_grouped_modes.py`
  - Outputs: `10x10_dynamically_grouped_diagrams_final_with_correct_norm_RESTORED2.png`,
    `10x10_dynamically_grouped_diagrams_final_bar_visualisation.png`.
- `fig_2` -> `scripts/fig2_projection_bars_and_modes.py`
  - Outputs: `9states_modes_bar_chart_with_modes_10x10_log_afterdebag_gamma01.png`.
- `fig_3` -> `scripts/fig3_relaxation_dynamics.py`
  - Outputs: `9states_modes_simulation_line_10x10_with_selection_ENG.png`.
- `fig_4` -> `scripts/fig4_average_path_length.py`
  - Outputs: `log10x10_p_L.png`.
- `fig_5` -> `scripts/fig5_topology_and_spectrum.py`
  - Outputs mode-dependent combined spectrum file (legacy naming pattern preserved).
- `fig_6` -> `scripts/fig6_lambda_visualization.py`
  - Reads existing `.npz`; outputs mode-wise plots and gap plots in `visualisation_plots_ENG_captions/`.
- `fig_7` -> `scripts/fig7_ipr_visualization.py`
  - Reads existing `.npz`; outputs IPR plots in `IPR from p images ENG/`.
- `fig_8` -> `PASS` (kept intentionally empty, as in legacy structure).
- `fig_9`/`fig_10` combined panel -> `scripts/fig9_task_b_heatmap.py`
  - Outputs: `Task_B_Strict_Layout_Fixed_test_p_0_ENG15.png`.
- `fig_10` benchmark analysis -> `scripts/fig10_random_benchmark.py`
  - Outputs benchmark artifacts in `res_gap/` (`benchmark.png`, `benchmark.txt`, checkpoint `.pkl`).
- `fig_11` -> `scripts/fig11_algorithm_visual.py`
  - Outputs: `Step1_Three_Modes.png`, `Step2_Hot_State.png`,
    `Step3_Cold_Iterations.png`, `Step4_Final_Algorithm_Selection.png`.
- `first-mode diagnostics` -> `scripts/first_mode_analysis.py`
  - Outputs console diagnostics and returns candidate states for downstream checks.

## Heavy precomputation

- `scripts/run_multicore_npz.py`
  - One-time sparse shift-invert ensemble run producing `rewiring_spectrum_data_...npz`.
  - Downstream consumers: `fig6`, `fig7`.

