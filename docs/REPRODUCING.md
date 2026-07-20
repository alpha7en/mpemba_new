# Reproducing the figures and benchmarks

Every figure in the paper is produced by a script in `scripts/` (run from that directory:
`python figN_*.py`). Heavy computations are cached in `scripts/precalc/`; the plotting layer can
always be re-run from the caches alone. This document maps each paper figure to its script, its
data dependency, and the source of randomness (if any).

## Randomness policy

Each stochastic artifact in this repository is in one of two states:

1. **Regenerable** — produced by a committed script with an explicit seed; deleting the data and
   re-running the script reproduces it (bit-exactly where stated).
2. **Stored realization** — a specific random graph realization used for the paper's
   illustrative figures is provided as a data artifact (with provenance and self-verification
   fingerprints where applicable). This is the usual practice for representative single-graph
   figures; no retroactive seed exists, the artifact itself is the source of truth.

## Figure map

| Paper figure | Output PDF(s) | Script | Data / randomness |
|---|---|---|---|
| Fig. 1 | `10x10main_without_numbers` | `fig1_grouped_modes.py` | deterministic (regular 10×10 grid, dense eig); cache `fig1_modes.npz` |
| Fig. 2 | `new_tabled_10x10_bar_chart_log_ENG_new_8STATES` | `fig2_projection_bars_and_modes.py` | deterministic; cache `fig2_projections.npz` |
| Fig. 3 | `9states_modes_simulation_line_10x10_with_selection_ENG` | `fig3_relaxation_dynamics.py` | deterministic (ODE integration, γ=0.1); cache `fig3_dynamics.npz` |
| Fig. 4 | `log10x10_p_L`, `..._SCALED` | `fig4_average_path_length.py` | ensemble statistics (1000 graphs/p, single-pass rewiring, L_avg on the largest connected component); cache `fig4_pathlength.npz`. Connectivity statistics backing the text: `check_connectivity_stats.py` (seed 1) |
| Fig. 5 | `with_examples_1..3` | `fig5_topology_and_spectrum.py` | representative realizations stored in cache `fig5_spectra.npz` |
| Fig. 6 | `re1_p_log`, `oscfrac_p_log`, `im1_p_log`, `sep_p_log` | `fig6_lambda_visualization.py` | corrected rightmost-mode sweep (below); reduction cached in `fig6_rightmost.npz` (git) |
| Fig. 7 | `10x10_IPR_group_k_1..3` | `fig7_ipr_visualization.py` | group reduction `fig78_groups.npz` (git), produced by `reduce_sweep_groups.py` from the corrected sweep |
| Fig. 8 | `10x10_overlap_group_k_1..3` | `fig8_overlap.py` | same group reduction; Louvain seeded (random_state=42) |
| Figs. 9–10 | `Task_B_..._part1/part2` | `fig9_task_b_heatmap.py` | representative 10×10 graph stored in cache `fig9_representative.npz` (incl. its `tau`); cacheless re-run is seeded (2026) but yields a different, reproducible representative |
| Figs. 11–12 | `Step1_Three_Modes`, `final_hot_state`, `final_cold_state` | `fig11_algorithm_visual.py` | the paper 6×6 graph (below) |
| Fig. 13 | `alg_compare` | `fig10_random_benchmark.py` | benchmark data `dirichlet_benchmark_30k.npz` (below) |

## Key data artifacts

**`precalc/paper_graph_6x6_p015.npz`** (git-tracked, ~16 KB) — the exact 6×6, p=0.15 disorder
realization used in Figs. 11–13 and the Sec. 8.4 benchmark: adjacency matrix, model parameters,
and physics fingerprints (Liouvillian spectrum, engineered hot/cold nodes, crossing time t*,
initial entropy gap) for self-verification.

**`precalc/validator_6x6_p015.pkl`** (~44 MB, not in git) — derived local cache (Liouvillian +
eigendecomposition of the above). Rebuild it with:

```
python rebuild_validator.py
```

The script recomputes everything from the canonical graph and refuses to write the checkpoint
unless five physics-identity checks pass (spectrum to ~1e-13, exact hot/cold node selection,
gap, t*, and a cross-check against the stored benchmark data). Eigenvector phases may differ
between LAPACK builds; all checks are phase-invariant, as is every quantity used in the paper.

**`precalc/dirichlet_benchmark_30k.npz`** (git-tracked) — the Fig. 13 / Sec. 8.4 Monte Carlo
benchmark: crossing times of 30 000 admissible Dirichlet(1,…,1) pairs. Regenerate bit-exactly
(~25 min on 6 cores) with:

```
python run_dirichlet_benchmark.py     # SEED = 7, draws in the main process only
```

Headline numbers: t*_algo = 44.66, and 1.21 % of admissible pairs cross faster than the
engineered pair. Crossings beyond t ≈ 150–160 sit at the numerical noise floor of D(t)
(~1e-13) and are not physically resolved (stated in the paper; the figure omits them).

**`precalc/sweep_rightmost_10x10.npz`** (~1.8 GB, not in git) — the CORRECTED p-sweep archive
behind Figs. 6–8: for every graph the 10 rightmost (largest Re λ) modes found by the verified
exp-transform solver (`spectral.analyze_liouvillian_modes_rightmost`; validation harness:
`verify_rightmost_solver.py`, 61/61 dense-truth graphs + 4/4 10×10 spot checks), with per-mode
residuals stored (worst 1.6e-12). Regenerate with `run_sweep_rightmost.py` (resumable,
per-p checkpoints; topologies seeded per (p, run_idx) exactly as the legacy archive, so the
graphs are identical). Small reductions used by the figures (`fig6_rightmost.npz`,
`fig78_groups.npz`) are git-tracked, so the figures rebuild without the big archive.

**`precalc/spectrum_sweep_10x10.npz`** (~670 MB, not in git) — the LEGACY sweep archive
(shift-invert near σ=0, modes ordered by |λ|). Kept for provenance/comparison only: it
misidentifies the slowest mode for p < 0.023 (see docs/revision_log.md, Block B1). Not used
by any current figure. Regenerate with `run_multicore_npz.py`.

## Model conventions

- N = number of graph nodes (10×10 lattice → N = 100); the Liouvillian is an N²×N² matrix.
- Code parameter J = 1 corresponds to the article's physical hopping J = 1/2 (verified:
  `L_code(J=1)` ≡ `L(J_phys=1/2)` exactly); with γ = 0.1 this gives the article's γ/J = 0.2.
- Eigenvalues are sorted by Re(λ) descending; mode k = 1 is the slowest non-trivial mode.
- Vectorization is row-major (`order="C"`) everywhere (`entropic_distance`, excitability maps,
  benchmark).
