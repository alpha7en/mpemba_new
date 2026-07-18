# Figure formatting report (Springer QIP)

Living log of the figure-restyling work. Goal: every figure in one uniform, journal-
compliant style. **Rule: computational kernels in `src/qdyn_research/` are never touched**
(only `plot_style.py` is added). Each figure change is presentation-only and confirmed with
`git status` (kernels clean) + `git diff` (only matplotlib lines changed).

## 1. The style engine — `src/qdyn_research/plot_style.py`

Single source of truth, imported by every figure script. Key pieces and *why*:

- **`apply_style()`** — installs `rcParams`: font Arial 9 pt (ticks 8 pt), thin axis lines,
  colorblind-safe line cycle (Wong), and **`pdf.fonttype = 42`** so fonts are embedded as
  TrueType (default is Type-3, which Springer rejects).
- **Width constants** `WIDTH_FULL=131.2 mm` (= the real sn-jnl `\textwidth`, measured as 372 pt —
  **corrected from a wrong 174 mm**, which had made every figure ~1.33× too wide → LaTeX shrank
  them to ~0.75× → fonts landed at ~6 pt instead of 8), `WIDTH_HALF≈63 mm` (0.48\linewidth),
  `WIDTH_THIRD≈42 mm` (0.32\linewidth). Figures are built *at their final on-page width* — the **WYSIWYG**
  principle: 9 pt in the script == 9 pt on paper, because LaTeX includes them at natural size
  with no scaling. This is what kills the old "font-doubling" hacks and the size mismatch.
- **Palettes**: `SEQ_CMAP = "viridis"` (intensity/excitability maps); signed mode maps keep
  the red(+)/white(0)/blue(-) convention; `LINE_COLORS` (Wong) for line plots.
- **`EDGE_WIDTH/EDGE_COLOR/EDGE_ALPHA`** — thin (0.8 pt, light grey) graph edges on network maps,
  so the graph structure shows without competing with node colours (author feedback).
- **`save_pdf(fig, path)`** — vector PDF, **no `bbox_inches='tight'`** (which would crop and
  re-introduce scaling); combined with per-figure `layout="constrained"` the output width
  equals the requested figsize exactly (verified: fig1 PDF is 118.1 mm = 0.9\linewidth to the mm,
  so LaTeX includes it at scale 1.0 and the 8 pt text is truly 8 pt on paper).

## 2. Figure status

| Fig | Script | Output (`paper/figures/*.pdf`) | Data source | State |
|----|----|----|----|----|
| **1** | fig1_grouped_modes | 10x10main_without_numbers | cached live 10x10 eig | **DONE** |
| **2** | fig2_projection_bars | new_tabled…8STATES | cached live 10x10 eig | **DONE** |
| **3** | fig3_relaxation_dynamics | 9states…selection_ENG | cached 10x10 ODE (γ=0.5) | **DONE** |
| **4** | fig4_average_path_length | log10x10_p_L, log10x10_p_L_SCALED | live ensemble | **DONE** |
| **5** | fig5_topology_and_spectrum | with_examples_1/2/3 | cached live 10x10 eig | **DONE** |
| **6** | fig6_lambda_visualization | mode_1/2/3_p_log_ENG, gap_p_log_ENG | **found npz** | **DONE** |
| **7** | fig7_ipr_visualization | 10x10_IPR_graph_mode_k_1/2/3 | **found npz** | **DONE** |
| **8** | fig8_overlap | 10x10_community_analyse_mode_k_1/2/3_ENG | **fresh npz (has tau)** | **DONE** |
| **9** | fig9_task_b_heatmap | Task_B…part1 (hist), part2 (maps) | cached live 10x10 | **DONE** |
| **11** | fig11_algorithm_visual (Step1) | Step1_Three_Modes | **validator pkl** | **DONE** |
| **12** | fig11_algorithm_visual (hot/cold) | final_hot_state, final_cold_state | **validator pkl** | **DONE** |
| **13** | fig10_random_benchmark | alg_compare | **pivoprosto.txt** | **DONE** |

## 3. What changed per finished figure

- **fig11 / fig12** (`scripts/fig11_algorithm_visual.py`): `draw_base_graph` → viridis + thick
  edges (`EDGE_*`) + returns the node collection so a **per-panel colorbar** can be drawn;
  Step1 rebuilt at `WIDTH_FULL` with `$k=1/2/3$` labels and one horizontal colorbar per panel;
  added a hot/cold block that saves `final_hot_state.pdf` (red = darkest node) and
  `final_cold_state.pdf` (blue = M brightest). `CHECKPOINT` now points at the migrated
  validator so the *exact graph from the article* is used. Computation (excitability map,
  `find_guaranteed_mpemba_dense`, hot/cold selection) unchanged.
- **fig13** (`scripts/fig10_random_benchmark.py`): loads the validator from the checkpoint
  (`MpembaValidator.load_state`) instead of drawing a new graph; loads the published 10^4
  crossing times from `precalc/pivoprosto.txt` (falls back to recomputing if absent);
  horizon widened to 120 so the algorithmic crossing (t*≈45) is detected; styled histogram +
  dashed vline for the algorithmic pair. `run_smart_strategy_score` / spectral methods unchanged.
- **fig6** (`scripts/fig6_lambda_visualization.py`): clean rewrite of the *plotting only*.
  Reduction preserved verbatim — `mean/std` of `arr[k].real` over runs, gap = `arr[1].real -
  arr[2].real`. Produces the 4 log-p panels at `WIDTH_COL`. Removed the old font-doubling
  helpers (`_apply_bold_double_text`, `_size_to_points`).
- **fig7** (`scripts/fig7_ipr_visualization.py`): clean rewrite of plotting only. IPR
  computation preserved (`calculate_ipr(V[:, mode], sites, "F")`). Produces the 3 IPR panels
  at `WIDTH_THIRD`.
- **fig4** (`scripts/fig4_average_path_length.py`): clean rewrite of plotting only; the
  ensemble call `run_average_path_experiment(10, 10, 1000, …, generate_rewired_grid_tau)` is
  unchanged. One computation → two panels at `0.37·WIDTH_FULL` (≈64 mm): full scale, and a
  crossover zoom (x∈[1e-3,1e-1], y auto-fit to the window) → `log10x10_p_L.pdf` /
  `log10x10_p_L_SCALED.pdf`.
- **fig8** (`scripts/fig8_overlap.py`): clean rewrite of plotting only; `calculate_max_overlap`
  (Fortran-reshape, |diag|^2 mass, Louvain community masses) preserved verbatim. Now reads
  **tau straight from the fresh npz** (matches the stored eigenvectors — no seed regeneration).
  3 panels at `WIDTH_THIRD`. Needs `python-louvain` (added to requirements.txt).
- **fig3** (`scripts/fig3_relaxation_dynamics.py`): relaxation trajectories D(rho(t)) for the 8
  states, bold for #3/4/5, crimson dots at the crossings of pairs (3,5) and (4,5). Rendered at
  **γ=0.5** (the author's choice to reproduce the published crossings at t≈0.85 and 4.5; the text
  still states γ=0.1 — a content decision to settle separately). ODE unchanged, cached to
  `precalc/fig3_dynamics.npz` (cache stores γ and is invalidated if γ changes). Full width, no
  baked title, RU→EN, 8–9 pt.
- **fig5** (`scripts/fig5_topology_and_spectrum.py`): 3 panels `with_examples_1/2/3` at p=0/0.1/0.3,
  each = topology (top, rewired edges in orange) + Liouvillian spectrum (bottom, complex plane).
  Computation (dense eig) unchanged, **cached to `precalc/fig5_spectra.npz`**. RU title removed,
  baked "Eigenvalue spectrum…" title dropped (caption covers it), fonts 20 pt → 8 pt, spectrum
  switched from spine-at-zero (which clipped the `Im λ` label at the edge) to a framed axes with a
  thin reference cross at Re=0/Im=0; common Re/Im limits across panels. `WIDTH_THIRD` (42 mm) each.
- **fig2** (`scripts/fig2_projection_bars_and_modes.py`): rebuilt. The projection computation is
  unchanged (`project_rho_on_modes`, `analyze_projections` W_k = |c_k|^2) and **cached to
  `precalc/fig2_projections.npz`**. Redesigned to fit: the per-panel **mode maps were removed**
  (not discussed in the body text, redundant with Fig.1 — caption updated accordingly) and the
  legacy "Boundary" state dropped (→ 8 states, matching the caption). Now a 2-column × 4-row grid,
  each panel = schematic (yellow=occupied) + log-scale W_k bar chart; RU→EN labels. Result:
  131.2 × 154.8 mm at 1\linewidth (scale 1.0, fonts true 8 pt) — was an impossible ~1.14 m-tall
  figure. Shared y-label (left column) / x-label (bottom row) small-multiples style.
- **fig1** (`scripts/fig1_grouped_modes.py`): the mode atlas. Computation unchanged (dense eig →
  `Re(diag(Mat(v_k)))` per mode), **cached to `precalc/fig1_modes.npz`** (only the 98 small
  population maps + eigenvalues, not the 1.6 GB eigenvectors). Layout **reflowed to the published
  2 super-columns** (k'=1..13 | k'=14..26), all 26 degeneracy groups; per-mode titles dropped
  (group label k'+Re(λ) kept), red/white/blue maps (opacity = amplitude), shared bottom colorbar.
  Output `0.9·WIDTH_FULL` (≈157×222 mm, aspect 1.42 ≈ published 1.46).
- **fig9** (`scripts/fig9_task_b_heatmap.py`): clean rewrite of plotting only; the computation
  (2000-graph search → median-L_avg representative → dense eig → `calculate_excitability_map`)
  is unchanged, but **cached to `precalc/fig9_representative.npz`** so the figure is reproducible
  and restylable without the ~minutes-long dense eig. Split into the article's two panels at
  `0.75·WIDTH_FULL`: part1 = L_avg histogram (median vline), part2 = B(k,i) maps for k=1,2,3
  (viridis + per-panel colorbars). Edges are drawn **thin/light** here (dense 10x10 → the graph
  should recede so the node excitability is the accent), unlike the thick edges on the sparse 6x6
  fig11. Note: the part2 caption in main.tex mentions "marker size" encoding — the figure encodes
  magnitude by color only, so that caption clause should be trimmed.

## 4. Data files (`scripts/precalc/`)

- `pivoprosto3.pkl` (legacy) → `validator_6x6_p015.pkl` — the exact 6x6 realization behind the
  published algorithm figures. Migrated by loading under a `__main__.MpembaValidator` shim,
  re-dumping under the `qdyn_research` class, and adding the missing `n` attribute.
- `rewiring_spectrum_data_10x10_20251008_*.npz` — the original "found" archive (exact article
  data, **no `tau`**). Kept for verification; **fig6/7/8 no longer use it**.
- `rewiring_spectrum_data_10x10_20260710_*.npz` — the **canonical** archive: our overnight run,
  has `tau`, statistically = the paper. `find_npz` picks the newest, so **fig6, fig7 and fig8 all
  read this one** (single reproducible dataset for the data-availability statement).
- `pivoprosto.txt` — the published 10^4 random crossing times (Fig.13 histogram data).

**Overnight run (`scripts/run_multicore_npz.py`, for Fig.8's tau-archive):** the script now pins
BLAS to 1 thread per worker (env vars set before `import numpy`). Without this, 8 pool workers
each spawned ~16 BLAS threads (128 vs 16 cores) and throughput collapsed (~40 h ETA). Pinned,
it should finish in ~3-5 h. Run from the repo root; the resulting `.npz` (with `tau`) lands there.

## 5. New-vs-old equivalence

Confirmed **statistically** identical (ensemble IPR / Re λ₁ match the paper to the digit; the
`L`/solver were verified exact earlier). **Not** bit-identical per random seed: the rewiring RNG
call-order differs, so the same seed yields a different graph (λ deviate ~1%). This is
scientifically irrelevant for ensemble-averaged figures, but it means Fig.8's tau must come from
a *fresh* archive rather than seed-regeneration.
