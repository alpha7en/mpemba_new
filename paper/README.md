# Paper build — Springer *Quantum Information Processing* (journal 11128)

Reformatted submission of **"Quantum Mpemba Effect in Disordered Lattices"** for the
Springer Nature `sn-jnl` template (Math & Physical Sciences, numbered references).

## Files

- `main.tex` — manuscript, class `sn-jnl` with `\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}`.
- `refs.bib` — 23 references in BibTeX (keys match `\cite` in `main.tex`).
- `sn-jnl.cls`, `sn-*.bst` — Springer Nature class + bibliography styles (from the official
  December-2024 template package; kept here so the paper compiles as-is).
- `figures/` — **placeholder PDFs** for now (Phase C/D replaces them with the real vector figures).

## How to compile

The Springer Nature class/style files are already in this folder, so it builds directly:

```
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

Produces `main.pdf` in the journal house style. **Verified locally with sn-jnl:**
28 pages, 0 errors, 0 undefined citations/references, 23 references rendered in
`sn-mathphys-num` (numbered) style.

> Alternative: the same `main.tex` + `refs.bib` can be uploaded to the
> [Springer Nature template on Overleaf](https://www.overleaf.com/latex/templates/springer-nature-latex-template/gsvvftmrppwq).

## Changes applied during the port (review these)

- **Fixed 2 broken citations** (case mismatch → rendered as `[?]`): `\cite{carollo2021}`→`Carollo2021`,
  `\cite{longhi2024}`→`Longhi2024`.
- **Added 6 previously-uncited references** at natural anchors (they were defined but never cited):
  `tang2018` (waveguide quantum-walk platform, Sec. 2.1), `lindblad1976,gks1976,breuer2002`
  (Lindblad master equation, Sec. 2.1), `gyamfi2020` (Liouville-space vectorization, Sec. 2.3),
  `neumann` (von Neumann entropy, Sec. 2.2), `watts1998` (Watts–Strogatz crossover, Sec. 6.2).
- Neutralized draft working-note colors: `\renewcommand{\textcolor}[2]{#2}` (green/dark-green text
  kept as normal black; the single red "delete" note was already commented out).
- Removed Russian TODO comment lines; normalized Unicode dashes/quotes to LaTeX.
- Front matter rebuilt with sn-jnl macros (`\author/\affil/\abstract/\keywords`); dropped the old
  `nsart_eng3` custom macros (`\volume,\nomer,\firstpage,\shortcite,\authorcopy,\corrauthor,\pacs`).
- Supplementary Material (1D chain) and the old author-info back matter are **not** included yet.

## Figure-name contract (Phase C/D targets)

`main.tex` references vector figures at `figures/<name>.pdf`. Phase D scripts must output PDFs with
exactly these basenames (spaces/`=` were sanitized to `_`):

| Fig | file (`figures/…`) |
|----|----|
| 1  | `10x10main_without_numbers` |
| 2  | `new_tabled_10x10_bar_chart_log_ENG_new_8STATES` |
| 3  | `9states_modes_simulation_line_10x10_with_selection_ENG` |
| 4  | `log10x10_p_L`, `log10x10_p_L_SCALED` |
| 5  | `with_examples_1/2/3` |
| 6  | `mode_1_p_log_ENG`, `mode_2_p_log_ENG`, `mode_3_p_log_ENG`, `gap_p_log_ENG` |
| 7  | `10x10_IPR_graph_mode_k_1/2/3` |
| 8  | `10x10_community_analyse_mode_k_1/2/3_ENG` |
| 9/10 | `Task_B_Strict_Layout_Fixed_test_p_0_ENG15_edited_part1/part2` |
| 11 | `Step1_Three_Modes` |
| 12 | `final_hot_state`, `final_cold_state` |
| 13 | `alg_compare` |

(These legacy names are kept for now to minimize churn; we can rename to `fig01…` etc. in Phase D
if you prefer — it is a one-line change to the contract.)
