# EGBS: Evolution-Guided Beam Search for Differential Trail Search

This repository contains the main experimental code and analysis scripts for the paper **"Evolution-Guided Beam Search for Differential Trail Search of Block Ciphers"**.

The project aims to build a unified framework for differential trail search by extending **Beam Search** with an **evolutionary diversity enhancement stage (EA stage)**, covering both ARX and SPN block ciphers, mainly **SPECK** and **PRESENT**.

---

## 1. Mapping Between the Paper and the Code

According to the paper, the core ideas are:

1. `delta`-optimal transition generation to control branch explosion
2. Structure-aware scoring, using look-ahead scoring for ARX and active S-box information for SPN
3. Evolutionary enhancement through perturbation and multi-parent filtering

These ideas are implemented mainly in:

- **Search core**: `core_search/`
- **Experiment scripts** (including MCTS/EGBS/EA comparisons and batch runs): `experiments/`
- **Statistics and plotting**: `analysis/`
- **MILP baseline tools**: `milp_tools/`
- **SAT baseline tools**: `sat_tools/`

---

## 2. Directory Structure

```text
EGBS/
├─ IEEE-Transactions-LaTeX2e-templates-and-instructions/  # Paper LaTeX source
├─ core_search/                                            # Core EGBS/BS search implementation
│  ├─ ea_initial_population.py                             # Initial population / path search for SPECK
│  ├─ ea_variation.py                                      # Mutation and multi-parent filtering
│  └─ present.py                                           # PRESENT search logic
├─ experiments/                                            # Main experiments, GA variants, diversity bridge analysis
├─ analysis/                                               # Statistics, visualization, table generation
├─ milp_tools/                                             # MILP modeling and path recovery tools
├─ sat_tools/                                              # SAT CNF modeling, solving, and decoding
├─ saved_initial_paths/                                    # Intermediate results
├─ saved_trail/                                            # Path results (pkl)
├─ saved_ea_txt/                                           # Path results (txt)
├─ saved_metrics/                                          # Metric data
├─ milp_output/                                            # MILP outputs
├─ sat_output/                                             # SAT outputs
└─ param_ablation/                                         # Parameter ablation summaries
```

---

## 3. Code Function Overview

### 3.1 Search Core: `core_search/`

- `ea_initial_population.py`: Base SPECK search entry point, including initial population generation, layer-wise expansion, path scoring, and core EGBS/sub-search procedures.
- `ea_variation.py`: Adds evolutionary enhancement on top of the base search, including perturbation candidate generation, multi-parent filtering, and single-parent/multi-parent variant experiments.
- `present.py`: PRESENT differential trail search implementation, including search and scoring based on S-box differential distributions and active S-box counts.

### 3.2 Experiment Scripts: `experiments/`

- `mcts_paper_aligned_timecap_stopflag.py`: Main experiment script aligned with the paper, used to run SPECK search under configurable rounds, directions, and time budgets.
- `mcts_mixed_onehour_cases.py`: Packs multiple SPECK settings into batch experiments and records results under a one-hour time budget.
- `mcts_speck96_r11_forward_batch.py`: Batch script for 11-round forward search on SPECK96, running over a set of input difference bit positions.
- `ga_variation.py`: GA variant experiment script used to test candidate expansion, crossover/mutation, and path retention behavior in the evolutionary stage.
- `ea_variation_param_ablation.py`: Parameter ablation script for comparing how settings such as `delta`, `top1`, and population size affect result quality and runtime.
- `speck_diversity_bridge_instrumentation.py`: Diversity-analysis experiment script that records round-level metrics, builds comparison payloads for `EGBS`, `EGBS+SP`, and `EGBS+MP`, and measures bridge-state rescue behavior.

### 3.3 Analysis and Plotting: `analysis/`

- `analyze_speck96_diversity.py`: Parses round-level states from SPECK96 text results and computes diversity metrics such as unique states, pairwise Hamming distance, and entropy.
- `plot_diversity_curves_pdf.py`: Reads round-level metrics from `saved_metrics/` and plots diversity curves and bridge-state bar charts.
- `speck_diff.py`: Extracts top-1 path scores from text outputs for later plotting or table generation.
- `speck_plots.py`: Unified SPECK plotting script with preset datasets and plotting helpers for line charts and comparisons.
- `speck96_plt.py`: Multi-method comparison plotting script for SPECK96, comparing `EGBS`, `BS`, `BS+SP`, and `BS+GA`.
- `present_diff.py`: Summarizes PRESENT search results and generates a table mapping input differences to best scores.
- `present_count_diff.py`: Computes output-difference probability distributions for PRESENT and filters valid differences under a score threshold.
- `present_plt.py`: Visualizes summarized PRESENT results by plotting the minimum-score curve.

### 3.4 MILP Baseline Tools: `milp_tools/`

- `speck_diff_find.py`: Generates MILP models for SPECK differential search, supporting normal search, fixed-input search, and fixed-output search.
- `speck_diff_sub_find.py`: Generates MILP subproblems under fixed input differences for fine-grained search over candidate differences.
- `MILPSbox.py`: Helper module for MILP constraints, providing building blocks such as XOR constraint generation.
- `get_route_from_sol.py`: Parses Gurobi `.sol` files and outputs round-wise differential trails, active bit positions, and reusable code-formatted results.
- `get_route_from_sol.sh`: Shell script version for extracting path information from `.sol` files.
- `get_route_from_sol_1.sh`: A more compatible variant of `get_route_from_sol.sh` with improved newline handling.
- `run_speck96_batch.sh`: Batch runner for Gurobi over multiple SPECK96 `.lp` models, with runtime and log recording.

### 3.5 SAT Baseline Tools: `sat_tools/`

- `speck_diff_cnf.py`: Generates CNF/SAT models for SPECK differential search and supports decoding solver outputs.
- `test_simple.smt2`: A minimal SMT example for validating basic bit-vector constraints, mainly used for SAT/SMT modeling debugging.
