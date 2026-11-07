# spem-experiments

Code, data, and results for the paper

> H. Okamura, K. Nakayama, and T. Dohi,
> **Sinkhorn Projection EM for Bernstein Copulas**,
> Graduate School of Advanced Science and Engineering, Hiroshima University.
> Submitted to *Computational Statistics & Data Analysis* (2026).

Every number in the paper is produced by the scripts in this repository.
The CSV files in `results/` are the exact outputs used for the submitted manuscript.

## What is here

The paper proposes the Sinkhorn projection EM (SP-EM) algorithm for the weights of a Bernstein copula.
SP-EM keeps the E-step of the EM algorithm of Dou et al. (2016) and replaces the exact M-step, a reverse KL projection onto the uniform-marginal constraint set, by the forward KL projection computed by Sinkhorn scaling.

`src/lib.jl` is a self-contained Julia implementation of the general-dimension EM harness:

| Component | Function(s) in `src/lib.jl` |
|---|---|
| Beta basis and E-step | `basis_matrix`, `basis_tensor`, `estep` |
| SP-EM M-step (Algorithm 1, Sinkhorn scaling) | `proj_sinkhorn` |
| Exact M-step, alternating bisection (Dou et al.) | `proj_dou` |
| Exact M-step, Newton on the dual (Appendix D) | `proj_newton` |
| EM driver with common stopping rule and M-step timing | `emfit` |
| SP-EM fixed point to high precision | `spem_fixedpoint` |
| Reference MAP estimate (Newton on the tangent space) | `map_newton`, `check_reference` |
| Quantities of Theorem 1 and Corollary 1 | `residuals`, `curvature_constants` |
| Synthetic data (independent, Gaussian, Clayton copulas) | `sample_copula` |
| Uranium data and mid-rank pseudo-observations | `load_uranium`, `pseudo_obs` |

All experiments use the MAP objective with pseudo-count alpha = 1e-3, the same initial value, the same stopping rule (change of the objective over 10 iterations below 1e-6 absolute and 1e-8 relative, at most 2500 iterations), and the same inner accuracy 1e-6 for the three M-step solvers.

## Requirements

* Julia 1.10 or later. The reported results were produced with Julia 1.12.2, single-threaded, on an Apple M4 processor.
  The only registered dependency is `Distributions` (pinned in `Manifest.toml`).
* Python 3 (standard library only) for `scripts/make_tables.py`.

## Reproducing the results

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'

julia --project=. scripts/exp_runtime.jl    # results/runtime.csv
julia --project=. scripts/exp_large.jl      # results/large.csv
julia --project=. scripts/exp_bound.jl      # results/bound.csv
julia --project=. scripts/exp_stopped.jl    # results/bound_stopped.csv
julia --project=. scripts/exp_init.jl       # results/init_runs.csv, init_summary.csv
julia --project=. scripts/exp_uranium.jl    # results/uranium_{pairs,bound,d3,monotone}.csv

python3 scripts/make_tables.py              # results/tables.tex
```

Each script overwrites its CSV files in `results/` and prints a log like those in `results/logs/`.
The random seeds are fixed, so iteration counts, objective values, errors, and bound quantities are reproduced exactly up to floating-point rounding.
Absolute times are machine dependent; the paper reports time ratios.
A consistency check of the three M-step solvers against each other and against the reference MAP solver is available as

```sh
julia --project=. -e 'include("src/lib.jl"); self_check()'
```

Running all six scripts takes a few hours on a laptop; `exp_runtime.jl` and `exp_uranium.jl` are the longest.

## Mapping to the paper

`scripts/make_tables.py` writes the table bodies to `results/tables.tex`, one snippet per table, separated by `%% ==== <name>` markers.

| Paper | Snippet in `tables.tex` | Source CSV |
|---|---|---|
| Table 1 (runtime, d = 2) | `runtime-d2` | `runtime.csv` |
| Table 2 (runtime, d = 3) | `runtime-d3` | `runtime.csv` |
| Table 3 (larger orders and dimensions) | `large` | `large.csv` |
| Table 4 (estimation accuracy) | `accuracy` | `runtime.csv` |
| Table 5 (bound quantities, Gaussian data) | `bound-gauss` | `bound.csv` |
| Table 6 (bound summary) | `bound-summary` | `bound.csv` |
| Table 7 (stopped iterates, Corollary 1) | `stopped-gauss` | `bound_stopped.csv` |
| Table 8 (initial values) | `init` | `init_summary.csv` |
| Table 9 (uranium, summary over 21 pairs) | `uranium-summary` | `uranium_pairs.csv` |
| Table 10 (uranium, effect of the order) | `uranium-orders` | `uranium_pairs.csv` |
| Table 11 (uranium, bound quantities) | `uranium-bound` | `uranium_bound.csv` |
| Table 12 (uranium, monotonicity of the objective) | `uranium-monotone` | `uranium_monotone.csv` |
| Table 13 (uranium, trivariate Co-Sc-Ti) | `uranium-d3` | `uranium_d3.csv` |
| Table 14 (Appendix B, all 21 pairs) | `uranium-full` | `uranium_pairs.csv` |

## Data

`data/uranium.csv` is the uranium exploration data set of Cook and Johnson (1986): log concentrations of seven chemical elements (U, Li, Co, K, Cs, Sc, Ti) in 655 water samples collected near Grand Junction, Colorado.
The file is the `uranium` data set distributed with the R package `copula`.
The scripts convert each column to mid-rank pseudo-observations before fitting.

R. D. Cook and M. E. Johnson (1986). Generalized Burr-Pareto-logistic distributions with applications to a uranium exploration data set. *Technometrics* 28(2), 123-131.

## License

MIT License. See `LICENSE`.
