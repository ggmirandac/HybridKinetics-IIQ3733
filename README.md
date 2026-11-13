# Modelo Hibrido

Repositorio de proyecto del ramo IIQ3733
:P

## Layout

```
src/                 one module per method (put on sys.path; there is no installable package)
  model/             the model, built in ONE place: parameters.py, kinetics.py (rate laws), stoichiometry.py
  settings.py        loader of settings/*.csv; default_params() gives the method settings as a dict
  paths.py           repo root and output-directory resolution
  solver.py          thread pinning, executable path, solver check
  data.py            data loading, sigmas, bounds of the estimation variables, check that the bounds contain the data
  estimation/        experiment.py (Pyomo model of one condition), estimator.py, parallel.py, multistart.py, refit.py, anchors.py
  sensitivity.py     sensitivities of the measured outputs (implicit function theorem, CasADi automatic differentiation)
  identifiability/   structural.py (Stigter and Molenaar), practical.py (Fisher information), rules.py
  selection.py       automated parameter selection (stage 0, structural step, practical loop); CLI
  final_analysis.py  the selected structure: classes, Fisher information, fit
  report.py          REPORT.md assembled from the saved files of a run
  uncertainty/       bootstrap.py, profiles.py, loocv.py
  metrics.py         fit metrics, likelihood-ratio threshold, misfit scale
settings/            run settings as CSV files: method.csv (the method), bounds_rules.csv, theta_bounds.csv,
                     protected_params.csv, solver_options.csv
notebooks/           one notebook per step of the method, calling src/ (see notebooks/README.md for the run order and switches)
  model, data, parameter_selection, structural_identifiability, final_fit, bootstrap, profiles, loocv,
  figure_data, figures_main, figures_si; paper_style.py (rcParams, colours, labels of the figures)
paper/               build_paper.py (figures, tables and numbers of the paper from results/flux_model), supplementary/
tests/test_pipeline.py, tests/fixtures/ (a fitted theta the tests start from)
environment.yml      conda environment (see below)
results/             final results of the method (results/flux_model)
```

## Method

1. **Stage 0**: weighted least squares (Pyomo parmest, IPOPT) with all 37 parameters free; multistart from the literature
   values and `n_starts` log-uniform starts.
2. **Structural identifiability** (Stigter and Molenaar 2015): SVD of the exact rate sensitivities (CasADi) at
   `structural_q` random parameter vectors; one parameter of every structural zero is fixed.
3. **Practical identifiability**: Fisher information of the output sensitivities (implicit function theorem, CasADi
   derivatives). While it is rank deficient (`rank_tol`) the parameter with the largest loading on the least identifiable
   direction is fixed; then, of the pair with the largest |r| > `corr_threshold`, the more sensitive member (mean
   normalized sensitivity |d ln y/d ln theta|) stays free and the other is fixed. One fix per iteration, refit after each. Every rate law keeps one free parameter.
4. **Characterization** of the final structure: case bootstrap, profile likelihood, leave-one-condition-out cross-validation.

Run order of the notebooks: `parameter_selection` -> `final_fit` -> `bootstrap`, `profiles`, `loocv` -> `figure_data` ->
`figures_main`, `figures_si` (`model`, `data`, `structural_identifiability` are independent). Each notebook loads the saved
results of `HK_OUT` (default `results/flux_model`) and recomputes only with `HK_RUN=1`; the settings are the dict
`params` shown in the cell before the run (`default_params()`, read from `settings/method.csv`); see `notebooks/README.md`.

Environment: `conda env create -f environment.yml && conda activate hybridkin` (python 3.12, pyomo 6.10, casadi 3.7, IPOPT 3.14,
numpy / pandas / scipy / matplotlib, jupyter and nbconvert). Register the notebook kernel with
`python -m ipykernel install --user --name hybridkin`.

Selection run from the command line: `python src/selection.py --out results/flux_model [--from_anchor <dir with
anchor_state.pkl>]` (a relative `--out` is relative to the repo root). Every run copies `settings/` into `<out>/settings_used`. Tests: `python tests/test_pipeline.py`.
