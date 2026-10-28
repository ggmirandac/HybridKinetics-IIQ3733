# Modelo Hibrido

Repositorio de proyecto del ramo IIQ3733
:P
## Layout (v3 pipeline)

```
src/                 one module per method (put on sys.path; there is no installable package)
  model/             the model, built in ONE place: parameters.py, kinetics.py (rate laws), stoichiometry.py
  settings.py        loader of settings/*.csv; default_params(step) gives the settings of one step as a dict
  paths.py           repo root, output-directory resolution and the guard of the frozen result folders
  solver.py          thread pinning, executable path, solver check
  data.py            data loading, sigmas, bounds of the estimation variables, check that the bounds contain the data
  estimation/        experiment.py (Pyomo model of one condition), estimator.py, parallel.py, multistart.py, refit.py, anchors.py
  sensitivity.py     reduced (profiled) sensitivities and FIM
  identifiability/   structural.py, output_level.py, check.py, practical.py, rules.py
  selection.py       automated parameter selection (stage 0 + reduction loop); CLI
  final_analysis.py  final report of the selected structure
  report.py          REPORT.md assembled from the saved files of a run
  uncertainty/       bootstrap.py, profiles.py, loocv.py
  metrics.py         fit metrics, LR threshold, misfit scaling
settings/            run settings as CSV files (name,value,type,description)
notebooks/           one notebook per step of the method, calling src/ (see notebooks/README.md for the run order and switches)
  model, data, parameter_selection, structural_identifiability, final_fit, bootstrap, profiles, loocv, isolation,
  figure_data, figures_main, figures_si; paper_style.py (rcParams, colours, labels of the figures)
tests/test_pipeline.py, tests/fixtures/ (small copies of old results the tests compare with)
environment.yml      conda environment (see below)
legacy/              paper / v2 / rounds 1-3 code and the old v3_*.py scripts (local only, not in the published repo)
results_v4/          output of new runs (results/, results_v2/, results_v3/ are frozen)
```

Run order of the notebooks: `parameter_selection` -> `final_fit` -> `bootstrap`, `profiles`, `loocv` -> `figure_data` ->
`figures_main`, `figures_si` (`model`, `data`, `structural_identifiability`, `isolation` are independent). Each notebook
loads the saved results of `HK_OUT` (default `results_v4/flux_model`) and recomputes only with `HK_RUN=1`; the settings of
a step are the dict `params` shown in the cell before its run (`default_params(step)`, read from `settings/*.csv`, so a
run that changes nothing reproduces the published results); see `notebooks/README.md`.

Environment: `conda env create -f environment.yml && conda activate hybridkin` (python 3.12, pyomo 6.10, casadi 3.7, IPOPT 3.14,
numpy / pandas / scipy / matplotlib, jupyter and nbconvert; `environment.yml` explains the optional k_aug extension). Register the
notebook kernel with `python -m ipykernel install --user --name hybridkin`.

Selection run from the command line: `python src/selection.py --out results_v4/flux_model [--from_anchor <dir with anchor_state.pkl>]
[--max_iter 30] [--n_jobs 8] [--resume]` (a relative `--out` is relative to the repo root; directories inside `results/`,
`results_v2/`, `results_v3/` are refused). Every run copies `settings/` into `<out>/settings_used` (a resume with edited
settings logs the differing files). Stage 0 warm starts from the saved anchor named in `settings/run.csv` (`stage0_warm_start`:
`Data/stage0_warm_start/anchor_state.pkl`, a copy of the stage-0 anchor of the v2 b2sd run); without it the multistart starts
from the literature values. Tests: `python tests/test_pipeline.py` (the comparisons with the original code need `legacy/`
and are skipped without it).
