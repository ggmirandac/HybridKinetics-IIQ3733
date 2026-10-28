# notebooks/

Every step of the method is a notebook that calls the library in `src/` and explains what it does. The notebooks add
`../src` to `sys.path` (no package install) and use the `hybridkin` kernel.

## Run order

| # | notebook | what it does | calls | writes to `OUT` |
|---|---|---|---|---|
| 1 | `model.ipynb` | the model in one place: parameters, groups, rate laws, S, Keq, bounds, settings | `model/`, `data`, `settings` | nothing |
| 2 | `data.ipynb` | rebuilds the `Data/` tables from the workbook and explores them | `data` | `data_rebuilt/` (RUN) |
| 3 | `parameter_selection.ipynb` | stage 0 (37 free) + the reduction loop; shows iterations, free sets, final structure | `selection.run_stage0`, `selection.run_reduction` | `stage0/`, `reduction/`, `settings_used/`, `fixing_log_with_classes.csv` |
| 4 | `structural_identifiability.ipynb` | Stigter-Molenaar / output-level analysis, spectra, null groups | `identifiability.structural`, saved checks | `structural/` (RUN) |
| 5 | `final_fit.ipynb` | final re-estimate, classes, covariance, metrics, `REPORT.md` | `final_analysis.run_final_analysis`, `report.build_report` | `final/`, `REPORT.md` |
| 6 | `bootstrap.ipynb` | case bootstrap of the final structure | `uncertainty.bootstrap.run_bootstrap` | `bootstrap/` |
| 7 | `profiles.ipynb` | profile likelihood of every free parameter | `uncertainty.profiles.run_profiles` | `profiles/` |
| 8 | `loocv.ipynb` | leak-free leave-one-condition-out CV | `uncertainty.loocv.run_loocv_analysis` | `loocv/` |
| 9 | `isolation.ipynb` | which bound change explains the drop in objective (optional; uses the v2 anchor of `Data/stage0_warm_start`) | `estimation`, `data` | `isolation/` |
| 10 | `figure_data.ipynb` | sensitivities, literature model, metrics per stage for the figures | `estimation.anchors.load_point`, `metrics` | `paper_figs_data/` |
| 11 | `figures_main.ipynb` | fig1, fig2, composite, landscape, pipeline diagram, fixing path | | `figures/`, `figures/paper/` |
| 12 | `figures_si.ipynb` | SI parity, parameter evolution, bootstrap, profiles | | `figures/paper/` |

Notes on the order:

* 3 -> 5 -> (6, 7, 8 in any order) -> 10 -> (11, 12). `figure_data` needs the LOOCV only for `stage_metrics.csv`, so run it
  after 8 (its sensitivities and the literature model, behind `HK_RUN=1`, only need 5).
* `REPORT.md` is assembled at the end of `final_fit.ipynb` from whatever exists; **re-run its last section (no `HK_RUN`)
  after 6, 7 and 8** to complete it. Likewise re-run `figure_data.ipynb` (no `HK_RUN`) after 8 to refresh `stage_metrics.csv`.
* `data`, `model`, `structural_identifiability` (without `HK_RUN`) and all figure notebooks are cheap; the figure
  notebooks stop with the list of missing files if a step has not run.

## Settings and switches

The settings of every step are a plain dict `params`, shown in the cell right before the run cell of the notebook:

    # Settings of the parameter selection (defaults from settings/thresholds.csv)
    params = default_params("selection")
    params["corr_threshold"] = 0.9   # |r| above this counts as correlated
    params["max_iter"] = 30
    params

`default_params(step)` (src/settings.py) reads the CSV files of `settings/`, so a run that changes nothing reproduces the
published results; `model.ipynb` lists the settings of every step (stage0, selection, structural, practical, sensitivity,
final, bootstrap, profiles, loocv, isolation). Change an entry in the notebook for one run, or the CSV file for good.

The environment variables only switch what runs and where:

| variable | default | meaning |
|---|---|---|
| `HK_OUT` | `out_dir` of `settings/run.csv` (`results_v4/flux_model`) | output directory (a relative path is relative to the repo root); with `HK_RUN=1` it must not be inside `results/`, `results_v2/`, `results_v3/` (the first cell refuses it) |
| `HK_RUN` | `0` | `0`: load the saved results of `OUT`; `1`: run the long computation of the notebook and save into `OUT` |
| `HK_N_JOBS` | `settings/run.csv` (8) | worker processes (copied into `params["n_jobs"]`) |
| `HK_FROM_ANCHOR` | none | `parameter_selection`: directory with `anchor_state.pkl`, skips stage 0 |
| `HK_RESUME` | `0` | `parameter_selection` / `bootstrap`: continue an interrupted run |

`RUN` only switches the long computations. Tables, figures and `REPORT.md` are rewritten into `OUT` every time unless
`OUT` is one of the frozen result folders (then they are only shown).

Execute a notebook in place, for example:

    cd notebooks
    HK_RUN=1 jupyter nbconvert --to notebook --execute --inplace parameter_selection.ipynb

The notebooks are committed WITH their outputs (tables, printed numbers and figures from the last execution), so a reader sees
the results without running anything; re-execute a notebook (as above) after changing the code or the results it reads, so that
the stored outputs stay in step with `OUT`. Plot code follows one rule: every figure has its own full cell (no plotting
helpers, no loops over figures or axes); `paper_style.py` only holds rcParams, colours and parameter labels.

## Not ported

* `gen_K_eq.ipynb` (root) stays where it is: it needs `equilibrator_api` and writes `Data/k_eq_values.pkl`.
* `data_analysis.ipynb` (root) and `fix_glc_row.py` stay: they write into `Data/`; `data.ipynb` only checks them.
