# notebooks/

Every step of the method is a notebook that calls the library in `src/` and explains what it does. The notebooks add
`../src` to `sys.path` (no package install) and use the `hybridkin` kernel.

## Run order

| # | notebook | what it does | calls | writes to `OUT` |
|---|---|---|---|---|
| 1 | `model.ipynb` | the model in one place: parameters, groups, rate laws, S, Keq, bounds, settings | `model/`, `data`, `settings` | nothing |
| 2 | `data.ipynb` | rebuilds the `Data/` tables from the workbook and explores them | `data` | `data_rebuilt/` (RUN) |
| 3 | `parameter_selection.ipynb` | stage 0 (37 free), structural step, practical loop; fixing path and selected structure | `selection.run_selection` | `stage0/`, `reduction/`, `settings_used/` |
| 4 | `structural_identifiability.ipynb` | Stigter-Molenaar spectra at stage 0 and at the final fit | `identifiability.structural` | nothing |
| 5 | `final_fit.ipynb` | final structure, Fisher information, fit metrics, `REPORT.md` | `final_analysis.run_final_analysis`, `report.build_report` | `final/`, `REPORT.md` |
| 6 | `bootstrap.ipynb` | case bootstrap of the final structure | `uncertainty.bootstrap.run_bootstrap` | `bootstrap/` |
| 7 | `profiles.ipynb` | profile likelihood of every free parameter | `uncertainty.profiles.run_profiles` | `profiles/` |
| 8 | `loocv.ipynb` | leave-one-condition-out cross-validation | `uncertainty.loocv.run_loocv_analysis` | `loocv/` |
| 9 | `figure_data.ipynb` | sensitivities, literature model, fit after the structural phase, metrics per stage | `sensitivity`, `estimation.refit`, `metrics` | `paper_figs_data/` |
| 10 | `figures_main.ipynb` | Fig. 2 of the paper | | `figures/paper/mainfigure/` |
| 11 | `figures_si.ipynb` | supplementary figures | | `figures/paper/supplementary/` |
| 12 | `../paper/build_paper.py` (script) | copies the paper figures into `paper/` and writes the generated tables and numbers of `paper/supplementary/` | | `paper/` |

Notes on the order:

* 3 -> 5 -> (6, 7, 8 in any order) -> 9 -> (10, 11) -> 12.
* `REPORT.md` is written by the last cell of `final_fit.ipynb` from whatever exists: **re-run it after 6, 7 and 8**.
  Likewise re-run `figure_data.ipynb` (no `HK_RUN`) after 8 to refresh `stage_metrics.csv`.
* `data`, `model`, `structural_identifiability` and the figure notebooks are cheap.

## Settings and switches

The settings of the method are a plain dict `params`, shown in a cell before the run cell of every notebook:

    # Settings of the method (defaults from settings/method.csv)
    params = settings.default_params()
    params["n_jobs"] = int(os.environ.get("HK_N_JOBS", params["n_jobs"]))
    params

Change an entry in the notebook for one run, or `settings/method.csv` for good. `model.ipynb` lists every settings file.

The environment variables only switch what runs and where:

| variable | default | meaning |
|---|---|---|
| `HK_OUT` | `out_dir` of `settings/method.csv` (`results/flux_model`) | output directory (relative to the repo root) |
| `HK_RUN` | `0` | `0`: load the saved results of `OUT`; `1`: run the computation of the notebook and save into `OUT` |
| `HK_N_JOBS` | `n_jobs` of `settings/method.csv` (8) | worker processes |
| `HK_FROM_ANCHOR` | none | `parameter_selection`: directory with `anchor_state.pkl`, skips stage 0 |
| `HK_RESUME` | `0` | `bootstrap`: continue an interrupted run |

Execute a notebook in place, for example:

    cd notebooks
    HK_RUN=1 jupyter nbconvert --to notebook --execute --inplace parameter_selection.ipynb

The notebooks are committed WITH their outputs, so a reader sees the results without running anything; re-execute a
notebook after changing the code or the results it reads. Plot code follows one rule: every figure has its own full cell
(no plotting helpers, no loops over figures or axes); `paper_style.py` only holds rcParams, colours and parameter labels.

## Other notebooks

* `gen_K_eq.ipynb` (in this folder) shows the equilibrium constants of `Data/k_eq_values.pkl`; with `HK_RUN=1`, in an
  environment that has `equilibrator_api`, it recomputes them and overwrites that file.
* `data_analysis.ipynb` (repo root) writes into `Data/`; `data.ipynb` only checks those tables (including the glucose row, taken by
  label from the sheet `Extracelular`).
