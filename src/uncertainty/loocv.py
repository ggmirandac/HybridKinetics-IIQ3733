"""
loocv.py -- leak-free leave-one-condition-out cross-validation of the final structure.

For every condition h (22 folds, RF03 included) the model is refitted on the other conditions, warm started at the
final fit restricted to them (theta, x, u, b).  The metabolite, cofactor and drain bounds are recomputed from the
training conditions only, so nothing of h leaks into the fold.  The held-out condition is predicted with the pts_fraction closure (alpha = training mean of
b / v_pts; u_ref, x_ref = geometric means of the fitted training state); the baseline is the training mean
(metabolites: mean of the measured values; fluxes: S^-1 b_mean).

Position in the pipeline: out-of-sample check of the final structure (after selection.py and final_analysis.py);
writes fold_table.csv, loocv_pred.csv, loocv_real.csv, loocv_baseline.csv, loocv_summary.json, ...
"""
import os
import json
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from settings import WARM_START_OPTIONS, default_params, snapshot_settings
from progress import log
from paths import resolve_out
from model.parameters import LI_FLUX_KEYS, BALANCED_KEYS, FLUX_KEYS
from model.stoichiometry import build_stoichiometric_matrix
from data import load_condition, variable_bounds, check_bounds_contain_data
from estimation.anchors import load_point, check_bounds, read_anchor

def loocv_baseline(train):
    """
    Training-mean baseline of a fold, a Series over the 18 outputs: metabolites, mean of the measured training
    values; fluxes, S^-1 b_mean with b_mean the training mean of the data drains.
    """
    S = build_stoichiometric_matrix()
    data = {c: load_condition(c) for c in train}
    row = {}
    for key in BALANCED_KEYS:
        values = [data[c]["x_real"][key] for c in train if key in data[c]["x_real"]]
        row[key] = float(np.mean(values)) if values else float("nan")
    mean_drains = np.array([np.mean([data[c]["b"].get(key, 0.0) for c in train]) for key in BALANCED_KEYS])
    fluxes = np.linalg.solve(S, mean_drains)
    for key, value in zip(FLUX_KEYS, fluxes):
        row[key] = float(value)
    return pd.Series(row)

def restrict(by_condition, train):
    return {c: by_condition[c] for c in train if c in by_condition}

def fit_fold(full_est, train):
    """The final fit restricted to ``train`` with the training-only bounds, warm started and refined."""
    u_bounds, x_bounds, drain_bounds = variable_bounds(train)
    check_bounds_contain_data(u_bounds, x_bounds, drain_bounds, conditions=train)
    x_all, u_all = full_est.primal_start()
    est = full_est.clone(theta_init=full_est.full_theta(), x_init=restrict(x_all, train),
                         u_init=restrict(u_all, train), b_init=restrict(full_est.primal_b_free(), train),
                         solver_options=WARM_START_OPTIONS, conditions=train, x_bounds=x_bounds, u_bounds=u_bounds,
                         drain_bounds=drain_bounds)
    est.estimate()
    est.refine_anchor()
    return est

def loocv_fold(full_est, held_out):
    """
    One fold of the LOOCV of the final fit ``full_est`` (``held_out`` is one of its conditions).  Returns a dict: h,
    error, obj_fold, theta, pred / real / b_implied (one-row DataFrames), baseline (Series), diag (closure solve
    status), wall_s.
    """
    started = time.time()
    train = [c for c in full_est.conditions if c != held_out]
    fold = {"h": held_out, "error": "", "obj_fold": float("nan")}
    try:
        est = fit_fold(full_est, train)
    except Exception as exc:   # noqa: BLE001  (IPOPT not converging is recorded in the fold table)
        fold["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:150])
        fold["wall_s"] = time.time() - started
        return fold
    fold["obj_fold"] = float(est.obj_value)
    fold["theta"] = est.theta.copy()
    prediction = est.predict_held_out(held_out, est.closure_inputs())
    fold["pred"] = prediction.predicted.loc[[held_out]]
    fold["real"] = prediction.real.loc[[held_out]]
    fold["diag"] = prediction.diagnostics.loc[held_out].to_dict()
    fold["b_implied"] = prediction.b.loc[[held_out]]
    fold["baseline"] = loocv_baseline(train)
    fold["wall_s"] = time.time() - started
    return fold

# The final fit, inherited by the forked workers (an estimator cannot be pickled)
FULL_FIT = None

def fold_task(held_out):
    return loocv_fold(FULL_FIT, held_out)

def run_loocv(full_est, params):
    """
    ``loocv_fold`` for every condition of ``full_est`` on params["n_jobs"] forked workers (the anchored estimator is
    inherited by the children, not pickled).  Returns the fold dicts in condition order.
    """
    global FULL_FIT
    FULL_FIT = full_est
    held_out = list(full_est.conditions)
    if params["n_jobs"] == 1:
        results = [fold_task(h) for h in held_out]
    else:
        with ProcessPoolExecutor(max_workers=int(params["n_jobs"]),
                                 mp_context=multiprocessing.get_context("fork")) as pool:
            results = list(pool.map(fold_task, held_out))
    for fold in results:
        log("fold %s: obj %.5f %s" % (fold["h"], fold["obj_fold"], fold["error"]))
    return results

def fold_table(results):
    rows = []
    for fold in results:
        diagnostics = fold.get("diag", {})
        rows.append({"h": fold["h"], "obj_fold": fold["obj_fold"], "wall_s": fold.get("wall_s"),
                     "closure_success": diagnostics.get("success"),
                     "closure_status": diagnostics.get("return_status"),
                     "closure_max_residual": diagnostics.get("max_residual"), "error": fold["error"]})
    return pd.DataFrame(rows)

def prediction_scores(real, predicted, baseline, rows):
    """RMSE of the model and of the baseline and R2 against the baseline over the cells measured in ``rows``."""
    measured = ~np.isnan(real) & ~np.isnan(baseline)
    y = real[measured]
    y_model = predicted[measured]
    y_baseline = baseline[measured]
    return {"rmse_model": float(np.sqrt(np.mean((y - y_model) ** 2))),
            "rmse_baseline": float(np.sqrt(np.mean((y - y_baseline) ** 2))),
            "r2_vs_baseline": float(1 - np.sum((y - y_model) ** 2) / np.sum((y - y_baseline) ** 2)),
            "n": int(measured.sum()), "n_folds": len(rows)}

def run_loocv_analysis(anchor_dir, out_dir, params=None):
    """
    LOOCV of the final structure saved in ``anchor_dir/anchor_state.pkl``: runs all folds and writes the fold
    table, predictions, baselines, per-output errors and the summary into ``out_dir`` (the settings of this run go
    to ``<parent of out_dir>/settings_used``).  Returns the fold dicts.
    """
    params = default_params() if params is None else params
    out_dir = resolve_out(out_dir)
    check_bounds()
    os.makedirs(out_dir, exist_ok=True)
    snapshot_settings(os.path.dirname(out_dir))
    results = run_loocv(load_point(read_anchor(anchor_dir)), params)
    folds = fold_table(results)
    folds.to_csv(os.path.join(out_dir, "fold_table.csv"), index=False)
    predicted_folds = [fold for fold in results if "pred" in fold]
    P = pd.concat([fold["pred"] for fold in predicted_folds])
    R = pd.concat([fold["real"] for fold in predicted_folds])
    B = pd.DataFrame({fold["h"]: fold["baseline"] for fold in predicted_folds}).T.reindex(P.index)
    P.to_csv(os.path.join(out_dir, "loocv_pred.csv"))
    R.to_csv(os.path.join(out_dir, "loocv_real.csv"))
    B.to_csv(os.path.join(out_dir, "loocv_baseline.csv"))
    pd.concat([fold["b_implied"] for fold in predicted_folds]).to_csv(os.path.join(out_dir, "loocv_b_implied.csv"))
    pd.DataFrame({fold["h"]: fold["theta"] for fold in predicted_folds}).T.to_csv(
        os.path.join(out_dir, "fold_theta.csv"))
    closure_success = [fold["h"] for fold in predicted_folds if fold.get("diag", {}).get("success")]
    met_columns = [c for c in P.columns if c.startswith("C_") and R[c].notna().any()]
    flux_columns = [c for c in LI_FLUX_KEYS if R[c].notna().any()]
    summary = {}
    for name, columns in (("met", met_columns), ("flux_li", flux_columns)):
        for tag, rows in (("all_folds", list(P.index)), ("closure_success_only", closure_success)):
            summary["%s_%s" % (name, tag)] = prediction_scores(R.loc[rows, columns].to_numpy(float),
                                                               P.loc[rows, columns].to_numpy(float),
                                                               B.loc[rows, columns].to_numpy(float), rows)
    per_output = {}
    for column in P.columns:
        if R[column].notna().any():
            measured = R[column].notna() & B[column].notna()
            per_output[column] = {
                "rmse_model": float(np.sqrt(np.mean((R[column][measured] - P[column][measured]) ** 2))),
                "rmse_baseline": float(np.sqrt(np.mean((R[column][measured] - B[column][measured]) ** 2))),
                "n": int(measured.sum())}
    pd.DataFrame(per_output).T.to_csv(os.path.join(out_dir, "loocv_per_output.csv"))
    summary["n_folds_failed"] = int(folds["error"].astype(bool).sum())
    summary["n_closure_failed"] = int(len(predicted_folds) - len(closure_success))
    with open(os.path.join(out_dir, "loocv_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    scores = {key: value for key, value in summary.items() if isinstance(value, dict) and "rmse_model" in value}
    pd.DataFrame(scores).T.to_csv(os.path.join(out_dir, "loocv_summary.csv"))
    log("LOOCV summary: %s" % summary)
    log("\n" + folds[["h", "obj_fold", "closure_success", "error"]].round(4).to_string())
    return results
