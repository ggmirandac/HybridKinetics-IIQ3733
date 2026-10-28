"""
loocv.py -- leak-free leave-one-condition-out cross-validation of the final structure.

For every condition h (22 folds, RF03 included) the training estimator is the full-data anchor restricted to the
other conditions, warm started at the anchor primal point restricted to them (theta, x, u, b), with the tight
options, then ``refine_anchor``.  The metabolite, cofactor and drain bounds are recomputed from the training
conditions only and must contain the training data, so nothing of h leaks into the fold.  The anchor theta with
re-optimized states is a feasible point of the fold problem, so the fold objective must not exceed the BOUND
(TOTAL - total of h) / n_train of the objective breakdown at the anchor: fold_status "anchor_ok" if
obj_fold <= bound (1 + loocv_bound_rtol), else a multistart from the restricted anchor ("multistart_ok" /
"above_bound").  The held-out condition is predicted with the pts_fraction closure (alpha = training mean of
b / v_pts; u_ref, x_ref = geometric means of the fitted training state); the baseline is the training mean
(metabolites: mean of the measured values; fluxes: S^-1 b_mean).

Position in the pipeline: out-of-sample check of the final structure (after selection.py and final_analysis.py);
writes fold_status.csv, loocv_pred.csv, loocv_real.csv, loocv_baseline.csv, loocv_summary.json, ...
"""
import os
import json
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from settings import TIGHT_OPTIONS, default_params, snapshot_settings
from progress import log
from paths import check_out_dir
from model.parameters import LI_FLUX_KEYS, BALANCED_KEYS, FLUX_KEYS
from model.stoichiometry import build_stoichiometric_matrix
from data import load_condition, variable_bounds, check_bounds_contain_data
from estimation.anchors import load_point, check_bounds, read_anchor
from estimation.multistart import multistart_recorded

# Seed of the multistart of a fold that does not reach its bound
FOLD_SEED = 0

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

def anchor_values_outside(train, x_all, u_all, b_all, x_bounds, u_bounds, drain_bounds):
    """'condition/key' of the restricted anchor primal values outside the training-only bounds (1e-9 slack)."""
    outside = []
    for condition in train:
        for key, value in x_all[condition].items():
            if not (x_bounds[key][0] - 1e-9 <= value <= x_bounds[key][1] + 1e-9):
                outside.append("%s/%s" % (condition, key))
        for key, value in u_all[condition].items():
            if not (u_bounds[key][0] - 1e-9 <= value <= u_bounds[key][1] + 1e-9):
                outside.append("%s/%s" % (condition, key))
        for key, value in b_all.get(condition, {}).items():
            if not (drain_bounds[key][0] - 1e-9 <= value <= drain_bounds[key][1] + 1e-9):
                outside.append("%s/b_%s" % (condition, key))
    return outside

def restricted_anchor_fit(full_est, train, bounds, fold):
    """The full-data anchor restricted to ``train``: tight warm start, refine.  None (error recorded) if it fails."""
    u_bounds, x_bounds, drain_bounds = bounds
    theta_anchor = {p: float(v) for p, v in full_est.full_theta().items()}
    x_all, u_all = full_est.primal_start()
    b_all = full_est.primal_b_free()
    base_options = dict(full_est.solver_options)
    try:
        est = full_est.clone(theta_init=theta_anchor, x_init=restrict(x_all, train), u_init=restrict(u_all, train),
                             b_init=restrict(b_all, train), solver_options={**base_options, **TIGHT_OPTIONS},
                             conditions=train, x_bounds=x_bounds, u_bounds=u_bounds, drain_bounds=drain_bounds)
        est.estimate()
        est.refine_anchor(primary_options=dict(TIGHT_OPTIONS))
    except Exception as exc:   # noqa: BLE001  (IPOPT not converging: the fold tries the multistart)
        fold["error"] = "restricted anchor fit: %s: %s" % (type(exc).__name__, str(exc)[:150])
        return None
    est.solver_options = dict(base_options)
    fold["obj_restricted_anchor_fit"] = float(est.obj_value)
    return est

def restrict(by_condition, train):
    return {c: by_condition[c] for c in train if c in by_condition}

def fold_multistart(full_est, train, bounds, best, n_starts, fold):
    """Multistart from the restricted anchor (its theta and random log-uniform ones); the better of it and ``best``."""
    u_bounds, x_bounds, drain_bounds = bounds
    theta_anchor = {p: float(v) for p, v in full_est.full_theta().items()}
    base_options = dict(full_est.solver_options)
    try:
        work = full_est.clone(theta_init=theta_anchor, conditions=train, x_bounds=x_bounds, u_bounds=u_bounds,
                              drain_bounds=drain_bounds)
        table = multistart_recorded(work, n_starts, FOLD_SEED, 1)
        converged = table[table["converged"]]
        fold["n_ms_converged"] = int(len(converged))
        if len(converged):
            theta_init = dict(theta_anchor)
            for p in full_est.free_params:
                theta_init[p] = float(converged.iloc[0][p])
            candidate = full_est.clone(theta_init=theta_init, conditions=train, x_bounds=x_bounds,
                                       u_bounds=u_bounds, drain_bounds=drain_bounds)
            candidate.estimate()
            candidate.refine_anchor(primary_options=dict(TIGHT_OPTIONS))
            candidate.solver_options = dict(base_options)
            if best is None or candidate.obj_value < best.obj_value:
                best = candidate
    except Exception as exc:   # noqa: BLE001  (IPOPT not converging is recorded in the fold table)
        message = "multistart: %s: %s" % (type(exc).__name__, str(exc)[:150])
        fold["error"] = (fold["error"] + " | " if fold["error"] else "") + message
    return best

def loocv_fold(full_est, held_out, breakdown, params):
    """
    One fold of the LOOCV of the anchored estimator ``full_est`` (fitted on all its conditions; ``held_out`` is one
    of them).  ``breakdown`` is its objective breakdown (the bound of the fold).  Errors are recorded in ``error`` /
    fold_status 'error', never swallowed.  Returns a dict: h, fold_status, error, obj_fold, bound,
    obj_restricted_anchor_fit, n_ms_converged, anchor_inside_training_bounds, n_anchor_values_outside_bounds,
    anchor_values_outside_bounds, theta, pred / real / b_implied (one-row DataFrames), baseline (Series), diag
    (closure solve status), closure_alpha_vpts, wall_s.
    """
    started = time.time()
    conditions = list(full_est.conditions)
    train = [c for c in conditions if c != held_out]
    fold = {"h": held_out, "fold_status": "error", "error": "", "obj_fold": float("nan"), "bound": float("nan"),
            "obj_restricted_anchor_fit": float("nan"), "n_ms_converged": 0}
    bounds = variable_bounds(train)
    u_bounds, x_bounds, drain_bounds = bounds
    check_bounds_contain_data(u_bounds, x_bounds, drain_bounds, conditions=train)
    totals = breakdown.loc[conditions]
    if totals["total"].isna().any():
        raise ValueError("objective_breakdown failed for %s: no valid LOOCV bound"
                         % list(totals.index[totals["total"].isna()]))
    fold["bound"] = float((totals["total"].sum() - float(totals.loc[held_out, "total"])) / len(train))
    x_all, u_all = full_est.primal_start()
    outside = anchor_values_outside(train, x_all, u_all, full_est.primal_b_free(), x_bounds, u_bounds, drain_bounds)
    fold["anchor_inside_training_bounds"] = len(outside) == 0
    fold["n_anchor_values_outside_bounds"] = len(outside)
    fold["anchor_values_outside_bounds"] = ";".join(outside[:20])
    limit = fold["bound"] * (1.0 + params["loocv_bound_rtol"])
    best = restricted_anchor_fit(full_est, train, bounds, fold)
    if best is not None and best.obj_value <= limit:
        fold["fold_status"] = "anchor_ok"
    else:
        best = fold_multistart(full_est, train, bounds, best, params["loocv_n_starts"], fold)
        if best is not None:
            fold["fold_status"] = "multistart_ok" if best.obj_value <= limit else "above_bound"
    if best is None:
        fold["wall_s"] = time.time() - started
        return fold
    fold["obj_fold"] = float(best.obj_value)
    fold["theta"] = best.theta.copy()
    closure = best.closure_inputs()
    prediction = best.predict_held_out(held_out, closure)
    fold["pred"] = prediction.predicted.loc[[held_out]]
    fold["real"] = prediction.real.loc[[held_out]]
    fold["diag"] = prediction.diagnostics.loc[held_out].to_dict()
    fold["b_implied"] = prediction.b.loc[[held_out]]
    fold["closure_alpha_vpts"] = {key: float(value) for key, value in closure["alpha_by_condition"].items()}
    fold["baseline"] = loocv_baseline(train)
    fold["wall_s"] = time.time() - started
    return fold

# The anchored estimator, its breakdown and the settings, inherited by the forked workers (an estimator cannot be
# pickled)
FOLD_INPUTS = None

def fold_task(held_out):
    full_est, breakdown, params = FOLD_INPUTS
    return loocv_fold(full_est, held_out, breakdown, params)

def run_loocv(full_est, params):
    """
    ``loocv_fold`` for every condition of ``full_est`` on params["n_jobs"] forked workers (the anchored estimator is
    inherited by the children, not pickled).  The objective breakdown at the anchor is computed once here.
    Returns the fold dicts in condition order.
    """
    global FOLD_INPUTS
    FOLD_INPUTS = (full_est, full_est.objective_breakdown(), params)
    held_out = list(full_est.conditions)
    if params["n_jobs"] == 1:
        results = [fold_task(h) for h in held_out]
    else:
        with ProcessPoolExecutor(max_workers=int(params["n_jobs"]),
                                 mp_context=multiprocessing.get_context("fork")) as pool:
            results = list(pool.map(fold_task, held_out))
    for fold in results:
        log("fold %s: %s obj %.5f bound %.5f" % (fold["h"], fold["fold_status"], fold["obj_fold"], fold["bound"]))
    return results

def fold_table(results):
    rows = []
    for fold in results:
        diagnostics = fold.get("diag", {})
        rows.append({"h": fold["h"], "fold_status": fold["fold_status"], "obj_fold": fold["obj_fold"],
                     "bound": fold["bound"], "obj_restricted_anchor_fit": fold["obj_restricted_anchor_fit"],
                     "n_ms_converged": fold["n_ms_converged"], "wall_s": fold.get("wall_s"),
                     "anchor_inside_training_bounds": fold.get("anchor_inside_training_bounds"),
                     "n_anchor_values_outside_bounds": fold.get("n_anchor_values_outside_bounds"),
                     "anchor_values_outside_bounds": fold.get("anchor_values_outside_bounds"),
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
    params = default_params("loocv") if params is None else params
    out_dir = check_out_dir(out_dir)
    check_bounds()
    os.makedirs(out_dir, exist_ok=True)
    snapshot_settings(os.path.dirname(out_dir))
    results = run_loocv(load_point(read_anchor(anchor_dir)), params)
    folds = fold_table(results)
    folds.to_csv(os.path.join(out_dir, "fold_status.csv"), index=False)
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
    summary["fold_status_counts"] = folds["fold_status"].value_counts().to_dict()
    summary["n_anchor_outside_training_bounds"] = int((~folds["anchor_inside_training_bounds"].astype(bool)).sum())
    with open(os.path.join(out_dir, "loocv_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    scores = {key: value for key, value in summary.items() if isinstance(value, dict) and "rmse_model" in value}
    pd.DataFrame(scores).T.to_csv(os.path.join(out_dir, "loocv_summary.csv"))
    log("LOOCV summary: %s" % summary)
    log("\n" + folds[["h", "fold_status", "obj_fold", "bound", "anchor_inside_training_bounds",
                      "n_anchor_values_outside_bounds", "closure_success"]].round(4).to_string())
    return results
