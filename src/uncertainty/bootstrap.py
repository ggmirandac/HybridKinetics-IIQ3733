"""
bootstrap.py -- case bootstrap of the final parameter structure.

The conditions are resampled with replacement (the conditions are the independent experimental units), every
resample is fitted warm started at the final point (theta and the primal point of each condition) with the tight
options, and a failed fit is retried once with a monotone barrier.  The summary gives percentiles, CV, the practical
identifiability call (97.5/2.5 percentile ratio at most bootstrap_ci_ratio_max, few resamples on a theta bound, the
interval not spread over most of the theta box) and the objective against the number of KO18 / KO05 copies in the
resample: the bootstrap objective is dominated by how often the hardest conditions are drawn.

Position in the pipeline: uncertainty of the final structure (after selection.py and final_analysis.py); writes
bootstrap_theta.csv, bootstrap_resamples.csv, bootstrap_summary.csv, objective_by_n_KO18.csv, bootstrap_info.json.
"""
import os
import json
import time
import pickle

import numpy as np
import pandas as pd

from settings import SOLVER_OPTIONS, TIGHT_OPTIONS, RETRY_MONOTONE_OPTIONS, THETA_BOUNDS, default_params
from settings import snapshot_settings
from progress import log
from paths import check_out_dir
from data import variable_bounds
from estimation.anchors import check_bounds, read_anchor
from estimation.parallel import fit_many

def run_bootstrap(anchor_dir, out_dir, params=None, resume=False):
    """
    Case bootstrap of the final structure saved in ``anchor_dir/anchor_state.pkl``: params["bootstrap_n"] resamples
    (seed bootstrap_seed), fitted in chunks of bootstrap_chunk with a checkpoint (bootstrap_theta.csv) after every
    chunk; ``resume`` continues from it.  Writes into ``out_dir``; the settings of this run are copied into
    ``<parent of out_dir>/settings_used``.
    """
    params = default_params("bootstrap") if params is None else params
    started = time.time()
    out_dir = check_out_dir(out_dir)
    check_bounds()
    os.makedirs(out_dir, exist_ok=True)
    snapshot_settings(os.path.dirname(out_dir))
    state = read_anchor(anchor_dir)
    with open(os.path.join(out_dir, "anchor_state.pkl"), "wb") as fh:
        pickle.dump(state, fh)
    log("bootstrap warm start: final point of the %s, obj %.6f, %d free" % (
        state.get("point", "reduction"), state["obj"], len(state["free"])))
    resamples = draw_resamples(state["conditions"], params["bootstrap_n"], params["bootstrap_seed"])
    pd.DataFrame({"resample": range(len(resamples)), "n_unique": [len(set(r)) for r in resamples],
                  "conditions": [";".join(r) for r in resamples]}).to_csv(
        os.path.join(out_dir, "bootstrap_resamples.csv"), index=False)
    fit_resamples(resamples, state, params, out_dir, resume)
    summarize(out_dir, state, params)
    log("DONE total %.0fs" % (time.time() - started))

def draw_resamples(conditions, n, seed):
    """``n`` lists of len(conditions) conditions drawn with replacement."""
    rng = np.random.default_rng(seed)
    resamples = []
    for _ in range(n):
        drawn = rng.integers(0, len(conditions), size=len(conditions))
        resamples.append([conditions[i] for i in drawn])
    return resamples

def fit_resamples(resamples, state, params, out_dir, resume):
    """Fit the resamples chunk by chunk (failures retried once with monotone mu) into bootstrap_theta.csv."""
    free = state["free"]
    u_bounds, x_bounds, drain_bounds = variable_bounds()
    estimator_args = {"free_params": free, "x_bounds": x_bounds, "u_bounds": u_bounds, "drain_bounds": drain_bounds,
                      "x_init": state["x_init"], "u_init": state["u_init"], "b_init": state["b_init"]}
    tight_args = {**estimator_args, "solver_options": {**SOLVER_OPTIONS, **TIGHT_OPTIONS}}
    retry_args = {**estimator_args, "solver_options": {**SOLVER_OPTIONS, **RETRY_MONOTONE_OPTIONS}}
    path = os.path.join(out_dir, "bootstrap_theta.csv")
    columns = ["resample", "converged", "obj", "wall_s", "retried", "error"] + free
    done = []
    if resume and os.path.exists(path):
        done = pd.read_csv(path).to_dict("records")
    finished = {int(row["resample"]) for row in done}
    todo = [i for i in range(len(resamples)) if i not in finished]
    chunk = params["bootstrap_chunk"]
    for start in range(0, len(todo), chunk):
        ids = todo[start:start + chunk]
        chunk_started = time.time()
        results = fit_many([resamples[i] for i in ids], [state["theta"]] * len(ids), tight_args, params["n_jobs"])
        rows = {}
        for i, result in zip(ids, results):
            rows[i] = resample_row(i, result, free)
        failed = [i for i in ids if not rows[i]["converged"]]
        if failed:
            log("chunk %d: %d failures on first attempt; retrying with monotone mu" % (start // chunk, len(failed)))
            retries = fit_many([resamples[i] for i in failed], [state["theta"]] * len(failed), retry_args,
                               params["n_jobs"])
            for i, result in zip(failed, retries):
                rows[i] = retried_row(rows[i], result, free)
        for i in ids:
            done.append(rows[i])
        table = pd.DataFrame(done)[columns].sort_values("resample")
        table.to_csv(path, index=False)
        log("saved %d/%d resamples (%d converged); chunk took %.0fs" % (
            len(table), len(resamples), int(table["converged"].sum()), time.time() - chunk_started))

def resample_row(i, result, free):
    theta, obj, converged, error, wall_s = result
    row = {"resample": i, "converged": bool(converged), "obj": obj, "wall_s": wall_s, "retried": False,
           "error": error}
    for p in free:
        row[p] = theta[p] if theta else np.nan
    return row

def retried_row(row, result, free):
    theta, obj, converged, error, wall_s = result
    if not converged:
        row["retried"] = True
        row["error"] = (row["error"] + " || RETRY: " + error)[:900]
        return row
    retried = {"resample": row["resample"], "converged": True, "obj": obj, "wall_s": row["wall_s"] + wall_s,
               "retried": True, "error": ""}
    for p in free:
        retried[p] = theta[p]
    return retried

def summarize(out_dir, state, params):
    """bootstrap_summary.csv, objective_by_n_KO18.csv and bootstrap_info.json from the saved fits."""
    fits = pd.read_csv(os.path.join(out_dir, "bootstrap_theta.csv"))
    resamples = pd.read_csv(os.path.join(out_dir, "bootstrap_resamples.csv"))
    free = state["free"]
    converged = fits[fits["converged"]]
    n_failed = int((~fits["converged"]).sum())
    rows = []
    for p in free:
        rows.append(parameter_summary(p, converged[p].to_numpy(float), float(state["theta"][p]), params))
    summary = pd.DataFrame(rows).set_index("param")
    # practically identifiable: a narrow percentile interval that is neither pinned to a theta bound nor spread over
    # most of the theta box
    summary["practically_identifiable"] = (
        (summary["ci_ratio_97.5_over_2.5"] <= params["bootstrap_ci_ratio_max"])
        & (summary["frac_at_theta_bound"] < params["bootstrap_frac_at_bound_max"])
        & (summary["ci_span_over_bounds_span"] < params["bootstrap_ci_span_max"]))
    summary.to_csv(os.path.join(out_dir, "bootstrap_summary.csv"))
    regression = objective_against_hard_conditions(converged, resamples, out_dir)
    info = {"n_resamples": int(len(fits)), "n_converged": int(len(converged)), "n_failed": n_failed,
            "frac_failed": n_failed / max(len(fits), 1),
            "n_retried_ok": int((fits["retried"] & fits["converged"]).sum()),
            "obj_anchor": state["obj"], "obj_boot_median": float(converged["obj"].median()),
            "obj_boot_p2.5_p97.5": [float(x) for x in np.percentile(converged["obj"], [2.5, 97.5])]}
    info.update(regression)
    info["n_practically_identifiable"] = int(summary["practically_identifiable"].sum())
    info["n_free"] = len(free)
    with open(os.path.join(out_dir, "bootstrap_info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    log("bootstrap summary: %s" % info)
    columns = ["anchor", "median", "p2.5", "p97.5", "cv_percent", "ci_ratio_97.5_over_2.5", "frac_at_theta_bound",
               "practically_identifiable"]
    log("\n" + summary[columns].round(4).to_string())
    return summary, info

def parameter_summary(p, values, anchor, params):
    lower, upper = THETA_BOUNDS[p]
    tol = params["theta_at_bound_tol"]
    percentiles = np.percentile(values, [2.5, 50, 97.5])
    at_bound = (np.abs(np.log(values / lower)) < tol) | (np.abs(np.log(upper / values)) < tol)
    return {"param": p, "anchor": anchor, "mean": float(values.mean()), "median": percentiles[1],
            "p2.5": percentiles[0], "p97.5": percentiles[2], "sd": float(values.std(ddof=1)),
            "cv_percent": 100 * float(values.std(ddof=1) / abs(values.mean())),
            "sd_log10": float(np.std(np.log10(values), ddof=1)),
            "ci_ratio_97.5_over_2.5": float(percentiles[2] / percentiles[0]),
            "frac_at_theta_bound": float(np.mean(at_bound)),
            "anchor_inside_ci": bool(percentiles[0] <= anchor <= percentiles[2]),
            "log_bounds_span_decades": float(np.log10(upper / lower)),
            "ci_span_over_bounds_span": float(np.log10(percentiles[2] / percentiles[0]) / np.log10(upper / lower))}

def copies_of(condition_list, condition):
    return condition_list.split(";").count(condition)

def objective_against_hard_conditions(converged, resamples, out_dir):
    """Least squares obj ~ c0 + c1 n_KO18 + c2 n_KO05 over the converged resamples; objective_by_n_KO18.csv."""
    joined = converged.merge(resamples, on="resample")
    joined["n_KO18"] = [copies_of(names, "KO18") for names in joined["conditions"]]
    joined["n_KO05"] = [copies_of(names, "KO05") for names in joined["conditions"]]
    X = np.column_stack([np.ones(len(joined)), joined["n_KO18"], joined["n_KO05"]])
    coefficients, *_ = np.linalg.lstsq(X, joined["obj"].to_numpy(float), rcond=None)
    fitted = X @ coefficients
    r2 = 1 - np.sum((joined["obj"] - fitted) ** 2) / np.sum((joined["obj"] - joined["obj"].mean()) ** 2)
    by_copies = joined.groupby("n_KO18")["obj"].agg(["count", "mean", "std"])
    by_copies.to_csv(os.path.join(out_dir, "objective_by_n_KO18.csv"))
    return {"obj_regression_intercept": float(coefficients[0]), "obj_per_KO18_copy": float(coefficients[1]),
            "obj_per_KO05_copy": float(coefficients[2]), "obj_regression_R2": float(r2)}

def summarize_saved(anchor_dir, out_dir, params=None):
    """Summary of an existing bootstrap in ``out_dir`` (bootstrap_theta.csv, bootstrap_resamples.csv)."""
    params = default_params("bootstrap") if params is None else params
    return summarize(out_dir, read_anchor(anchor_dir), params)
