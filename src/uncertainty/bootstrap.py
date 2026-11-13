"""
bootstrap.py -- case bootstrap of the final parameter structure.

The conditions are resampled with replacement (the conditions are the independent experimental units) and every
resample is fitted warm started at the final fit (theta and the primal point of each condition).  With
bootstrap_local_starts > 0 every resample is also fitted from that many local starts (every free parameter at the
estimate times 10**U(-bootstrap_local_span, bootstrap_local_span), clipped to its bounds, from the primal point of the
final fit) and from bootstrap_profile_seeds profile points drawn at random among those within the likelihood-ratio
threshold (their full theta), and the best converged fit of the resample is kept: a local optimizer started only at
the estimate stays near it where the objective is flat, which makes the intervals too narrow.  Every fit is saved, and
the near-optimal set of a resample (its fits within the likelihood-ratio threshold of its best) gives a second
interval, comparable with the profile likelihood interval.  A free parameter
is practically identifiable if the ratio of the 97.5 and 2.5 percentiles of its estimates is at most
bootstrap_ci_ratio, the interval does not reach a parameter bound (an interval cut by the bound says nothing about the
parameter) and the selection did not keep it free by design.  The objective is also regressed on the number
of copies of the two worst-fitted conditions (KO18, KO05) in each resample.

Position in the pipeline: uncertainty of the final structure (after selection.py and final_analysis.py); writes
bootstrap_theta.csv, bootstrap_fits_all.csv, bootstrap_resamples.csv, bootstrap_summary.csv, objective_by_n_KO18.csv, bootstrap_info.json.
"""
import os
import json
import time

import numpy as np
import pandas as pd

from settings import SOLVER_OPTIONS, WARM_START_OPTIONS, THETA_BOUNDS, default_params, snapshot_settings
from progress import log
from paths import resolve_out
from data import variable_bounds
from estimation.anchors import check_bounds, read_anchor
from estimation.parallel import fit_many
from metrics import lr_threshold

# An interval end within this log distance (1%) of a parameter bound reaches the bound
BOUND_LOG_DISTANCE = 0.01

# Resamples fitted between two saves of bootstrap_theta.csv (an interrupted run continues with ``resume``)
CHUNK = 100

def run_bootstrap(anchor_dir, out_dir, params=None, resume=False, profile_dir=None):
    """
    Case bootstrap of the final structure saved in ``anchor_dir`` (anchor_state.pkl, final_summary.json):
    params["bootstrap_n"] resamples (seed params["seed"]), saved every CHUNK resamples; ``resume`` continues from the
    saved fits.  The profile seeds come from ``profile_dir`` (default: the "profiles" folder next to ``out_dir``).
    Writes into ``out_dir``.
    """
    params = default_params() if params is None else params
    started = time.time()
    out_dir = resolve_out(out_dir)
    check_bounds()
    os.makedirs(out_dir, exist_ok=True)
    snapshot_settings(os.path.dirname(out_dir))
    state = read_anchor(anchor_dir)
    resamples = draw_resamples(state["conditions"], params["bootstrap_n"], params["seed"])
    pd.DataFrame({"resample": range(len(resamples)), "conditions": [";".join(r) for r in resamples]}).to_csv(
        os.path.join(out_dir, "bootstrap_resamples.csv"), index=False)
    if profile_dir is None:
        profile_dir = os.path.join(os.path.dirname(out_dir), "profiles")
    seeds = profile_seed_pool(profile_dir, state, params)
    fit_resamples(resamples, state, params, out_dir, resume, seeds)
    summarize(anchor_dir, out_dir, params)
    log("bootstrap done in %.0fs" % (time.time() - started))

def draw_resamples(conditions, n, seed):
    """``n`` lists of len(conditions) conditions drawn with replacement."""
    rng = np.random.default_rng(seed)
    resamples = []
    for _ in range(n):
        drawn = rng.integers(0, len(conditions), size=len(conditions))
        resamples.append([conditions[i] for i in drawn])
    return resamples

def local_starts(theta, free, n_starts, span, rng):
    """``n_starts`` copies of ``theta`` with every free parameter times 10**U(-span, span), clipped to its bounds."""
    starts = []
    for _ in range(n_starts):
        start = dict(theta)
        for p in free:
            lower, upper = THETA_BOUNDS[p]
            start[p] = float(np.clip(theta[p] * 10.0 ** rng.uniform(-span, span), lower, upper))
        starts.append(start)
    return starts

def profile_seed_pool(profile_dir, state, params):
    """
    The profile points within the likelihood-ratio threshold of the lowest objective (full theta of the free
    parameters, columns theta_<p> of profiles_all.csv), as a list of (label, theta dict).  Empty when
    params["bootstrap_profile_seeds"] is 0.
    """
    if int(params["bootstrap_profile_seeds"]) == 0:
        return []
    points = pd.read_csv(os.path.join(profile_dir, "profiles_all.csv"))
    points = points[points["status"] == "ok"]
    threshold = lr_threshold(params["lr_alpha"], len(state["conditions"]))
    obj_hat = min(float(state["obj"]), float(points["obj"].min()))
    near = points[points["obj"] <= obj_hat + threshold]
    pool = []
    for _, point in near.iterrows():
        theta = dict(state["theta"])
        for p in state["free"]:
            theta[p] = float(point["theta_" + p])
        pool.append(("profile %s %+d %d" % (point["param"], int(point["arm"]), int(point["k"])), theta))
    log("%d profile points within the threshold are bootstrap seeds" % len(pool))
    return pool

def starts_of_resample(i, state, params, seeds):
    """
    The starts of resample ``i`` as (label, theta, cold): the estimate (warm), params["bootstrap_profile_seeds"] profile
    points drawn without replacement from ``seeds`` and params["bootstrap_local_starts"] local starts (cold), drawn
    from a generator seeded by params["seed"] and ``i``.
    """
    rng = np.random.default_rng([params["seed"], i])
    starts = [("warm", state["theta"], False)]
    n_seeds = min(int(params["bootstrap_profile_seeds"]), len(seeds))
    if n_seeds > 0:
        for j in rng.choice(len(seeds), size=n_seeds, replace=False):
            label, theta = seeds[int(j)]
            starts.append((label, theta, True))
    local = local_starts(state["theta"], state["free"], int(params["bootstrap_local_starts"]),
                         params["bootstrap_local_span"], rng)
    for k, theta in enumerate(local):
        starts.append(("local %d" % k, theta, True))
    return starts

def fit_rows(i, free, starts, results):
    """Rows of bootstrap_fits_all.csv: every start of resample ``i`` with its fit."""
    rows = []
    for (label, _theta, _cold), (theta, obj, converged, error, wall_s) in zip(starts, results):
        row = {"resample": i, "start": label, "converged": converged, "obj": obj, "wall_s": wall_s, "error": error}
        for p in free:
            row[p] = theta[p] if converged else np.nan
        rows.append(row)
    return rows

def best_fit_row(i, free, all_rows):
    """
    One row of bootstrap_theta.csv: the converged fit of resample ``i`` with the lowest objective (best_start: its
    start), the objective of the warm-started fit and the number of converged fits.
    """
    converged = [row for row in all_rows if row["converged"]]
    best = min(converged, key=lambda row: row["obj"]) if converged else None
    warm = all_rows[0]
    row = {"resample": i, "converged": best is not None, "obj": best["obj"] if best else None,
           "wall_s": sum(r["wall_s"] for r in all_rows), "error": warm["error"],
           "obj_warm": warm["obj"] if warm["converged"] else None, "best_start": best["start"] if best else "",
           "n_converged_fits": len(converged)}
    for p in free:
        row[p] = best[p] if best else np.nan
    return row

def fit_resamples(resamples, state, params, out_dir, resume, seeds):
    """
    Fit the resamples into bootstrap_theta.csv (best fit per resample) and bootstrap_fits_all.csv (every fit): the
    starts of starts_of_resample, all of a chunk in one process pool (warm options for the warm start, default options
    for the others).
    """
    free = state["free"]
    u_bounds, x_bounds, drain_bounds = variable_bounds()
    warm_args = {"free_params": free, "x_bounds": x_bounds, "u_bounds": u_bounds, "drain_bounds": drain_bounds,
                 "x_init": state["x_init"], "u_init": state["u_init"], "b_init": state["b_init"],
                 "solver_options": WARM_START_OPTIONS}
    cold_args = dict(warm_args, solver_options=dict(SOLVER_OPTIONS))
    path = os.path.join(out_dir, "bootstrap_theta.csv")
    all_path = os.path.join(out_dir, "bootstrap_fits_all.csv")
    rows = []
    all_rows = []
    if resume and os.path.exists(path):
        rows = pd.read_csv(path).to_dict("records")
        all_rows = pd.read_csv(all_path, keep_default_na=False, na_values=[""]).to_dict("records")
    finished = {int(row["resample"]) for row in rows}
    todo = [i for i in range(len(resamples)) if i not in finished]
    for chunk_start in range(0, len(todo), CHUNK):
        ids = todo[chunk_start:chunk_start + CHUNK]
        starts_by_resample = []
        conditions = []
        thetas = []
        args_indices = []
        for i in ids:
            starts = starts_of_resample(i, state, params, seeds)
            starts_by_resample.append(starts)
            for _label, theta, cold in starts:
                conditions.append(resamples[i])
                thetas.append(theta)
                args_indices.append(1 if cold else 0)
        results = fit_many(conditions, thetas, [warm_args, cold_args], params["n_jobs"], args_indices)
        position = 0
        for i, starts in zip(ids, starts_by_resample):
            mine = fit_rows(i, free, starts, results[position:position + len(starts)])
            position += len(starts)
            all_rows.extend(mine)
            rows.append(best_fit_row(i, free, mine))
        table = pd.DataFrame(rows).sort_values("resample")
        table.to_csv(path, index=False)
        pd.DataFrame(all_rows).sort_values(["resample"], kind="stable").to_csv(all_path, index=False)
        n_moved = int((table["best_start"].fillna("warm") != "warm").sum())
        log("saved %d/%d resamples (%d converged; best fit not from the warm start in %d)" % (
            len(table), len(resamples), int(table["converged"].sum()), n_moved))

def summarize(anchor_dir, out_dir, params=None):
    """bootstrap_summary.csv, objective_by_n_KO18.csv and bootstrap_info.json from the saved fits."""
    params = default_params() if params is None else params
    state = read_anchor(anchor_dir)
    with open(os.path.join(anchor_dir, "final_summary.json")) as fh:
        kept_free_by_design = json.load(fh)["kept_free_by_design"]
    fits = pd.read_csv(os.path.join(out_dir, "bootstrap_theta.csv"))
    resamples = pd.read_csv(os.path.join(out_dir, "bootstrap_resamples.csv"))
    converged = fits[fits["converged"]]
    near = near_optimal_fits(out_dir, state, params)
    rows = []
    for p in state["free"]:
        values = converged[p].to_numpy(float)
        low, median, high = np.percentile(values, [2.5, 50, 97.5])
        ci_ratio = float(high / low)
        lower, upper = THETA_BOUNDS[p]
        reaches_bound = np.log(low / lower) < BOUND_LOG_DISTANCE or np.log(upper / high) < BOUND_LOG_DISTANCE
        rows.append({"param": p, "estimate": float(state["theta"][p]), "median": median, "p2.5": low, "p97.5": high,
                     "cv_percent": 100 * float(values.std(ddof=1) / abs(values.mean())),
                     "ci_ratio_97.5_over_2.5": ci_ratio, "interval_reaches_bound": bool(reaches_bound),
                     "kept_free_by_design": p in kept_free_by_design,
                     "practically_identifiable": (ci_ratio <= params["bootstrap_ci_ratio"]) and not reaches_bound
                     and p not in kept_free_by_design})
        if near is not None:
            near_low, near_high = np.percentile(near[p].to_numpy(float), [2.5, 97.5])
            rows[-1].update({"near_p2.5": near_low, "near_p97.5": near_high,
                             "near_ci_ratio": float(near_high / near_low)})
    summary = pd.DataFrame(rows).set_index("param")
    summary.to_csv(os.path.join(out_dir, "bootstrap_summary.csv"))
    info = {"n_resamples": int(len(fits)), "n_converged": int(len(converged)),
            "obj_estimate": state["obj"], "obj_boot_median": float(converged["obj"].median()),
            "obj_boot_p2.5_p97.5": [float(x) for x in np.percentile(converged["obj"], [2.5, 97.5])],
            "practically_identifiable": list(summary.index[summary["practically_identifiable"]])}
    if "best_start" in converged.columns:
        info["n_best_not_from_warm_start"] = int((converged["best_start"].fillna("warm") != "warm").sum())
        info["best_start_counts"] = {kind: int(n) for kind, n in
                                     converged["best_start"].fillna("warm").str.split(" ").str[0].value_counts().items()}
    if near is not None:
        info["n_near_optimal_fits"] = int(len(near))
    info.update(objective_against_hard_conditions(converged, resamples, out_dir))
    with open(os.path.join(out_dir, "bootstrap_info.json"), "w") as fh:
        json.dump(info, fh, indent=1)
    log("bootstrap summary: %s\n%s" % (info, summary.round(4).to_string()))
    return summary, info

def near_optimal_fits(out_dir, state, params):
    """
    The near-optimal set: every converged fit of bootstrap_fits_all.csv within the likelihood-ratio threshold of the
    best fit of its resample (None without that file).
    """
    all_path = os.path.join(out_dir, "bootstrap_fits_all.csv")
    if not os.path.exists(all_path):
        return None
    fits = pd.read_csv(all_path)
    fits = fits[fits["converged"]]
    threshold = lr_threshold(params["lr_alpha"], len(state["conditions"]))
    best_of_resample = fits.groupby("resample")["obj"].transform("min")
    return fits[fits["obj"] <= best_of_resample + threshold]

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
    joined.groupby("n_KO18")["obj"].agg(["count", "mean", "std"]).to_csv(
        os.path.join(out_dir, "objective_by_n_KO18.csv"))
    return {"obj_regression_intercept": float(coefficients[0]), "obj_per_KO18_copy": float(coefficients[1]),
            "obj_per_KO05_copy": float(coefficients[2]), "obj_regression_R2": float(r2)}
