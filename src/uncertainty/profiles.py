"""
profiles.py -- profile likelihood of every free parameter of the final structure.

For each free parameter two continuation arms leave the estimate (up and down), each with profile_points steps over
profile_span decades (clipped to the theta bounds): every point fixes the parameter and re-optimizes the other free
parameters, warm started at the previous point's theta and primal point, plus profile_extra_starts starts around it (the best
fit is kept).  A point that a point farther out on its arm beats by more than the threshold (or that failed) missed the
minimum: it is refitted backward, from the theta of the next point farther out.  The threshold is
obj_hat + lr_threshold(lr_alpha, N); an arm crosses where it reaches the threshold and stays above it farther out (the
crossing of the lowest objective at or beyond every point).  A parameter is "identifiable" if both arms cross within
log10(bootstrap_ci_ratio) decades in total (the bootstrap criterion), "wide" if both cross farther apart, "one_sided"
if one does, "flat" if none does.

Position in the pipeline: uncertainty of the final structure (after selection.py and final_analysis.py); writes
profiles_all.csv, profile_summary.csv, profile_thresholds.json.
"""
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from settings import SOLVER_OPTIONS, WARM_START_OPTIONS, THETA_BOUNDS, default_params
from settings import snapshot_settings
from progress import log
from paths import resolve_out
from metrics import lr_threshold
from estimation.anchors import make_est, check_bounds, read_anchor

def fit_point(conditions, param, value, others, theta, x_init, u_init, b_init, solver_options=WARM_START_OPTIONS):
    """
    One profile point: ``param`` fixed at ``value``, the ``others`` re-optimized from (theta, x, u, b) with
    ``solver_options`` (the warm-start options by default).  Returns the estimator, or None if IPOPT does not converge.
    """
    theta_init = dict(theta)
    theta_init[param] = float(value)
    est = make_est(conditions, theta_init, others, solver_options=solver_options, x_init=x_init, u_init=u_init,
                   b_init=b_init)
    try:
        est.estimate()
        est.refine_anchor(max_rounds=3)
    except Exception:   # noqa: BLE001  (IPOPT not converging: the point is recorded as failed)
        return None
    return est

def extra_starts(theta, others, n_starts, span, rng):
    """``n_starts`` copies of ``theta`` with every other free parameter times 10**U(-span, span), clipped to its bounds."""
    starts = []
    for _ in range(n_starts):
        start = dict(theta)
        for p in others:
            lower, upper = THETA_BOUNDS[p]
            start[p] = float(np.clip(theta[p] * 10.0 ** rng.uniform(-span, span), lower, upper))
        starts.append(start)
    return starts

def best_point(conditions, param, value, others, theta, x_init, u_init, b_init, n_extra, extra_span, rng):
    """
    The best of the warm-started fit of a profile point and ``n_extra`` fits from extra starts around the previous
    point (default solver options, primal point of the previous point).  Returns (estimator or None, winning start).
    """
    best = fit_point(conditions, param, value, others, theta, x_init, u_init, b_init)
    winner = "warm"
    for k, start in enumerate(extra_starts(theta, others, n_extra, extra_span, rng)):
        est = fit_point(conditions, param, value, others, start, x_init, u_init, b_init, solver_options=dict(SOLVER_OPTIONS))
        if est is not None and (best is None or est.obj_value < best.obj_value):
            best = est
            winner = "extra %d" % k
    return best, winner

def profile_arm(task):
    """
    One continuation arm (param, sign +1 / -1, steps, span in decades, anchor directory, extra starts, their span,
    seed): the rows of profiles_all.csv, with the full theta of every point (columns theta_<p>).  Every point starts at
    the previous point, plus the extra starts; a point beyond a theta bound repeats the previous value and is skipped.
    """
    param, sign, steps, span, anchor_dir, n_extra, extra_span, seed = task
    state = read_anchor(anchor_dir)
    others = [p for p in state["free"] if p != param]
    theta = dict(state["theta"])
    x_init, u_init, b_init = state["x_init"], state["u_init"], state["b_init"]
    lower, upper = THETA_BOUNDS[param]
    anchor_value = float(theta[param])
    rng = np.random.default_rng([seed, state["free"].index(param), 1 if sign > 0 else 0])
    rows = []
    previous = None
    for k in range(0, steps + 1):
        offset = sign * k * span / steps
        value = float(min(max(anchor_value * 10.0 ** offset, lower), upper))
        row = {"param": param, "arm": sign, "k": k, "log10_offset": offset, "value": value, "obj": np.nan}
        if previous is not None and value == previous:
            row["status"] = "at_bound_skipped"
            rows.append(row)
            continue
        previous = value
        started = time.time()
        est, winner = best_point(state["conditions"], param, value, others, theta, x_init, u_init, b_init, n_extra,
                                 extra_span, rng)
        if est is None:
            row["status"] = "failed"
            row["wall_s"] = time.time() - started
            rows.append(row)
            continue
        theta = {**theta, **{p: float(est.theta[p]) for p in others}, param: value}
        x_init, u_init = est.primal_start()
        b_init = est.primal_b_free()
        row["obj"] = float(est.obj_value)
        row["status"] = "ok"
        row["start"] = winner
        for p in state["free"]:
            row["theta_" + p] = float(theta[p])
        row["wall_s"] = time.time() - started
        rows.append(row)
    return rows

def outer_minimum(values):
    """For every point of an arm (sorted outward), the lowest value at that point or farther out (NaN ignored)."""
    filled = np.where(np.isfinite(values), values, np.inf)
    return np.minimum.accumulate(filled[::-1])[::-1]

def points_to_refit(arm, threshold):
    """
    k of the fitted points of one arm (rows sorted by k) that a point farther out beats by more than ``threshold``, or
    that failed while a point farther out converged: fits that missed the minimum of the profile.
    """
    obj = arm["obj"].to_numpy(float)
    best_farther = np.append(outer_minimum(obj)[1:], np.inf)
    bad = []
    for k, value, farther in zip(arm["k"], obj, best_farther):
        if np.isfinite(farther) and (not np.isfinite(value) or value > farther + threshold):
            bad.append(int(k))
    return bad

def refit_arm(task):
    """
    Backward refit of the points ``bad`` of one arm (param, sign, rows of the arm, bad k, anchor directory, extra
    starts, their span, seed).  From the outermost inward, every point restarts from the theta of the next fitted point
    farther out, plus the extra starts around it (default solver options, primal point of the estimate), and keeps the
    best fit if it is lower than the one it has.  Returns the rows of the arm (column "refitted").
    """
    param, sign, rows, bad, anchor_dir, n_extra, extra_span, seed = task
    state = read_anchor(anchor_dir)
    others = [p for p in state["free"] if p != param]
    rng = np.random.default_rng([seed, state["free"].index(param), 3 if sign > 0 else 2])
    by_k = {int(row["k"]): row for row in rows}
    for k in sorted(bad, reverse=True):
        row = by_k[k]
        source = next(by_k[j] for j in sorted(by_k) if j > k and by_k[j]["status"] == "ok")
        theta = dict(state["theta"])
        for p in state["free"]:
            theta[p] = float(source["theta_" + p])
        started = time.time()
        best = None
        winner = ""
        for label, start in [("neighbour", theta)] + [("extra %d" % i, s) for i, s in
                                                     enumerate(extra_starts(theta, others, n_extra, extra_span, rng))]:
            est = fit_point(state["conditions"], param, row["value"], others, start, state["x_init"], state["u_init"],
                            state["b_init"], solver_options=dict(SOLVER_OPTIONS))
            if est is not None and (best is None or est.obj_value < best.obj_value):
                best = est
                winner = label
        row["refitted"] = True
        row["wall_s"] = (float(row["wall_s"]) if pd.notna(row.get("wall_s")) else 0.0) + time.time() - started
        if best is not None and (row["status"] != "ok" or best.obj_value < row["obj"]):
            row["obj"] = float(best.obj_value)
            row["status"] = "ok"
            row["start"] = "backward " + winner
            for p in others:
                row["theta_" + p] = float(best.theta[p])
            row["theta_" + param] = float(row["value"])
    return [by_k[k] for k in sorted(by_k)]

def repair_profiles(anchor_dir, out_dir, params):
    """
    Refit the points of profiles_all.csv that missed the minimum (points_to_refit) by backward continuation, one arm
    per task on params["n_jobs"] workers, and rewrite the file.
    """
    state = read_anchor(anchor_dir)
    path = os.path.join(out_dir, "profiles_all.csv")
    points = pd.read_csv(path)
    threshold = lr_threshold(params["lr_alpha"], len(state["conditions"]))
    arms = []
    tasks = []
    for (param, sign), arm in points.groupby(["param", "arm"], sort=False):
        arm = arm.sort_values("k")
        arms.append((param, sign))
        bad = points_to_refit(arm[arm["status"] != "at_bound_skipped"], threshold)
        if bad:
            tasks.append((param, int(sign), arm.to_dict("records"), bad, anchor_dir, params["profile_extra_starts"],
                          params["profile_extra_span"], params["seed"]))
            log("arm %s %+d: %d points to refit backward (k = %s)" % (param, sign, len(bad), bad))
    if not tasks:
        return
    refitted = {}
    with ProcessPoolExecutor(max_workers=min(params["n_jobs"], len(tasks))) as pool:
        for rows in pool.map(refit_arm, tasks):
            refitted[(rows[0]["param"], rows[0]["arm"])] = rows
    rows = []
    for key in arms:
        if key in refitted:
            rows.extend(refitted[key])
        else:
            rows.extend(points[(points["param"] == key[0]) & (points["arm"] == key[1])].sort_values("k").to_dict("records"))
    pd.DataFrame(rows).to_csv(path, index=False)

def crossing(x, y, threshold):
    """First x (moving away from x[0]) where y reaches ``threshold`` (linear interpolation), else NaN."""
    for i in range(1, len(x)):
        if np.isfinite(y[i]) and y[i] >= threshold:
            j = i - 1
            while j > 0 and not np.isfinite(y[j]):
                j -= 1
            if np.isfinite(y[j]):
                return float(x[j] + (threshold - y[j]) / (y[i] - y[j]) * (x[i] - x[j]))
            return float(x[i])
    return float("nan")

def profile_class(upper, lower, max_width):
    """identifiable: both arms cross within max_width decades in total; wide: both cross, wider; one_sided; flat."""
    if np.isfinite(upper) and np.isfinite(lower):
        if upper + lower <= max_width:
            return "identifiable"
        return "wide"
    if np.isfinite(upper) or np.isfinite(lower):
        return "one_sided"
    return "flat"

def summarize(out_dir, state, params):
    """profile_summary.csv and profile_thresholds.json from profiles_all.csv."""
    points = pd.read_csv(os.path.join(out_dir, "profiles_all.csv"))
    threshold = lr_threshold(params["lr_alpha"], len(state["conditions"]))
    # reference objective: the lowest objective found (the estimate or a profile point)
    obj_estimate = float(state["obj"])
    obj_hat = float(min(obj_estimate, np.nanmin(points["obj"].to_numpy(float))))
    rows = []
    for p in state["free"]:
        rows.append(parameter_profile(p, points[points["param"] == p], state, obj_hat, threshold,
                                      np.log10(params["bootstrap_ci_ratio"])))
    summary = pd.DataFrame(rows).set_index("param")
    summary.to_csv(os.path.join(out_dir, "profile_summary.csv"))
    with open(os.path.join(out_dir, "profile_thresholds.json"), "w") as fh:
        json.dump({"lr_threshold": threshold, "obj_hat": obj_hat, "obj_estimate": obj_estimate}, fh, indent=1)
    log("profile summary (threshold %.4f):\n%s" % (threshold, summary.round(4).to_string()))

def parameter_profile(p, points, state, obj_hat, threshold, max_width):
    """
    Crossings of the threshold (decades from the estimate) and class of one parameter: an arm crosses where the lowest
    objective at or beyond each point reaches the threshold, so a point above it with a lower point farther out is not
    a crossing.
    """
    up = points[points["arm"] == 1].sort_values("k")
    down = points[points["arm"] == -1].sort_values("k")
    upper = crossing(up["log10_offset"].to_numpy(float), outer_minimum((up["obj"] - obj_hat).to_numpy(float)),
                     threshold)
    lower = crossing((-down["log10_offset"]).to_numpy(float), outer_minimum((down["obj"] - obj_hat).to_numpy(float)),
                     threshold)
    return {"param": p, "estimate": float(state["theta"][p]), "upper_decades": upper, "lower_decades": lower,
            "profile_class": profile_class(upper, lower, max_width), "n_failed": int((points["status"] == "failed").sum()),
            "max_dobj_up": float(np.nanmax((up["obj"] - obj_hat).to_numpy(float))),
            "max_dobj_down": float(np.nanmax((down["obj"] - obj_hat).to_numpy(float)))}

def run_profiles(anchor_dir, out_dir, params=None):
    """
    Profile likelihood of every free parameter of the final structure in ``anchor_dir/anchor_state.pkl``: 2 arms x
    (profile_points + 1) points per parameter on params["n_jobs"] workers, then the backward refit of the points that
    missed the minimum (repair_profiles).  Writes into ``out_dir``; the
    settings of this run are copied into ``<parent of out_dir>/settings_used``.
    """
    params = default_params() if params is None else params
    out_dir = resolve_out(out_dir)
    check_bounds()
    os.makedirs(out_dir, exist_ok=True)
    snapshot_settings(os.path.dirname(out_dir))
    state = read_anchor(anchor_dir)
    steps = params["profile_points"]
    tasks = []
    for p in state["free"]:
        for sign in (1, -1):
            tasks.append((p, sign, steps, params["profile_span"], anchor_dir, params["profile_extra_starts"],
                          params["profile_extra_span"], params["seed"]))
    log("%d profile arms (%d params x 2), %d points each" % (len(tasks), len(state["free"]), steps + 1))
    rows = []
    with ProcessPoolExecutor(max_workers=params["n_jobs"]) as pool:
        for arm in pool.map(profile_arm, tasks):
            rows.extend(arm)
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "profiles_all.csv"), index=False)
            n_failed = sum(row["status"] == "failed" for row in arm)
            log("arm %s %+d done (%d points, %d failed)" % (arm[0]["param"], arm[0]["arm"], len(arm), n_failed))
    repair_profiles(anchor_dir, out_dir, params)
    summarize(out_dir, state, params)
