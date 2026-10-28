"""
profiles.py -- profile likelihood of every free parameter of the final structure.

For each free parameter two continuation arms leave the anchor (up and down), each with profile_half_points steps
over profile_span decades (clipped to the theta bounds): every point fixes the parameter, re-optimizes the other free
parameters warm started at the previous point's theta and primal point, with the tight options (default and monotone
options as retries).  The threshold is obj_hat + lr_threshold(lr_alpha, N), and the misfit-scaled one (times s2); a
parameter is "identifiable" if both arms cross the threshold, "one_sided" if one does, "flat" if none does.

Position in the pipeline: uncertainty of the final structure (after selection.py and final_analysis.py); writes
profiles_all.csv, profile_summary.csv, profile_thresholds.json.
"""
import os
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from settings import TIGHT_OPTIONS, SOLVER_OPTIONS, RETRY_MONOTONE_OPTIONS, THETA_BOUNDS, default_params
from settings import snapshot_settings
from progress import log
from paths import check_out_dir
from metrics import lr_threshold
from estimation.anchors import make_est, check_bounds, read_anchor

def fit_point(conditions, param, value, others, theta, x_init, u_init, b_init):
    """
    One profile point: ``param`` fixed at ``value``, the ``others`` re-optimized from (theta, x, u, b), with the
    tight options, then the default ones, then monotone mu.  Returns (estimator, options tag) or (None, "failed").
    """
    theta_init = dict(theta)
    theta_init[param] = float(value)
    for tag, extra_options in (("tight", dict(TIGHT_OPTIONS)), ("default", {}),
                               ("monotone", dict(RETRY_MONOTONE_OPTIONS))):
        est = make_est(conditions, theta_init, others, solver_options={**SOLVER_OPTIONS, **extra_options},
                       x_init=x_init, u_init=u_init, b_init=b_init)
        try:
            est.estimate()
            est.refine_anchor(max_iter=3, primary_options=dict(TIGHT_OPTIONS))
        except Exception:   # noqa: BLE001  (IPOPT not converging: try the next options)
            continue
        return est, tag
    return None, "failed"

def profile_arm(task):
    """
    One continuation arm (param, sign +1 / -1, steps, span in decades, anchor directory): the rows of
    profiles_all.csv.  Every point starts at the previous point; a point beyond a theta bound repeats the previous
    value and is skipped.
    """
    param, sign, steps, span, anchor_dir = task
    state = read_anchor(anchor_dir)
    others = [p for p in state["free"] if p != param]
    theta = dict(state["theta"])
    x_init, u_init, b_init = state["x_init"], state["u_init"], state["b_init"]
    lower, upper = THETA_BOUNDS[param]
    anchor_value = float(theta[param])
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
        est, tag = fit_point(state["conditions"], param, value, others, theta, x_init, u_init, b_init)
        if est is None:
            row["status"] = "failed"
            row["wall_s"] = time.time() - started
            rows.append(row)
            continue
        theta = {**theta, **{p: float(est.theta[p]) for p in others}, param: value}
        x_init, u_init = est.primal_start()
        b_init = est.primal_b_free()
        row["obj"] = float(est.obj_value)
        row["status"] = tag
        row["wall_s"] = time.time() - started
        rows.append(row)
    return rows

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

def profile_class(upper, lower):
    if np.isfinite(upper) and np.isfinite(lower):
        return "identifiable"
    if np.isfinite(upper) or np.isfinite(lower):
        return "one_sided"
    return "flat"

def summarize(out_dir, state, anchor_dir, params):
    """profile_summary.csv and profile_thresholds.json from profiles_all.csv (s2 from final_summary.json)."""
    points = pd.read_csv(os.path.join(out_dir, "profiles_all.csv"))
    n_conditions = len(state["conditions"])
    s2 = float(pd.read_json(os.path.join(anchor_dir, "final_summary.json"), typ="series")["s2"])
    threshold = lr_threshold(params["lr_alpha"], n_conditions)
    threshold_s2 = lr_threshold(params["lr_alpha"], n_conditions, s2=s2)
    # reference objective: the lowest objective found (the anchor or a profile point); a re-fit from the anchor
    # primal can land a few 1e-2 above or below the anchor because IPOPT starts without duals
    obj_anchor = float(state["obj"])
    obj_hat = float(min(obj_anchor, np.nanmin(points["obj"].to_numpy(float))))
    rows = []
    for p in state["free"]:
        rows.append(parameter_profile(p, points[points["param"] == p], state, obj_hat, obj_anchor, threshold,
                                      threshold_s2))
    summary = pd.DataFrame(rows).set_index("param")
    summary.to_csv(os.path.join(out_dir, "profile_summary.csv"))
    # the "0.95" of the two keys is historical: the level is lr_alpha of settings/thresholds.csv
    thresholds = {"lr_threshold_0.95": threshold, "lr_threshold_0.95_s2": threshold_s2, "s2": s2,
                  "obj_hat": obj_hat, "obj_anchor": obj_anchor}
    with open(os.path.join(out_dir, "profile_thresholds.json"), "w") as fh:
        json.dump(thresholds, fh, indent=1)
    columns = ["upper_decades_lr", "lower_decades_lr", "class_lr", "class_lr_s2", "max_dobj_up", "max_dobj_down",
               "min_dobj_anywhere", "n_failed"]
    log("profile summary (threshold %.4f, scaled %.4f):\n%s" % (threshold, threshold_s2,
                                                                 summary[columns].round(4).to_string()))

def parameter_profile(p, points, state, obj_hat, obj_anchor, threshold, threshold_s2):
    """Crossings (decades from the anchor) and class of one parameter under both thresholds."""
    up = points[points["arm"] == 1].sort_values("k")
    down = points[points["arm"] == -1].sort_values("k")
    row = {"param": p, "anchor": float(state["theta"][p]), "obj_hat": obj_hat, "obj_anchor": obj_anchor,
           "n_failed": int((points["status"] == "failed").sum()), "n_points": int(points["obj"].notna().sum())}
    for tag, level in (("lr", threshold), ("lr_s2", threshold_s2)):
        upper = crossing(up["log10_offset"].to_numpy(float), (up["obj"] - obj_hat).to_numpy(float), level)
        lower = crossing((-down["log10_offset"]).to_numpy(float), (down["obj"] - obj_hat).to_numpy(float), level)
        row["upper_decades_" + tag] = upper
        row["lower_decades_" + tag] = lower
        row["class_" + tag] = profile_class(upper, lower)
    row["max_dobj_up"] = float(np.nanmax((up["obj"] - obj_hat).to_numpy(float))) if len(up) else np.nan
    row["max_dobj_down"] = float(np.nanmax((down["obj"] - obj_hat).to_numpy(float))) if len(down) else np.nan
    row["min_dobj_anywhere"] = float(np.nanmin((points["obj"] - obj_hat).to_numpy(float)))
    return row

def run_profiles(anchor_dir, out_dir, params=None):
    """
    Profile likelihood of every free parameter of the final structure in ``anchor_dir/anchor_state.pkl``: 2 arms x
    (profile_half_points + 1) points per parameter on params["n_jobs"] workers.  Writes into ``out_dir``; the
    settings of this run are copied into ``<parent of out_dir>/settings_used``.
    """
    params = default_params("profiles") if params is None else params
    out_dir = check_out_dir(out_dir)
    check_bounds()
    os.makedirs(out_dir, exist_ok=True)
    snapshot_settings(os.path.dirname(out_dir))
    state = read_anchor(anchor_dir)
    steps = params["profile_half_points"]
    tasks = []
    for p in state["free"]:
        for sign in (1, -1):
            tasks.append((p, sign, steps, params["profile_span"], anchor_dir))
    log("%d profile arms (%d params x 2), %d points each" % (len(tasks), len(state["free"]), steps + 1))
    rows = []
    with ProcessPoolExecutor(max_workers=params["n_jobs"]) as pool:
        for arm in pool.map(profile_arm, tasks):
            rows.extend(arm)
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "profiles_all.csv"), index=False)
            n_failed = sum(row["status"] == "failed" for row in arm)
            log("arm %s %+d done (%d points, %d failed)" % (arm[0]["param"], arm[0]["arm"], len(arm), n_failed))
    summarize(out_dir, state, anchor_dir, params)

def summarize_saved(anchor_dir, out_dir, params=None):
    """Summary of existing profiles in ``out_dir`` (profiles_all.csv)."""
    params = default_params("profiles") if params is None else params
    summarize(out_dir, read_anchor(anchor_dir), anchor_dir, params)
