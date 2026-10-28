"""
multistart.py -- multistart of the parameter estimation (global search in the theta box).

``multistart_recorded``: start 0 is the current theta, the other starts sample every free parameter log-uniformly
within its theta bounds (fixed parameters keep their values); each start is a full fit with the same configuration,
run in a process pool (estimation/parallel.py), and EVERY run is recorded (start, converged, objective, error, initial
and final theta).  ``multistart_candidate`` re-runs and refines the best non-warm starts of such a table in this
process and returns a strictly better estimator or None.

Position in the pipeline: stage 0 (all 37 parameters free), the periodic multistart of the selection loop and the
LOOCV folds that do not reach their bound.
"""
import numpy as np
import pandas as pd

from solver import check_solver
from estimation.parallel import fit_many

def random_starts(est, n_starts, seed):
    """[current theta] + ``n_starts`` draws with every free parameter log-uniform in its theta bounds."""
    free = list(est.free_params)
    base = est.full_theta()
    rng = np.random.default_rng(seed)
    lower = np.array([est.theta_bounds[p][0] for p in free], dtype=float)
    upper = np.array([est.theta_bounds[p][1] for p in free], dtype=float)
    starts = [{p: float(v) for p, v in base.items()}]
    for _ in range(int(n_starts)):
        start = dict(base)
        draw = np.exp(rng.uniform(np.log(lower), np.log(upper)))
        for p, value in zip(free, draw):
            start[p] = float(value)
        starts.append(start)
    return starts

def multistart_recorded(est, n_starts, seed, n_jobs):
    """
    Fit the conditions of ``est`` from its current theta (start 0, the warm start) and from ``n_starts`` random
    log-uniform starts, every fit recorded.  Returns a DataFrame sorted by objective (failed runs last) with columns
    start, is_warm_start, converged, obj_value, error, wall_s, init_<p> and <p> (final value, NaN if not converged)
    for every free parameter.  ``est`` itself is not changed.
    """
    check_solver("ipopt")
    free = list(est.free_params)
    starts = random_starts(est, n_starts, seed)
    estimator_args = {"free_params": free, "solver_options": est.solver_options, "x_bounds": est.x_bounds,
                      "u_bounds": est.u_bounds, "drain_bounds": est.drain_bounds, "x_init": None, "u_init": None,
                      "b_init": None}
    results = fit_many([list(est.conditions)] * len(starts), starts, estimator_args, n_jobs)
    rows = []
    for i, (start, result) in enumerate(zip(starts, results)):
        theta_fit, obj, converged, error, wall = result
        row = {"start": i, "is_warm_start": i == 0, "converged": bool(converged),
               "obj_value": float(obj) if converged else np.nan, "error": error, "wall_s": float(wall)}
        for p in free:
            row["init_" + p] = float(start[p])
        for p in free:
            row[p] = float(theta_fit[p]) if converged else np.nan
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values("obj_value", na_position="last", kind="stable").reset_index(drop=True)

def multistart_candidate(est, params, seed, log):
    """
    ``multistart_recorded`` (params["n_starts"] starts) from the anchor of ``est``, then the best
    params["multistart_top_n"] non-warm starts whose table objective is within params["multistart_max_ratio"] of the
    incumbent are re-run here and refined.  Returns (the best candidate if it is strictly better than ``est``, else
    None; the table).  ``est`` is not changed.
    """
    theta_anchor = {p: float(v) for p, v in est.full_theta().items()}
    work = est.clone(theta_init=theta_anchor)
    table = multistart_recorded(work, params["n_starts"], seed, params["n_jobs"])
    converged = table[table["converged"]]
    best_table = float(converged["obj_value"].min()) if len(converged) else np.nan
    log("    multistart: %d/%d converged, table best %.5f (warm start %.5f), incumbent %.5f" % (
        len(converged), len(table), best_table, float(table.loc[table["is_warm_start"], "obj_value"].min()),
        est.obj_value))
    best, best_obj = None, np.inf
    for _, row in converged[~converged["is_warm_start"]].head(params["multistart_top_n"]).iterrows():
        if float(row["obj_value"]) > params["multistart_max_ratio"] * est.obj_value:
            continue
        theta_init = dict(theta_anchor)
        for p in est.free_params:
            theta_init[p] = float(row["init_" + p])
        candidate = est.clone(theta_init=theta_init)
        try:
            candidate.estimate()
            candidate.refine_anchor()
        except Exception as exc:   # noqa: BLE001  (a start that does not converge is skipped)
            log("    multistart rerun failed: %s" % str(exc)[:100])
            continue
        log("    multistart start %d: table %.5f, rerun+refine %.5f" % (
            int(row["start"]), float(row["obj_value"]), candidate.obj_value))
        if candidate.obj_value < best_obj:
            best, best_obj = candidate, candidate.obj_value
    if best is not None and not best_obj < float(est.obj_value):
        best = None
    return best, table
