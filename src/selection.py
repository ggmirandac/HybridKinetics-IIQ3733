"""
selection.py -- automated parameter selection of the glycolysis model.

Stage 0 fits all 37 parameters: a multistart (n_starts log-uniform starts in the parameter bounds plus the literature
values) and the best start refined.

The selection then fixes parameters until the free set is identifiable:
  1. structural step (identifiability/structural.py, Stigter and Molenaar 2015): while the stacked rate sensitivity
     matrix has a structural zero singular value, the rule of identifiability/rules.py acts on the parameter with
     the largest loading in its singular vector; then one refit;
  2. practical loop (identifiability/practical.py), one decision per iteration:
       - if the Fisher information is rank deficient, the rule acts on the parameter with the largest loading in the
         singular vector of the smallest singular value;
       - otherwise, in the pair with the largest |r| > corr_threshold, the more sensitive member stays free and the
         rule acts on the less sensitive one (mean |d ln y*/d ln theta|; the other member if that one is protected);
       - otherwise the free set is identifiable and the loop stops.
     After a fix the model is refitted (estimation/refit.py) and the sensitivities are recomputed.
The rule fixes the parameter at its estimated value if the protections allow it; a protected parameter stays free,
is reported as not identifiable ("kept free by design") and is left out of the later criteria.  One decision per
iteration, because every fix changes the Fisher information the next one is read from.

Outputs (``<out>``): stage0/ (anchor_state.pkl, theta_full.csv, multistart_recorded.csv, objective_breakdown.csv,
b_fit_vs_data.csv, pred_insample_*.csv, stage_row.json); reduction/ (selection.log, fixing_log.csv,
iterations.csv, structural_spectrum.csv, iter_XX/analysis.csv, final_anchor_state.pkl, final_theta_full.csv,
reduction_summary.json); settings_used/.

Position in the pipeline: the selection step; its final fit feeds the final analysis, bootstrap, profiles and LOOCV.
Command line (a relative --out is relative to the repo root):
    python src/selection.py --out results/flux_model [--from_anchor <dir with anchor_state.pkl>]
"""
import os
import json
import time
import argparse

import numpy as np
import pandas as pd

from settings import THETA_BOUNDS, default_params, snapshot_settings
from paths import resolve_out
from progress import log
from model.parameters import ALL_PARAMS, LITERATURE_THETA
from data import available_conditions
from metrics import b_vs_data, pred_metrics, theta_at_bounds, misfit_s2
from estimation.anchors import make_est, save_anchor, restore_anchor, check_bounds
from estimation.refit import refit_with_fixed
from estimation.multistart import multistart_recorded
from identifiability.structural import RateLaws, condition_states, stacked_sensitivity_matrix, singular_spectrum
from identifiability.practical import sensitivity_matrices, fim_analysis
from identifiability.rules import can_fix, largest_loading

class SelectionRecord:
    """The tables and the log of the selection (reduction/)."""

    def __init__(self, red_dir):
        self.dir = red_dir
        self.decisions = []
        self.iterations = []
        self.started = time.time()
        open(os.path.join(red_dir, "selection.log"), "w").close()

    def log(self, message):
        line = "[%6.0fs] %s" % (time.time() - self.started, message)
        print(line, flush=True)
        with open(os.path.join(self.dir, "selection.log"), "a") as fh:
            fh.write(line + "\n")

    def decision(self, est, name, stage, rule, iteration, action, loading=np.nan, partner="", r=np.nan,
                 sensitivity=np.nan):
        """One row of fixing_log.csv: the parameter a rule acted on and whether it was fixed or kept free."""
        value = float(est.full_theta()[name])
        self.decisions.append({"param": name, "iter": iteration, "stage": stage, "rule": rule, "action": action,
                               "loading": loading, "partner": partner, "r": r, "sensitivity": sensitivity,
                               "value": value, "obj_before": float(est.obj_value), "obj_after": np.nan})
        self.log("  %-9s %-12s rule %-10s loading %s partner %s r %s value %.4g" % (
            action.upper(), name, rule, "%.3f" % loading if np.isfinite(loading) else "-", partner or "-",
            "%.3f" % r if np.isfinite(r) else "-", value))

    def save(self):
        pd.DataFrame(self.decisions).to_csv(os.path.join(self.dir, "fixing_log.csv"), index=False)
        pd.DataFrame(self.iterations).to_csv(os.path.join(self.dir, "iterations.csv"), index=False)

def apply_rule(est, name, kept_free, record):
    """
    Fix ``name`` (refit with it held at its estimated value) if the protections allow it; otherwise add it to
    ``kept_free``.  Returns (estimator, action).
    """
    if not can_fix(name, est.free_params):
        kept_free.append(name)
        return est, "kept_free"
    new_free = [p for p in est.free_params if p != name]
    return refit_with_fixed(est, new_free, record.log), "fixed"

def structural_step(est, params, record):
    """
    Step 1: the structural zeros of the stacked rate sensitivity matrix at the stage-0 fit, one parameter at a time,
    then one refit with all structural fixes.  Returns (estimator, parameters kept free by design).
    """
    free = list(est.free_params)
    S = stacked_sensitivity_matrix(RateLaws(), est.full_theta(), condition_states(est), free, params)
    analysed = list(free)
    fixed = []
    kept_free = []
    spectra = {}
    while analysed:
        columns = [free.index(p) for p in analysed]
        ratio, Vt = singular_spectrum(S[:, columns])
        spectra["%d_free" % len(analysed)] = pd.Series(ratio, index=np.arange(1, len(ratio) + 1))
        if ratio[-1] >= params["structural_tol"]:
            break
        name, loading = largest_loading(analysed, Vt[-1])
        free_now = [p for p in free if p not in fixed]
        action = "fixed" if can_fix(name, free_now) else "kept_free"
        record.decision(est, name, "structural", "structural_zero", 0, action, loading=loading)
        if action == "fixed":
            fixed.append(name)
        else:
            kept_free.append(name)
        analysed.remove(name)
    pd.DataFrame(spectra).rename_axis("k").to_csv(os.path.join(record.dir, "structural_spectrum.csv"))
    record.log("structural step: %d fixed %s, %d kept free by design %s" % (len(fixed), fixed, len(kept_free),
                                                                           kept_free))
    if fixed:
        obj_before = float(est.obj_value)
        est = refit_with_fixed(est, [p for p in free if p not in fixed], record.log)
        record.log("  refit: %d free, obj %.5f -> %.5f" % (len(est.free_params), obj_before, est.obj_value))
    for row in record.decisions:
        row["obj_after"] = float(est.obj_value)
    return est, kept_free

def practical_decision(analysis, free):
    """(parameter, rule, loading, partner, r) the practical rules act on, or None if the free set is identifiable."""
    if not analysis["full_rank"]:
        name, loading = largest_loading(analysis["names"], analysis["null_vector"])
        return name, "svd", loading, "", np.nan
    if not analysis["pairs"]:
        return None
    first, second, r = analysis["pairs"][0]
    sensitivity = analysis["sensitivity"]
    lower, higher = (first, second) if sensitivity[first] <= sensitivity[second] else (second, first)
    if can_fix(lower, free) or not can_fix(higher, free):
        return lower, "correlation", np.nan, higher, r
    return higher, "correlation", np.nan, lower, r

def write_iteration(record, iteration, est, analysis):
    """iter_XX/analysis.csv: value, sensitivity, loading on the least identifiable direction and CV of the analysed."""
    folder = os.path.join(record.dir, "iter_%02d" % iteration)
    os.makedirs(folder, exist_ok=True)
    full = est.full_theta()
    table = pd.DataFrame({"value": [float(full[p]) for p in analysis["names"]],
                          "sensitivity": analysis["sensitivity"].to_numpy(),
                          "null_loading": analysis["null_vector"]}, index=pd.Index(analysis["names"], name="param"))
    if analysis["cv_percent"] is not None:
        table["cv_percent"] = analysis["cv_percent"]
    table.to_csv(os.path.join(folder, "analysis.csv"))
    pd.Series(analysis["sigma_ratio"], name="sigma_ratio").rename_axis("k").to_csv(
        os.path.join(folder, "spectrum.csv"))

def practical_loop(est, kept_free, params, record):
    """Step 2: one practical decision per iteration until the free set is identifiable.  Returns the estimator."""
    J, relative, failed = sensitivity_matrices(est)
    iteration = 0
    while True:
        iteration += 1
        analysed = [p for p in est.free_params if p not in kept_free]
        if not analysed:
            record.log("  every free parameter is kept free by design: nothing left to analyse")
            return est
        analysis = fim_analysis(J, relative, analysed, params)
        write_iteration(record, iteration, est, analysis)
        decision = practical_decision(analysis, est.free_params)
        row = {"iter": iteration, "n_free": len(est.free_params), "n_analysed": len(analysed),
               "obj": float(est.obj_value), "sigma_ratio_min": float(analysis["sigma_ratio"][-1]),
               "full_rank": analysis["full_rank"], "max_abs_r": analysis["max_abs_r"],
               "n_pairs": len(analysis["pairs"]), "sens_failed_conditions": ";".join(failed)}
        record.log("iteration %d: %d free (%d analysed), obj %.5f, sigma_min/max %.2e, max|r| %.3f" % (
            iteration, row["n_free"], row["n_analysed"], row["obj"], row["sigma_ratio_min"], row["max_abs_r"]))
        if decision is None:
            row["decision"] = "identifiable"
            record.iterations.append(row)
            record.log("  the free set is identifiable")
            return est
        name, rule, loading, partner, r = decision
        before = est
        est, action = apply_rule(est, name, kept_free, record)
        record.decision(before, name, "practical", rule, iteration, action, loading=loading, partner=partner, r=r,
                        sensitivity=float(analysis["sensitivity"][name]))
        record.decisions[-1]["obj_after"] = float(est.obj_value)
        row["decision"] = "%s %s (%s)" % (action, name, rule)
        record.iterations.append(row)
        record.save()
        if action == "fixed":
            record.log("  refit: %d free, obj %.5f" % (len(est.free_params), est.obj_value))
            J, relative, failed = sensitivity_matrices(est)

def run_reduction(est, out_dir, params=None):
    """
    The structural step and the practical loop from the stage-0 fit ``est``.  Writes ``<out_dir>/reduction/`` and
    returns (the final estimator, the parameters kept free by design).
    """
    params = default_params() if params is None else params
    red_dir = os.path.join(resolve_out(out_dir), "reduction")
    os.makedirs(red_dir, exist_ok=True)
    record = SelectionRecord(red_dir)
    record.log("selection from %d free, obj %.5f" % (len(est.free_params), est.obj_value))
    est, kept_free = structural_step(est, params, record)
    record.save()
    est = practical_loop(est, kept_free, params, record)
    record.save()
    save_anchor(est, red_dir)
    os.replace(os.path.join(red_dir, "anchor_state.pkl"), os.path.join(red_dir, "final_anchor_state.pkl"))
    os.replace(os.path.join(red_dir, "theta_full.csv"), os.path.join(red_dir, "final_theta_full.csv"))
    free = list(est.free_params)
    summary = {"n_free": len(free), "free": free, "fixed": [p for p in ALL_PARAMS if p not in free],
               "kept_free_by_design": kept_free, "obj": float(est.obj_value), "wall_s": time.time() - record.started}
    with open(os.path.join(red_dir, "reduction_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    record.log("selection finished: %d free %s; kept free by design %s; obj %.6f" % (len(free), free, kept_free,
                                                                                     est.obj_value))
    return est, kept_free

def literature_theta_in_bounds():
    theta = {}
    for p in ALL_PARAMS:
        theta[p] = float(min(max(LITERATURE_THETA[p], THETA_BOUNDS[p][0]), THETA_BOUNDS[p][1]))
    return theta

def run_stage0(out_dir, params=None, conditions=None):
    """
    Stage 0, all 37 parameters free: the recorded multistart (the literature values and params["n_starts"]
    log-uniform starts), then its best start re-run here and refined.  Writes ``<out_dir>/stage0/`` and returns it.
    """
    params = default_params() if params is None else params
    stage0_dir = os.path.join(resolve_out(out_dir), "stage0")
    os.makedirs(stage0_dir, exist_ok=True)
    check_bounds()
    conditions = available_conditions() if conditions is None else list(conditions)
    theta_start = literature_theta_in_bounds()
    started = time.time()
    table = multistart_recorded(make_est(conditions, theta_start, ALL_PARAMS), params["n_starts"], params["seed"],
                                params["n_jobs"])
    table.to_csv(os.path.join(stage0_dir, "multistart_recorded.csv"), index=False)
    converged = table[table["converged"]]
    log("multistart done in %.0fs: %d/%d converged, best obj %.5f, median %.5f" % (
        time.time() - started, len(converged), len(table), converged["obj_value"].min(),
        converged["obj_value"].median()))
    best = converged.iloc[0]
    theta = {p: float(best["init_" + p]) for p in ALL_PARAMS}
    est = make_est(conditions, theta, ALL_PARAMS)
    est.estimate()
    est.refine_anchor()
    log("best start %d re-run and refined: obj %.5f (multistart table %.5f)" % (int(best["start"]), est.obj_value,
                                                                              float(best["obj_value"])))
    write_stage0_results(est, table, stage0_dir)
    return stage0_dir

def write_stage0_results(est, table, stage0_dir):
    """anchor_state.pkl, theta_full.csv, objective_breakdown.csv, b_fit_vs_data.csv, pred_insample_*.csv and
    stage_row.json of the stage-0 fit."""
    save_anchor(est, stage0_dir)
    breakdown = est.objective_breakdown()
    breakdown.to_csv(os.path.join(stage0_dir, "objective_breakdown.csv"))
    drains = b_vs_data(est)
    drains.to_csv(os.path.join(stage0_dir, "b_fit_vs_data.csv"), index=False)
    prediction = est.predict_fitted()
    prediction.predicted.to_csv(os.path.join(stage0_dir, "pred_insample_predicted.csv"))
    prediction.real.to_csv(os.path.join(stage0_dir, "pred_insample_real.csv"))
    converged = table[table["converged"]]
    row = {"n_free": len(est.free_params), "obj": float(est.obj_value), "s2": float(misfit_s2(est)),
           "multistart_best_table": float(converged["obj_value"].min()),
           "multistart_n_converged": int(len(converged)), "multistart_n_total": int(len(table)),
           "multistart_median": float(converged["obj_value"].median()),
           "b_rmse_vs_data": float(np.sqrt(np.mean(drains["err"] ** 2))),
           "n_drains_at_bound": int(drains["at_bound"].sum()),
           "theta_at_bounds": ";".join(theta_at_bounds(est)), "n_pred_failed": int(prediction.n_failed)}
    row.update(pred_metrics(prediction.real, prediction.predicted))
    with open(os.path.join(stage0_dir, "stage_row.json"), "w") as fh:
        json.dump(row, fh, indent=1)
    log("stage0 saved: obj %.5f, rmse_met %.4f (baseline %.4f), LI flux rmse %.4f (baseline %.4f)" % (
        est.obj_value, row["rmse_met"], row["rmse_met_meanbaseline"], row["rmse_flux_li"],
        row["rmse_flux_li_meanbaseline"]))

def run_selection(out_dir, params=None, from_anchor=None):
    """
    Stage 0 (skipped with ``from_anchor``, a directory with anchor_state.pkl), then the reduction from that fit.
    Returns (the final estimator, the parameters kept free by design).
    """
    params = default_params() if params is None else params
    out_dir = resolve_out(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    snapshot_settings(out_dir)
    anchor_dir = from_anchor if from_anchor else run_stage0(out_dir, params)
    est, _state = restore_anchor(anchor_dir)
    return run_reduction(est, out_dir, params)

def main(argv=None):
    params = default_params()
    parser = argparse.ArgumentParser(description="Automated parameter selection: stage 0 and the reduction.")
    parser.add_argument("--out", default=params["out_dir"], help="output directory (relative to the repo root)")
    parser.add_argument("--from_anchor", default=None, help="directory with anchor_state.pkl: skip stage 0")
    args = parser.parse_args(argv)
    run_selection(args.out, params, from_anchor=args.from_anchor)

if __name__ == "__main__":
    main()
