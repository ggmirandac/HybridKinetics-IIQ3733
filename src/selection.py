"""
selection.py -- automated parameter selection of the glycolysis model: stage 0 and the identifiability reduction.

Stage 0 fits all 37 parameters: the saved anchor of settings/run.csv (stage0_warm_start) refined with the tight
options, against the best start of a recorded multistart; the better one is the stage-0 anchor.  Without the saved
anchor the multistart starts from the literature values.

The reduction (``run_reduction``) fixes ONE parameter per iteration until the free set is identifiable:
  - a structural check at the start of every iteration (rate level and output level, identifiability/check.py);
    its fixes are refitted and the check repeated until it is clean;
  - the practical analysis on the Schur complement of the reduced-sensitivity FIM (identifiability/practical.py)
    and the first rule that fires: R0 zero influence (a batch, the FIM does not change), R1 SVD while the FIM is
    rank deficient, R2 |r| > corr_threshold at full rank, R3 a flagged parameter;
  - a monotone refit at the anchored values after the fix (estimation/refit.py), a multistart every
    multistart_every iterations and once more before the loop stops.
Protections: kcat_f_* are never fixed and every reaction keeps a free parameter.  Fixed parameters keep their
anchored values: the anchor stays feasible with the same objective, so a refit can only improve on it.  One fix per
iteration, because every fix changes the FIM the next rule is read from.  All settings come from settings/*.csv
(``default_params``).

Outputs (``<out>``): stage0/ (anchor_state.pkl, theta_full.csv, multistart_recorded.csv, objective_breakdown.csv,
stage_row.json, ...); reduction/ with iterations.csv, fixing_log.csv, reduction.log, structural_per_iteration.csv,
structural_spectra/, checkpoint.pkl, iter_XX/ (free_params.csv: the free set at the START of iteration XX;
next_free_params.csv: the free set after all fixes of the iteration; theta_full.csv, influence.csv, spectrum.csv,
corr_pairs.csv, reduced_sensitivity_params.csv), final_free_params.csv, final_structure.csv, final_anchor_state.pkl,
reduction_summary.json; settings_used/ (the settings CSV files of the latest run).

Position in the pipeline: the selection step; its final anchor feeds the final analysis, bootstrap, profiles and
LOOCV.  Command line (a relative --out is relative to the repo root; results/, results_v2/, results_v3/ are refused):
    python src/selection.py --out results_v4/flux_model [--from_anchor <dir with anchor_state.pkl>]
                            [--max_iter 30] [--n_jobs 8] [--resume]
"""
import os
import json
import time
import pickle
import shutil
import argparse

import numpy as np
import pandas as pd

from settings import TIGHT_OPTIONS, SOLVER_OPTIONS, THETA_BOUNDS, RUN, default_params, snapshot_settings
from paths import REPO_ROOT, check_out_dir
from progress import log
from model.parameters import ALL_PARAMS, RXN_OF_PARAM, LITERATURE_THETA
from data import available_conditions
from metrics import b_vs_data, pred_metrics, theta_at_bounds, misfit_s2
from estimation.anchors import make_est, save_anchor, restore_anchor, check_bounds, read_anchor
from estimation.refit import refit_with_fixed
from estimation.multistart import multistart_recorded, multistart_candidate
from identifiability.practical import practical_analysis
from identifiability.rules import can_fix
from identifiability.check import StructuralCheck, plan_structural_fixes

STRUCTURAL_RULE = {"exact": "structural_exact", "mixed": "structural_mixed", "point-dependent": "structural_pointdep",
                   "output-level": "structural_output"}

class ReductionRecord:
    """
    What the reduction writes into reduction/: the rows of iterations.csv, fixing_log.csv and
    structural_per_iteration.csv, the checkpoint and reduction.log.  It also keeps the last practical analysis:
    the reduced sensitivities take about 30 s, and the structural check and the final analysis often need the
    analysis of the same anchor with the same nuisance set again.
    """

    def __init__(self, red_dir):
        self.dir = red_dir
        self.iterations = []
        self.fixes = []
        self.structural = []
        self.multistart_done_for = None
        self.started = time.time()
        self.analysis = None
        self.analysis_est = None
        self.analysis_nuisance = None

    def log(self, message):
        line = "[%6.0fs] %s" % (time.time() - self.started, message)
        print(line, flush=True)
        with open(os.path.join(self.dir, "reduction.log"), "a") as fh:
            fh.write(line + "\n")

    def practical_analysis(self, est, nuisance, params):
        """The practical analysis at the anchor of ``est``; reused if the last one has the same anchor and nuisance."""
        if self.analysis_est is not est or self.analysis_nuisance != list(nuisance):
            self.analysis = practical_analysis(est, nuisance, params)
            self.analysis_est = est
            self.analysis_nuisance = list(nuisance)
        return self.analysis

    def save(self, est, iteration):
        """The three tables and checkpoint.pkl (written to a temporary file first, so a crash never leaves half)."""
        pd.DataFrame(self.iterations).to_csv(os.path.join(self.dir, "iterations.csv"), index=False)
        pd.DataFrame(self.fixes).to_csv(os.path.join(self.dir, "fixing_log.csv"), index=False)
        pd.DataFrame(self.structural).to_csv(os.path.join(self.dir, "structural_per_iteration.csv"), index=False)
        x_init, u_init = est.primal_start()
        checkpoint = {"it": iteration, "free": list(est.free_params), "theta": est.full_theta(), "x_init": x_init,
                      "u_init": u_init, "b_init": est.primal_b_free(), "obj": float(est.obj_value),
                      "hist": self.iterations, "fixrows": self.fixes, "ms_done_for": self.multistart_done_for,
                      "srows": self.structural}
        path = os.path.join(self.dir, "checkpoint.pkl")
        with open(path + ".tmp", "wb") as fh:
            pickle.dump(checkpoint, fh)
        os.replace(path + ".tmp", path)

def write_free_set(red_dir, iteration, file_name, est, free):
    """iter_XX/<file_name>: status (free / fixed), anchored value and reaction of every parameter."""
    folder = os.path.join(red_dir, "iter_%02d" % iteration)
    os.makedirs(folder, exist_ok=True)
    write_free_table(os.path.join(folder, file_name), est, free)

def write_free_table(path, est, free):
    full = est.full_theta()
    free_set = set(free)
    status = ["free" if p in free_set else "fixed" for p in ALL_PARAMS]
    values = [float(full[p]) for p in ALL_PARAMS]
    reactions = [RXN_OF_PARAM[p] for p in ALL_PARAMS]
    table = pd.DataFrame({"status": status, "value": values, "reaction": reactions},
                         index=pd.Index(ALL_PARAMS, name="param"))
    table.to_csv(path)

def run_reduction(est, out_dir, params=None, resume=False):
    """
    Fix one parameter per iteration until the free set is identifiable or nothing can be fixed.  ``est`` holds the
    stage-0 anchor (``restore_anchor``); ``resume`` continues from reduction/checkpoint.pkl.  Writes
    ``<out_dir>/reduction/`` and returns (the final anchor, status).

    status: converged (full rank at eig_cut, no |r| > corr_threshold pair among the non-nuisance free parameters,
    none flagged, nothing unresolved, and the final structural check flags at most the nuisance parameters);
    stuck (no rule can fix anything although a criterion is violated; the violations are listed as unresolved and
    never reported as identifiable); structural_unresolved (a flagged structural group without a fixable member, or
    still flagged after max_structural_rounds rounds, or a final check that flags something the rules cannot
    explain); max_iter.
    """
    params = default_params("selection") if params is None else params
    red_dir = os.path.join(check_out_dir(out_dir), "reduction")
    os.makedirs(red_dir, exist_ok=True)
    checker = StructuralCheck(params, log)
    record = ReductionRecord(red_dir)
    iteration = 0
    stop = None
    unresolved = []
    nuisance = []
    if resume and os.path.exists(os.path.join(red_dir, "checkpoint.pkl")):
        est, iteration = resume_from_checkpoint(est, record, params)
    else:
        est, stop, unresolved = structural_prepass(est, checker, record, params)
    iterations_run = 0
    while stop is None and iteration < params["max_iter"]:
        iteration += 1
        iterations_run += 1
        est, outcome = run_iteration(est, iteration, checker, record, params)
        stop = outcome["stop"]
        unresolved = outcome["unresolved"]
        nuisance = outcome["nuisance"]
    status = stop or "max_iter"
    if status == "max_iter" and iteration > 0 and iterations_run == 0:
        # a run resumed at max_iter: record the free set of its last iteration again
        write_free_set(red_dir, iteration, "next_free_params.csv", est, est.free_params)
    status, final_check = final_structural_check(est, iteration, nuisance, status, unresolved, checker, record,
                                                 params)
    write_reduction_results(est, iteration, status, unresolved, final_check, record, params)
    return est, status

def resume_from_checkpoint(est, record, params):
    """
    Rebuild the anchor of reduction/checkpoint.pkl on a clone of ``est`` (same free set, anchored theta and primal
    point, tight warm start, then refine_anchor) and restore the tables.  A warm-started solve can drift in the flat
    valley, so the solved point is accepted only if obj <= checkpoint obj (1 + refit_rtol) + refit_atol; otherwise
    the checkpoint itself is kept as an anchor-only estimator.  Returns (estimator, iteration of the checkpoint).
    """
    with open(os.path.join(record.dir, "checkpoint.pkl"), "rb") as fh:
        checkpoint = pickle.load(fh)
    record.iterations = checkpoint["hist"]
    record.fixes = checkpoint["fixrows"]
    record.structural = checkpoint["srows"]
    record.multistart_done_for = checkpoint["ms_done_for"]
    record.log("restoring checkpoint at iteration %d (%d free)" % (checkpoint["it"], len(checkpoint["free"])))
    base_options = dict(est.solver_options)
    restored = est.clone(free_params=checkpoint["free"], theta_init=checkpoint["theta"],
                         x_init=checkpoint["x_init"], u_init=checkpoint["u_init"], b_init=checkpoint["b_init"],
                         solver_options={**base_options, **TIGHT_OPTIONS})
    limit = checkpoint["obj"] * (1.0 + params["refit_rtol"]) + params["refit_atol"]
    accepted = False
    try:
        restored.estimate()
        restored.refine_anchor(primary_options=dict(TIGHT_OPTIONS))
        accepted = restored.obj_value <= limit
        record.log("restored (tight warm start) obj %.5f (checkpoint %.5f) %s" % (
            restored.obj_value, checkpoint["obj"], "accepted" if accepted else "REJECTED (higher than the checkpoint)"))
    except Exception as exc:   # noqa: BLE001  (IPOPT not converging: the checkpoint is kept as it is)
        record.log("restore solve failed: %s" % str(exc)[:100])
    restored.solver_options = base_options
    if accepted:
        restored.refit_status = "restored"
        return restored, checkpoint["it"]
    kept = est.clone(free_params=checkpoint["free"], theta_init=checkpoint["theta"], x_init=checkpoint["x_init"],
                     u_init=checkpoint["u_init"], b_init=checkpoint["b_init"], solver_options=base_options)
    kept.anchor_only(checkpoint["x_init"], checkpoint["u_init"], checkpoint["b_init"], checkpoint["theta"],
                     checkpoint["obj"])
    kept.refit_status = "kept_anchor"
    record.log("kept the checkpoint as an anchor-only estimator, obj %.5f" % checkpoint["obj"])
    return kept, checkpoint["it"]

def structural_prepass(est, checker, record, params):
    """
    Iteration 0: the structural check at the stage-0 anchor with its fixes and refits.  Returns (estimator, stop,
    unresolved) with stop "structural_unresolved" or None.
    """
    started = time.time()
    record.log("iteration 0 (structural check at the initial anchor): %d free, obj %.5f" % (
        len(est.free_params), est.obj_value))
    write_free_set(record.dir, 0, "free_params.csv", est, est.free_params)
    est, phase = structural_phase(est, 0, checker, record, params)
    row = {"iter": 0, "n_free": phase["n_free_start"], "obj": phase["obj_start"]}
    if phase["analysis_at_start"] is not None:
        row.update(analysis_columns(phase["analysis_at_start"]))
    row.update({"fixed_this_iter": ";".join(phase["fixed"]),
                "reason": ";".join("%s:structural" % p for p in phase["fixed"]),
                "unresolved": " | ".join(phase["unresolved"]), "obj_after": float(est.obj_value), "multistart": "",
                "wall_s": time.time() - started})
    row.update(structural_columns(phase))
    record.iterations.append(row)
    record.save(est, 0)
    write_free_set(record.dir, 0, "next_free_params.csv", est, est.free_params)
    if phase["status"] != "clean":
        return est, "structural_unresolved", list(phase["unresolved"])
    return est, None, []

def analysis_columns(analysis):
    """The practical-analysis columns of the iteration-0 row (the analysis at the stage-0 anchor)."""
    return {"rank_1e-6": analysis["ranks"][1e-6], "rank_1e-8": analysis["ranks"][1e-8],
            "rank_1e-10": analysis["ranks"][1e-10], "cond": analysis["cond"], "sigma_ratio": analysis["sigma_ratio"],
            "sigma_ratio_nz": analysis["sigma_ratio_nz"], "n_pairs_gt": len(analysis["pairs"]),
            "n_zero": len(analysis["zero"]), "max_abs_r": max_abs_r(analysis)}

def max_abs_r(analysis):
    """Largest |r| off the diagonal of the correlation matrix of the analysis (NaN without parameters)."""
    corr = analysis["corr"].to_numpy()
    off_diagonal = np.abs(corr - np.diag(np.diag(corr)))
    if off_diagonal.size:
        return float(off_diagonal.max())
    return float("nan")

def structural_phase(est, iteration, checker, record, params):
    """
    The structural check(s) at the start of an iteration: while the check flags something fixable, fix it, refit
    and check again (at most max_structural_rounds rounds).  Returns (estimator, phase) with phase = {status
    (clean / unresolved), n_fixed, fixed, rounds, obj_start, n_free_start, first_row, last_row, unresolved,
    analysis_at_start (the practical analysis at the first anchor if the check flagged it), nuisance,
    output_level, refit_status}.
    """
    phase = {"status": "clean", "n_fixed": 0, "fixed": [], "rounds": 0, "obj_start": float(est.obj_value),
             "n_free_start": len(est.free_params), "first_row": None, "last_row": None, "unresolved": [],
             "analysis_at_start": None, "nuisance": [], "output_level": None, "refit_status": ""}
    round_number = 0
    while True:
        check = checker.run(est, iteration, round_number)
        if check["flagged"]:
            # the fix plan keeps the highest-influence member of every null group free
            analysis = record.practical_analysis(est, [], params)
            plan_structural_fixes(check, analysis["infl"], params)
            if round_number == 0:
                phase["analysis_at_start"] = analysis
        record_check(record, check, iteration)
        if phase["first_row"] is None:
            phase["first_row"] = check["row"]
        phase["last_row"] = check["row"]
        phase["output_level"] = check["output_level"]
        phase["nuisance"] = list(check["kept_free_by_rule"])
        record.log(structural_line(check["row"], ""))
        if check["kept_free_by_rule"]:
            record.log("  kept_free_by_rule (nuisance in the practical criteria): %s" % check["kept_free_by_rule"])
        if not check["flagged"]:
            return est, phase
        if not check["fixes"] and check["kept_free_by_rule"] and not check["unresolved"]:
            # the remaining nulls only load on parameters the protections keep free
            return est, phase
        if not check["fixes"] or round_number >= params["max_structural_rounds"]:
            phase["status"] = "unresolved"
            phase["unresolved"] = check["unresolved"] or ["structural: no fixable parameter in the flagged groups"]
            if round_number >= params["max_structural_rounds"]:
                phase["unresolved"].append("structural: still flagged after %d rounds"
                                           % params["max_structural_rounds"])
            for message in phase["unresolved"]:
                record.log("  unresolved: %s" % message)
            return est, phase
        est = apply_structural_fixes(est, iteration, round_number, check, record, params)
        fixed = [fix["param"] for fix in check["fixes"]]
        phase["n_fixed"] += len(fixed)
        phase["fixed"].extend(fixed)
        phase["rounds"] += 1
        phase["refit_status"] = est.refit_status
        round_number += 1
        record.save(est, iteration)

def record_check(record, check, iteration):
    """The row of structural_per_iteration.csv and the two spectra of one check (structural_spectra/)."""
    record.structural.append(check["row"])
    folder = os.path.join(record.dir, "structural_spectra")
    os.makedirs(folder, exist_ok=True)
    name = "it%s_r%s" % (str(iteration).zfill(2), check["row"]["round"])
    check["spectra"].to_csv(os.path.join(folder, name + "_anchored.csv"))
    check["spectra_perturbed"].to_csv(os.path.join(folder, name + "_perturbed.csv"))

def structural_line(row, tag):
    groups = ("; groups: " + row["groups"]) if row["groups"] else ""
    return ("  structural%s it %s round %s: %d free, sigma_min/max anchor %.2e, min over %d runs %.2e, gap to next "
            "%.1f decades, runs below cut %d, null dim %d (perturbed states %d)%s" % (
                tag, row["iter"], row["round"], row["n_free"], row["sigma_ratio_anchor"], row["n_runs"],
                row["sigma_ratio_min_over_runs"], row["gap_decades_to_next_anchor"],
                row["n_runs_with_ratio_below_cut"], row["null_dim_consensus"], row["null_dim_perturbed_states"],
                groups))

def apply_structural_fixes(est, iteration, round_number, check, record, params):
    """Log the planned fixes of a check, refit with them held at their anchored values and return the refit."""
    free_before = list(est.free_params)
    for fix in check["fixes"]:
        rule = STRUCTURAL_RULE[fix["cls"]]
        trigger = "sigma_ratio<%.0e in all %d runs (dim %d, %s)" % (check["row"]["sv_cut"], check["row"]["n_runs"],
                                                                    fix["null_dim"], fix["vectors"])
        record.fixes.append({"param": fix["param"], "iter": iteration, "round": round_number, "stage": "structural",
                             "rule": rule, "partner": fix["kept"], "group": fix["members"], "r": np.nan,
                             "influence_param": fix["influence"], "influence_partner": np.nan,
                             "value_fixed": fix["value"], "svd_loading": np.nan, "trigger": trigger,
                             "influence_tie_within_25pct": fix["tied"], "output_class_at_fix": fix["output_class"]})
        record.log("  STRUCTURAL FIX %-10s %-16s keep %s (group %s) value=%.4g infl=%.3g" % (
            fix["param"], rule, fix["kept"], fix["members"], fix["value"], fix["influence"]))
    for message in check["unresolved"]:
        record.log("  unresolved: %s" % message)
    fixed = [fix["param"] for fix in check["fixes"]]
    new_free = [p for p in free_before if p not in fixed]
    obj_before = float(est.obj_value)
    est = refit_with_fixed(est, new_free, params, record.log)
    record.log("  structural refit: %d free, obj %.5f (before %.5f), refit_status %s" % (
        len(est.free_params), est.obj_value, obj_before, est.refit_status))
    return est

def structural_columns(phase):
    """The structural columns of a row of iterations.csv (first and last check of the iteration's phase)."""
    first = phase["first_row"]
    last = phase["last_row"]
    return {"n_free_start": phase["n_free_start"], "obj_start": phase["obj_start"],
            "struct_rounds": phase["rounds"], "struct_n_flagged_params_first_check": first["n_flagged_params"],
            "struct_groups_first_check": first["groups"], "struct_fixed_this_iter": ";".join(phase["fixed"]),
            "struct_sigma_ratio_anchor": last["sigma_ratio_anchor"],
            "struct_sigma_ratio_min_over_runs": last["sigma_ratio_min_over_runs"],
            "struct_gap_decades_to_next": last["gap_decades_to_next_anchor"],
            "struct_null_dim_perturbed": last["null_dim_perturbed_states"],
            "struct_out_null_dim": last["out_null_dim"],
            "struct_out_null_dim_active_set": last["out_null_dim_active_set"],
            "struct_out_groups": last["out_groups"],
            "struct_kept_free_by_rule": last["kept_free_by_rule"],
            "struct_refit_status": phase["refit_status"]}

def run_iteration(est, iteration, checker, record, params):
    """
    One iteration: the structural phase, the practical analysis, at most one practical fix and its refit.  Returns
    (estimator, outcome) with outcome = {stop: None while the loop goes on, else the status; unresolved; nuisance}.
    """
    started = time.time()
    write_free_set(record.dir, iteration, "free_params.csv", est, est.free_params)
    record.log("iteration %d: structural check at the start (%d free, obj %.5f)" % (
        iteration, len(est.free_params), est.obj_value))
    est, phase = structural_phase(est, iteration, checker, record, params)
    if phase["status"] != "clean":
        stop_after_structural_phase(est, iteration, phase, record, started)
        return est, {"stop": "structural_unresolved", "unresolved": list(phase["unresolved"]),
                     "nuisance": list(phase["nuisance"])}
    free = list(est.free_params)
    nuisance = list(phase["nuisance"])
    record.log("iteration %d: %d free, obj %.5f" % (iteration, len(free), est.obj_value))
    analysis = record.practical_analysis(est, nuisance, params)
    record.log(analysis_line(analysis, nuisance, params))
    fixes, unresolved = choose_practical_fix(analysis, free, iteration, params)
    fixed = [fix["param"] for fix in fixes]
    row = iteration_row(iteration, est, free, analysis, nuisance, fixed, fixes, unresolved, params)
    row["max_abs_r"] = max_abs_r(analysis)
    row.update(structural_columns(phase))
    write_iteration_files(record.dir, iteration, est, free, analysis, nuisance)
    write_free_set(record.dir, iteration, "next_free_params.csv", est, [p for p in free if p not in fixed])
    outcome = {"stop": None, "unresolved": unresolved, "nuisance": nuisance}
    if fixes:
        est = refit_after_fix(est, iteration, free, fixes, unresolved, row, record, params)
        row["wall_s"] = time.time() - started
        record.iterations.append(row)
        record.fixes.extend(fixes)
        record.save(est, iteration)
        return est, outcome
    # nothing fixable: one last multistart around this free set before the loop stops
    if record.multistart_done_for != tuple(free):
        record.multistart_done_for = tuple(free)
        candidate, _table = multistart_candidate(est, params, 1000 + iteration, record.log)
        if candidate is not None and candidate.obj_value < est.obj_value - params["end_multistart_rtol"] * abs(
                est.obj_value):
            record.log("  end multistart improved obj %.5f -> %.5f; re-analysing" % (est.obj_value,
                                                                                      candidate.obj_value))
            row["obj_after"] = float(candidate.obj_value)
            row["multistart"] = "end_improved"
            row["unresolved"] = " | ".join(unresolved)
            row["wall_s"] = time.time() - started
            record.iterations.append(row)
            record.save(candidate, iteration)
            return candidate, outcome
        row["multistart"] = "end_kept"
    row["obj_after"] = float(est.obj_value)
    row["wall_s"] = time.time() - started
    record.iterations.append(row)
    outcome["stop"] = stopping_status(analysis, unresolved)
    record.log("  STOP: %s; unresolved: %s" % (outcome["stop"], unresolved))
    return est, outcome

def stop_after_structural_phase(est, iteration, phase, record, started):
    """The row of an iteration whose structural phase left a group unresolved (the loop stops there)."""
    unresolved = list(phase["unresolved"])
    row = {"iter": iteration, "n_free": len(est.free_params), "obj": float(est.obj_value),
           "fixed_this_iter": ";".join(phase["fixed"]),
           "reason": ";".join("%s:structural" % p for p in phase["fixed"]),
           "unresolved": " | ".join(unresolved), "obj_after": float(est.obj_value), "wall_s": time.time() - started}
    row.update(structural_columns(phase))
    record.iterations.append(row)
    write_free_set(record.dir, iteration, "next_free_params.csv", est, est.free_params)
    record.save(est, iteration)
    record.log("  STOP: %s; unresolved: %s" % ("structural_unresolved", unresolved))

def analysis_line(analysis, nuisance, params):
    eig_cut = params["eig_cut"]
    report = analysis["report"]
    note = "" if analysis["full_rank"] else " (not used: rank deficient)"
    return ("  eig cut %.0e (singular ratio %.0e): rank %d/%d (%d nuisance), sigma_min/max %.2e, %d pairs |r|>%.2f%s, "
            "%d structural zeros, %d flagged %s; sens: failed %d, re-solved %d, one-sided %d, inconsistent %d, "
            "NaN %d" % (eig_cut, np.sqrt(eig_cut), analysis["ranks"][eig_cut], len(analysis["fa"]), len(nuisance),
                        analysis["sigma_ratio"], len(analysis["pairs"]), params["corr_threshold"], note,
                        len(analysis["zero"]), len(analysis["flagged"]), analysis["flagged"],
                        report["n_failed_first_pass"], report["n_recovered_by_retry"], report["n_one_sided"],
                        report["n_inconsistent"], report["n_nan_entries"]))

def practical_fix_row(analysis, name, iteration, rule, partner="", r=np.nan, loading=np.nan, trigger=""):
    """The row of fixing_log.csv of a practical fix."""
    influence = analysis["infl"]
    influence_partner = float(influence[partner]) if partner in influence.index else np.nan
    return {"param": name, "iter": iteration, "rule": rule, "partner": partner, "r": r,
            "influence_param": float(influence[name]), "influence_partner": influence_partner,
            "value_fixed": float(analysis["theta"][name]), "svd_loading": loading, "trigger": trigger,
            "round": np.nan, "stage": "practical", "group": ""}

def choose_practical_fix(analysis, free, iteration, params):
    """
    The practical rules in order; the first one that fixes something acts.  R0 fixes every structurally zero
    parameter (a batch: the FIM does not change), R1-R3 fix one parameter.  Returns (rows of fixing_log.csv,
    unresolved violations).
    """
    free_now = set(free)
    unresolved = []
    fixes = zero_influence_fixes(analysis, free_now, iteration, unresolved)
    if not fixes and not analysis["full_rank"] and rank_deficient_on_nonzero(analysis, params["eig_cut"]):
        fixes = svd_fix(analysis, free_now, iteration, unresolved, params)
    if not fixes and analysis["full_rank"]:
        fixes = correlation_fix(analysis, free_now, iteration, unresolved)
    if not fixes:
        fixes = flag_fix(analysis, free_now, iteration, unresolved)
    return fixes, unresolved

def zero_influence_fixes(analysis, free_now, iteration, unresolved):
    """R0: every parameter with influence below zero_cut times the largest, lowest influence first."""
    fixes = []
    for name in sorted(analysis["zero"], key=analysis["infl"].get):
        if can_fix(name, free_now):
            free_now.discard(name)
            fixes.append(practical_fix_row(analysis, name, iteration, "zero_influence"))
        else:
            unresolved.append("zero:%s (not fixable)" % name)
    return fixes

def rank_deficient_on_nonzero(analysis, eig_cut):
    """True if F_S restricted to the parameters with nonzero influence is rank deficient at the eigenvalue cut."""
    singular = analysis["s_nz"]
    rank = int(np.sum(singular ** 2 / singular[0] ** 2 > eig_cut))
    return rank < len(analysis["nz"])

def svd_fix(analysis, free_now, iteration, unresolved, params):
    """
    R1 (the FIM is rank deficient): fix the fixable parameter with the largest |loading| in the last right singular
    vector; loadings within svd_tie_tol of the top one are tied and the tied parameter with the lowest influence is
    fixed.  The vector is unresolved if the top fixable loading is below svd_min_loading.
    """
    influence = analysis["infl"]
    # the last right singular vector is the direction the data cannot see
    null_vector = analysis["Vt_nz"][-1]
    loadings = {}
    for name, loading in zip(analysis["nz"], null_vector):
        if can_fix(name, free_now):
            loadings[name] = loading
    top = max([abs(loading) for loading in loadings.values()], default=0.0)
    if top < params["svd_min_loading"]:
        largest = np.argsort(-np.abs(null_vector))[:4]
        listed = ", ".join("%s=%.2f" % (analysis["nz"][i], null_vector[i]) for i in largest)
        unresolved.append("svd: no fixable parameter in the null vector (%s)" % listed)
        return []
    tied = [name for name, loading in loadings.items() if abs(loading) >= (1.0 - params["svd_tie_tol"]) * top]
    param_to_fix = tied[0]
    for name in tied:
        if influence[name] < influence[param_to_fix]:
            param_to_fix = name
    partners = [name for name in tied if name != param_to_fix]
    free_now.discard(param_to_fix)
    return [practical_fix_row(analysis, param_to_fix, iteration, "svd", partner=";".join(partners),
                              loading=float(loadings[param_to_fix]),
                              trigger="rank_deficient@%.0e" % params["eig_cut"])]

def correlation_fix(analysis, free_now, iteration, unresolved):
    """
    R2 (full rank only: the pseudo-inverse of a rank-deficient FIM gives meaningless correlations): in the pair with
    the largest |r| that has a fixable member, fix the member with the lower influence (the other one if that is
    protected).
    """
    influence = analysis["infl"]
    for first, second, r in analysis["pairs"]:
        if influence[first] <= influence[second]:
            lower, higher = first, second
        else:
            lower, higher = second, first
        if can_fix(lower, free_now):
            free_now.discard(lower)
            return [practical_fix_row(analysis, lower, iteration, "corr", partner=higher, r=r)]
        if can_fix(higher, free_now):
            free_now.discard(higher)
            return [practical_fix_row(analysis, higher, iteration, "corr", partner=lower, r=r)]
        unresolved.append("pair:%s~%s r=%.3f (neither fixable)" % (first, second, r))
    return []

def flag_fix(analysis, free_now, iteration, unresolved):
    """
    R3: a free non-nuisance parameter flagged at_bound / fd_unstable / fd_failed carries no trustworthy information;
    the fixable one with the lowest influence is fixed, the others are listed as unresolved.
    """
    fixes = []
    for name in sorted(analysis["flagged"], key=analysis["infl"].get):
        if can_fix(name, free_now):
            free_now.discard(name)
            trigger = ";".join(analysis["flags"][name])
            fixes.append(practical_fix_row(analysis, name, iteration, "flag", trigger=trigger))
            break
    fixed = [fix["param"] for fix in fixes]
    for name in analysis["flagged"]:
        if name in fixed:
            continue
        note = "" if can_fix(name, free_now) else " (not fixable)"
        unresolved.append("flagged:%s [%s]%s" % (name, ";".join(analysis["flags"][name]), note))
    return fixes

def iteration_row(iteration, est, free, analysis, nuisance, fixed, fixes, unresolved, params):
    """The row of iterations.csv of a practical iteration (before the refit)."""
    eig_cut = params["eig_cut"]
    report = analysis["report"]
    return {"iter": iteration, "n_free": len(free), "obj": float(est.obj_value), "obj_before": float(est.obj_value),
            "s2": float(misfit_s2(est)), "eig_cut": float(eig_cut),
            "rank_1e-6": analysis["ranks"][1e-6], "rank_1e-8": analysis["ranks"][1e-8],
            "rank_1e-10": analysis["ranks"][1e-10], "rank_at_cut": analysis["ranks"][eig_cut],
            "n_analysed": len(analysis["fa"]), "cond": analysis["cond"], "sigma_ratio": analysis["sigma_ratio"],
            "sigma_ratio_nz": analysis["sigma_ratio_nz"],
            "n_pairs_gt": len(analysis["pairs"]) if analysis["full_rank"] else 0, "n_zero": len(analysis["zero"]),
            "n_nuisance": len(nuisance), "nuisance": ";".join(nuisance),
            "n_flagged": len(analysis["flagged"]), "flagged": ";".join(analysis["flagged"]),
            "fixed_this_iter": ";".join(fixed),
            "reason": ";".join("%s:%s" % (fix["param"], fix["rule"]) for fix in fixes),
            "unresolved": " | ".join(unresolved),
            "sens_failed_first": report["n_failed_first_pass"], "sens_retry_ok": report["n_recovered_by_retry"],
            "sens_one_sided": report["n_one_sided"], "sens_inconsistent": report["n_inconsistent"],
            "sens_nan": report["n_nan_entries"], "refit_status": ""}

def write_iteration_files(red_dir, iteration, est, free, analysis, nuisance):
    """iter_XX/: theta_full.csv, influence.csv, spectrum.csv, corr_pairs.csv, reduced_sensitivity_params.csv."""
    folder = os.path.join(red_dir, "iter_%02d" % iteration)
    os.makedirs(folder, exist_ok=True)
    full = est.full_theta()
    theta = pd.DataFrame({"value": pd.Series({p: float(full[p]) for p in ALL_PARAMS}),
                          "free": [p in free for p in ALL_PARAMS]})
    theta.rename_axis("param").to_csv(os.path.join(folder, "theta_full.csv"))
    names = analysis["infl"].index
    influence = pd.DataFrame({"influence": analysis["infl"], "influence_dlny": analysis["infl_dlny"],
                              "influence_schur": analysis["infl_S"].reindex(names),
                              "kept_free_by_rule": [p in nuisance for p in names],
                              "flags": [";".join(analysis["flags"][p]) for p in names]})
    influence.rename_axis("param").to_csv(os.path.join(folder, "influence.csv"))
    spectrum = pd.DataFrame({"sigma": analysis["s"], "sigma_ratio": analysis["s"] / analysis["s"][0]})
    spectrum.rename_axis("k").to_csv(os.path.join(folder, "spectrum.csv"))
    pd.DataFrame(analysis["pairs"], columns=["p1", "p2", "r"]).to_csv(os.path.join(folder, "corr_pairs.csv"),
                                                                      index=False)
    analysis["report"]["param_table"].to_csv(os.path.join(folder, "reduced_sensitivity_params.csv"))

def refit_after_fix(est, iteration, free, fixes, unresolved, row, record, params):
    """Log the fix, refit with the fixed parameter held at its anchored value; every multistart_every iterations
    a multistart around the new structure replaces the refit if it is strictly better."""
    for fix in fixes:
        r_text = "%.3f" % fix["r"] if np.isfinite(fix["r"]) else "-"
        record.log("  FIX %-10s rule=%-4s partner=%s r=%s value=%.4g infl=%.3g" % (
            fix["param"], fix["rule"], fix["partner"], r_text, fix["value_fixed"], fix["influence_param"]))
    for message in unresolved:
        record.log("  unresolved: %s" % message)
    fixed = [fix["param"] for fix in fixes]
    new_free = [p for p in free if p not in fixed]
    est = refit_with_fixed(est, new_free, params, record.log)
    row["obj_after"] = float(est.obj_value)
    row["refit_status"] = est.refit_status
    row["multistart"] = ""
    record.log("  refit: %d free, obj_before %.5f -> obj_after %.5f (refit_status %s)" % (
        len(est.free_params), row["obj_before"], est.obj_value, row["refit_status"]))
    if params["multistart_every"] and iteration % params["multistart_every"] == 0:
        candidate, _table = multistart_candidate(est, params, iteration, record.log)
        if candidate is not None and candidate.obj_value < est.obj_value:
            record.log("  multistart improved obj %.5f -> %.5f" % (est.obj_value, candidate.obj_value))
            est = candidate
            row["obj_after"] = float(est.obj_value)
            row["multistart"] = "improved"
        else:
            row["multistart"] = "kept"
    return est

def stopping_status(analysis, unresolved):
    """
    converged if the FIM is full rank, no correlated pair, no flag and nothing unresolved; else stuck, with the
    violations added to ``unresolved`` (they are never reported as identifiable).
    """
    if analysis["full_rank"] and not analysis["pairs"] and not analysis["flagged"] and not unresolved:
        return "converged"
    if not analysis["full_rank"]:
        unresolved.append("rank %d < %d analysed at eigenvalue cut %.0e" % (
            analysis["ranks"][analysis["eig_cut"]], len(analysis["fa"]), analysis["eig_cut"]))
    else:
        for first, second, r in analysis["pairs"]:
            tag = "pair:%s~%s" % (first, second)
            if not any(message.startswith(tag) for message in unresolved):
                unresolved.append("%s r=%.3f remains" % (tag, r))
    return "stuck"

def final_structural_check(est, iteration, nuisance, status, unresolved, checker, record, params):
    """
    The structural check at the final anchor (it never fixes anything).  A flag that the protected parameters do not
    explain turns a converged run into structural_unresolved.  Returns (status, check).
    """
    check = checker.run(est, iteration, "final")
    if check["flagged"]:
        analysis = record.practical_analysis(est, nuisance, params)
        plan_structural_fixes(check, analysis["infl"], params)
    record_check(record, check, "final")
    record.log(structural_line(check["row"], " FINAL"))
    explained = bool(check["kept_free_by_rule"]) and not check["unresolved"] and not check["fixes"]
    if check["flagged"] and not explained:
        unresolved.append("final structural check flags: %s" % check["row"]["groups"])
        if status == "converged":
            status = "structural_unresolved"
    return status, check

def write_reduction_results(est, iteration, status, unresolved, final_check, record, params):
    """
    Final tables (iterations.csv, fixing_log.csv, final_structure.csv, final_correlation.csv,
    final_singular_spectrum.csv, final_reduced_sensitivity_params.csv, final_free_params.csv), the checkpoint, the
    final anchor (final_anchor_state.pkl, final_theta_full.csv) and reduction_summary.json.
    """
    nuisance = list(final_check["kept_free_by_rule"])
    analysis = record.practical_analysis(est, nuisance, params)
    free = list(est.free_params)
    fix_log = pd.DataFrame(record.fixes)
    pd.DataFrame(record.iterations).to_csv(os.path.join(record.dir, "iterations.csv"), index=False)
    fix_log.to_csv(os.path.join(record.dir, "fixing_log.csv"), index=False)
    write_final_structure(est, analysis, fix_log, record.dir, final_check["output_level"], nuisance)
    write_free_table(os.path.join(record.dir, "final_free_params.csv"), est, free)
    record.save(est, iteration)
    wall_s = time.time() - record.started
    log("reduction finished: status %s, %d free, wall %.0fs" % (status, len(free), wall_s))
    summary = {"status": status, "identifiable": status == "converged", "n_free": len(free), "free": free,
               "fixed": [p for p in ALL_PARAMS if p not in free], "unresolved": unresolved,
               "kept_free_by_rule": nuisance, "wall_s": wall_s, "last_refit_status": est.refit_status,
               "obj_final_anchor": float(est.obj_value)}
    save_anchor(est, record.dir, extra={"status": status})
    os.replace(os.path.join(record.dir, "anchor_state.pkl"), os.path.join(record.dir, "final_anchor_state.pkl"))
    os.replace(os.path.join(record.dir, "theta_full.csv"), os.path.join(record.dir, "final_theta_full.csv"))
    with open(os.path.join(record.dir, "reduction_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    log("saved final anchor: %d free, obj %.6f, status %s, kept_free_by_rule %s" % (
        len(free), est.obj_value, status, nuisance))

def write_final_structure(est, analysis, fix_log, out_dir, output_level, nuisance):
    """
    final_structure.csv (all 37 parameters), final_correlation.csv, final_singular_spectrum.csv and
    final_reduced_sensitivity_params.csv.  Class of a free parameter (``ident_class``): "kept_free_by_rule"
    (nuisance; its CV is the conditional, bound-driven one, cv_basis "nuisance_conditional"), "flagged" (at_bound /
    fd_unstable / fd_failed: never reported identifiable), else the output-level class (identifiable /
    bound_identified / non_identifiable).  A fixed parameter gets the iteration and the rule that fixed it.
    """
    full = est.full_theta()
    free = list(analysis["fp"])
    variances = np.diag(analysis["cov_abs"].to_numpy())
    sd = np.sqrt(np.where(variances > 0, variances, np.nan))
    s2 = misfit_s2(est)
    rows = []
    for p in ALL_PARAMS:
        value = float(full[p])
        row = {"param": p, "status": "free" if p in free else "fixed", "value": value}
        if p in free:
            i = free.index(p)
            row.update(free_parameter_columns(p, value, sd[i], s2, analysis, output_level["classes"], nuisance))
        else:
            fixed_rows = fix_log[fix_log["param"] == p] if len(fix_log) else fix_log
            if len(fixed_rows):
                last = fixed_rows.iloc[-1]
                row.update({"influence": float(last["influence_param"]), "fixed_iter": int(last["iter"]),
                            "fixed_rule": last["rule"]})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "final_structure.csv"), index=False)
    analysis["corr"].to_csv(os.path.join(out_dir, "final_correlation.csv"))
    singular = analysis["s"]
    spectrum = pd.DataFrame({"sigma": singular, "sigma_ratio": singular / singular[0], "eig_F_rel": singular ** 2,
                             "eig_ratio": singular ** 2 / singular[0] ** 2})
    spectrum.rename_axis("k").to_csv(os.path.join(out_dir, "final_singular_spectrum.csv"))
    analysis["report"]["param_table"].to_csv(os.path.join(out_dir, "final_reduced_sensitivity_params.csv"))

def free_parameter_columns(p, value, sd, s2, analysis, classes, nuisance):
    flags = analysis["flags"].get(p, [])
    if p in nuisance:
        ident_class = "kept_free_by_rule"
    elif flags:
        ident_class = "flagged"
    else:
        ident_class = classes.get(p, "identifiable_by_fim")
    return {"influence": float(analysis["infl"][p]), "influence_dlny": float(analysis["infl_dlny"][p]),
            "ident_class": ident_class, "output_class": classes.get(p, ""), "flags": ";".join(flags),
            "cv_basis": "nuisance_conditional" if p in nuisance else "schur",
            "sd": sd, "ci_low": value - 1.96 * sd, "ci_high": value + 1.96 * sd, "cv_percent": 100.0 * sd / abs(value),
            "sd_s2scaled": sd * np.sqrt(s2), "cv_percent_s2scaled": 100.0 * sd * np.sqrt(s2) / abs(value)}

def run_stage0(out_dir, params=None, conditions=None):
    """
    Stage 0 of the selection, all 37 parameters free.  Candidate A: the saved anchor of params["stage0_warm_start"]
    (its anchor_state.pkl) refitted under the current bounds and refined with the tight options.  Candidate B: the
    best start of a recorded multistart (params["n_starts_stage0"] starts, warm start included) re-run here and
    refined.  The better one is the stage-0 anchor.  Without the saved anchor (a fresh clone) candidate A is skipped
    and the multistart is warm started at the literature theta.  The names "b2sd_primal" / obj_b2sd_primal of
    candidate A are historical (the saved anchor is the stage-0 anchor of the v2 b2sd run).  Writes
    ``<out_dir>/stage0/`` and returns its path.
    """
    params = default_params("stage0") if params is None else params
    out_dir = check_out_dir(out_dir)
    check_bounds()
    stage0_dir = os.path.join(out_dir, "stage0")
    os.makedirs(stage0_dir, exist_ok=True)
    conditions = available_conditions() if conditions is None else list(conditions)
    warm_start_dir = os.path.join(REPO_ROOT, params["stage0_warm_start"])
    saved = None
    if os.path.exists(os.path.join(warm_start_dir, "anchor_state.pkl")):
        saved = read_anchor(warm_start_dir)
        theta_start = {p: float(saved["theta"][p]) for p in ALL_PARAMS}
        log("stage0: %d conditions, warm start from %s (obj %.4f under its own bounds)" % (
            len(conditions), warm_start_dir, saved["obj"]))
    else:
        theta_start = literature_theta_in_bounds()
        log("stage0: %d conditions; warm start file %s not found: candidate A (saved anchor) skipped, the multistart "
            "is warm started at the literature theta" % (len(conditions), os.path.join(warm_start_dir,
                                                                                       "anchor_state.pkl")))
    candidates = {}
    if saved is not None:
        candidates["b2sd_primal"] = refined_saved_anchor(conditions, theta_start, saved)
    multistart = stage0_multistart(conditions, theta_start, params, stage0_dir)
    candidates["multistart_best"] = multistart["estimator"]
    used = "multistart_best"
    if "b2sd_primal" in candidates and candidates["b2sd_primal"].obj_value <= candidates["multistart_best"].obj_value:
        used = "b2sd_primal"
    est = candidates[used]
    est.solver_options = dict(SOLVER_OPTIONS)
    log("using candidate %s: obj %.5f" % (used, est.obj_value))
    write_stage0_results(est, used, candidates, multistart, stage0_dir, params)
    return stage0_dir

def literature_theta_in_bounds():
    theta = {}
    for p in ALL_PARAMS:
        theta[p] = float(min(max(LITERATURE_THETA[p], THETA_BOUNDS[p][0]), THETA_BOUNDS[p][1]))
    return theta

def refined_saved_anchor(conditions, theta_start, saved):
    """Candidate A: the saved anchor (theta and primal point, clipped into the current bounds) refitted, refined."""
    est = make_est(conditions, theta_start, ALL_PARAMS, x_init=saved["x_init"], u_init=saved["u_init"],
                   b_init=saved["b_init"])
    est.estimate()
    log("candidate A (saved anchor primal, clipped): single fit obj %.5f" % est.obj_value)
    est.refine_anchor(primary_options=dict(TIGHT_OPTIONS))
    log("candidate A refined obj %.5f" % est.obj_value)
    return est

def stage0_multistart(conditions, theta_start, params, stage0_dir):
    """Candidate B: the recorded multistart (multistart_recorded.csv) and its best start re-run here and refined."""
    started = time.time()
    table = multistart_recorded(make_est(conditions, theta_start, ALL_PARAMS), params["n_starts_stage0"],
                                params["stage0_seed"], params["n_jobs"])
    table.to_csv(os.path.join(stage0_dir, "multistart_recorded.csv"), index=False)
    converged = table[table["converged"]]
    warm_start_obj = float(table.loc[table["is_warm_start"], "obj_value"].min())
    log("multistart done in %.0fs: %d/%d converged, best obj %.5f, warm start %.5f, median %.5f" % (
        time.time() - started, len(converged), len(table), converged["obj_value"].min(), warm_start_obj,
        converged["obj_value"].median()))
    best = converged.iloc[0]
    theta = dict(theta_start)
    for p in ALL_PARAMS:
        theta[p] = float(best["init_" + p])
    est = make_est(conditions, theta, ALL_PARAMS)
    est.estimate()
    log("candidate B (best multistart start re-run in-process): obj %.5f (table %.5f)" % (
        est.obj_value, float(best["obj_value"])))
    est.refine_anchor(primary_options=dict(TIGHT_OPTIONS))
    log("candidate B refined obj %.5f" % est.obj_value)
    return {"estimator": est, "table": table, "converged": converged, "best": best, "warm_start_obj": warm_start_obj}

def write_stage0_results(est, used, candidates, multistart, stage0_dir, params):
    """anchor_state.pkl, theta_full.csv, objective_breakdown.csv, b_fit_vs_data.csv, pred_insample_*.csv and
    stage_row.csv / .json of the stage-0 anchor."""
    save_anchor(est, stage0_dir, extra={"used": used})
    breakdown = est.objective_breakdown()
    breakdown.to_csv(os.path.join(stage0_dir, "objective_breakdown.csv"))
    drains = b_vs_data(est)
    drains.to_csv(os.path.join(stage0_dir, "b_fit_vs_data.csv"), index=False)
    prediction = est.predict_fitted()
    prediction.predicted.to_csv(os.path.join(stage0_dir, "pred_insample_predicted.csv"))
    prediction.real.to_csv(os.path.join(stage0_dir, "pred_insample_real.csv"))
    metrics = pred_metrics(prediction.real, prediction.predicted)
    converged = multistart["converged"]
    at_bounds = theta_at_bounds(est, params["theta_at_bound_tol"])
    obj_saved = float(candidates["b2sd_primal"].obj_value) if "b2sd_primal" in candidates else float("nan")
    row = {"n_free": len(est.free_params), "obj": float(est.obj_value), "s2": float(misfit_s2(est)),
           "candidate_used": used, "obj_b2sd_primal": obj_saved,
           "obj_multistart_best": float(candidates["multistart_best"].obj_value),
           "multistart_best_table": float(multistart["best"]["obj_value"]),
           "multistart_warm_start": multistart["warm_start_obj"],
           "multistart_n_converged": int(len(converged)), "multistart_n_total": int(len(multistart["table"])),
           "multistart_median": float(converged["obj_value"].median()),
           "n_within_0.01_of_best": int((converged["obj_value"] <= converged["obj_value"].min() * 1.01).sum()),
           "b_rmse_vs_data": float(np.sqrt(np.mean(drains["err"] ** 2))),
           "n_drains_at_bound": int(drains["at_bound"].sum()),
           "n_theta_at_bounds": len(at_bounds), "theta_at_bounds": ";".join(at_bounds),
           "n_pred_failed": int(prediction.n_failed)}
    row.update(metrics)
    pd.DataFrame([row]).to_csv(os.path.join(stage0_dir, "stage_row.csv"), index=False)
    with open(os.path.join(stage0_dir, "stage_row.json"), "w") as fh:
        json.dump(row, fh, indent=1, default=str)
    log("stage0 saved: obj %.5f (mean breakdown %.5f) s2 %.2f; rmse_met %.4f (baseline %.4f); LI flux rmse %.4f "
        "(baseline %.4f); %d drain values at bound; %d theta at bounds" % (
            est.obj_value, breakdown.loc["MEAN", "total"], row["s2"], metrics["rmse_met"],
            metrics["rmse_met_meanbaseline"], metrics["rmse_flux_li"], metrics["rmse_flux_li_meanbaseline"],
            row["n_drains_at_bound"], row["n_theta_at_bounds"]))

def copy_anchor_files(from_anchor, stage0_dir):
    """
    The directory with the anchor of --from_anchor (``from_anchor`` itself, or its stage0/ for a run directory), and
    a copy of every file of it into ``stage0_dir`` (the figure notebooks read them there).  A file that already
    exists in ``stage0_dir`` is never overwritten.  Returns the anchor directory.
    """
    anchor_dir = os.path.abspath(from_anchor)
    if not os.path.exists(os.path.join(anchor_dir, "anchor_state.pkl")):
        if os.path.exists(os.path.join(anchor_dir, "stage0", "anchor_state.pkl")):
            anchor_dir = os.path.join(anchor_dir, "stage0")
    if not os.path.exists(os.path.join(anchor_dir, "anchor_state.pkl")):
        raise FileNotFoundError("no anchor_state.pkl in %s" % anchor_dir)
    if os.path.realpath(anchor_dir) == os.path.realpath(stage0_dir):
        return anchor_dir
    os.makedirs(stage0_dir, exist_ok=True)
    copied, kept = [], []
    for name in sorted(os.listdir(anchor_dir)):
        source = os.path.join(anchor_dir, name)
        target = os.path.join(stage0_dir, name)
        if not os.path.isfile(source):
            continue
        if os.path.exists(target):
            kept.append(name)
        else:
            shutil.copy(source, target)
            copied.append(name)
    log("selection: copied %d files of the anchor directory into %s (%d existing files not overwritten)" % (
        len(copied), stage0_dir, len(kept)))
    return anchor_dir

def run_selection(out_dir, params=None, stage0_params=None, from_anchor=None, resume=False):
    """
    The whole selection: stage 0 (skipped with ``from_anchor``, a directory with anchor_state.pkl or a run directory
    whose stage0/ has it, whose files are copied into <out_dir>/stage0; with ``resume`` an existing <out_dir>/stage0
    is reused), then the reduction from that anchor.  <out_dir>/settings_used is refreshed with the settings of this
    run.  Returns (the final anchor, status).
    """
    out_dir = check_out_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    check_bounds()
    snapshot_settings(out_dir)
    stage0_dir = os.path.join(out_dir, "stage0")
    if from_anchor:
        anchor_dir = copy_anchor_files(from_anchor, stage0_dir)
        log("selection: starting from the saved anchor %s (stage 0 skipped)" % anchor_dir)
    elif resume and os.path.exists(os.path.join(stage0_dir, "anchor_state.pkl")):
        anchor_dir = stage0_dir
        log("selection: resuming, reusing the stage 0 anchor in %s" % stage0_dir)
    else:
        anchor_dir = run_stage0(out_dir, stage0_params)
    est, _state = restore_anchor(anchor_dir)
    return run_reduction(est, out_dir, params, resume)

def main(argv=None):
    params = default_params("selection")
    stage0_params = default_params("stage0")
    parser = argparse.ArgumentParser(
        description="Automated parameter selection: stage 0 (all 37 free) + identifiability reduction.")
    parser.add_argument("--out", default=RUN["out_dir"],
                        help="output directory (default: out_dir of settings/run.csv; relative to the repo root)")
    parser.add_argument("--from_anchor", default=None,
                        help="directory with anchor_state.pkl: start the reduction from it and skip stage 0")
    parser.add_argument("--max_iter", type=int, default=params["max_iter"])
    parser.add_argument("--n_jobs", type=int, default=params["n_jobs"])
    parser.add_argument("--n_starts", type=int, default=params["n_starts"], help="starts of the reduction multistarts")
    parser.add_argument("--n_starts_stage0", type=int, default=stage0_params["n_starts_stage0"])
    parser.add_argument("--multistart_every", type=int, default=params["multistart_every"])
    parser.add_argument("--resume", action="store_true", help="continue from <out>/reduction/checkpoint.pkl")
    args = parser.parse_args(argv)
    params["max_iter"] = args.max_iter
    params["n_jobs"] = args.n_jobs
    params["n_starts"] = args.n_starts
    params["multistart_every"] = args.multistart_every
    stage0_params["n_jobs"] = args.n_jobs
    stage0_params["n_starts_stage0"] = args.n_starts_stage0
    run_selection(args.out, params, stage0_params, from_anchor=args.from_anchor, resume=args.resume)

if __name__ == "__main__":
    main()
