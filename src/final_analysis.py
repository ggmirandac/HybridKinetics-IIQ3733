"""
final_analysis.py -- final report of the selected structure: re-estimate, identifiability classes, covariance check.

The final anchor of the selection is re-solved with the tight warm start; the solved point is accepted if its
objective is within final_accept_lr_fraction of the LR threshold of the anchor, otherwise the anchor primal point is
used.  At that point: the final structural check (rate level + output level) gives the classes identifiable /
bound_identified / non_identifiable and the nuisance parameters kept free by the protections; the practical
analysis on the Schur complement gives rank, max |r| and a CV per parameter; the k_aug covariance of the extensive
form is a cross-check (in a forked child with a timeout; no fallback); and the fit metrics (objective per condition,
in-sample prediction, drains against the data) close the report.

Position in the pipeline: after selection.py; its ``final/anchor_state.pkl`` is the point the bootstrap, the profiles
and the LOOCV start from.  Reads ``<out>/reduction``, writes ``<out>/final``.
"""
import os
import json
import pickle
import multiprocessing

import numpy as np
import pandas as pd

from settings import SOLVER_OPTIONS, TIGHT_OPTIONS, default_params, snapshot_settings
from progress import log
from paths import check_out_dir
from metrics import lr_threshold, pred_metrics, b_vs_data, theta_at_bounds, misfit_s2
from estimation.anchors import load_point, save_anchor, check_bounds
from identifiability.check import StructuralCheck, plan_structural_fixes
from identifiability.practical import practical_analysis
from selection import write_final_structure, max_abs_r

# Iteration number of the final structural check: it fixes the seed (final_seed_base + 99), apart from the checks of
# the selection loop
FINAL_CHECK_ITERATION = 99

def run_final_analysis(out_dir, params=None):
    """
    Final analysis of the selection in ``<out_dir>/reduction`` (final_anchor_state.pkl, fixing_log.csv,
    reduction_summary.json); writes ``<out_dir>/final`` including its anchor_state.pkl and final_summary.json, whose
    dict it returns.  ``<out_dir>/settings_used`` is refreshed with the settings of this run.
    """
    params = default_params("final") if params is None else params
    out_dir = check_out_dir(out_dir)
    check_bounds()
    snapshot_settings(out_dir)
    red_dir = os.path.join(out_dir, "reduction")
    final_dir = os.path.join(out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    with open(os.path.join(red_dir, "final_anchor_state.pkl"), "rb") as fh:
        anchor_state = pickle.load(fh)
    threshold = lr_threshold(params["lr_alpha"], len(anchor_state["conditions"]))
    est, point = re_estimate(load_point(anchor_state), anchor_state, threshold, params)
    save_anchor(est, final_dir, extra={"saved_reduction_obj": anchor_state["obj"], "point": point["name"],
                                       "obj_re_estimated": point["obj"], "obj_anchor": anchor_state["obj"]})
    fix_log = pd.read_csv(os.path.join(red_dir, "fixing_log.csv"))
    with open(os.path.join(red_dir, "reduction_summary.json")) as fh:
        reduction_summary = json.load(fh)
    log("final point: %d free, obj %.6f (reduction saved %.6f)" % (len(est.free_params), est.obj_value,
                                                                     anchor_state["obj"]))
    check = final_structural_check(est, final_dir, params)
    nuisance = list(check["kept_free_by_rule"])
    analysis = practical_analysis(est, nuisance, params)
    write_final_structure(est, analysis, fix_log, final_dir, check["output_level"], nuisance)
    analysis["report"]["param_table"].to_csv(os.path.join(final_dir, "reduced_sensitivity_params.csv"))
    s2 = float(misfit_s2(est))
    summary = summary_of_point(est, point, anchor_state, threshold, s2, analysis, nuisance, check,
                               reduction_summary["status"], params)
    covariance_crosscheck(est, analysis, s2, point["name"] == "re_estimated", final_dir, summary)
    fit_metrics(est, final_dir, summary, params)
    with open(os.path.join(final_dir, "final_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    log_final_structure(final_dir, summary, est, s2)
    return summary

def re_estimate(anchor_est, anchor_state, threshold, params):
    """
    Re-solve the final anchor from its primal point with the tight warm start.  The solved point is used if
    obj <= obj_anchor + final_accept_lr_fraction * LR threshold; otherwise the anchor primal is kept (and the k_aug
    covariance, which needs a solved extensive form, is not available).  Returns (estimator, point info).
    """
    limit = anchor_state["obj"] + params["final_accept_lr_fraction"] * threshold
    x_init, u_init = anchor_est.primal_start()
    solved = anchor_est.clone(theta_init=anchor_est.full_theta(), x_init=x_init, u_init=u_init,
                              b_init=anchor_est.primal_b_free(), solver_options={**SOLVER_OPTIONS, **TIGHT_OPTIONS})
    error = ""
    try:
        solved.estimate()
        solved.refine_anchor(primary_options=dict(TIGHT_OPTIONS))
        solved.solver_options = dict(SOLVER_OPTIONS)
        obj = float(solved.obj_value)
    except Exception as exc:   # noqa: BLE001  (IPOPT not converging: the anchor primal point is used)
        solved, obj, error = None, float("nan"), str(exc)[:150]
    accepted = solved is not None and obj <= limit
    if accepted:
        verdict = "ACCEPTED, the solved point is used"
    else:
        verdict = "REJECTED, the anchor primal is used for all point estimates / classes; kaug covariance not available"
    log("re-estimate: obj_anchor %.6f, obj_re_estimated %.6f (%s), acceptance limit %.6f -> %s" % (
        anchor_state["obj"], obj, error or "ok", limit, verdict))
    if accepted:
        return solved, {"name": "re_estimated", "obj": obj, "limit": limit}
    return anchor_est, {"name": "anchor_primal", "obj": obj, "limit": limit}

def final_structural_check(est, final_dir, params):
    """The structural check at the final point; writes final_structural_check.csv and output_level_classes.csv."""
    checker = StructuralCheck(params, log)
    check = checker.run(est, FINAL_CHECK_ITERATION, "final")
    if check["flagged"]:
        analysis = practical_analysis(est, [], params)
        plan_structural_fixes(check, analysis["infl"], params)
    row = check["row"]
    pd.Series(row).to_csv(os.path.join(final_dir, "final_structural_check.csv"), header=["value"])
    log("final structural check: rate-level null dim %d, output-level null dim %d (active set %d, generic states %d); "
        "kept_free_by_rule %s" % (row["null_dim_consensus"], row["out_null_dim"], row["out_null_dim_active_set"],
                                  row["out_null_dim_generic_states"], check["kept_free_by_rule"]))
    classes = check["output_level"]["classes"]
    pd.DataFrame({"param": list(classes), "output_class": list(classes.values())}).to_csv(
        os.path.join(final_dir, "output_level_classes.csv"), index=False)
    return check

def summary_of_point(est, point, anchor_state, threshold, s2, analysis, nuisance, check, reduction_status, params):
    """The first entries of final_summary.json: the point, the practical criteria and the structural classes."""
    n_conditions = len(anchor_state["conditions"])
    flags = {}
    for name, flag_list in analysis["flags"].items():
        if flag_list:
            flags[name] = flag_list
    class_counts = {}
    for name in ("identifiable", "bound_identified", "non_identifiable"):
        class_counts[name] = sum(1 for value in check["output_level"]["classes"].values() if value == name)
    off_diagonal_size = len(analysis["nz"]) ** 2
    return {"final_point": point["name"], "obj_anchor": anchor_state["obj"], "obj_re_estimated": point["obj"],
            "re_estimate_acceptance_limit": point["limit"], "n_free": len(est.free_params),
            "free": list(est.free_params), "obj": float(est.obj_value), "s2": s2, "n_analysed": len(analysis["fa"]),
            "kept_free_by_rule": nuisance, "eig_cut": analysis["eig_cut"],
            "rank_at_cut": analysis["ranks"][analysis["eig_cut"]], "full_rank": analysis["full_rank"],
            "sigma_ratio": analysis["sigma_ratio"], "cond_F": analysis["cond"],
            "max_abs_r": max_abs_r(analysis) if off_diagonal_size else None,
            # the "0.9" of the key is historical: the threshold is corr_threshold of settings/thresholds.csv
            "n_pairs_gt_0.9": len(analysis["pairs"]) if analysis["full_rank"] else None,
            "flagged": analysis["flagged"], "flagged_nuisance": analysis["flagged_nuisance"], "flags": flags,
            "reduction_status": reduction_status,
            # the "0.95" of the two keys is historical: the level is lr_alpha of settings/thresholds.csv
            "lr_threshold_0.95": threshold,
            "lr_threshold_0.95_s2": lr_threshold(params["lr_alpha"], n_conditions, s2=s2),
            "output_level": class_counts}

def covariance_in_child(est, queue):
    """Target of the forked child: the k_aug covariance, or the error message."""
    try:
        cov = est.covariance()
    except Exception as exc:   # noqa: BLE001  (k_aug failing is reported as "not available")
        queue.put((None, "%s: %s" % (type(exc).__name__, str(exc)[:200]), None))
        return
    queue.put((np.asarray(cov, float).tolist(), list(cov.columns), est.cov_psd))

def kaug_covariance(est, final_dir, timeout):
    """
    The k_aug covariance computed in a forked child (k_aug can hang); returns (psd or None, method or 'none',
    DataFrame or None) and writes covariance_kaug.csv.  There is no fallback.
    """
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    child = context.Process(target=covariance_in_child, args=(est, queue))
    child.start()
    child.join(timeout)
    if child.is_alive():
        child.terminate()
        log("covariance (kaug) timed out: not available")
        return None, "none", None
    if queue.empty():
        return None, "none", None
    values, columns, psd = queue.get()
    if values is None:
        log("covariance (kaug) not available: %s" % columns)
        return None, "none", None
    cov = pd.DataFrame(values, index=columns, columns=columns)
    cov.to_csv(os.path.join(final_dir, "covariance_kaug.csv"))
    return bool(psd), "automatic_differentiation_kaug", cov

def covariance_crosscheck(est, analysis, s2, solved, final_dir, summary):
    """covariance_crosscheck.csv: the SD of the Schur-complement FIM (also times s) against the k_aug SD."""
    if solved:
        psd, method, cov = kaug_covariance(est, final_dir, timeout=600)
    else:
        psd, method, cov = None, "none", None
        log("covariance (kaug): not available (anchor-primal point, no extensive-form solve)")
    summary["cov_method"] = method
    summary["cov_psd"] = psd
    free = list(est.free_params)
    variances = np.diag(analysis["cov_abs"].to_numpy())
    table = pd.DataFrame({"sd_schur": np.sqrt(np.where(variances > 0, variances, np.nan))}, index=free)
    table["sd_schur_s2scaled"] = table["sd_schur"] * np.sqrt(s2)
    if cov is not None:
        kaug_variance = pd.Series(np.diag(cov.to_numpy()), index=cov.index)
        table["var_kaug"] = kaug_variance
        table["sd_kaug"] = np.sqrt(kaug_variance.where(kaug_variance > 0))
        theta = pd.Series({p: abs(float(est.theta[p])) for p in free})
        table["cv_percent_kaug"] = 100 * table["sd_kaug"] / theta
        table["ratio_kaug_over_schur_s2"] = table["sd_kaug"] / table["sd_schur_s2scaled"]
        summary["cov_n_negative_variances"] = int((kaug_variance < 0).sum())
    table.rename_axis("param").to_csv(os.path.join(final_dir, "covariance_crosscheck.csv"))

def fit_metrics(est, final_dir, summary, params):
    """objective_breakdown.csv, pred_insample_*.csv (at the fitted point), b_fit_vs_data.csv and their summary."""
    breakdown = est.objective_breakdown()
    breakdown.to_csv(os.path.join(final_dir, "objective_breakdown.csv"))
    prediction = est.predict_fitted()
    prediction.predicted.to_csv(os.path.join(final_dir, "pred_insample_predicted.csv"))
    prediction.real.to_csv(os.path.join(final_dir, "pred_insample_real.csv"))
    metrics = pred_metrics(prediction.real, prediction.predicted)
    drains = b_vs_data(est)
    drains.to_csv(os.path.join(final_dir, "b_fit_vs_data.csv"), index=False)
    summary.update(metrics)
    summary.update({"breakdown_mean_total": float(breakdown.loc["MEAN", "total"]),
                    "n_pred_failed": int(prediction.n_failed),
                    "b_rmse_vs_data": float(np.sqrt(np.mean(drains["err"] ** 2))),
                    "n_drains_at_bound": int(drains["at_bound"].sum()), "n_drain_values": int(len(drains)),
                    "theta_at_bounds": theta_at_bounds(est, params["theta_at_bound_tol"])})

def log_final_structure(final_dir, summary, est, s2):
    structure = pd.read_csv(os.path.join(final_dir, "final_structure.csv"), index_col=0)
    columns = [c for c in ("status", "ident_class", "output_class", "flags", "fixed_rule", "fixed_iter", "value",
                           "influence", "cv_percent", "cv_basis") if c in structure.columns]
    log("final structure:\n" + structure[columns].to_string())
    log("rank %d/%d at eig cut %.0e, max|r| %.3f, obj %.5f, s2 %.2f, rmse_met %.4f (baseline %.4f), LI flux %.4f "
        "(baseline %.4f)" % (summary["rank_at_cut"], summary["n_analysed"], summary["eig_cut"],
                             summary["max_abs_r"] or np.nan, est.obj_value, s2, summary["rmse_met"],
                             summary["rmse_met_meanbaseline"], summary["rmse_flux_li"],
                             summary["rmse_flux_li_meanbaseline"]))
