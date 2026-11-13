"""
final_analysis.py -- the selected structure: identifiability classes, uncertainty from the Fisher information and fit.

At the final fit of the selection: the structural check of all free parameters (Stigter and Molenaar), the Fisher
information of the analysed free parameters (all free ones except those kept free by design) with rank, correlations
and CV, and the fit metrics (objective per condition, in-sample prediction, drains against the data).

Classes of the 37 parameters (final_structure.csv): "free" (analysed: together they pass the structural and practical
criteria; whether each one is practically identifiable is decided by the bootstrap and the profile likelihood),
"kept_free_by_design" (free but not identifiable; the protections keep it), "fixed" (with the stage, rule and
iteration that fixed it).

Position in the pipeline: after selection.py; its ``final/anchor_state.pkl`` is the point the bootstrap, the profiles
and the LOOCV start from.  Reads ``<out>/reduction``, writes ``<out>/final``.
"""
import os
import json
import pickle

import numpy as np
import pandas as pd

from settings import default_params, snapshot_settings
from progress import log
from paths import resolve_out
from model.parameters import ALL_PARAMS
from metrics import pred_metrics, b_vs_data, theta_at_bounds, misfit_s2, lr_threshold
from estimation.anchors import load_point, save_anchor, check_bounds
from identifiability.structural import RateLaws, condition_states, stacked_sensitivity_matrix, singular_spectrum
from identifiability.practical import sensitivity_matrices, fim_analysis

def run_final_analysis(out_dir, params=None):
    """
    Final analysis of the selection in ``<out_dir>/reduction``; writes ``<out_dir>/final`` (anchor_state.pkl,
    final_structure.csv, final_correlation.csv, final_singular_spectrum.csv, final_structural_spectrum.csv,
    objective_breakdown.csv, pred_insample_*.csv, b_fit_vs_data.csv, final_summary.json) and returns the summary.
    """
    params = default_params() if params is None else params
    out_dir = resolve_out(out_dir)
    check_bounds()
    snapshot_settings(out_dir)
    red_dir = os.path.join(out_dir, "reduction")
    final_dir = os.path.join(out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    with open(os.path.join(red_dir, "reduction_summary.json")) as fh:
        reduction = json.load(fh)
    with open(os.path.join(red_dir, "final_anchor_state.pkl"), "rb") as fh:
        est = load_point(pickle.load(fh))
    save_anchor(est, final_dir)
    kept_free = reduction["kept_free_by_design"]
    structural_zeros = final_structural_check(est, final_dir, params)
    analysed = [p for p in est.free_params if p not in kept_free]
    J, relative, failed = sensitivity_matrices(est)
    analysis = fim_analysis(J, relative, analysed, params)
    fix_log = pd.read_csv(os.path.join(red_dir, "fixing_log.csv"))
    write_final_structure(est, analysis, kept_free, fix_log, final_dir)
    s2 = float(misfit_s2(est))
    summary = {"n_free": len(est.free_params), "free": list(est.free_params), "kept_free_by_design": kept_free,
               "n_analysed": len(analysed), "obj": float(est.obj_value), "s2": s2,
               "n_structural_zeros": structural_zeros, "full_rank": analysis["full_rank"],
               "sigma_ratio_min": float(analysis["sigma_ratio"][-1]), "max_abs_r": analysis["max_abs_r"],
               "n_pairs_above_threshold": len(analysis["pairs"]), "sens_failed_conditions": failed,
               "lr_threshold": lr_threshold(params["lr_alpha"], len(est.conditions)),
               "theta_at_bounds": theta_at_bounds(est)}
    summary.update(fit_metrics(est, final_dir))
    with open(os.path.join(final_dir, "final_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    structure = pd.read_csv(os.path.join(final_dir, "final_structure.csv"), index_col=0)
    log("final structure:\n" + structure[structure["status"] == "free"].to_string())
    log("structural zeros %d; rank full %s (sigma_min/max %.2e); max|r| %.3f; obj %.5f; rmse_met %.4f (baseline "
        "%.4f); LI flux %.4f (baseline %.4f)" % (
            structural_zeros, analysis["full_rank"], analysis["sigma_ratio"][-1], analysis["max_abs_r"],
            est.obj_value, summary["rmse_met"], summary["rmse_met_meanbaseline"], summary["rmse_flux_li"],
            summary["rmse_flux_li_meanbaseline"]))
    return summary

def final_structural_check(est, final_dir, params):
    """Singular values of the stacked rate sensitivity matrix of all free parameters; returns the structural zeros."""
    free = list(est.free_params)
    S = stacked_sensitivity_matrix(RateLaws(), est.full_theta(), condition_states(est), free, params)
    ratio, _Vt = singular_spectrum(S)
    pd.Series(ratio, index=np.arange(1, len(ratio) + 1), name="sigma_ratio").rename_axis("k").to_csv(
        os.path.join(final_dir, "final_structural_spectrum.csv"))
    return int(np.sum(ratio < params["structural_tol"]))

def write_final_structure(est, analysis, kept_free, fix_log, final_dir):
    """final_structure.csv (all 37 parameters), final_correlation.csv and final_singular_spectrum.csv."""
    full = est.full_theta()
    rows = []
    for p in ALL_PARAMS:
        row = {"param": p, "value": float(full[p])}
        if p in est.free_params:
            row["status"] = "free"
            row["ident_class"] = "kept_free_by_design" if p in kept_free else "free"
            if p in analysis["names"]:
                row["sensitivity"] = float(analysis["sensitivity"][p])
                if analysis["cv_percent"] is not None:
                    row["cv_percent"] = float(analysis["cv_percent"][p])
        else:
            last = fix_log[(fix_log["param"] == p) & (fix_log["action"] == "fixed")].iloc[-1]
            row.update({"status": "fixed", "ident_class": "fixed", "fixed_stage": last["stage"],
                        "fixed_rule": last["rule"], "fixed_iter": int(last["iter"])})
        rows.append(row)
    pd.DataFrame(rows).set_index("param").to_csv(os.path.join(final_dir, "final_structure.csv"))
    if analysis["corr"] is not None:
        analysis["corr"].to_csv(os.path.join(final_dir, "final_correlation.csv"))
    pd.Series(analysis["sigma_ratio"], name="sigma_ratio").rename_axis("k").to_csv(
        os.path.join(final_dir, "final_singular_spectrum.csv"))

def fit_metrics(est, final_dir):
    """objective_breakdown.csv, pred_insample_*.csv (at the fitted point), b_fit_vs_data.csv and their summary."""
    breakdown = est.objective_breakdown()
    breakdown.to_csv(os.path.join(final_dir, "objective_breakdown.csv"))
    prediction = est.predict_fitted()
    prediction.predicted.to_csv(os.path.join(final_dir, "pred_insample_predicted.csv"))
    prediction.real.to_csv(os.path.join(final_dir, "pred_insample_real.csv"))
    drains = b_vs_data(est)
    drains.to_csv(os.path.join(final_dir, "b_fit_vs_data.csv"), index=False)
    metrics = pred_metrics(prediction.real, prediction.predicted)
    metrics.update({"n_pred_failed": int(prediction.n_failed),
                    "b_rmse_vs_data": float(np.sqrt(np.mean(drains["err"] ** 2))),
                    "n_drains_at_bound": int(drains["at_bound"].sum())})
    return metrics
