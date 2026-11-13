"""
report.py -- REPORT.md of a run, assembled from the saved files of every step.

Every number and parameter name in the text is read from the files of the run directory: final analysis, fixing log,
bootstrap, profiles and LOOCV.  A step that has not run yet is marked "not run".

Position in the pipeline: the last section of final_fit.ipynb (re-run it after the bootstrap, profiles and LOOCV).
"""
import os
import json

import pandas as pd

from paths import REPO_ROOT

def read_csv_if_present(path, **options):
    if os.path.exists(path):
        return pd.read_csv(path, **options)
    return None

def read_json_if_present(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None

def build_report(out_dir):
    """The text of REPORT.md of the run in ``out_dir``."""
    summary = read_json_if_present(os.path.join(out_dir, "final", "final_summary.json"))
    structure = read_csv_if_present(os.path.join(out_dir, "final", "final_structure.csv"), index_col=0)
    fixes = read_csv_if_present(os.path.join(out_dir, "reduction", "fixing_log.csv"))
    boot_info = read_json_if_present(os.path.join(out_dir, "bootstrap", "bootstrap_info.json"))
    boot = read_csv_if_present(os.path.join(out_dir, "bootstrap", "bootstrap_summary.csv"), index_col=0)
    profiles = read_csv_if_present(os.path.join(out_dir, "profiles", "profile_summary.csv"), index_col=0)
    loocv = read_json_if_present(os.path.join(out_dir, "loocv", "loocv_summary.json"))
    relative = os.path.relpath(out_dir, REPO_ROOT)
    lines = ["# Flux model (%s)" % relative, ""]
    lines += structure_lines(summary, structure, boot, profiles)
    lines += fixing_lines(fixes)
    lines += bootstrap_lines(boot_info)
    lines += fit_lines(summary)
    lines += loocv_lines(loocv)
    return "\n".join(lines) + "\n"

def structure_lines(summary, structure, boot, profiles):
    lines = ["## Identifiable structure"]
    if summary is None or structure is None:
        return lines + ["not run (final/final_summary.json or final/final_structure.csv missing)", ""]
    n_fixed = int((structure["status"] == "fixed").sum())
    lines += ["%d free / %d fixed. Structural zeros of the rate laws: %d. Fisher information of the %d analysed free "
              "parameters: full rank %s (smallest singular value ratio %.2e), max |r| %.3f. Kept free by design (not "
              "identifiable): %s." % (summary["n_free"], n_fixed, summary["n_structural_zeros"],
                                      summary["n_analysed"], summary["full_rank"], summary["sigma_ratio_min"],
                                      summary["max_abs_r"], ", ".join(summary["kept_free_by_design"]) or "none"),
              "```"]
    free = structure[structure["status"] == "free"].copy()
    columns = [c for c in ("ident_class", "value", "sensitivity", "cv_percent") if c in free.columns]
    table = free[columns]
    if boot is not None:
        table = table.join(boot[["ci_ratio_97.5_over_2.5", "practically_identifiable"]])
    if profiles is not None:
        table = table.join(profiles[["profile_class"]])
    lines += [table.round(4).to_string(), "```", ""]
    return lines

def fixing_lines(fixes):
    lines = ["## Fixing path"]
    if fixes is None:
        return lines + ["not run (reduction/fixing_log.csv missing)", ""]
    lines += ["```", fixes[["iter", "stage", "rule", "param", "action", "partner", "r", "loading", "obj_after"]]
              .round(4).to_string(index=False), "```", ""]
    return lines

def bootstrap_lines(info):
    lines = ["## Bootstrap"]
    if info is None:
        return lines + ["not run", ""]
    lines += ["%d/%d resamples converged; objective median %.3f (95%% interval %.3f-%.3f; estimate %.3f). Practically "
              "identifiable: %s. Objective = %.2f + %.2f n_KO18 + %.2f n_KO05 (R2 %.2f)." % (
                  info["n_converged"], info["n_resamples"], info["obj_boot_median"], info["obj_boot_p2.5_p97.5"][0],
                  info["obj_boot_p2.5_p97.5"][1], info["obj_estimate"],
                  ", ".join(info["practically_identifiable"]) or "none", info["obj_regression_intercept"],
                  info["obj_per_KO18_copy"], info["obj_per_KO05_copy"], info["obj_regression_R2"]), ""]
    return lines

def fit_lines(summary):
    lines = ["## Fit"]
    if summary is None:
        return lines + ["not run", ""]
    lines += ["Objective %.4f (misfit s2 %.2f). Metabolites RMSE %.3f (mean predictor %.3f, R2 %.3f); LI fluxes RMSE "
              "%.3f (mean predictor %.3f, R2 %.3f). Parameters on a bound: %s." % (
                  summary["obj"], summary["s2"], summary["rmse_met"], summary["rmse_met_meanbaseline"],
                  summary["r2_met_permean"], summary["rmse_flux_li"], summary["rmse_flux_li_meanbaseline"],
                  summary["r2_flux_li_permean"], ", ".join(summary["theta_at_bounds"]) or "none"), ""]
    return lines

def loocv_lines(loocv):
    lines = ["## Leave-one-condition-out cross-validation"]
    if loocv is None:
        return lines + ["not run", ""]
    met = loocv["met_all_folds"]
    flux = loocv["flux_li_all_folds"]
    lines += ["Metabolites RMSE %.3f (training-mean predictor %.3f); LI fluxes RMSE %.3f (training-mean predictor "
              "%.3f). Folds failed: %d; closure solves failed: %d." % (
                  met["rmse_model"], met["rmse_baseline"], flux["rmse_model"], flux["rmse_baseline"],
                  loocv["n_folds_failed"], loocv["n_closure_failed"]), ""]
    return lines
