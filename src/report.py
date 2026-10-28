"""
report.py -- REPORT.md of a run, assembled from the saved files of every step.

Every number, parameter name and threshold in the text is read from the files of the run directory: final analysis,
fixing log, bootstrap, profiles, LOOCV and the figures present.  A step that has not run yet is marked "not run".  The
Fit section compares the final fit with the reference run of settings/run.csv (compare_to) when that folder exists.

Position in the pipeline: the last section of final_fit.ipynb (re-run it after the bootstrap, profiles and LOOCV).
"""
import os
import json
import glob

import numpy as np
import pandas as pd

from settings import THRESHOLDS, THETA_BOUNDS, RUN, UNCERTAINTY, uncertainty_settings
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

def sci(x, digits=0):
    """Compact scientific notation: 1e-04 -> 1e-4, 3.5e+04 -> 3.5e4."""
    mantissa, exponent = ("%.*e" % (digits, x)).split("e")
    return "%se%d" % (mantissa, int(exponent))

def fixing_log_with_classes(out_dir):
    """fixing_log_with_classes.csv (parameter_selection.ipynb), else fixing_log.csv with class_at_fix added."""
    with_classes = read_csv_if_present(os.path.join(out_dir, "fixing_log_with_classes.csv"))
    if with_classes is not None:
        return with_classes
    fix_log = read_csv_if_present(os.path.join(out_dir, "reduction", "fixing_log.csv"))
    if fix_log is not None and "output_class_at_fix" in fix_log.columns:
        fix_log["class_at_fix"] = fix_log["output_class_at_fix"]
    return fix_log

def reference_run(out_dir):
    """(name, final_summary, bootstrap_info) of the run named by compare_to; Nones when it is missing or is OUT."""
    name = RUN["compare_to"]
    folder = name if os.path.isabs(name) else os.path.join(REPO_ROOT, name)
    if os.path.abspath(folder) == os.path.abspath(out_dir):
        return name, None, None
    if not os.path.isdir(folder):
        print("reference run not found: %s (the report has no comparison)" % folder)
        return name, None, None
    summary = read_json_if_present(os.path.join(folder, "final", "final_summary.json"))
    bootstrap = read_json_if_present(os.path.join(folder, "bootstrap", "bootstrap_info.json"))
    return name, summary, bootstrap

def load_report_inputs(out_dir):
    """The saved files the report reads (None where a step has not run)."""
    used_dir = os.path.join(out_dir, "settings_used")
    if os.path.exists(os.path.join(used_dir, "uncertainty.csv")):
        uncertainty = uncertainty_settings(used_dir)
    else:
        uncertainty = UNCERTAINTY
    reference_name, reference_summary, reference_bootstrap = reference_run(out_dir)
    return {"summary": read_json_if_present(os.path.join(out_dir, "final", "final_summary.json")),
            "structure": read_csv_if_present(os.path.join(out_dir, "final", "final_structure.csv")),
            "cross": read_csv_if_present(os.path.join(out_dir, "final", "covariance_crosscheck.csv"), index_col=0),
            "fixes": fixing_log_with_classes(out_dir),
            "boot_info": read_json_if_present(os.path.join(out_dir, "bootstrap", "bootstrap_info.json")),
            "boot_summary": read_csv_if_present(os.path.join(out_dir, "bootstrap", "bootstrap_summary.csv"),
                                                index_col=0),
            "profile_summary": read_csv_if_present(os.path.join(out_dir, "profiles", "profile_summary.csv"),
                                                   index_col=0),
            "profile_thresholds": read_json_if_present(os.path.join(out_dir, "profiles", "profile_thresholds.json")),
            "loocv_summary": read_json_if_present(os.path.join(out_dir, "loocv", "loocv_summary.json")),
            "loocv_folds": read_csv_if_present(os.path.join(out_dir, "loocv", "fold_status.csv")),
            "stage_row": read_json_if_present(os.path.join(out_dir, "stage0", "stage_row.json")),
            "uncertainty": uncertainty, "reference_name": reference_name, "reference_summary": reference_summary,
            "reference_bootstrap": reference_bootstrap}

def build_report(out_dir):
    """The text of REPORT.md of the run in ``out_dir``."""
    inputs = load_report_inputs(out_dir)
    relative = os.path.relpath(out_dir, REPO_ROOT)
    lines = ["# Flux model, repaired pipeline (%s)" % (out_dir if relative.startswith("..") else relative), ""]
    lines += identifiability_lines(inputs)
    lines += fixing_lines(inputs)
    lines += bootstrap_lines(inputs)
    lines += profile_lines(inputs)
    lines += fit_lines(inputs)
    lines += loocv_lines(inputs)
    lines += limitation_lines(inputs, out_dir)
    return "\n".join(lines) + "\n"

def correlated_pair_text(summary):
    threshold = THRESHOLDS["corr_threshold"]
    if summary.get("n_pairs_gt_0.9") == 0:
        return "no |r|>%.1f pair" % threshold
    return "%s pairs |r|>%.1f" % (summary.get("n_pairs_gt_0.9"), threshold)

def identifiability_lines(inputs):
    summary = inputs["summary"]
    structure = inputs["structure"]
    lines = ["## Identifiability first"]
    if summary is None or structure is None:
        return lines + ["not run (final/final_summary.json or final/final_structure.csv missing)", ""]
    n_fixed = int((structure["status"] == "fixed").sum())
    lines += ["Final structure: %d free / %d fixed, reduction status `%s`; final point = %s (obj %.4f; reduction anchor "
              "%.4f; acceptance limit %.4f)." % (summary["n_free"], n_fixed, summary["reduction_status"],
                                                 summary["final_point"], summary["obj"], summary["obj_anchor"],
                                                 summary["re_estimate_acceptance_limit"])]
    lines += ["Practical criteria on the Schur complement (%d kept_free_by_rule parameters as nuisance): rank %d/%d at "
              "eigenvalue cut %.0e (singular ratio %s), max |r| %.3f, %s. Output-level classes (bound-free / active "
              "set): %s." % (len(summary["kept_free_by_rule"]), summary["rank_at_cut"], summary["n_analysed"],
                             summary["eig_cut"], sci(np.sqrt(summary["eig_cut"])), summary["max_abs_r"],
                             correlated_pair_text(summary), summary["output_level"])]
    lines += ["Flags: %s." % summary["flags"], "```"]
    columns = ["param", "ident_class", "output_class", "flags", "value", "influence", "cv_percent", "cv_basis"]
    table = structure[structure["status"] == "free"][columns].copy()
    if inputs["boot_summary"] is not None:
        table["bootstrap_cv_%"] = table["param"].map(inputs["boot_summary"]["cv_percent"])
        table["boot_ci_ratio"] = table["param"].map(inputs["boot_summary"]["ci_ratio_97.5_over_2.5"])
    if inputs["profile_summary"] is not None:
        table["profile_class"] = table["param"].map(inputs["profile_summary"]["class_lr"])
    lines += [table.round(4).to_string(index=False), "```", ""]
    cross = inputs["cross"]
    if cross is not None and "ratio_kaug_over_schur_s2" in cross.columns:
        lines += [covariance_sentence(summary, cross["ratio_kaug_over_schur_s2"].dropna()), ""]
    elif summary.get("cov_method") == "none":
        lines += ["kaug covariance: not available for this point.", ""]
    return lines

def covariance_sentence(summary, ratio):
    worst = ratio.idxmax()
    rest = ratio.drop(worst)
    disagrees = bool(((ratio < 0.5) | (ratio > 2.0)).any())
    if len(rest):
        spread = "ratios %.1g-%.1g, %s for %s" % (rest.min(), rest.max(), sci(ratio.max(), 1), worst)
    else:
        spread = "ratio %s for %s" % (sci(ratio.max(), 1), worst)
    return ("kaug covariance: PSD %s, %s negative variances (final/covariance_crosscheck.csv). It %s the Schur-FIM SDs "
            "(%s), so CVs are indicative only." % (summary.get("cov_psd"), summary.get("cov_n_negative_variances", "0"),
                                                   "disagrees with" if disagrees else "agrees with", spread))

def fixing_lines(inputs):
    fixes = inputs["fixes"]
    if fixes is None:
        return []
    class_column = "class_at_fix" if "class_at_fix" in fixes.columns else None
    columns = ["param", "iter", "rule", "influence_param"]
    if class_column:
        columns.append(class_column)
    table = fixes[columns].rename(columns={"influence_param": "influence_at_fix"})
    if class_column:
        table[class_column] = table[class_column].fillna("")
    lines = ["### Which rule fixed which parameter", "```", table.round(5).to_string(index=False), "```"]
    if class_column:
        sentence = output_level_fix_sentence(fixes, class_column, inputs["structure"])
        if sentence:
            lines += [sentence]
    return lines + [""]

def output_level_fix_sentence(fixes, class_column, structure):
    """The classes of the output-level fixes, and the nuisance parameters that are not non_identifiable."""
    output_fixes = fixes[fixes["rule"] == "structural_output"]
    counts = output_fixes[class_column].value_counts().to_dict()
    if not counts:
        return ""
    if len(counts) == 1:
        class_name = next(iter(counts))
        active_set_note = " (null also with the active set)" if class_name == "non_identifiable" else ""
        sentence = "Output-level fixes: all %d were `%s`%s" % (len(output_fixes), class_name, active_set_note)
    else:
        sentence = "Output-level fixes: " + ", ".join("%d `%s`" % (n, c) for c, n in counts.items())
    if "bound_identified" not in counts:
        sentence += ", none only bound_identified"
    if "round" in output_fixes.columns and output_fixes["round"].notna().any():
        anchors = set()
        for iteration, round_number in zip(output_fixes["iter"], output_fixes["round"]):
            if pd.notna(round_number):
                anchors.add((int(iteration), int(round_number)))
        sentence += ", at the " + " / ".join("iteration-%d round-%d" % anchor for anchor in sorted(anchors)) + " anchor"
    sentence += "."
    if structure is not None:
        odd = structure[(structure["status"] == "free") & (structure["ident_class"] == "kept_free_by_rule")
                        & (structure["output_class"] != "non_identifiable")]
        for name, output_class in zip(odd["param"], odd["output_class"]):
            sentence += " %s is `%s` at the final point (kept_free_by_rule)." % (name, output_class)
    return sentence

def bootstrap_lines(inputs):
    info = inputs["boot_info"]
    summary = inputs["boot_summary"]
    if info is not None:
        lines = ["## Bootstrap (N=%d, case resampling, seed %d, warm start at the final point)" % (
            info["n_resamples"], inputs["uncertainty"]["bootstrap_seed"])]
    else:
        lines = ["## Bootstrap (case resampling of the conditions, warm start at the final point)"]
    if info is None or summary is None:
        return lines + ["not run", ""]
    ko18 = "KO18 still dominates"
    if inputs["reference_bootstrap"] is not None:
        ko18 += " (%.2f per copy; %.2f in %s)" % (info["obj_per_KO18_copy"],
                                                  inputs["reference_bootstrap"]["obj_per_KO18_copy"],
                                                  inputs["reference_name"])
    lines += ["%d/%d converged (%d needed the monotone retry). Objective: median %.2f, 95%% [%.2f, %.2f]; obj ~ %.2f + "
              "%.2f n_KO18 + %.2f n_KO05 (R2 %.2f); %s." % (
                  info["n_converged"], info["n_resamples"], info["n_retried_ok"], info["obj_boot_median"],
                  info["obj_boot_p2.5_p97.5"][0], info["obj_boot_p2.5_p97.5"][1], info["obj_regression_intercept"],
                  info["obj_per_KO18_copy"], info["obj_per_KO05_copy"], info["obj_regression_R2"], ko18)]
    lines += [bootstrap_caveat(summary, inputs["profile_summary"])]
    columns = ["anchor", "median", "p2.5", "p97.5", "cv_percent", "ci_ratio_97.5_over_2.5", "frac_at_theta_bound",
               "practically_identifiable"]
    return lines + ["```", summary[columns].round(4).reset_index().to_string(index=False), "```", ""]

def bootstrap_caveat(summary, profile_summary):
    """
    Parameters the data cannot move: a warm-started refit never leaves its start, so the bootstrap median equals the
    anchor to 7 digits (parking, not information).  Credible = profile identifiable AND bootstrap identifiable.
    """
    parked = []
    for name in summary.index:
        if abs(summary.loc[name, "median"] - summary.loc[name, "anchor"]) <= 1e-6 * abs(summary.loc[name, "anchor"]):
            parked.append(name)
    kcats = [name[len("kcat_f_"):] for name in parked if name.startswith("kcat_f_")]
    parked_names = (["kcat_f_" + "/".join(kcats)] if kcats else []) + [n for n in parked if not n.startswith("kcat_f_")]
    if profile_summary is not None:
        profile_ok = profile_summary["class_lr"] == "identifiable"
    else:
        profile_ok = summary["practically_identifiable"]
    credible = []
    partly = []
    for name in summary.index:
        if name in parked or not bool(profile_ok.get(name, False)):
            continue
        if bool(summary.loc[name, "practically_identifiable"]):
            credible.append(name)
        else:
            partly.append(name)
    caveat = ("Caveat: for parameters the data cannot move%s the bootstrap median equals the warm start to 7 digits: "
              "the narrow percentile interval reflects parking at the start, not information." % (
                  " (%s)" % ", ".join(parked_names) if parked else ""))
    if credible or partly:
        if credible and partly:
            names = ", ".join(credible) + " (and partly %s)" % ", ".join(partly)
        elif partly:
            names = ", ".join(credible) + "partly " + ", ".join(partly)
        else:
            names = ", ".join(credible)
        caveat += " Only %s %s credible." % (names, "is" if len(credible) + len(partly) == 1 else "are")
    else:
        caveat += " Compare with the profiles before calling a parameter practically identifiable."
    return caveat

def profile_lines(inputs):
    uncertainty = inputs["uncertainty"]
    summary = inputs["profile_summary"]
    thresholds = inputs["profile_thresholds"]
    lines = ["## Profiles (%d points, +/-%g decades, continuation, tight)" % (
        2 * uncertainty["profile_half_points"] + 1, uncertainty["profile_span"])]
    if summary is None or thresholds is None:
        return lines + ["not run", ""]
    lines += ["Thresholds %.4f (and %.4f misfit scaled). Re-fits start without duals, so profile points sit about %.2f "
              "above the best objective: crossings are uncertain at the level of 1/3 of the threshold." % (
                  thresholds["lr_threshold_0.95"], thresholds["lr_threshold_0.95_s2"],
                  summary["min_dobj_anywhere"].median())]
    columns = ["upper_decades_lr", "lower_decades_lr", "class_lr", "class_lr_s2", "n_failed"]
    return lines + ["```", summary[columns].round(3).reset_index().to_string(index=False), "```", ""]

def fit_lines(inputs):
    summary = inputs["summary"]
    lines = ["## Fit"]
    if summary is None:
        return lines
    fit = ("rmse_met %.4f (training-mean baseline %.4f), LI flux rmse %.4f (baseline %.4f), s2 %.2f, drains at a bound "
           "%s/%s." % (summary["rmse_met"], summary["rmse_met_meanbaseline"], summary["rmse_flux_li"],
                       summary["rmse_flux_li_meanbaseline"], summary["s2"], summary.get("n_drains_at_bound"),
                       summary.get("n_drain_values")))
    if inputs["stage_row"] is not None:
        fit += " Stage0 (%d free): obj %.3f." % (inputs["stage_row"]["n_free"], inputs["stage_row"]["obj"])
    reference = inputs["reference_summary"]
    if reference is not None:
        fit += " Comparison with %s (%d free): obj %.1f, s2 %.1f, rmse_met %.3f, LI flux %.3f." % (
            inputs["reference_name"], reference["n_free"], reference["obj"], reference["s2"], reference["rmse_met"],
            reference["rmse_flux_li"])
    return lines + [fit, ""]

def loocv_lines(inputs):
    folds = inputs["loocv_folds"]
    summary = inputs["loocv_summary"]
    if folds is not None:
        lines = ["## LOOCV (%d folds%s)" % (len(folds), ", RF03 included" if "RF03" in set(folds["h"]) else "")]
    else:
        lines = ["## LOOCV (leak-free, closure prediction of the held-out condition)"]
    if summary is None:
        return lines + ["not run", ""]
    met_all, flux_all = summary["met_all_folds"], summary["flux_li_all_folds"]
    met_ok, flux_ok = summary["met_closure_success_only"], summary["flux_li_closure_success_only"]
    worse = met_all["rmse_model"] > met_all["rmse_baseline"] and flux_all["rmse_model"] > flux_all["rmse_baseline"]
    lines += ["Model is %s the training-mean baselines: metabolites rmse %.4f vs %.4f, LI flux %.3f vs %.3f (all folds); "
              "closure-success folds only: %.4f vs %.4f and %.3f vs %.3f.%s" % (
                  "WORSE than" if worse else "not uniformly worse than", met_all["rmse_model"],
                  met_all["rmse_baseline"], flux_all["rmse_model"], flux_all["rmse_baseline"], met_ok["rmse_model"],
                  met_ok["rmse_baseline"], flux_ok["rmse_model"], flux_ok["rmse_baseline"],
                  " Known limitation (pts_fraction closure ill-posed, S invertible)." if worse else "")]
    n_folds = len(folds) if folds is not None else sum(summary["fold_status_counts"].values())
    status_line = "Fold status: %s; the restricted anchor lies outside the training-only bounds in %s/%d folds." % (
        summary["fold_status_counts"], summary["n_anchor_outside_training_bounds"], n_folds)
    if folds is not None:
        status_line += above_bound_text(folds)
    lines += [status_line]
    if folds is not None:
        columns = ["h", "fold_status", "obj_fold", "bound", "anchor_inside_training_bounds",
                   "n_anchor_values_outside_bounds", "closure_success"]
        lines += ["```", folds[columns].round(4).to_string(index=False), "```", ""]
    return lines

def above_bound_text(folds):
    """Why the above_bound folds miss their bound: anchor outside the training bounds, or warm-start drift."""
    above = folds[folds["fold_status"] == "above_bound"]
    inside = above["anchor_inside_training_bounds"].astype(bool)
    parts = []
    if (~inside).any():
        names = list(above.loc[~inside, "h"])
        parts.append("%s %s the anchor outside the training bounds (bound unattainable)" % (
            ", ".join(names), "has" if len(names) == 1 else "have"))
    if inside.any():
        drifted = above[inside]
        excess = drifted["obj_fold"] - drifted["bound"]
        if len(drifted) == 1:
            parts.append("%s has it inside and is above the bound by %.3f (ipopt warm-start drift)" % (
                drifted["h"].iloc[0], excess.iloc[0]))
        else:
            parts.append("%s have it inside and are above the bound by %s (ipopt warm-start drift)" % (
                ", ".join(drifted["h"]), ", ".join("%.3f" % x for x in excess)))
    if parts:
        return " above_bound folds: " + "; ".join(parts) + "."
    return ""

def limitation_lines(inputs, out_dir):
    structure = inputs["structure"]
    at_bound = []
    if structure is not None:
        for name in structure.loc[structure["status"] == "free", "param"]:
            row = structure[structure["param"] == name].iloc[0]
            if "at_bound" in str(row["flags"]):
                value = float(row["value"])
                lower, upper = THETA_BOUNDS[name]
                side = "lower" if abs(np.log(value / lower)) < abs(np.log(upper / value)) else "upper"
                at_bound.append("%s at its %s bound" % (name, side))
    lines = ["## Known limitations",
             "closure ill-posed (S invertible, pts_fraction closure); PGI thermodynamic floor (the data violate the PGI "
             "driving force in 6 conditions, reported only); isozymes ignored; cofactor bounds (Pi literature range); "
             "ipopt warm-start drift of about 0.3%% (kept_anchor semantics, no duals); profile and bootstrap "
             "behaviour of parameters at a bound%s." % (" (%s)" % ", ".join(at_bound) if at_bound else ""), ""]
    figures = sorted(glob.glob(os.path.join(out_dir, "figures", "*.png"))
                     + glob.glob(os.path.join(out_dir, "figures", "paper", "*.png")))
    if figures:
        lines += ["Figures: " + ", ".join(os.path.relpath(f, out_dir) for f in figures) + "."]
    else:
        lines += ["Figures: not generated yet (figures_main.ipynb, figures_si.ipynb)"]
    return lines
