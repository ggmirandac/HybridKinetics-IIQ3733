"""
build_paper.py -- the figure files and the generated parts of the supplementary material, from the final run.

- paper/figures/: the main figures (fig2_conference.pdf) from results/flux_model/figures/paper/mainfigure;
- paper/supplementary/figures/: the supplementary figures (figS_*.pdf) from results/flux_model/figures/paper/supplementary;
- paper/supplementary/generated/: numbers.tex (the numbers the supplementary captions quote, as macros), the table of
  the parameter values across the stages and the table of the free parameters of the final model;
- paper/overleaf/ (and paper/overleaf.zip): a self-contained copy for the Overleaf project, with the generated tables
  pasted in, the numbers written out, the figures in supplementary_figures/ and the main figure next to it.

supplementary.tex is written by hand and inputs the generated files, so running this script after a new run of the
pipeline updates every figure, table and number of the supplementary material.  The notebooks figures_main and
figures_si must have been run on the same results.

usage (repo root): python paper/build_paper.py
"""
import os
import re
import sys
import json
import shutil
import zipfile

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "notebooks"))

import paper_style as S
from model.parameters import ALL_PARAMS
from settings import THETA_BOUNDS
from uncertainty.bootstrap import BOUND_LOG_DISTANCE

RESULTS = os.path.join(REPO, "results", "flux_model")
PAPER = os.path.join(REPO, "paper")
SUPPLEMENTARY = os.path.join(PAPER, "supplementary")
OVERLEAF = os.path.join(PAPER, "overleaf")
OVERLEAF_FIGURES = "supplementary_figures/"        # folder of the supplementary figures in the Overleaf project
OVERLEAF_ECOLI_MAP = "figuras/glycolisis_modified.png"   # the map already in the Overleaf project
MAIN_FIGURES = ["fig2_conference.pdf"]
SUPPLEMENTARY_FIGURES = ["figS_selection_path.pdf", "figS_final_identifiability.pdf", "figS_parity_stages.pdf",
                         "figS_parameter_evolution.pdf", "figS_bootstrap_all.pdf", "figS_profiles_all.pdf"]
PROFILE_CLASS_TEXT = {"identifiable": "identifiable", "wide": "wide", "one_sided": "one-sided", "flat": "flat"}

def result(*parts):
    return os.path.join(RESULTS, *parts)

def latex_number(value):
    """Three significant figures; a power of ten below 0.01 and from 10^4 on."""
    if value == 0:
        return "0"
    value = float("%.3g" % value)   # round first, so that 9.996 becomes 10.0 and not 10.00
    exponent = int(np.floor(np.log10(abs(value))))
    if -2 <= exponent < 4:
        decimals = max(0, 2 - exponent)
        return "%.*f" % (decimals, value)
    mantissa = value / 10.0 ** exponent
    return "$%.2f \\times 10^{%d}$" % (mantissa, exponent)

def method_setting(name, default):
    """A value of the settings used by the run (results/flux_model/settings_used/method.csv)."""
    table = pd.read_csv(result("settings_used", "method.csv"), dtype=str).set_index("name")
    if name not in table.index:
        return default
    return type(default)(float(table.loc[name, "value"]))

def copy_figures():
    os.makedirs(os.path.join(PAPER, "figures"), exist_ok=True)
    os.makedirs(os.path.join(SUPPLEMENTARY, "figures"), exist_ok=True)
    for name in MAIN_FIGURES:
        shutil.copy2(result("figures", "paper", "mainfigure", name), os.path.join(PAPER, "figures", name))
    for name in SUPPLEMENTARY_FIGURES:
        shutil.copy2(result("figures", "paper", "supplementary", name), os.path.join(SUPPLEMENTARY, "figures", name))
    print("copied %d main and %d supplementary figures" % (len(MAIN_FIGURES), len(SUPPLEMENTARY_FIGURES)))

def status_text(param, fix_log):
    """How the selection treated a parameter: free, or the rule (and practical iteration) that fixed it."""
    rows = fix_log[(fix_log["param"] == param) & (fix_log["action"] == "fixed")]
    if len(rows) == 0:
        return "free"
    row = rows.iloc[0]
    if row["rule"] == "structural_zero":
        return "structural zero"
    if row["rule"] == "svd":
        return "rank of $\\mathbf{F}$ (it. %d)" % int(row["iter"])
    return "correlation with %s (it. %d)" % (S.label(row["partner"]), int(row["iter"]))

def parameter_values_table():
    """Table: every parameter in the literature, after the initial estimation, the structural phase and the final model."""
    literature = pd.read_csv(os.path.join(REPO, "Data", "theta_init_sampled.csv"), index_col=0)
    initial = pd.read_csv(result("stage0", "theta_full.csv"), index_col=0)["value"]
    structural_free = pd.read_csv(result("reduction", "iter_01", "analysis.csv"), index_col=0)["value"]
    final = pd.read_csv(result("final", "theta_full.csv"), index_col=0)["value"]
    fix_log = pd.read_csv(result("reduction", "fixing_log.csv"))
    n_structural = int(len(structural_free))
    n_free = int((pd.read_csv(result("final", "final_structure.csv"))["status"] == "free").sum())
    lines = ["\\begin{longtable}{lrrrrl}",
             "\\caption{Values of the %d kinetic parameters in the literature model, after the initial estimation (all "
             "parameters free), after the structural phase (%d free) and in the final model (%d free). Fixed parameters "
             "keep the value estimated when they were fixed. Status: free in the final model, or the rule that fixed the "
             "parameter (practical iteration in parentheses). $^\\dagger$No reported value; a sampled value was used as "
             "the literature value.}" % (len(ALL_PARAMS), n_structural, n_free),
             "\\label{tab:param_evolution}\\\\",
             "\\toprule",
             "Parameter & Literature & Initial estimation & Structural phase & Final model & Status \\\\",
             "\\midrule",
             "\\endfirsthead",
             "\\toprule",
             "Parameter & Literature & Initial estimation & Structural phase & Final model & Status \\\\",
             "\\midrule",
             "\\endhead",
             "\\bottomrule",
             "\\endlastfoot"]
    for p in ALL_PARAMS:
        literature_text = latex_number(float(literature.loc[p, "value"]))
        if literature.loc[p, "source"] != "literature":
            literature_text += "$^\\dagger$"
        structural = float(structural_free[p]) if p in structural_free.index else float(initial[p])
        lines.append("%s & %s & %s & %s & %s & %s \\\\" % (S.label(p), literature_text, latex_number(float(initial[p])),
                     latex_number(structural), latex_number(float(final[p])), status_text(p, fix_log)))
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"

def joint_verdict(p, bootstrap, profiles):
    """What the two analyses bound together.  A side is bounded when the profile crosses the threshold on that side
    and the bootstrap interval does not reach the parameter bound there; a parameter bounded on one side only is
    constrained by the data from that side, and further data are needed to bound the other."""
    lower, upper = THETA_BOUNDS[p]
    low = float(bootstrap.loc[p, "p2.5"])
    high = float(bootstrap.loc[p, "p97.5"])
    bounded_below = np.isfinite(profiles.loc[p, "lower_decades"]) and np.log(low / lower) >= BOUND_LOG_DISTANCE
    bounded_above = np.isfinite(profiles.loc[p, "upper_decades"]) and np.log(upper / high) >= BOUND_LOG_DISTANCE
    if bool(bootstrap.loc[p, "kept_free_by_design"]):
        return "not identifiable"
    if bounded_below and bounded_above:
        if bool(bootstrap.loc[p, "practically_identifiable"]) and profiles.loc[p, "profile_class"] == "identifiable":
            return "identifiable"
        return "wide"
    if bounded_below:
        return "bounded below"
    if bounded_above:
        return "bounded above"
    return "not identifiable"

def free_parameters_table():
    """Table: estimate, bootstrap interval and verdict, and profile class of every free parameter of the final model."""
    final_structure = pd.read_csv(result("final", "final_structure.csv"))
    bootstrap = pd.read_csv(result("bootstrap", "bootstrap_summary.csv"), index_col=0)
    profiles = pd.read_csv(result("profiles", "profile_summary.csv"), index_col=0)
    free_in_table = set(final_structure.loc[final_structure["status"] == "free", "param"])
    free = [p for p in ALL_PARAMS if p in free_in_table]
    lines = ["\\begin{table}[h]",
             "\\centering",
             "\\caption{Free parameters of the final model, with the verdict of each uncertainty analysis: estimate, "
             "bootstrap 95\\% percentile interval, the ratio of its ends and the bootstrap verdict (practically "
             "identifiable if the ratio is at most \\SIbootRatio{} and the interval does not reach a parameter bound), and "
             "the class of the profile likelihood from $\\hat\\theta/\\SIprofileFactor$ to $\\SIprofileFactor\\,\\hat\\theta$ "
             "(an arm crosses where it exceeds the threshold and stays above it farther out; identifiable: both arms cross "
             "within a factor \\SIbootRatio{} in total; wide: both cross, farther "
             "apart; one-sided: one arm; flat: none). Near-optimal: 95\\% percentile interval of every bootstrap fit within "
             "the likelihood-ratio threshold of the best fit of its resample. Both: the sides on which the two analyses "
             "bound the parameter together, a side being bounded when the profile crosses the threshold on it and the "
             "bootstrap interval does not reach the parameter bound there (identifiable: both sides, and identifiable "
             "by each analysis; wide: both sides, farther apart; bounded below or above: one side only, so the data "
             "constrain the parameter from that side and further data are needed to bound the other; not "
             "identifiable: neither side). $^\\ddagger$The interval reaches a parameter bound.}",
             "\\label{tab:free_parameters}",
             "\\begin{tabular}{lrrrrlll}",
             "\\toprule",
             "Parameter & Estimate & Bootstrap 95\\% interval & Ratio & Near-optimal interval & Bootstrap & Profile & Both \\\\",
             "\\midrule"]
    for p in free:
        interval = "%s -- %s" % (latex_number(float(bootstrap.loc[p, "p2.5"])), latex_number(float(bootstrap.loc[p, "p97.5"])))
        if bool(bootstrap.loc[p, "interval_reaches_bound"]):
            interval += "$^\\ddagger$"
        near = "%s -- %s" % (latex_number(float(bootstrap.loc[p, "near_p2.5"])), latex_number(float(bootstrap.loc[p, "near_p97.5"])))
        verdict = "practically identifiable" if bool(bootstrap.loc[p, "practically_identifiable"]) else "not identifiable"
        both = joint_verdict(p, bootstrap, profiles)
        lines.append("%s & %s & %s & %s & %s & %s & %s & %s \\\\" % (S.label(p), latex_number(float(bootstrap.loc[p, "estimate"])), interval,
                     latex_number(float(bootstrap.loc[p, "ci_ratio_97.5_over_2.5"])), near, verdict, PROFILE_CLASS_TEXT[profiles.loc[p, "profile_class"]], both))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"

def numbers():
    """The numbers quoted by the supplementary captions, as LaTeX macros."""
    final_summary = json.load(open(result("final", "final_summary.json")))
    bootstrap_info = json.load(open(result("bootstrap", "bootstrap_info.json")))
    thresholds = json.load(open(result("profiles", "profile_thresholds.json")))
    fix_log = pd.read_csv(result("reduction", "fixing_log.csv"))
    fixed = fix_log[fix_log["action"] == "fixed"]
    n_local = method_setting("bootstrap_local_starts", 0)
    local_span = method_setting("bootstrap_local_span", 1.0)
    n_seeds = method_setting("bootstrap_profile_seeds", 0)
    if n_local > 0:
        starts = ("warm started at the final estimate, from %d profile likelihood points drawn at random among those "
                  "within the likelihood-ratio threshold and from %d further starts with every free parameter at the "
                  "estimate times $10^{u}$, $u \\sim U(-%g, %g)$; the best fit of each resample is kept" % (n_seeds, n_local, local_span, local_span))
    else:
        starts = "warm started at the final estimate"
    values = {
        "SInAll": "%d" % len(ALL_PARAMS),
        "SInFree": "%d" % int(final_summary["n_free"]),
        "SInStructural": "%d" % (len(ALL_PARAMS) - int((fixed["rule"] == "structural_zero").sum())),
        "SInStructuralZeros": "%d" % int((fixed["rule"] == "structural_zero").sum()),
        "SInRankFixes": "%d" % int((fixed["rule"] == "svd").sum()),
        "SInCorrelationFixes": "%d" % int((fixed["rule"] == "correlation").sum()),
        "SInConditions": "%d" % len(pd.read_csv(result("bootstrap", "bootstrap_resamples.csv"))["conditions"].iloc[0].split(";")),
        "SIbootN": "%d" % int(bootstrap_info["n_resamples"]),
        "SIbootConverged": "%d" % int(bootstrap_info["n_converged"]),
        "SIbootStarts": starts,
        "SIbootRatio": "%g" % method_setting("bootstrap_ci_ratio", 10.0),
        "SIprofileSpan": "%g" % method_setting("profile_span", 1.0),
        "SIprofilePoints": "%d" % method_setting("profile_points", 30),
        "SIprofileFactor": "10" if method_setting("profile_span", 1.0) == 1.0 else "10^{%g}" % method_setting("profile_span", 1.0),
        "SIthreshold": "%.3f" % float(thresholds["lr_threshold"]),
        "SIepsOne": "10^{%d}" % round(np.log10(method_setting("structural_tol", 1e-10))),
        "SIepsTwo": "10^{%d}" % round(2 * np.log10(method_setting("rank_tol", 1e-4))),
        "SIcorrelation": "%g" % method_setting("corr_threshold", 0.9),
        "SIobjFinal": "%.3f" % float(final_summary["obj"]),
    }
    lines = ["%% generated by paper/build_paper.py from results/flux_model; do not edit"]
    for name, value in values.items():
        lines.append("\\newcommand{\\%s}{%s}" % (name, value))
    return "\n".join(lines) + "\n", values

def overleaf_copy(values):
    """
    paper/overleaf/supplementary.tex: supplementary.tex with the generated tables pasted in and every number macro
    replaced by its value, so that it needs no other file but its figures; with the figures and a zip of the folder.
    """
    text = open(os.path.join(SUPPLEMENTARY, "supplementary.tex"), encoding="utf-8").read()
    text = text.replace(text[text.index("% Supplementary material."):text.index("\\input{\\SIdir generated/numbers}\n") + len("\\input{\\SIdir generated/numbers}\n")],
                        "% Supplementary material, made by paper/build_paper.py from the results of the pipeline (results/flux_model).\n")
    for name in ["table_parameter_values", "table_free_parameters"]:
        table = open(os.path.join(SUPPLEMENTARY, "generated", name + ".tex"), encoding="utf-8").read()
        text = text.replace("\\input{\\SIdir generated/%s}" % name, table.rstrip("\n"))
    text = text.replace("\\SIdir figures/glycolisis_modified.png", OVERLEAF_ECOLI_MAP)
    text = text.replace("\\SIdir figures/", OVERLEAF_FIGURES)
    # the longest names first, so that no macro name is replaced inside a longer one
    for name in sorted(values, key=len, reverse=True):
        text = text.replace("\\%s{}" % name, values[name])
        text = re.sub(r"\\%s(?![A-Za-z])" % name, lambda match: values[name], text)
    left = re.findall(r"\\SI[A-Za-z]+", text)
    if left:
        raise ValueError("macros left in the Overleaf copy: %s" % sorted(set(left)))
    os.makedirs(os.path.join(OVERLEAF, OVERLEAF_FIGURES), exist_ok=True)
    with open(os.path.join(OVERLEAF, "supplementary.tex"), "w", encoding="utf-8") as fh:
        fh.write(text)
    for name in SUPPLEMENTARY_FIGURES:
        shutil.copy2(os.path.join(SUPPLEMENTARY, "figures", name), os.path.join(OVERLEAF, OVERLEAF_FIGURES, name))
    for name in MAIN_FIGURES:
        shutil.copy2(os.path.join(PAPER, "figures", name), os.path.join(OVERLEAF, name))
    with zipfile.ZipFile(os.path.join(PAPER, "overleaf.zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(os.path.join(OVERLEAF, "supplementary.tex"), "supplementary.tex")
        for name in MAIN_FIGURES:
            archive.write(os.path.join(OVERLEAF, name), name)
        for name in SUPPLEMENTARY_FIGURES:
            archive.write(os.path.join(OVERLEAF, OVERLEAF_FIGURES, name), OVERLEAF_FIGURES + name)
    print("Overleaf copy: paper/overleaf/ and paper/overleaf.zip")

if __name__ == "__main__":
    copy_figures()
    generated = os.path.join(SUPPLEMENTARY, "generated")
    os.makedirs(generated, exist_ok=True)
    text, values = numbers()
    with open(os.path.join(generated, "numbers.tex"), "w") as fh:
        fh.write(text)
    with open(os.path.join(generated, "table_parameter_values.tex"), "w") as fh:
        fh.write(parameter_values_table())
    with open(os.path.join(generated, "table_free_parameters.tex"), "w") as fh:
        fh.write(free_parameters_table())
    overleaf_copy(values)
    for name, value in values.items():
        print("  \\%-20s %s" % (name, value))
