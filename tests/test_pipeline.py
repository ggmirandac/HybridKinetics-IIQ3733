"""
Tests of the pipeline (one module per method under src/, settings in settings/*.csv).  Prints PASS / FAIL / SKIP per
test.  The checks: the settings reproduce the constants of the original code, default_params of every step, one
reaction group per rate equation, import of every module, the selection command line, the run infrastructure, the
bounds, the monotone refit, the reduced sensitivities, the output-level test, the in-sample prediction, a LOOCV fold,
a profile point and one iteration of the reduction.

Usage (repo root):  python tests/test_pipeline.py [--only name1,name2] [--n_jobs 8]
Reads Data/ and tests/fixtures/ (copies of the old checkpoint, the paper theta and the v3 fixing log); the
comparisons with the original code need legacy/ and are skipped when it is missing.  Writes only into temporary
directories.  Wall time: a few minutes (toy fits on 4 conditions, one reduced-sensitivity check at the old
checkpoint, one reduction iteration on the toy fit).
"""
import os
import sys
import copy
import json
import time
import shutil
import pickle
import inspect
import argparse
import warnings
import logging
import filecmp
import tempfile
import importlib
import importlib.util
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)
os.chdir(REPO)

import settings  # noqa: E402  (pins BLAS / OpenMP to one thread before numpy is loaded)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from settings import default_params  # noqa: E402
from data import (available_conditions, variable_bounds, check_bounds_contain_data, metabolite_bounds,  # noqa: E402
                  default_drain_bounds)
from model.parameters import ALL_PARAMS, PARAM_RXN_MAP, FREE_DRAIN_KEYS, reaction_groups  # noqa: E402
from model.kinetics import EcoliCarbonKinetics  # noqa: E402
from estimation.estimator import GlycolysisParameterEstimator  # noqa: E402
from estimation.refit import refit_with_fixed  # noqa: E402
from identifiability.structural import RateLaws, condition_states  # noqa: E402
from sensitivity import reduced_sensitivity  # noqa: E402
import paths  # noqa: E402
import identifiability.output_level as output_level  # noqa: E402
import identifiability.rules as rules  # noqa: E402
import selection  # noqa: E402
import final_analysis  # noqa: E402
import uncertainty.bootstrap as bootstrap  # noqa: E402
import uncertainty.loocv as loocv  # noqa: E402
import uncertainty.profiles as profiles  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("pyomo").setLevel(logging.ERROR)

TOY = ["KO02", "KO11", "KO13", "KO24"]
FIXTURES = os.path.join(REPO, "tests", "fixtures")
CHECKPOINT = os.path.join(FIXTURES, "v2_b2sd_checkpoint.pkl")
LEGACY_SRC = os.path.join(REPO, "legacy", "src_original")
LEGACY_V3 = os.path.join(REPO, "legacy", "v3_scripts_original")
RESULTS = []
TOY_FIT = []
LEGACY = {}

MODULES = ["paths", "settings", "solver", "progress", "data", "metrics", "sensitivity", "selection", "final_analysis",
           "report", "model", "model.parameters", "model.kinetics", "model.stoichiometry", "estimation",
           "estimation.experiment", "estimation.estimator", "estimation.parallel", "estimation.multistart",
           "estimation.refit", "estimation.anchors", "identifiability", "identifiability.structural",
           "identifiability.output_level", "identifiability.check", "identifiability.practical",
           "identifiability.rules", "uncertainty", "uncertainty.bootstrap", "uncertainty.profiles",
           "uncertainty.loocv"]

def report(name, ok, message):
    RESULTS.append((name, "PASS" if ok else "FAIL"))
    print("%s  %-34s %s" % ("PASS" if ok else "FAIL", name, message), flush=True)

def skip(name, message):
    RESULTS.append((name, "SKIP"))
    print("SKIP  %-34s %s" % (name, message), flush=True)

def paper_theta():
    """The paper's fitted theta (fixture copies of results/third_estimation): literature values updated by the fit."""
    sources = pd.read_csv(os.path.join(FIXTURES, "paper_theta_init_sources.csv"), index_col=0)["value"]
    fitted = pd.read_csv(os.path.join(FIXTURES, "paper_theta_fitted.csv"), index_col=0)["value"]
    theta = {p: float(sources[p]) for p in ALL_PARAMS}
    for name, value in fitted.items():
        if name in theta:
            theta[name] = float(value)
    return theta

def drain_bounds_2sd():
    """Drain bounds of the v2 b2sd run: mean +/- 2 sample sd of each cellular_needs.csv row."""
    needs = pd.read_csv(os.path.join(REPO, "Data", "cellular_needs.csv"), index_col=0)
    bounds = {}
    for key in FREE_DRAIN_KEYS:
        row = needs.loc[key].astype(float)
        mean, sd = float(row.mean()), float(row.std(ddof=1))
        bounds[key] = (mean - 2.0 * sd, mean + 2.0 * sd)
    return bounds

def v2_estimator():
    """
    Anchor-only estimator at the old checkpoint, on the v2 b2sd footing (metabolite bounds n_std = 2, drains
    mean +/- 2 sd).
    """
    with open(CHECKPOINT, "rb") as fh:
        checkpoint = pickle.load(fh)
    u_bounds, x_bounds = metabolite_bounds(n_std=2)
    est = GlycolysisParameterEstimator(conditions=available_conditions(), theta_init=dict(checkpoint["theta"]),
                                       free_params=list(checkpoint["free"]), x_bounds=x_bounds, u_bounds=u_bounds,
                                       drain_bounds=drain_bounds_2sd())
    return est.anchor_only(checkpoint["x_init"], checkpoint["u_init"], checkpoint["b_init"], checkpoint["theta"],
                           checkpoint["obj"])

def toy_est():
    """Quick 37-free fit on the 4 toy conditions with the bounds of these conditions (built once)."""
    if TOY_FIT:
        return TOY_FIT[0]
    u_bounds, x_bounds, drain_bounds = variable_bounds(TOY)
    est = GlycolysisParameterEstimator(conditions=TOY, theta_init=paper_theta(), x_bounds=x_bounds,
                                       u_bounds=u_bounds, drain_bounds=drain_bounds)
    est.estimate()
    est.refine_anchor(max_iter=5, primary_options=dict(settings.TIGHT_OPTIONS))
    TOY_FIT.append(est)
    return est

def legacy():
    """The original modules of legacy/src_original and the old v3_common, loaded under their own names."""
    if LEGACY:
        return LEGACY
    sys.path.insert(0, LEGACY_SRC)
    try:
        for name in ("kinetics_noor", "param_estimation_parmest", "structural_identifiability"):
            LEGACY[name] = importlib.import_module(name)
        spec = importlib.util.spec_from_file_location("old_v3_common", os.path.join(LEGACY_V3, "v3_common.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        LEGACY["v3_common"] = module
    finally:
        sys.path.remove(LEGACY_SRC)
    return LEGACY

def legacy_present():
    return os.path.isdir(LEGACY_SRC) and os.path.isdir(LEGACY_V3)

def t_glc_row():
    import fix_glc_row
    labelled = fix_glc_row.labelled_glucose()
    imbalanced = pd.read_csv("Data/imbalanced_metabolites.csv", index_col=0)
    row = imbalanced.loc["C_glc"].astype(float)
    reference = pd.Series({c: labelled[c] for c in imbalanced.columns})
    n_equal = int((np.abs(row - reference) < 1e-12).sum())
    expected = {"KO16": 0.00695, "KO05": 0.00983, "RF03": 0.00633}
    expected_ok = all(abs(row[k] - v) < 1e-9 and abs(reference[k] - v) < 1e-9 for k, v in expected.items())
    backup = os.path.exists("Data/imbalanced_metabolites_glc_shifted_backup.csv")
    report("glc_row", n_equal == 22 and expected_ok and backup,
           "%d/22 equal the labelled sheet; KO16 %.5f KO05 %.5f RF03 %.5f; backup present %s" % (
               n_equal, row["KO16"], row["KO05"], row["RF03"], backup))

def t_bounds():
    conditions = available_conditions()
    u_bounds, x_bounds, drain_bounds = variable_bounds()
    violations = check_bounds_contain_data(u_bounds, x_bounds, drain_bounds, raise_on_fail=False)
    u_v2, x_v2 = metabolite_bounds(n_std=2)
    violations_v2 = check_bounds_contain_data(u_v2, x_v2, drain_bounds, raise_on_fail=False)
    pi_ok = u_bounds["C_pi"] == (1.0, 20.0)
    # a fold (21 training conditions) must contain its own training data too
    fold_ok = True
    for held_out in ("KO18", "KO05", "RF03"):
        train = [c for c in conditions if c != held_out]
        u_fold, x_fold, d_fold = variable_bounds(train)
        fold_ok &= len(check_bounds_contain_data(u_fold, x_fold, d_fold, train, raise_on_fail=False)) == 0
    report("bounds_contain_data", len(violations) == 0 and len(violations_v2) >= 6 and pi_ok and fold_ok,
           "bounds: %d violations (n_std=2 control: %d); Pi bounds %s; fold bounds contain training data %s" % (
               len(violations), len(violations_v2), u_bounds["C_pi"], fold_ok))

def t_refit():
    est = toy_est()
    params = default_params("selection")
    obj_anchor = float(est.obj_value)
    new_free = [p for p in ALL_PARAMS if p not in ("Ks_pgp_7", "Kp_adp_3")]
    refit = refit_with_fixed(est, new_free, params, lambda message: None)
    limit = obj_anchor * (1 + params["refit_rtol"]) + params["refit_atol"]
    theta = refit.full_theta()
    theta_anchor = est.full_theta()
    fixed_ok = all(abs(theta[p] - theta_anchor[p]) < 1e-12 * max(1.0, abs(theta_anchor[p]))
                   for p in ("Ks_pgp_7", "Kp_adp_3"))
    report("refit_monotone", refit.obj_value <= limit and fixed_ok,
           "obj_anchor %.6f -> obj_after %.6f (status %s, %d free, fixed values kept %s)" % (
               obj_anchor, refit.obj_value, refit.refit_status, len(refit.free_params), fixed_ok))
    # every attempt fails (an option IPOPT does not know): the anchor itself must be kept, not a rise accepted
    broken = copy.copy(est)
    broken.solver_options = {"nonexistent_ipopt_option": 1}
    kept = refit_with_fixed(broken, new_free, params, lambda message: None)
    kept_theta = kept.full_theta()
    same = all(abs(kept_theta[p] - theta_anchor[p]) <= 1e-12 * max(1.0, abs(theta_anchor[p])) for p in ALL_PARAMS)
    report("refit_kept_anchor_fallback", kept.refit_status == "kept_anchor" and abs(kept.obj_value - obj_anchor) < 1e-12
           and len(kept.free_params) == len(new_free) and same,
           "all attempts failed -> status %s, obj %.6f (anchor %.6f), %d free, theta == anchor %s" % (
               kept.refit_status, kept.obj_value, obj_anchor, len(kept.free_params), same))

def t_reduced_sensitivity(n_jobs):
    est = v2_estimator()
    params = default_params("sensitivity")
    params["n_jobs"] = n_jobs
    names = ["Kp_f6p_pgi", "kcat_f_4", "kcat_f_3", "Ka3_1"]
    started = time.time()
    result = reduced_sensitivity(est, params, param_names=names)
    theta = est.full_theta()
    blocks = []
    for condition, columns in result["sensitivities"].items():
        values = np.where(np.isnan(columns.to_numpy(float)), 0.0, columns.to_numpy(float))
        weight = 1.0 / result["sigma"][condition].to_numpy(float)
        blocks.append(values * np.array([theta[p] for p in names])[None, :] * weight[:, None])
    J = np.vstack(blocks)
    influence = dict(zip(names, np.sqrt((J ** 2).mean(axis=0))))
    table = result["report"]["param_table"]
    flags = {p: table.loc[p, "flags"] for p in names if table.loc[p, "flags"]}
    ok = (abs(influence["Kp_f6p_pgi"] - 9.88) > 1.0 and influence["Kp_f6p_pgi"] < 1.0
          and 3.3 < influence["kcat_f_4"] < 3.9 and result["report"]["n_nan_entries"] == 0)
    report("A3_reduced_sensitivity", ok,
           "influence Kp_f6p_pgi %.4f (old artefact 9.88), kcat_f_4 %.3f (expected ~3.6); flags %s; "
           "inconsistent %d, re-solved %d, NaN %d; %.0fs" % (
               influence["Kp_f6p_pgi"], influence["kcat_f_4"], flags, result["report"]["n_inconsistent"],
               result["report"]["n_recovered_by_retry"], result["report"]["n_nan_entries"], time.time() - started))

def t_reduced_sensitivity_at_bound():
    """A step that would leave the theta bounds is replaced by the inward one-sided difference and flagged."""
    est = toy_est()
    param = "Ks_f6p_3"
    theta = est.full_theta()
    original = est.theta_bounds[param]
    est.theta_bounds[param] = (original[0], float(theta[param]) * 1.005)       # upper bound 0.5% above the fit
    params = default_params("sensitivity")
    params["n_jobs"] = 1
    try:
        result = reduced_sensitivity(est, params, param_names=[param, "kcat_f_4"])
    finally:
        est.theta_bounds[param] = original
    table = result["report"]["param_table"]
    finite = all(np.isfinite(columns[param].to_numpy(float)).all() for columns in result["sensitivities"].values())
    n_one_sided = result["report"]["n_one_sided"]
    n_nan = result["report"]["n_nan_entries"]
    report("A3_at_bound_inward_step", bool(table.loc[param, "at_bound"]) and finite
           and not bool(table.loc["kcat_f_4", "at_bound"]) and n_one_sided >= len(est.conditions) - n_nan,
           "upper bound 0.5%% above theta: at_bound flag %s, finite inward one-sided entries %s, n_one_sided %d, "
           "NaN %d" % (bool(table.loc[param, "at_bound"]), finite, n_one_sided, n_nan))

def t_output_level():
    est = v2_estimator()
    params = default_params("structural")
    result = output_level.output_level_test(est, est.free_params, RateLaws(), condition_states(est),
                                            settings.THETA_BOUNDS, params, 0)
    free = result["free"]
    loading = sorted(p for j, p in enumerate(free) if result["P_i"][j, j] > 0.5)
    expected = sorted(["kcat_f_7", "kcat_f_6", "Ka3_1", "kcat_f_3"])
    ratio_ok = float(result["ratio_anchor_i"][-4]) < 1e-10
    counts = {c: sum(1 for v in result["classes"].values() if v == c)
              for c in ("identifiable", "bound_identified", "non_identifiable")}
    report("A4_output_level", result["k_i"] == 4 and loading == expected and result["k_ii"] == 3 and ratio_ok,
           "version (i) nulls %d on %s (smallest-4 ratios %s); version (ii) nulls %d; classes %s" % (
               result["k_i"], loading, " ".join("%.0e" % x for x in result["ratio_anchor_i"][-4:]), result["k_ii"],
               counts))

def t_profile_point():
    est = toy_est()
    theta = est.full_theta()
    param = "Ks_f6p_3"
    x_init, u_init = est.primal_start()
    others = [q for q in est.free_params if q != param]
    point, tag = profiles.fit_point(list(est.conditions), param, float(theta[param]) * 1.2, others, theta, x_init,
                                    u_init, est.primal_b_free())
    ok = point is not None and np.isfinite(point.obj_value) and param not in point.free_params
    report("profile_point", ok, "profile point (%s x1.2, options '%s') -> obj %.5f (%d others re-optimized)" % (
        param, tag, float(point.obj_value) if point is not None else float("nan"),
        0 if point is None else len(point.free_params)))

def t_predict_fitted():
    est = toy_est()
    prediction = est.predict_fitted()
    x_by, _u_by, _b_by = est.fitted_state()
    condition = TOY[0]
    same = max(abs(float(prediction.predicted.loc[condition, k]) - x_by[condition][k]) for k in x_by[condition]) < 1e-12
    evaluated = (prediction.diagnostics["return_status"] == "evaluated").all()
    report("predict_evaluates_fitted_primal", same and evaluated and prediction.n_failed == 0,
           "predicted states == fitted x %s; diagnostics max residual %.1e; n_failed %d" % (
               same, float(prediction.diagnostics["max_residual"].max()), prediction.n_failed))

def t_loocv():
    est = toy_est()
    held_out = TOY[-1]
    params = default_params("loocv")
    params["loocv_n_starts"] = 3
    fold = loocv.loocv_fold(est, held_out, est.objective_breakdown(), params)
    ok = fold["fold_status"] in ("anchor_ok", "multistart_ok") and fold["obj_fold"] <= fold["bound"] * (1 + 1e-6)
    predicted_ok = "pred" in fold and np.isfinite(fold["pred"].to_numpy(float)).all()
    report("loocv_toy_fold", ok and predicted_ok,
           "fold %s: status %s, obj_fold %.6f <= bound %.6f; closure solve success %s, max residual %.1e; error '%s'" % (
               held_out, fold["fold_status"], fold["obj_fold"], fold["bound"], fold.get("diag", {}).get("success"),
               fold.get("diag", {}).get("max_residual", float("nan")), fold["error"]))

def t_reduction_smoke(n_jobs):
    """
    run_reduction on a fit of 5 conditions: the structural pre-pass and one iteration; the per-iteration files agree.
    (The rate-level check needs at least as many rate rows, 9 per condition, as free parameters: 5 conditions.)
    """
    conditions = TOY + ["RF03"]
    u_bounds, x_bounds, drain_bounds = variable_bounds(conditions)
    est = GlycolysisParameterEstimator(conditions=conditions, theta_init=paper_theta(), x_bounds=x_bounds,
                                       u_bounds=u_bounds, drain_bounds=drain_bounds)
    est.estimate()
    est.refine_anchor(max_iter=5, primary_options=dict(settings.TIGHT_OPTIONS))
    params = default_params("selection")
    params["max_iter"] = 1
    params["multistart_every"] = 0
    params["n_starts"] = 2
    params["n_jobs"] = n_jobs
    with tempfile.TemporaryDirectory() as tmp:
        final, status = selection.run_reduction(est, tmp, params)
        red = os.path.join(tmp, "reduction")
        iterations = pd.read_csv(os.path.join(red, "iterations.csv"))
        summary = json.load(open(os.path.join(red, "reduction_summary.json")))
        free_now = pd.read_csv(os.path.join(red, "iter_01", "free_params.csv"), index_col=0)
        free_next = pd.read_csv(os.path.join(red, "iter_01", "next_free_params.csv"), index_col=0)
        last = iterations[iterations["iter"] == 1].iloc[0]
        fixed_iteration = [p for p in str(last["fixed_this_iter"]).split(";") if p and p != "nan"]
        structural_fixed = [p for p in str(last.get("struct_fixed_this_iter", "")).split(";") if p and p != "nan"]
        now = list(free_now.index[free_now["status"] == "free"])
        after = list(free_next.index[free_next["status"] == "free"])
        files_ok = (set(now) - set(after) == set(fixed_iteration) | set(structural_fixed)
                    and after == summary["free"] == list(final.free_params)
                    and os.path.exists(os.path.join(red, "final_free_params.csv"))
                    and os.path.exists(os.path.join(red, "final_anchor_state.pkl")))
        columns_ok = "refit_status" in iterations.columns and "obj_before" in iterations.columns
    report("reduction_smoke", columns_ok and files_ok and status == summary["status"],
           "status %s, %d free after the pre-pass and 1 iteration, fixed in iteration 1 '%s'; per-iteration free set "
           "files consistent %s" % (status, len(summary["free"]), ";".join(fixed_iteration), files_ok))

def legacy_constants():
    """{setting name: value of the original code} for the settings the legacy code fixes, and the checks done."""
    old = legacy()
    estimation, structural, kinetics, common = (old["param_estimation_parmest"], old["structural_identifiability"],
                                                old["kinetics_noor"], old["v3_common"])
    reduction = inspect.signature(estimation.identifiability_reduction).parameters
    check = inspect.signature(structural.IterStructural.__init__).parameters
    sensitivity = inspect.signature(estimation.GlycolysisParameterEstimator.reduced_sensitivity).parameters
    values = {}
    for name in ("corr_threshold", "zero_cut", "eig_cut", "svd_tie_tol", "svd_min_loading", "max_structural_rounds",
                 "multistart_every", "n_starts"):
        values[name] = reduction[name].default
    values["max_iter"] = 30
    values["structural_q"] = check["q"].default
    values["seed_base"] = check["seed_base"].default
    values["structural_q_out"] = check["q_out"].default
    values["structural_sv_tol"] = structural.SV_TOL
    values["structural_guard_rel"] = structural.GUARD_REL
    values["structural_load_min"] = structural.LOAD_MIN
    values["active_rtol"] = structural.ACTIVE_RTOL
    values["rank_rcond"] = structural.RANK_RCOND
    values["sens_step"] = sensitivity["h"].default
    values["sens_step_check"] = sensitivity["h_check"].default
    values["refit_rtol"] = estimation.REFIT_RTOL
    values["refit_atol"] = estimation.REFIT_ATOL
    values["loocv_bound_rtol"] = estimation.LOOCV_BOUND_RTOL
    values["loocv_n_starts"] = inspect.signature(estimation.loocv_fold).parameters["n_starts"].default
    values["tie_rtol"] = inspect.signature(structural.choose_fixes).parameters["tie_rtol"].default
    values["realization_lo"] = inspect.signature(structural.realizations).parameters["lo"].default
    values["realization_hi"] = inspect.signature(structural.realizations).parameters["hi"].default
    values["perturb_lo"] = inspect.signature(structural.perturb_states).parameters["lo"].default
    values["perturb_hi"] = inspect.signature(structural.perturb_states).parameters["hi"].default
    values.update(legacy_text_constants(structural, estimation))
    values.update(legacy_script_defaults())
    values["n_jobs"] = common.N_JOBS
    values["bootstrap_seed"] = 0
    values["stage0_seed"] = 0
    values["isolation_n_starts"] = 30
    return values, {"estimation": estimation, "kinetics": kinetics, "common": common, "reduction": reduction}

def legacy_text_constants(structural, estimation):
    """Constants that were literals in the legacy code, read off its source text."""
    import re
    structural_text = open(structural.__file__).read()
    estimation_text = open(estimation.__file__).read()
    values = {}
    match = re.search(r"realizations\(anchor, bnds, q=q, seed=seed, free=free, lo=([0-9.]+), hi=([0-9.]+)\)",
                      structural_text)
    values["output_realization_lo"] = float(match.group(1))
    values["output_realization_hi"] = float(match.group(2))
    match = re.search(r"np.random.default_rng\(seed \+ 7000 \+ i\), ([0-9.]+), ([0-9.]+)\)", structural_text)
    values["output_perturb_lo"] = float(match.group(1))
    values["output_perturb_hi"] = float(match.group(2))
    values["final_seed_base"] = int(re.search(r"return ([0-9]+) \+ int\(it\)", structural_text).group(1))
    values["perturb_seed_offset"] = int(re.search(r"state_seed=seed \+ ([0-9]+)\)", structural_text).group(1))
    match = re.search(r"> ([0-9.]+) \* mx \+ ([0-9.e-]+)\)", estimation_text)
    values["fd_consistency_rtol"] = float(match.group(1))
    values["fd_consistency_atol"] = float(match.group(2))
    match = re.search(r"> ([0-9.]+) \* max\(float\(np.linalg.norm\(a_\)\),\s+float\(np.linalg.norm\(b_\)\)\) \+ "
                      r"([0-9.e-]+)", estimation_text)
    values["fd_consistency_rtol (inconsistent pairs)"] = float(match.group(1))
    values["fd_consistency_atol (inconsistent pairs)"] = float(match.group(2))
    values["multistart_top_n"] = int(re.search(r'conv\[~conv\["is_warm_start"\]\]\.head\(([0-9]+)\)',
                                               estimation_text).group(1))
    values["multistart_max_ratio"] = float(re.search(r'float\(row\["obj_value"\]\) > ([0-9.]+) \* est.obj_value',
                                                     estimation_text).group(1))
    values["end_multistart_rtol"] = float(re.search(
        r"cand.obj_value < est.obj_value - ([0-9.e-]+) \* abs\(est.obj_value\)", estimation_text).group(1))
    return values

def legacy_script_defaults():
    """argparse defaults and literals of the old v3 scripts."""
    texts = {}
    for name in ("v3_stage0.py", "v3_reduce.py", "v3_bootstrap.py", "v3_profiles.py", "v3_final.py"):
        texts[name] = open(os.path.join(LEGACY_V3, name)).read()
    needles = {"n_starts_stage0": ("v3_stage0.py", '"--n_starts", type=int, default=50'),
               "bootstrap_n": ("v3_bootstrap.py", '"--n", type=int, default=1000'),
               "bootstrap_chunk": ("v3_bootstrap.py", '"--chunk", type=int, default=100'),
               "profile_half_points": ("v3_profiles.py", '"--half_points", type=int, default=15'),
               "profile_span": ("v3_profiles.py", '"--span", type=float, default=0.5')}
    values = {}
    for name, (script, needle) in needles.items():
        if needle in texts[script]:
            values[name] = float(needle.split("default=")[-1])
    values["final_accept_lr_fraction"] = 0.5 if "0.5 * thr" in texts["v3_final.py"] else None
    values["lr_alpha"] = 0.95 if "lr_threshold(0.95" in texts["v3_final.py"] else None
    return values

def t_settings_match_legacy():
    """The settings CSV files reproduce the constants of legacy/src_original and of the old v3 scripts."""
    if not legacy_present():
        skip("settings_match_legacy", "legacy/ is not in this checkout (it is not part of the published repo)")
        return
    values, old = legacy_constants()
    estimation, kinetics, common, reduction = old["estimation"], old["kinetics"], old["common"], old["reduction"]
    thresholds = settings.THRESHOLDS
    mismatches = []
    n_checks = 0
    pairs = [("ALL_PARAMS", ALL_PARAMS, kinetics.ALL_PARAMS), ("PARAM_RXN_MAP", PARAM_RXN_MAP, kinetics.PARAM_RXN_MAP),
             ("THETA_BOUNDS keys", list(settings.THETA_BOUNDS), list(common.THETA_BOUNDS)),
             ("SOLVER_OPTIONS", settings.SOLVER_OPTIONS, common.SOLVER_OPTIONS),
             ("TIGHT_OPTIONS", settings.TIGHT_OPTIONS, estimation.TIGHT_OPTIONS),
             ("TIGHT_OPTIONS (v3_common)", settings.TIGHT_OPTIONS, common.TIGHT),
             ("RETRY_MONOTONE", settings.RETRY_MONOTONE_OPTIONS, {"mu_strategy": "monotone", "max_iter": 15000}),
             ("REFIT_MONOTONE", settings.REFIT_MONOTONE_OPTIONS, {"mu_strategy": "monotone", "max_iter": 10000}),
             ("n_std", settings.BOUNDS_RULES["n_std"], inspect.signature(common.v3_bounds).parameters["n_std"].default),
             ("drain_widen", settings.BOUNDS_RULES["drain_widen"],
              inspect.signature(common.v3_bounds).parameters["drain_widen"].default),
             ("pi bounds", {"C_pi": (settings.BOUNDS_RULES["pi_lo"], settings.BOUNDS_RULES["pi_hi"])},
              estimation.LITERATURE_U_BOUNDS),
             ("protected prefixes", settings.PROTECTED_PREFIXES, tuple(reduction["never_fix_prefix"].default)),
             ("svd_on_rank_deficiency (removed setting: always on)", True,
              reduction["svd_on_rank_deficiency"].default),
             ("casadi options", {"ipopt": settings.CASADI_OPTIONS, "print_time": 0},
              inspect.signature(kinetics.EcoliCarbonKinetics.__init__).parameters["options"].default),
             ("theta_bounds rows", len(pd.read_csv(os.path.join(paths.SETTINGS_DIR, "theta_bounds.csv"))), 37)]
    for p in ALL_PARAMS:
        pairs.append(("THETA_BOUNDS[%s]" % p, [float(x) for x in settings.THETA_BOUNDS[p]],
                      [float(x) for x in common.THETA_BOUNDS[p]]))
    u_bounds, x_bounds, drain_bounds = variable_bounds()
    pairs += [("U_BOUNDS", u_bounds, common.U_BOUNDS), ("X_BOUNDS", x_bounds, common.X_BOUNDS),
              ("DRAIN_BOUNDS", drain_bounds, common.DRAIN_BOUNDS)]
    for name, value in values.items():
        current = name.split(" (")[0]
        setting = thresholds.get(current, settings.UNCERTAINTY.get(current, settings.RUN.get(current)))
        pairs.append((name, float(setting) if isinstance(value, float) else setting, value))
    for label, new, old_value in pairs:
        n_checks += 1
        if new != old_value:
            mismatches.append("%s: new %r != old %r" % (label, new, old_value))
    warm_start = os.path.join(REPO, settings.RUN["stage0_warm_start"], "anchor_state.pkl")
    old_warm_start = os.path.join(REPO, "results_v2", "flux_model_b2sd", "stage0", "anchor_state.pkl")
    n_checks += 1
    if settings.RUN["stage0_warm_start"] != "Data/stage0_warm_start" or (
            os.path.exists(old_warm_start) and not filecmp.cmp(warm_start, old_warm_start, shallow=False)):
        mismatches.append("stage0 warm start is not the copy of the v2 b2sd stage-0 anchor")
    report("settings_match_legacy", not mismatches,
           "%d settings / constants compared with legacy/src_original and the old v3 scripts; mismatches: %s" % (
               n_checks, mismatches if mismatches else "none"))

def t_default_params():
    """default_params of every step: the CSV values, and (with legacy/) the constants of the original code."""
    sources = {}
    sources.update(settings.THRESHOLDS)
    sources.update(settings.UNCERTAINTY)
    sources.update(settings.RUN)
    legacy_values = legacy_constants()[0] if legacy_present() else {}
    problems = []
    n_compared = 0
    for step in settings.STEP_SETTINGS:
        params = default_params(step)
        expected = [name for names in settings.STEP_SETTINGS[step].values() for name in names]
        if list(params) != expected:
            problems.append("%s: keys %s" % (step, list(params)))
        for name, value in params.items():
            if value != sources[name]:
                problems.append("%s/%s: %r != CSV %r" % (step, name, value, sources[name]))
            if name in legacy_values:
                n_compared += 1
                if float(value) != float(legacy_values[name]):
                    problems.append("%s/%s: %r != original %r" % (step, name, value, legacy_values[name]))
    note = "%d values compared with the original code" % n_compared if legacy_values else "legacy/ absent: CSV only"
    report("default_params", not problems, "%d steps; %s; problems: %s" % (
        len(settings.STEP_SETTINGS), note, problems if problems else "none"))

def t_reaction_groups():
    """One group per reaction; the protections against the original PTS-only rule (with legacy/) and on the v3
    fixing sequence (fixture copy of results_v3 fixing_log.csv)."""
    groups = reaction_groups()
    pts_ok = groups["pts"] == ["v_max_1", "Ka1_1", "Ka2_1", "Ka3_1", "K_g6p_1"]
    cover_ok = len(groups) == 9 and sorted(p for g in groups.values() for p in g) == sorted(ALL_PARAMS)
    prefixes = settings.PROTECTED_PREFIXES
    fixing_log = pd.read_csv(os.path.join(FIXTURES, "v3_fixing_log.csv"))
    free = list(ALL_PARAMS)
    replay = True
    for name in fixing_log["param"]:
        replay &= rules.can_fix(name, free)
        free = [q for q in free if q != name]
    message = "9 groups, PTS group %s, cover all 37 %s; replay of %d v3 fixes: all fixable %s" % (
        pts_ok, cover_ok, len(fixing_log), replay)
    if not legacy_present():
        report("reaction_groups_all9", pts_ok and cover_ok and replay, message + "; legacy comparison skipped")
        return
    estimation, structural = legacy()["param_estimation_parmest"], legacy()["structural_identifiability"]
    pts_only = {"pts": list(estimation.PTS_GROUP)}
    rng = np.random.default_rng(0)
    n_diff = 0
    for _ in range(3000):
        # a random free set that keeps every protected parameter (the selection never fixes kcat_f_*)
        free = [p for p in ALL_PARAMS if p.startswith(prefixes) or rng.random() < rng.uniform(0.05, 1.0)]
        for p in ALL_PARAMS:
            n_diff += int(rules.can_fix(p, free) != estimation._can_fix(p, free, prefixes, pts_only))
    n_choices_diff = 0
    bounds = settings.THETA_BOUNDS
    for _ in range(400):
        reaction = list(PARAM_RXN_MAP)[int(rng.integers(0, 9))]
        all_members = list(PARAM_RXN_MAP[reaction])
        k = int(rng.integers(2, len(all_members) + 1)) if len(all_members) >= 2 else 1
        members = [all_members[i] for i in sorted(rng.choice(len(all_members), size=k, replace=False))]
        d = int(rng.integers(1, min(2, len(members)) + 1))
        basis = rng.normal(size=(d, len(members)))
        influence = {p: float(np.exp(rng.normal())) for p in members}
        if rng.random() < 0.5:                           # exact ties
            first = influence[members[0]]
            influence = {p: first * (1 + 0.05 * rng.random()) for p in members}
        values = {p: float(np.exp(rng.uniform(np.log(bounds[p][0]), np.log(bounds[p][1])))) for p in ALL_PARAMS}
        free_now = [p for p in ALL_PARAMS if p.startswith(prefixes) or rng.random() < 0.8]
        old_choice = structural.choose_fixes({"members": members}, influence, basis, members, values, bounds,
                                             never_fix_prefix=prefixes, pts_group=estimation.PTS_GROUP,
                                             free_now=list(free_now))
        new_choice = rules.choose_structural_fixes(influence, basis, members, values, bounds, list(free_now),
                                                   settings.THRESHOLDS["tie_rtol"])
        n_choices_diff += int(old_choice != new_choice)
    report("reaction_groups_all9", pts_ok and cover_ok and replay and n_diff == 0 and n_choices_diff == 0,
           message + "; can_fix vs the original PTS-only rule: %d differences in 3000 random free sets; structural "
           "fix choice old vs new: %d differences in 400 random groups" % (n_diff, n_choices_diff))

def t_import_smoke():
    failures = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:   # noqa: BLE001  (the test lists every module that fails to import)
            failures.append("%s: %s" % (name, str(exc)[:80]))
    report("import_smoke", not failures, "%d modules imported; failures: %s" % (
        len(MODULES), failures if failures else "none"))

def refused(path):
    try:
        paths.check_out_dir(path)
    except ValueError:
        return True
    return False

def t_selection_cli():
    result = subprocess.run([sys.executable, os.path.join(REPO, "src", "selection.py"), "--help"],
                            capture_output=True, text=True, cwd=REPO)
    help_ok = result.returncode == 0 and "--from_anchor" in result.stdout and "--resume" in result.stdout
    frozen_refused = all(refused(p) for p in ("results/x", "results_v2/x", "results_v3/flux_model",
                                              os.path.join(REPO, "results_v3")))
    fine = paths.check_out_dir("results_v4/flux_model") == os.path.join(REPO, "results_v4", "flux_model")
    working_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        # a relative path is relative to the repo root, not to the working directory
        os.chdir(tmp)
        try:
            relative_ok = paths.resolve_out("results_v4/x") == os.path.join(REPO, "results_v4", "x")
            relative_refused = refused("results_v3/x")
        finally:
            os.chdir(working_dir)
        # a symbolic link into results_v3 (a dangling one when results_v3 is not in the checkout)
        link = os.path.join(tmp, "link_to_v3")
        os.symlink(os.path.join(REPO, "results_v3"), link)
        link_refused = refused(os.path.join(link, "flux_model"))
        # the frozen folders are refused even where they do not exist: a repo root without them
        repo_root = paths.REPO_ROOT
        paths.REPO_ROOT = tmp
        try:
            absent_refused = all(refused(name + "/x") for name in ("results", "results_v2", "results_v3"))
            absent_allowed = not refused("results_v4/x")
        finally:
            paths.REPO_ROOT = repo_root
    case_refused = refused("Results_v3/x") if sys.platform in ("darwin", "win32") else True
    guard_ok = relative_ok and relative_refused and link_refused and case_refused and absent_refused and absent_allowed
    report("selection_cli", help_ok and frozen_refused and fine and guard_ok,
           "selection.py --help works %s; results/, results_v2/, results_v3/ refused %s (also when absent %s); "
           "results_v4/ accepted %s; relative --out against the repo root %s (refused from another cwd %s); symlink "
           "into results_v3 refused %s; other-case spelling refused %s" % (
               help_ok, frozen_refused, absent_refused, fine, relative_ok, relative_refused, link_refused,
               case_refused))

def refuses_frozen(call, frozen):
    try:
        call(frozen)
    except ValueError as exc:
        return "refusing" in str(exc)
    return False

def stage0_into(frozen):
    selection.run_stage0(frozen)

def reduction_into(frozen):
    selection.run_reduction(None, frozen)

def selection_into(frozen):
    selection.run_selection(frozen)

def final_into(frozen):
    final_analysis.run_final_analysis(frozen)

def bootstrap_into(frozen):
    bootstrap.run_bootstrap(frozen, frozen)

def profiles_into(frozen):
    profiles.run_profiles(frozen, frozen)

def loocv_into(frozen):
    loocv.run_loocv_analysis(frozen, frozen)

def t_run_infrastructure():
    """settings_used snapshot (overwrite + warning), --from_anchor file copy, the writers refuse frozen folders."""
    with tempfile.TemporaryDirectory() as tmp:
        first = settings.snapshot_settings(tmp)
        csv_names = sorted(n for n in os.listdir(paths.SETTINGS_DIR) if n.endswith(".csv"))
        first_ok = first == [] and sorted(os.listdir(os.path.join(tmp, "settings_used"))) == csv_names
        with open(os.path.join(tmp, "settings_used", "run.csv"), "a") as fh:
            fh.write("extra,1,int,edited\n")
        second = settings.snapshot_settings(tmp)
        same_after = filecmp.cmp(os.path.join(tmp, "settings_used", "run.csv"),
                                 os.path.join(paths.SETTINGS_DIR, "run.csv"), shallow=False)
        snapshot_ok = first_ok and second == ["run.csv"] and same_after
        # --from_anchor: every file is copied, nothing existing is overwritten
        source = os.path.join(tmp, "src_stage0")
        stage0 = os.path.join(tmp, "out", "stage0")
        os.makedirs(source)
        os.makedirs(stage0)
        for name in ("anchor_state.pkl", "theta_full.csv", "stage_row.json", "b_fit_vs_data.csv"):
            open(os.path.join(source, name), "w").write("new")
        open(os.path.join(stage0, "theta_full.csv"), "w").write("keep")
        found = selection.copy_anchor_files(source, stage0)
        copy_ok = (found == source and sorted(os.listdir(stage0)) == sorted(os.listdir(source))
                   and open(os.path.join(stage0, "theta_full.csv")).read() == "keep"
                   and open(os.path.join(stage0, "stage_row.json")).read() == "new")
        # a run directory (the anchor in its stage0/) is accepted too
        run_dir = os.path.join(tmp, "run")
        shutil.copytree(source, os.path.join(run_dir, "stage0"))
        stage0_b = os.path.join(tmp, "out2", "stage0")
        copy_ok = copy_ok and selection.copy_anchor_files(run_dir, stage0_b) == os.path.join(run_dir, "stage0")
        copy_ok = copy_ok and len(os.listdir(stage0_b)) == 4
    frozen = os.path.join(REPO, "results_v3", "x")
    calls = (stage0_into, reduction_into, selection_into, final_into, bootstrap_into, profiles_into, loocv_into)
    refusals = [refuses_frozen(call, frozen) for call in calls]
    report("run_infrastructure", snapshot_ok and copy_ok and all(refusals),
           "settings_used overwritten + differing files reported %s; --from_anchor copies all files, never overwrites "
           "%s; stage0 / reduction / selection / final / bootstrap / profiles / loocv refuse results_v3: %s" % (
               snapshot_ok, copy_ok, refusals))

def t_stage0_no_warm_start():
    """run_stage0 without the saved anchor: logged, candidate A skipped, multistart from the literature theta."""
    params = default_params("stage0")
    params["n_starts_stage0"] = 1
    params["n_jobs"] = 1
    with tempfile.TemporaryDirectory() as tmp:
        params["stage0_warm_start"] = os.path.join(tmp, "no_such_dir")
        stage0 = selection.run_stage0(tmp, params, TOY)
        row = json.load(open(os.path.join(stage0, "stage_row.json")))
        files = sorted(os.listdir(stage0))
    ok = (row["candidate_used"] == "multistart_best" and np.isnan(float(row["obj_b2sd_primal"]))
          and "anchor_state.pkl" in files and "stage_row.csv" in files and row["n_free"] == 37)
    report("stage0_no_warm_start", ok, "candidate used %s, obj_b2sd_primal %s (kept key, NaN), obj %.4f, %d files "
           "written" % (row["candidate_used"], row["obj_b2sd_primal"], row["obj"], len(files)))

def t_report():
    """REPORT.md is rebuilt from the saved files; a missing reference run (compare_to) only drops the comparison."""
    run_dir = os.path.join(REPO, settings.RUN["out_dir"])
    if not os.path.exists(os.path.join(run_dir, "REPORT.md")):
        skip("report", "no REPORT.md in %s" % run_dir)
        return
    import report as report_module
    same = report_module.build_report(run_dir) == open(os.path.join(run_dir, "REPORT.md")).read()
    if not os.path.isdir(os.path.join(REPO, settings.RUN["compare_to"])):
        same = True   # the saved report has the comparison with a reference run that is not in this checkout
    compare_to = settings.RUN["compare_to"]
    settings.RUN["compare_to"] = "no_such_reference_run"
    try:
        without_reference = report_module.build_report(run_dir)
    finally:
        settings.RUN["compare_to"] = compare_to
    degraded = "Comparison with" not in without_reference and "## Fit" in without_reference
    report("report", degraded and same, "rebuilt REPORT.md identical to the saved one %s; without the reference run "
           "the comparison is left out %s" % (same, degraded))

def t_estimator_defaults():
    """An estimator built without options takes the solver options, bounds and drain widening from settings."""
    est = GlycolysisParameterEstimator(conditions=TOY, free_params=["kcat_f_2", "Ks_g6p_pgi", "Kp_f6p_pgi"])
    u_bounds, x_bounds, drain_bounds = variable_bounds()
    setup = est.exp_list[0].setup
    options_ok = est.solver_options == settings.SOLVER_OPTIONS
    bounds_ok = (est.x_bounds == x_bounds and est.u_bounds == u_bounds and setup.x_bounds == x_bounds
                 and setup.u_bounds == u_bounds)
    drain_ok = default_drain_bounds() == drain_bounds and setup.drain_bounds == drain_bounds
    kinetics = EcoliCarbonKinetics(u_bounds, x_bounds)
    casadi_ok = kinetics.casadi_options == {"ipopt": settings.CASADI_OPTIONS, "print_time": 0}
    report("estimator_defaults_settings", options_ok and bounds_ok and drain_ok and casadi_ok,
           "default solver options == SOLVER_OPTIONS %s; default x/u bounds == variable_bounds %s; default drain "
           "bounds (widen %s) == variable_bounds %s; kinetics CasADi options == 'casadi' set %s" % (
               options_ok, bounds_ok, settings.BOUNDS_RULES["drain_widen"], drain_ok, casadi_ok))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()
    tests = [("imports", t_import_smoke), ("cli", t_selection_cli), ("infra", t_run_infrastructure),
             ("defaults", t_estimator_defaults), ("stage0", t_stage0_no_warm_start),
             ("settings", t_settings_match_legacy), ("params", t_default_params), ("groups", t_reaction_groups),
             ("glc_row", t_glc_row), ("bounds", t_bounds), ("refit", t_refit),
             ("a3", lambda: t_reduced_sensitivity(args.n_jobs)), ("a3b", t_reduced_sensitivity_at_bound),
             ("a4", t_output_level), ("predict", t_predict_fitted), ("loocv", t_loocv), ("profile", t_profile_point),
             ("report", t_report), ("reduction", lambda: t_reduction_smoke(args.n_jobs))]
    only = [name for name in args.only.split(",") if name]
    started = time.time()
    for name, test in tests:
        if only and name not in only:
            continue
        try:
            test()
        except Exception as exc:   # noqa: BLE001  (a crashing test is reported as FAIL with its traceback)
            import traceback
            report(name, False, "EXCEPTION %s: %s\n%s" % (type(exc).__name__, str(exc)[:300],
                                                          traceback.format_exc()[-800:]))
    n_pass = sum(1 for _, status in RESULTS if status == "PASS")
    n_skip = sum(1 for _, status in RESULTS if status == "SKIP")
    n_fail = sum(1 for _, status in RESULTS if status == "FAIL")
    print("\n%d passed, %d skipped, %d failed in %.0fs" % (n_pass, n_skip, n_fail, time.time() - started))
    return 0 if n_fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
