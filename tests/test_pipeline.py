"""
Tests of the pipeline (one module per method under src/, settings in settings/*.csv).  Prints PASS / FAIL per test.
The checks: import of every module, the settings, one reaction group per rate equation, the glucose data row, the
bounds, the refit, the exact sensitivities against finite differences, the structural zeros of the rate laws, the
practical criteria on known matrices, the protections, the in-sample prediction, a LOOCV fold, a profile point and
the report.

Usage (repo root):  python tests/test_pipeline.py [--only name1,name2]
Reads Data/ and writes only into temporary directories.  Wall time: about a minute (fits on 4 conditions).
"""
import os
import sys
import time
import argparse
import warnings
import logging
import importlib

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)
os.chdir(REPO)

import settings  # noqa: E402  (pins BLAS / OpenMP to one thread before numpy is loaded)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from settings import default_params  # noqa: E402
from data import available_conditions, variable_bounds, check_bounds_contain_data  # noqa: E402
from metrics import theta_at_bounds  # noqa: E402
from model.parameters import ALL_PARAMS, PARAM_RXN_MAP, reaction_groups  # noqa: E402
from estimation.estimator import GlycolysisParameterEstimator  # noqa: E402
from estimation.refit import refit_with_fixed  # noqa: E402
from identifiability.structural import (RateLaws, condition_states, stacked_sensitivity_matrix,  # noqa: E402
                                        singular_spectrum)
from identifiability.practical import fim_analysis  # noqa: E402
from identifiability.rules import is_protected, largest_loading  # noqa: E402
from selection import practical_decision  # noqa: E402
import sensitivity  # noqa: E402
import report  # noqa: E402
import uncertainty.loocv as loocv  # noqa: E402
import uncertainty.profiles as profiles  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("pyomo").setLevel(logging.ERROR)

TOY = ["KO02", "KO11", "KO13", "KO24"]
FIXTURES = os.path.join(REPO, "tests", "fixtures")
RESULTS = []
TOY_FIT = []

MODULES = ["paths", "settings", "solver", "progress", "data", "metrics", "sensitivity", "selection", "final_analysis",
           "report", "model", "model.parameters", "model.kinetics", "model.stoichiometry", "estimation",
           "estimation.experiment", "estimation.estimator", "estimation.parallel", "estimation.multistart",
           "estimation.refit", "estimation.anchors", "identifiability", "identifiability.structural",
           "identifiability.practical", "identifiability.rules", "uncertainty", "uncertainty.bootstrap",
           "uncertainty.profiles", "uncertainty.loocv"]

def report_result(name, ok, message):
    RESULTS.append((name, "PASS" if ok else "FAIL"))
    print("%s  %-30s %s" % ("PASS" if ok else "FAIL", name, message), flush=True)

def fitted_theta():
    """A fitted theta (fixture files paper_theta_*.csv): literature values updated by a fit."""
    sources = pd.read_csv(os.path.join(FIXTURES, "paper_theta_init_sources.csv"), index_col=0)["value"]
    fitted = pd.read_csv(os.path.join(FIXTURES, "paper_theta_fitted.csv"), index_col=0)["value"]
    theta = {p: float(sources[p]) for p in ALL_PARAMS}
    for name, value in fitted.items():
        if name in theta:
            theta[name] = float(value)
    return theta

def toy_est():
    """A 37-free fit on the 4 toy conditions with the bounds of these conditions (built once)."""
    if TOY_FIT:
        return TOY_FIT[0]
    u_bounds, x_bounds, drain_bounds = variable_bounds(TOY)
    est = GlycolysisParameterEstimator(conditions=TOY, theta_init=fitted_theta(), x_bounds=x_bounds,
                                       u_bounds=u_bounds, drain_bounds=drain_bounds)
    est.estimate()
    est.refine_anchor(max_rounds=5)
    TOY_FIT.append(est)
    return est

def t_imports():
    failed = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:   # noqa: BLE001  (a failing import is the test result)
            failed.append("%s (%s)" % (name, exc))
    report_result("imports", not failed, "%d modules; failed: %s" % (len(MODULES), failed or "none"))

def t_settings():
    params = default_params()
    names = pd.read_csv(os.path.join(REPO, "settings", "method.csv"))["name"].tolist()
    warm_ok = all(settings.WARM_START_OPTIONS[k] == v for k, v in settings.solver_options("warm_start").items())
    bounds_ok = sorted(settings.THETA_BOUNDS) == sorted(ALL_PARAMS)
    report_result("settings", list(params) == names and warm_ok and bounds_ok,
                  "%d method settings; warm-start set contains its options %s; bounds of all 37 parameters %s" % (
                      len(params), warm_ok, bounds_ok))

def t_reaction_groups():
    groups = reaction_groups()
    members = [p for group in groups.values() for p in group]
    ok = len(groups) == 9 and sorted(members) == sorted(ALL_PARAMS) and len(members) == len(set(members))
    ok &= all(groups[r] == list(PARAM_RXN_MAP[r]) for r in groups)
    report_result("reaction_groups", ok, "%d groups, every parameter in exactly one: %s" % (len(groups), ok))

def t_glc_row():
    raw = pd.read_excel("Data/data_important.xlsx", sheet_name="Extracelular", header=None)
    glucose_row = raw.iloc[:, 0].tolist().index("Glucose")
    labelled = {}
    for column, label in enumerate(raw.iloc[0].tolist()):
        if isinstance(label, str) and label != "Sample ID":
            labelled[label] = float(raw.iloc[glucose_row, column])
    imbalanced = pd.read_csv("Data/imbalanced_metabolites.csv", index_col=0)
    row = imbalanced.loc["C_glc"].astype(float)
    reference = pd.Series({c: labelled[c] for c in imbalanced.columns})
    n_equal = int((np.abs(row - reference) < 1e-12).sum())
    report_result("glc_row", n_equal == 22, "%d/22 glucose values equal the labelled sheet" % n_equal)

def t_bounds():
    conditions = available_conditions()
    u_bounds, x_bounds, drain_bounds = variable_bounds()
    violations = check_bounds_contain_data(u_bounds, x_bounds, drain_bounds, raise_on_fail=False)
    fold_ok = True
    for held_out in ("KO18", "KO05", "RF03"):
        train = [c for c in conditions if c != held_out]
        u_fold, x_fold, d_fold = variable_bounds(train)
        fold_ok &= len(check_bounds_contain_data(u_fold, x_fold, d_fold, train, raise_on_fail=False)) == 0
    report_result("bounds_contain_data", len(violations) == 0 and u_bounds["C_pi"] == (1.0, 20.0) and fold_ok,
                  "%d violations; Pi bounds %s; fold bounds contain their training data %s" % (
                      len(violations), u_bounds["C_pi"], fold_ok))

def t_refit():
    est = toy_est()
    fixed = ("Ks_pgp_7", "Kp_adp_3")
    new_free = [p for p in ALL_PARAMS if p not in fixed]
    refit = refit_with_fixed(est, new_free, lambda message: None)
    theta = refit.full_theta()
    theta_before = est.full_theta()
    fixed_ok = all(abs(theta[p] - theta_before[p]) < 1e-12 * max(1.0, abs(theta_before[p])) for p in fixed)
    report_result("refit", refit.obj_value <= est.obj_value and fixed_ok and len(refit.free_params) == 35,
                  "obj %.6f -> %.6f, fixed values kept %s, %d free" % (est.obj_value, refit.obj_value, fixed_ok,
                                                                       len(refit.free_params)))

def t_exact_sensitivity():
    """
    The CasADi implicit-function-theorem sensitivities against central differences of re-solved inner problems for
    two parameters inside their bounds (a pair whose re-solves do not both converge is skipped).
    """
    est = toy_est()
    names = ["kcat_f_4", "Ks_g6p_pgi"]
    result = sensitivity.reduced_sensitivity(est, param_names=names)
    theta = est.full_theta()
    x_init, u_init = est.primal_start()
    problems = sensitivity.InnerProblems(est.setup, est.conditions, theta, (x_init, u_init, est.primal_b_free()))
    step = 1e-3
    n_compared = 0
    worst = 0.0
    for condition in est.conditions:
        record = problems.models[condition]
        for name in names:
            outputs = []
            converged = []
            for sign in (1.0, -1.0):
                getattr(record["model"], name).set_value(theta[name] * np.exp(sign * step))
                converged.append(problems.solve(condition))
                outputs.append(np.array([component() for component in record["outputs"]]))
            getattr(record["model"], name).set_value(theta[name])
            if not all(converged):
                continue
            n_compared += 1
            difference = (outputs[0] - outputs[1]) / (2 * step) / record["sigma"]
            exact = result["sensitivities"][condition][name].to_numpy(float) * theta[name] / record["sigma"]
            size = max(float(np.linalg.norm(difference)), float(np.linalg.norm(exact)), 1e-2)
            worst = max(worst, float(np.linalg.norm(exact - difference)) / size)
    at_bound = [p for p in names if p in theta_at_bounds(est)]
    report_result("exact_sensitivity", worst < 1e-2 and n_compared >= 6 and not result["failed"] and not at_bound,
                  "largest relative difference to central differences %.1e (%d pairs); failed conditions %s; on a "
                  "bound %s" % (worst, n_compared, result["failed"], at_bound))

def t_structural():
    """The Noor rate laws have structural zeros: the product of the PFK substrate constants is all that matters."""
    est = toy_est()
    free = list(est.free_params)
    S = stacked_sensitivity_matrix(RateLaws(), est.full_theta(), condition_states(est), free, default_params())
    ratio, Vt = singular_spectrum(S)
    n_zero = int(np.sum(ratio < default_params()["structural_tol"]))
    pfk = [free.index("Ks_f6p_3"), free.index("Ks_atp_3")]
    pair_in_null = bool(np.linalg.norm(Vt[ratio < 1e-10][:, pfk]) > 0.5)
    name, loading = largest_loading(free, Vt[-1])
    report_result("structural", n_zero >= 8 and pair_in_null,
                  "%d structural zeros (37 free), gap %.1e -> %.1e; PFK pair in the null space %s; first rule "
                  "target %s (%.3f)" % (n_zero, ratio[-n_zero], ratio[-n_zero - 1], pair_in_null, name, loading))

def t_practical():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 3))
    correlated = pd.DataFrame(np.column_stack([base, base[:, 0] + 0.1 * rng.normal(size=200)]),
                              columns=["a", "b", "c", "d"])
    relative = correlated.abs()
    relative["d"] = 0.5 * relative["d"]
    analysis = fim_analysis(correlated, relative, ["a", "b", "c", "d"], default_params())
    pair_ok = analysis["full_rank"] and analysis["pairs"][0][:2] == ("a", "d")
    fixes_less_sensitive = practical_decision(analysis, ["a", "b", "c", "d"])[0] == "d"
    deficient = correlated.copy()
    deficient["d"] = 2.0 * deficient["a"]
    analysis = fim_analysis(deficient, relative, ["a", "b", "c", "d"], default_params())
    name, _loading = largest_loading(analysis["names"], analysis["null_vector"])
    rank_ok = not analysis["full_rank"] and name == "a"
    report_result("practical", pair_ok and fixes_less_sensitive and rank_ok,
                  "correlated pair found %s; the less sensitive member 'd' is fixed %s; rank deficiency found and "
                  "rule acts on 'a' %s" % (pair_ok, fixes_less_sensitive, rank_ok))

def t_protection():
    pts = list(PARAM_RXN_MAP["pts"])
    others = [p for p in ALL_PARAMS if p not in pts]
    last_ok = is_protected("Ka1_1", others + ["Ka1_1"]) and not is_protected("Ka1_1", ALL_PARAMS)
    report_result("protection", last_ok, "the last free parameter of a rate law is protected: %s" % last_ok)

def t_predict_fitted():
    est = toy_est()
    prediction = est.predict_fitted()
    x_by, _u_by, _b_by = est.fitted_state()
    condition = TOY[0]
    same = max(abs(float(prediction.predicted.loc[condition, k]) - x_by[condition][k]) for k in x_by[condition]) < 1e-12
    report_result("predict_fitted", same and prediction.n_failed == 0,
                  "predicted states == fitted x %s; n_failed %d" % (same, prediction.n_failed))

def t_loocv():
    fold = loocv.loocv_fold(toy_est(), TOY[-1])
    predicted_ok = "pred" in fold and np.isfinite(fold["pred"].to_numpy(float)).all()
    report_result("loocv_fold", predicted_ok and not fold["error"],
                  "fold %s: obj %.5f, closure success %s, error '%s'" % (
                      TOY[-1], fold["obj_fold"], fold.get("diag", {}).get("success"), fold["error"]))

def t_profile_point():
    est = toy_est()
    theta = est.full_theta()
    param = "kcat_f_4"
    x_init, u_init = est.primal_start()
    others = [q for q in est.free_params if q != param]
    point = profiles.fit_point(list(est.conditions), param, float(theta[param]) * 1.2, others, theta, x_init,
                               u_init, est.primal_b_free())
    ok = point is not None and np.isfinite(point.obj_value) and param not in point.free_params
    report_result("profile_point", ok, "%s x1.2 -> obj %.5f (%d others re-optimized)" % (
        param, float("nan") if point is None else point.obj_value, 0 if point is None else len(point.free_params)))

def t_report():
    text = report.build_report(os.path.join(REPO, "results", "flux_model"))
    report_result("report", text.startswith("# Flux model") and "## Identifiable structure" in text,
                  "%d lines" % len(text.splitlines()))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    tests = [("imports", t_imports), ("settings", t_settings), ("groups", t_reaction_groups), ("glc_row", t_glc_row),
             ("bounds", t_bounds), ("refit", t_refit), ("exact", t_exact_sensitivity), ("structural", t_structural),
             ("practical", t_practical), ("protection", t_protection), ("predict", t_predict_fitted),
             ("loocv", t_loocv), ("profile", t_profile_point), ("report", t_report)]
    only = [name for name in args.only.split(",") if name]
    started = time.time()
    for name, test in tests:
        if only and name not in only:
            continue
        try:
            test()
        except Exception as exc:   # noqa: BLE001  (a crashing test is reported as FAIL with its traceback)
            import traceback
            report_result(name, False, "EXCEPTION %s: %s\n%s" % (type(exc).__name__, str(exc)[:300],
                                                                 traceback.format_exc()[-800:]))
    n_pass = sum(1 for _, status in RESULTS if status == "PASS")
    n_fail = sum(1 for _, status in RESULTS if status == "FAIL")
    print("\n%d passed, %d failed in %.0fs" % (n_pass, n_fail, time.time() - started))
    return 0 if n_fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
