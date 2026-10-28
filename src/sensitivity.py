"""
sensitivity.py -- reduced (profiled) sensitivities of the measured outputs.

For each condition the inner problem min_{x, u, b} weighted SSE is solved with ALL kinetic parameters fixed at theta;
y*(theta) are the model values of the measured outputs at that optimum.  This is the problem that is actually
estimated: the free cofactors and drains absorb part of every parameter change, which sensitivities at fixed
(x, u, b) would miss.  The reduced sensitivities dy*/dtheta are finite differences in ln theta with
  - forward / backward consistency checks (an inconsistent pair is re-solved, then the smaller one-sided value is
    used: never invented information),
  - one-sided steps at the theta bounds (flag at_bound),
  - a column check at a second step (flag fd_unstable),
  - NaN entries counted and never set to zero (flag fd_failed for NaN in more than 2 conditions).
The solves are parallelised over (parameter, step) with a process pool.

Position in the pipeline: input of the practical identifiability analysis (identifiability/practical.py) in every
iteration of the parameter selection; also the sensitivity heatmap of the paper figures.
"""
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import pyomo.environ as pyo

from settings import TIGHT_OPTIONS
from solver import check_solver
from model.parameters import ALL_PARAMS
from estimation.experiment import GlycolysisExperiment

INNER_BASE = {"print_level": 0, "max_iter": 10000}

# Every inner solve starts at the FITTED primal point with the tight monotone warm start, so all solves stay in one
# basin: the adaptive barrier update of the default options switches branches between neighbouring solves, which
# showed up as huge finite-difference artefacts.  No single barrier start works for every condition (the inner
# problems are ill-conditioned: KO05 only converges from mu_init ~1e-3, KO08 / KO20 from 1e-7), so each solve walks a
# fixed LADDER of option sets, in order, until one terminates optimal.  Ladder 0 is the primary one, ladder 1
# (tighter) re-solves failed entries and inconsistent forward / backward pairs.
PRIMARY_LADDER = [
    {**TIGHT_OPTIONS, **INNER_BASE, "tol": 1e-8, "acceptable_tol": 1e-8, "mu_init": 1e-9},
    {**TIGHT_OPTIONS, **INNER_BASE, "tol": 1e-8, "acceptable_tol": 1e-8, "mu_init": 1e-7},
    {**TIGHT_OPTIONS, **INNER_BASE, "tol": 1e-8, "acceptable_tol": 1e-8, "mu_init": 1e-3},
    {**INNER_BASE, "warm_start_init_point": "yes", "mu_strategy": "adaptive", "tol": 1e-8, "acceptable_tol": 1e-8},
]
RESOLVE_LADDER = [
    {**TIGHT_OPTIONS, **INNER_BASE, "max_iter": 20000, "tol": 1e-9, "acceptable_tol": 1e-9, "acceptable_iter": 30,
     "mu_init": 1e-11},
    {**TIGHT_OPTIONS, **INNER_BASE, "max_iter": 20000, "tol": 1e-9, "acceptable_tol": 1e-9, "acceptable_iter": 30,
     "mu_init": 1e-8},
    {**TIGHT_OPTIONS, **INNER_BASE, "max_iter": 20000, "tol": 1e-9, "acceptable_tol": 1e-9, "acceptable_iter": 30,
     "mu_init": 1e-4},
    {**INNER_BASE, "max_iter": 20000, "warm_start_init_point": "yes", "mu_strategy": "adaptive", "tol": 1e-9,
     "acceptable_tol": 1e-9},
]
LADDERS = (PRIMARY_LADDER, RESOLVE_LADDER)

class InnerProblems:
    """The inner problems (all theta fixed, objective = weighted SSE) of every condition, re-solvable at any theta."""

    def __init__(self, setup, conditions, theta_full, primal, solver_options):
        self.solver = pyo.SolverFactory("ipopt")
        self.base_options = dict(solver_options)
        x_init, u_init, b_init = primal
        self.models = {}
        for condition in conditions:
            experiment = GlycolysisExperiment(condition, setup, theta_init=theta_full, x_init=x_init.get(condition),
                                              u_init=u_init.get(condition), b_init=b_init.get(condition))
            model = experiment.get_labeled_model()
            for key in ALL_PARAMS:
                getattr(model, key).set_value(float(theta_full[key]))
                getattr(model, key).fix()
            outputs = list(model.experiment_outputs.keys())
            model.wsse_obj = pyo.Objective(expr=0.5 * sum(
                ((float(model.experiment_outputs[c]) - c) / float(model.measurement_error[c])) ** 2 for c in outputs))
            free_vars = [v for v in model.component_data_objects(pyo.Var)
                         if not v.fixed and v.local_name not in ALL_PARAMS]
            self.models[condition] = {
                "model": model, "outputs": outputs, "names": [c.local_name for c in outputs],
                "sigma": np.array([float(model.measurement_error[c]) for c in outputs]),
                "free_vars": free_vars, "start": [float(pyo.value(v)) for v in free_vars]}

    def solve(self, condition, ladder):
        """Optimal outputs of one condition at the current theta (NaN if no option set of the ladder converges)."""
        record = self.models[condition]
        for options in ladder:
            for name, value in {**self.base_options, **options}.items():
                self.solver.options[name] = value
            for variable, start in zip(record["free_vars"], record["start"]):
                variable.set_value(start)
            try:
                result = self.solver.solve(record["model"], tee=False)
            except Exception:   # noqa: BLE001  (an IPOPT error at one rung: try the next options)
                continue
            if str(result.solver.termination_condition) == "optimal":
                return np.array([float(pyo.value(c)) for c in record["outputs"]])
        return np.full(len(record["outputs"]), np.nan)

    def outputs(self, theta_full, conditions=None, family=0):
        """{condition: optimal outputs} at ``theta_full`` (``conditions``: only these; ``family``: which ladder)."""
        result = {}
        for condition, record in self.models.items():
            if conditions is not None and condition not in conditions:
                continue
            for key in ALL_PARAMS:
                getattr(record["model"], key).set_value(float(theta_full[key]))
            result[condition] = self.solve(condition, LADDERS[family])
        return result

def perturbed_theta(theta_full, param, step):
    """theta with ``param`` multiplied by exp(step) (param None: theta itself)."""
    theta = dict(theta_full)
    if param is not None:
        theta[param] = theta[param] * float(np.exp(step))
    return theta

# The inner problems and the base theta of a worker process, built once by start_worker
WORKER_PROBLEMS = None
WORKER_THETA = None

def start_worker(setup, conditions, theta_full, primal, solver_options):
    global WORKER_PROBLEMS, WORKER_THETA
    WORKER_PROBLEMS = InnerProblems(setup, conditions, theta_full, primal, solver_options)
    WORKER_THETA = dict(theta_full)

def solve_task(task):
    """task = (param or None, step, ladder family, conditions or None): the outputs at theta[param] * exp(step)."""
    param, step, family, conditions = task
    theta = perturbed_theta(WORKER_THETA, param, step)
    return task, WORKER_PROBLEMS.outputs(theta, conditions=conditions, family=family)

class TaskRunner:
    """Runs solve tasks in this process (n_jobs = 1) or in a process pool that is reused by every batch."""

    def __init__(self, problems, setup, conditions, theta_full, primal, solver_options, n_jobs):
        self.problems = problems
        self.theta_full = theta_full
        self.pool = None
        if n_jobs != 1:
            self.pool = ProcessPoolExecutor(max_workers=int(n_jobs), initializer=start_worker,
                                            initargs=(setup, conditions, theta_full, primal, solver_options))

    def run(self, tasks):
        """[(task, {condition: outputs})] in task order."""
        if self.pool is not None:
            return list(self.pool.map(solve_task, tasks))
        results = []
        for task in tasks:
            param, step, family, conditions = task
            theta = perturbed_theta(self.theta_full, param, step)
            results.append((task, self.problems.outputs(theta, conditions=conditions, family=family)))
        return results

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()

def step_allowed(theta_full, bounds, param, step):
    """True if theta[param] * exp(step) stays inside the theta bounds (relative tolerance 1e-12)."""
    value = float(theta_full[param]) * float(np.exp(step))
    lower, upper = bounds[param]
    return (value >= lower * (1.0 - 1e-12)) and (value <= upper * (1.0 + 1e-12))

def retry_failed(results, conditions, runner):
    """
    Re-solve once, with the re-solve ladder, every (step, condition) whose solve failed; ``results`` is updated in
    place.  Returns (number failed in the first pass, number recovered).
    """
    failed_by_key = {}
    n_failed = 0
    for key, outputs in results.items():
        for condition in conditions:
            if np.isnan(outputs[condition]).any():
                failed_by_key.setdefault(key, []).append(condition)
                n_failed += 1
    tasks = []
    for (param, step), failed in failed_by_key.items():
        tasks.append((param, step, 1, tuple(failed)))
    n_recovered = 0
    for task, outputs in runner.run(tasks):
        for condition, values in outputs.items():
            if not np.isnan(values).any():
                results[(task[0], task[1])][condition] = values
                n_recovered += 1
    return n_failed, n_recovered

class Differences:
    """Forward / backward differences in ln theta of the outputs, from the solves of every step."""

    def __init__(self, results, resolved, sigma, params):
        self.results = results
        self.resolved = resolved
        self.y0 = results[(None, 0.0)]
        self.sigma = sigma
        self.rtol = params["fd_consistency_rtol"]
        self.atol = params["fd_consistency_atol"]

    def pair(self, store, param, condition, step):
        """(forward, backward) difference in dy / d ln theta at ``step`` (None where a solve is missing or failed)."""
        y_plus = store.get((param, step), {}).get(condition)
        y_minus = store.get((param, -step), {}).get(condition)
        base_ok = not np.isnan(self.y0[condition]).any()
        forward = None
        backward = None
        if y_plus is not None and base_ok and not np.isnan(y_plus).any():
            forward = (y_plus - self.y0[condition]) / step
        if y_minus is not None and base_ok and not np.isnan(y_minus).any():
            backward = (self.y0[condition] - y_minus) / step
        return forward, backward

    def inconsistent(self, forward, backward, condition):
        """||D+ - D-|| > rtol max(||D+||, ||D-||) + atol, on the sigma-weighted vectors."""
        a = forward / self.sigma[condition]
        b = backward / self.sigma[condition]
        largest = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
        return float(np.linalg.norm(a - b)) > self.rtol * largest + self.atol

    def entry(self, param, condition, step, store, flagged_pairs):
        """
        (dy / d ln theta or None, status): 'ok' (central difference), 'resolved' (consistent after the re-solve),
        'inconsistent' (the one-sided value of smaller magnitude), 'one_sided' or 'nan'.
        """
        forward, backward = self.pair(store, param, condition, step)
        if forward is not None and backward is not None:
            if (param, condition) not in flagged_pairs:
                return 0.5 * (forward + backward), "ok"
            forward_2, backward_2 = self.pair(self.resolved, param, condition, step)
            if forward_2 is None:
                forward_2 = forward
            if backward_2 is None:
                backward_2 = backward
            if not self.inconsistent(forward_2, backward_2, condition):
                return 0.5 * (forward_2 + backward_2), "resolved"
            size_forward = np.linalg.norm(forward_2 / self.sigma[condition])
            size_backward = np.linalg.norm(backward_2 / self.sigma[condition])
            return (forward_2 if size_forward <= size_backward else backward_2), "inconsistent"
        if forward is not None:
            return forward, "one_sided"
        if backward is not None:
            return backward, "one_sided"
        return None, "nan"

def find_inconsistent(differences, param_names, conditions, step):
    pairs = []
    for param in param_names:
        for condition in conditions:
            forward, backward = differences.pair(differences.results, param, condition, step)
            if forward is not None and backward is not None and differences.inconsistent(forward, backward, condition):
                pairs.append((param, condition))
    return pairs

def resolve_inconsistent(pairs, step, runner):
    """Re-solve both sides of the inconsistent pairs with the re-solve ladder: {(param, step): {condition: y}}."""
    conditions_of = {}
    for param, condition in pairs:
        conditions_of.setdefault(param, []).append(condition)
    tasks = []
    for param, conditions in conditions_of.items():
        for sign in (1.0, -1.0):
            tasks.append((param, sign * step, 1, tuple(conditions)))
    resolved = {}
    for task, outputs in runner.run(tasks):
        resolved[(task[0], task[1])] = outputs
    return resolved

def weighted_column_rms(columns_by_condition, sigma, conditions, n_params):
    """RMS of every column of the sigma-weighted sensitivities (dy / sigma per ln theta); NaN counts as 0 here."""
    total = np.zeros(n_params)
    n_rows = 0
    for condition in conditions:
        weighted = np.where(np.isnan(columns_by_condition[condition]), 0.0, columns_by_condition[condition])
        weighted = weighted / sigma[condition][:, None]
        total += (weighted ** 2).sum(axis=0)
        n_rows += weighted.shape[0]
    return np.sqrt(total / max(n_rows, 1))

def reduced_sensitivity(est, params, theta=None, param_names=None, relative=False, check=True):
    """
    Reduced (profiled) sensitivities of the measured outputs at ``theta`` (default: the fit of ``est``) for the
    columns ``param_names`` (default: the free parameters); see the module docstring.

    The step h in ln theta is params["sens_step"]; consistent forward / backward pairs give the central difference.
    With ``check`` every column is recomputed at params["sens_step_check"] and flagged fd_unstable when the weighted
    column RMS differs by more than fd_consistency_rtol (+ fd_consistency_atol).  params["n_jobs"] workers.
    ``relative``: d ln y / d ln theta (rows with |y*| < 1e-8 set to 0); otherwise dy / dtheta.

    Returns a dict: sensitivities {condition: DataFrame (measured outputs x param_names)}, sigma and y0
    {condition: Series over the measured outputs: measurement sigma, model value y*(theta)} and report (n_entries,
    n_failed_first_pass, n_recovered_by_retry, n_one_sided, n_inconsistent, n_resolved, n_nan_entries, nan_entries,
    flags {param: [flag, ...]}, param_table, h, h_check).
    """
    check_solver("ipopt")
    conditions = list(est.conditions)
    param_names = list(est.free_params) if param_names is None else list(param_names)
    theta_full = est.full_theta(theta)
    x_init, u_init = est.primal_start()
    primal = (x_init, u_init, est.primal_b_free())
    step = float(params["sens_step"])
    check_step = float(params["sens_step_check"])
    steps = [step, check_step] if check else [step]
    tasks = [(None, 0.0, 0, None)]
    for param in param_names:
        for size in steps:
            for sign in (1.0, -1.0):
                if step_allowed(theta_full, est.theta_bounds, param, sign * size):
                    tasks.append((param, sign * size, 0, None))
    problems = InnerProblems(est.setup, conditions, theta_full, primal, est.solver_options)
    names = {c: problems.models[c]["names"] for c in conditions}
    sigma = {c: problems.models[c]["sigma"] for c in conditions}
    runner = TaskRunner(problems, est.setup, conditions, theta_full, primal, est.solver_options, params["n_jobs"])
    try:
        results = {}
        for task, outputs in runner.run(tasks):
            results[(task[0], task[1])] = outputs
        n_failed, n_recovered = retry_failed(results, conditions, runner)
        differences = Differences(results, {}, sigma, params)
        flagged_pairs = find_inconsistent(differences, param_names, conditions, step)
        differences.resolved = resolve_inconsistent(flagged_pairs, step, runner)
    finally:
        runner.close()
    columns, table, nan_entries = sensitivity_columns(differences, param_names, conditions, names, step,
                                                      check_step if check else None, set(flagged_pairs), theta_full,
                                                      est.theta_bounds, params)
    y0 = differences.y0
    sensitivities = {}
    for condition in conditions:
        values = columns[condition].copy()
        if relative:
            safe = np.abs(y0[condition]) >= 1e-8
            values = np.where(safe[:, None], values / np.where(safe, y0[condition], 1.0)[:, None], 0.0)
        else:
            values = values / np.array([float(theta_full[p]) for p in param_names])[None, :]
        sensitivities[condition] = pd.DataFrame(values, index=names[condition], columns=param_names)
    flags = {}
    for param in param_names:
        flags[param] = [f for f in ("at_bound", "fd_unstable", "fd_failed") if bool(table.loc[param, f])]
    table["flags"] = [";".join(flags[p]) for p in param_names]
    report = {"n_entries": int(len(param_names) * len(conditions)), "n_failed_first_pass": int(n_failed),
              "n_recovered_by_retry": int(n_recovered), "n_one_sided": int(table["n_one_sided"].sum()),
              "n_inconsistent": int(table["n_inconsistent"].sum()), "n_resolved": int(table["n_resolved"].sum()),
              "n_nan_entries": len(nan_entries), "nan_entries": nan_entries, "flags": flags, "param_table": table,
              "h": step, "h_check": check_step if check else None}
    if nan_entries:
        warnings.warn("reduced_sensitivity: %d (condition, parameter) entries could not be computed (NaN): %s ..." % (
            len(nan_entries), nan_entries[:5]))
    return {"sensitivities": sensitivities,
            "sigma": {c: pd.Series(sigma[c], index=names[c]) for c in conditions},
            "y0": {c: pd.Series(y0[c], index=names[c]) for c in conditions},
            "report": report}

def sensitivity_columns(differences, param_names, conditions, names, step, check_step, flagged_pairs, theta_full,
                        bounds, params):
    """
    The dy / d ln theta columns of every condition, the per-parameter table (n_inconsistent, n_resolved,
    n_one_sided, n_nan, at_bound, rms_h, rms_check, rel_diff_check, fd_unstable, fd_failed) and the NaN entries.
    """
    columns = {c: np.full((len(names[c]), len(param_names)), np.nan) for c in conditions}
    check_columns = {c: np.full((len(names[c]), len(param_names)), np.nan) for c in conditions}
    rows = []
    nan_entries = []
    for j, param in enumerate(param_names):
        at_bound = not (step_allowed(theta_full, bounds, param, step)
                        and step_allowed(theta_full, bounds, param, -step))
        counts = {"ok": 0, "resolved": 0, "one_sided": 0, "inconsistent": 0, "nan": 0}
        for condition in conditions:
            value, status = differences.entry(param, condition, step, differences.results, flagged_pairs)
            counts[status] += 1
            if value is None:
                nan_entries.append((condition, param))
            else:
                columns[condition][:, j] = value
            if check_step is not None:
                # the column check uses the first-pass solves only (no re-solve of inconsistent pairs)
                value, _status = differences.entry(param, condition, check_step, differences.results, set())
                if value is not None:
                    check_columns[condition][:, j] = value
        rows.append({"param": param, "n_inconsistent": counts["inconsistent"], "n_resolved": counts["resolved"],
                     "n_one_sided": counts["one_sided"], "n_nan": counts["nan"], "at_bound": bool(at_bound)})
    table = pd.DataFrame(rows).set_index("param")
    sigma = differences.sigma
    table["rms_h"] = weighted_column_rms(columns, sigma, conditions, len(param_names))
    if check_step is not None:
        table["rms_check"] = weighted_column_rms(check_columns, sigma, conditions, len(param_names))
        largest = np.maximum(table["rms_h"], table["rms_check"])
        difference = np.abs(table["rms_h"] - table["rms_check"])
        table["rel_diff_check"] = difference / np.maximum(largest, 1e-300)
        table["fd_unstable"] = difference > params["fd_consistency_rtol"] * largest + params["fd_consistency_atol"]
    else:
        table["rms_check"] = np.nan
        table["rel_diff_check"] = np.nan
        table["fd_unstable"] = False
    table["fd_failed"] = table["n_nan"] > 2
    return columns, table, nan_entries
