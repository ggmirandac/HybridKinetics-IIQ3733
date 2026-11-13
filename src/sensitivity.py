"""
sensitivity.py -- reduced sensitivities of the measured outputs by the implicit function theorem (CasADi).

For each condition the inner problem
    min_w  phi(w, theta) = 1/2 sum_k ((y_k - h_k(w, theta)) / sigma_k)^2   s.t.  F(w, theta) = S v - b = 0,  lo <= w <= hi
is solved once at theta (w = balanced metabolites x, free cofactors u, free drains b).  y*(theta) = h(w*(theta), theta)
are the model values of the measured outputs at that optimum: the free cofactors and drains absorb part of every
parameter change, so these are the sensitivities of the problem that is actually estimated.  At the solution the
variables on a bound stay fixed and the others (I) satisfy the optimality conditions
    grad_I phi + A_I^T lam = 0,   F = 0,          A_I = dF / dw_I.
Differentiating them with respect to theta gives one linear system per condition (the KKT matrix):
    [ H_II  A_I^T ] [ dw_I / dtheta  ]     [ d grad_I L / dtheta ]
    [ A_I   0     ] [ dlam / dtheta  ] = - [ dF / dtheta         ],     L = phi + lam^T F,  H = d2 L / dw2,
and dy* / dtheta = dh / dw_I dw_I / dtheta + dh / dtheta.  It is solved in its null-space form (reduced Hessian), which
is better conditioned than the full KKT matrix.  CasADi builds the rate laws of model/kinetics.py as symbols, so every
derivative is exact (automatic differentiation): there is no step size.

Position in the pipeline: input of the practical identifiability analysis (identifiability/practical.py) in every
iteration of the parameter selection, and of the sensitivity figure.
"""
import warnings

import numpy as np
import pandas as pd
import casadi as ca
import pyomo.environ as pyo

from settings import WARM_START_OPTIONS
from solver import check_solver
from model.kinetics import GlycolysisRateLaws
from model.parameters import ALL_PARAMS, BALANCED_KEYS, IMBALANCED_KEYS, ENZYME_KEYS, FLUX_KEYS, FREE_DRAIN_KEYS
from data import load_condition
from estimation.experiment import GlycolysisExperiment

# The inner problems are solved from the fitted point with the warm-start options of the estimation and a tight
# complementarity tolerance, so that the variables of an active bound sit on it (the default of IPOPT leaves them up
# to 1e-3 away); a variable within ACTIVE_RTOL of a bound is then on the bound.
INNER_OPTIONS = {**WARM_START_OPTIONS, "compl_inf_tol": 1e-10, "max_iter": 10000}
ACTIVE_RTOL = 1e-5
# The stationarity residual of an optimal solve, relative to the size of the objective gradient (scaled variables)
STATIONARITY_RTOL = 1e-4
# Eigenvalues of the reduced Hessian below this fraction of the largest are numerically zero: exactly flat
# directions of w (they leave the outputs unchanged and are dropped).  The inner problems are nearly degenerate
# (eigenvalue ratios down to ~1e-10 that do move the outputs), so the cut sits at the level of rounding errors.
FLAT_RCOND = 1e-14

class InnerProblems:
    """The inner problems (all theta fixed, objective = weighted SSE) of every condition, started at the fit."""

    def __init__(self, setup, conditions, theta_full, primal):
        self.solver = pyo.SolverFactory("ipopt")
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

    def solve(self, condition):
        """Solve the inner problem of one condition from the fitted point; True if IPOPT terminates optimal."""
        record = self.models[condition]
        for name, value in INNER_OPTIONS.items():
            self.solver.options[name] = value
        for variable, start in zip(record["free_vars"], record["start"]):
            variable.set_value(start)
        result = self.solver.solve(record["model"], tee=False, load_solutions=False)
        if str(result.solver.termination_condition) != "optimal":
            return False
        record["model"].solutions.load_from(result)
        return True

def symbolic_condition(record, condition, setup, theta_full, param_names):
    """
    CasADi function of one condition: (w, k, lam) -> (phi, F, h, grad_w L, dF/dw, dF/dk, d2L/dw2, d2L/dw dk, dh/dw,
    dh/dk), with w the free variables of the Pyomo model (in its order) and k the parameters ``param_names``.
    """
    data = load_condition(condition)
    model = record["model"]
    free_names = [variable.local_name for variable in record["free_vars"]]
    w = ca.SX.sym("w", len(free_names))
    k = ca.SX.sym("k", len(param_names))
    lam = ca.SX.sym("lam", len(BALANCED_KEYS))
    symbols = {}
    for i, name in enumerate(free_names):
        symbols[name] = w[i]
    for key in BALANCED_KEYS + IMBALANCED_KEYS + ["b_" + key for key in FREE_DRAIN_KEYS]:
        if key not in symbols:
            symbols[key] = float(getattr(model, key).value)
    constants = {}
    for key in ALL_PARAMS:
        constants[key] = float(theta_full[key])
    for j, key in enumerate(param_names):
        constants[key] = k[j]
    concentrations = {}
    for key in BALANCED_KEYS + IMBALANCED_KEYS:
        concentrations[key] = symbols[key]
    enzymes = {}
    for key in ENZYME_KEYS:
        enzymes[key] = float(data["e"][key])
    rates = GlycolysisRateLaws(setup.keq).compute_fluxes(concentrations, enzymes, constants)
    for flux_key, rate in zip(FLUX_KEYS, rates):
        symbols[flux_key] = rate
    constraints = []
    for i, key in enumerate(BALANCED_KEYS):
        rate_sum = 0
        for j in range(len(FLUX_KEYS)):
            rate_sum = rate_sum + float(setup.S[i, j]) * rates[j]
        if key in FREE_DRAIN_KEYS:
            constraints.append(rate_sum - symbols["b_" + key])
        else:
            constraints.append(rate_sum - float(data["b"].get(key, 0.0)))
    F = ca.vertcat(*constraints)
    measured = np.array([float(model.experiment_outputs[c]) for c in record["outputs"]])
    h = ca.vertcat(*[symbols[name] for name in record["names"]])
    residual = (measured - h) / record["sigma"]
    phi = 0.5 * ca.sumsqr(residual)
    lagrangian = phi + ca.dot(lam, F)
    grad_w = ca.gradient(lagrangian, w)
    outputs = [phi, F, h, grad_w, ca.jacobian(F, w), ca.jacobian(F, k), ca.jacobian(grad_w, w),
               ca.jacobian(grad_w, k), ca.jacobian(h, w), ca.jacobian(h, k)]
    return ca.Function("inner_" + condition, [w, k, lam], outputs)

def active_variables(record, w_star):
    """Boolean array: the free variable sits on one of its bounds (relative distance ACTIVE_RTOL)."""
    active = np.zeros(len(w_star), dtype=bool)
    for i, variable in enumerate(record["free_vars"]):
        lower, upper = variable.lb, variable.ub
        if lower is not None and w_star[i] - lower <= ACTIVE_RTOL * max(abs(lower), 1e-9):
            active[i] = True
        if upper is not None and upper - w_star[i] <= ACTIVE_RTOL * max(abs(upper), 1e-9):
            active[i] = True
    return active

def condition_sensitivity(function, record, k_values):
    """
    (dy*/dk, y*, diagnostics) of one solved condition (the Pyomo model holds w*), by the null-space form of the KKT
    system.  With the variables scaled by their values, dw = dw_p + Z dz: dw_p solves the linearised steady state
    A dw_p = -dF/dk (least norm), Z is an orthonormal basis of the null space of A and the reduced Hessian Z^T H Z
    gives dz = -(Z^T H Z)^-1 Z^T (d grad L / dk + H dw_p).  dy*/dk is None when the solve is not stationary.
    """
    w_star = np.array([float(variable.value) for variable in record["free_vars"]])
    n_constraints = len(BALANCED_KEYS)
    values = function(w_star, k_values, np.zeros(n_constraints))
    _phi, F, h, grad_phi, F_w, _F_k, _H, _L_wk, _h_w, _h_k = [np.array(v) for v in values]
    active = active_variables(record, w_star)
    inactive = np.where(~active)[0]
    scale = np.maximum(np.abs(w_star[inactive]), 1e-12)
    A = F_w[:, inactive] * scale[None, :]
    gradient = grad_phi[inactive, 0] * scale
    # multipliers from the stationarity of the inactive variables: gradient + A^T lam = 0
    lam, *_ = np.linalg.lstsq(A.T, -gradient, rcond=None)
    stationarity = float(np.linalg.norm(gradient + A.T @ lam)) / max(float(np.linalg.norm(gradient)), 1e-300)
    values = function(w_star, k_values, lam)
    _phi, F, h, _grad_L, F_w, F_k, H, L_wk, h_w, h_k = [np.array(v) for v in values]
    H_scaled = H[np.ix_(inactive, inactive)] * np.outer(scale, scale)
    L_wk_scaled = L_wk[inactive, :] * scale[:, None]
    h_w_scaled = h_w[:, inactive] * scale[None, :]
    # particular solution of the linearised steady state and the null space of A
    dw_particular, *_ = np.linalg.lstsq(A, -F_k, rcond=None)
    _U, singular, Vt = np.linalg.svd(A)
    rank = int(np.sum(singular > 1e-12 * singular[0]))
    Z = Vt[rank:, :].T
    reduced_hessian = Z.T @ H_scaled @ Z
    eigenvalues, vectors = np.linalg.eigh(0.5 * (reduced_hessian + reduced_hessian.T))
    # no null space (the steady state and the active bounds fix every variable): dw is the particular solution
    largest = 1e-300
    if eigenvalues.size:
        largest = max(float(np.abs(eigenvalues).max()), 1e-300)
    kept = np.abs(eigenvalues) > FLAT_RCOND * largest
    flat_output_change = 0.0
    if not kept.all():
        flat_directions = Z @ vectors[:, ~kept]
        flat_output_change = float(np.abs((h_w_scaled @ flat_directions) / record["sigma"][:, None]).max())
    diagnostics = {"n_active": int(active.sum()), "stationarity": stationarity,
                   "steady_state_residual": float(np.abs(F).max()), "constraint_rank": rank,
                   "reduced_hessian_ratio": float(eigenvalues[kept].min() / largest) if kept.any() else 0.0,
                   "n_flat": int((~kept).sum()), "flat_output_change": flat_output_change,
                   "n_negative_curvature": int(np.sum(eigenvalues < -1e-8 * largest))}
    if stationarity > STATIONARITY_RTOL:
        return None, h[:, 0], diagnostics
    projected = Z.T @ (L_wk_scaled + H_scaled @ dw_particular)
    coefficients = (vectors[:, kept].T @ projected) / eigenvalues[kept][:, None]
    dz = -vectors[:, kept] @ coefficients
    dw_scaled = dw_particular + Z @ dz
    dy = h_w_scaled @ dw_scaled + h_k
    return dy, h[:, 0], diagnostics

def reduced_sensitivity(est, theta=None, param_names=None, relative=False):
    """
    Reduced sensitivities of the measured outputs at ``theta`` (default: the fit of ``est``) for the columns
    ``param_names`` (default: the free parameters); see the module docstring.  ``relative``: d ln y / d ln theta (rows
    with |y*| < 1e-8 set to 0); otherwise dy / dtheta.

    Returns a dict: sensitivities {condition: DataFrame (measured outputs x param_names)}, sigma and y0 {condition:
    Series}, failed (conditions without sensitivities: NaN entries) and condition_table (solver diagnostics).
    """
    check_solver("ipopt")
    conditions = list(est.conditions)
    param_names = list(est.free_params) if param_names is None else list(param_names)
    theta_full = est.full_theta(theta)
    x_init, u_init = est.primal_start()
    problems = InnerProblems(est.setup, conditions, theta_full, (x_init, u_init, est.primal_b_free()))
    k_values = np.array([float(theta_full[p]) for p in param_names])
    sensitivities = {}
    sigma = {}
    y0 = {}
    condition_rows = []
    failed = []
    for condition in conditions:
        record = problems.models[condition]
        sigma[condition] = pd.Series(record["sigma"], index=record["names"])
        converged = problems.solve(condition)
        dy = None
        diagnostics = {}
        y = np.full(len(record["names"]), np.nan)
        if converged:
            function = symbolic_condition(record, condition, est.setup, theta_full, param_names)
            dy, y, diagnostics = condition_sensitivity(function, record, k_values)
        y0[condition] = pd.Series(y, index=record["names"])
        if dy is None:
            dy = np.full((len(record["names"]), len(param_names)), np.nan)
            failed.append(condition)
        if relative:
            safe = np.abs(y) >= 1e-8
            dy = np.where(safe[:, None], dy * k_values[None, :] / np.where(safe, y, 1.0)[:, None], 0.0)
        sensitivities[condition] = pd.DataFrame(dy, index=record["names"], columns=param_names)
        condition_rows.append({"condition": condition, "converged": converged, **diagnostics})
    if failed:
        warnings.warn("reduced_sensitivity: no sensitivities for %s (solve or KKT failed)" % failed)
    return {"sensitivities": sensitivities, "sigma": sigma, "y0": y0, "failed": failed,
            "condition_table": pd.DataFrame(condition_rows).set_index("condition")}
