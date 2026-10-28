"""
estimator.py -- parameter estimation of the glycolysis model with Pyomo + parmest (weighted least squares).

One scenario e per experimental condition:

    min_{ {x_e, u_e, b_e}_e , theta }   (1/N) sum_e  sum_outputs  ((y_meas - y_model) / sigma)^2 / 2
    s.t.   S v(x_e, u_e, E_e; Keq, theta) = b_e      for all e
           bounds on x_e, u_e, b_e and theta

x_e (balanced metabolites), u_e (cofactors) and b_e (free drains) are per-scenario decision variables, theta (the
kinetic parameters) is shared by all scenarios (parmest's ``unknown_parameters``).  ``GlycolysisParameterEstimator``
builds the scenarios (estimation/experiment.py), solves the extensive form with IPOPT (``estimate``), re-optimizes
at the fit until it is a fixed point of the optimizer (``refine_anchor``), exposes the fitted primal point
(x, u, b), predicts in sample and for a held-out condition, decomposes the objective per condition
(``objective_breakdown``) and gives the k_aug covariance.

Position in the pipeline: the core object of every later step (selection, refits, bootstrap, profiles, LOOCV).
Written against the parmest API of Pyomo 6.10 (Experiment-based interface, ``obj_function='SSE_weighted'``,
``cov_est``).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import pyomo.environ as pyo
import pyomo.contrib.parmest.parmest as parmest

from settings import THETA_BOUNDS, TIGHT_OPTIONS, SOLVER_OPTIONS
from solver import check_solver
from model.kinetics import EcoliCarbonKinetics
from model.parameters import (ALL_PARAMS, LITERATURE_THETA, FREE_DRAIN_KEYS, BALANCED_KEYS, IMBALANCED_KEYS,
                              ENZYME_KEYS, FLUX_KEYS)
from model.stoichiometry import load_keq, build_stoichiometric_matrix
from data import available_conditions, load_condition, load_measurement_sigmas, imbalanced_fallbacks, variable_bounds
from estimation.experiment import GlycolysisExperiment, EstimationSetup

OUTPUT_COLUMNS = BALANCED_KEYS + FLUX_KEYS
DIAGNOSTIC_COLUMNS = ["return_status", "success", "ipopt_success", "ss_residual", "max_residual"]

@dataclass
class Prediction:
    """
    Model outputs of ``predict_fitted`` / ``predict_held_out``: predicted and measured values (conditions x the 18
    outputs C_* then v_*; ``real`` is NaN where nothing is measured), the solver diagnostics per condition
    (return_status, success, ipopt_success, ss_residual, max_residual), the drains used per condition and the
    number of conditions whose solve failed.
    """
    predicted: pd.DataFrame
    real: pd.DataFrame
    diagnostics: pd.DataFrame
    b: pd.DataFrame
    n_failed: int

def ordered_free_params(free_params):
    """The free parameters in ALL_PARAMS order (all 37 when None); unknown names raise."""
    if free_params is None:
        return list(ALL_PARAMS)
    unknown = [p for p in free_params if p not in ALL_PARAMS]
    if unknown:
        raise ValueError("Unknown free_params (not in ALL_PARAMS): %s" % unknown)
    return [p for p in ALL_PARAMS if p in set(free_params)]

def measured_table(conditions):
    """Measured values (conditions x the 18 outputs), NaN where a value is not measured."""
    rows = {}
    for condition in conditions:
        data = load_condition(condition)
        row = {}
        for key in FLUX_KEYS:
            row[key] = data["v_real"].get(key, float("nan"))
        for key in BALANCED_KEYS:
            row[key] = data["x_real"].get(key, float("nan"))
        rows[condition] = row
    table = pd.DataFrame(rows, index=OUTPUT_COLUMNS).T.astype(float)
    return table[OUTPUT_COLUMNS]

def enzyme_levels(condition):
    data = load_condition(condition)
    enzymes = {}
    for key in ENZYME_KEYS:
        enzymes[key] = float(data["e"][key])
    return enzymes

def data_drains(condition):
    """Drains b of the data, {balanced_key: value} over all 9 balanced keys (0 for the zero-drain rows)."""
    drains = load_condition(condition)["b"]
    return {key: float(drains.get(key, 0.0)) for key in BALANCED_KEYS}

class GlycolysisParameterEstimator:
    """
    parmest estimator of the glycolysis model.

    conditions      conditions to fit (default: all conditions of Data/)
    free_params     parameters to estimate (default all 37); the others are held at ``theta_init``
    theta_init      {param: value} start of theta (missing parameters: the literature values)
    solver_options  IPOPT options (default: the 'default' set of settings/solver_options.csv)
    x_bounds, u_bounds, drain_bounds
                    bounds of the balanced metabolites, cofactors and free drains (default: data.variable_bounds
                    over all conditions)
    x_init, u_init, b_init
                    primal warm start {condition: {key: value}} of x, u and the free drains (default: measured
                    values, else the geometric mean of the bounds / the dataset fallback / the data drains).
                    ``primal_start()`` returns the solved values in this format: theta_init alone does not
                    restart at an optimum, because x and u are re-initialized from the data.
    """

    def __init__(self, conditions=None, free_params=None, theta_init=None, solver_options=None, x_bounds=None,
                 u_bounds=None, drain_bounds=None, x_init=None, u_init=None, b_init=None):
        self.conditions = list(conditions) if conditions else available_conditions()
        self.free_params = ordered_free_params(free_params)
        if not self.free_params:
            raise ValueError("free_params is empty: nothing to estimate.")
        missing = [c for c in self.conditions if c not in set(available_conditions())]
        if missing:
            raise ValueError("Conditions not present in all data files: %s" % missing)
        self.theta_init = theta_init
        self.solver_options = solver_options or dict(SOLVER_OPTIONS)
        if x_bounds is None or u_bounds is None or drain_bounds is None:
            default_u, default_x, default_drains = variable_bounds()
        self.x_bounds = default_x if x_bounds is None else x_bounds
        self.u_bounds = default_u if u_bounds is None else u_bounds
        self.drain_bounds = dict(default_drains if drain_bounds is None else drain_bounds)
        self.theta_bounds = {p: tuple(THETA_BOUNDS[p]) for p in ALL_PARAMS}
        self.x_init = dict(x_init) if x_init else {}
        self.u_init = dict(u_init) if u_init else {}
        self.b_init = dict(b_init) if b_init else {}
        sigma_flux, sigma_met = load_measurement_sigmas()
        self.setup = EstimationSetup(keq=load_keq(), S=build_stoichiometric_matrix(), sigma_flux=sigma_flux,
                                     sigma_met=sigma_met, u_fallback=imbalanced_fallbacks(), x_bounds=self.x_bounds,
                                     u_bounds=self.u_bounds, drain_bounds=self.drain_bounds,
                                     free_params=self.free_params)
        self.exp_list = self.build_experiments()
        self.pest = self.build_parmest()
        self.obj_value = None
        self.theta = None
        self.cov = None
        self.cov_psd = None
        # (x_init, u_init, b_init) of an anchor-only estimator (see anchor_only)
        self.anchor_primal = None
        self.kinetics = None
        # how the parameter selection obtained this point ("" for a plain fit; see estimation/refit.py)
        self.refit_status = ""

    def build_experiments(self):
        experiments = []
        for condition in self.conditions:
            experiments.append(GlycolysisExperiment(condition, self.setup, theta_init=self.theta_init,
                                                    x_init=self.x_init.get(condition),
                                                    u_init=self.u_init.get(condition),
                                                    b_init=self.b_init.get(condition)))
        return experiments

    def build_parmest(self):
        return parmest.Estimator(self.exp_list, obj_function="SSE_weighted", tee=False,
                                 solver_options=self.solver_options)

    def clone(self, free_params=None, theta_init=None, x_init=None, u_init=None, b_init=None, solver_options=None,
              conditions=None, x_bounds=None, u_bounds=None, drain_bounds=None):
        """
        A fresh, unsolved estimator with the configuration of this one; any argument given replaces it (the
        warm start x_init / u_init / b_init is NOT copied: it starts from the data unless given).
        """
        if conditions is None:
            conditions = self.conditions
        if free_params is None:
            free_params = self.free_params
        if theta_init is None:
            theta_init = self.theta_init or {}
        if solver_options is None:
            solver_options = self.solver_options
        if x_bounds is None:
            x_bounds = self.x_bounds
        if u_bounds is None:
            u_bounds = self.u_bounds
        if drain_bounds is None:
            drain_bounds = self.drain_bounds
        return GlycolysisParameterEstimator(conditions=list(conditions), free_params=list(free_params),
                                            theta_init=dict(theta_init), solver_options=dict(solver_options),
                                            x_bounds=x_bounds, u_bounds=u_bounds, drain_bounds=drain_bounds,
                                            x_init=x_init, u_init=u_init, b_init=b_init)

    def anchor_only(self, x_init, u_init, b_init, theta_full, obj_value):
        """
        Turn this unsolved estimator into an ANCHOR-ONLY one: it represents the fitted point (theta, x, u, b) with
        objective ``obj_value`` without solving the extensive form.  A warm-started IPOPT solve from an anchor drifts
        by about 0.3% in the flat valley because the duals are unknown, so saved points are used as they are.
        The primal accessors read the stored point; ``covariance`` is not available.  Returns self.
        """
        self.anchor_primal = (x_init, u_init, b_init)
        self.x_init = x_init
        self.u_init = u_init
        self.b_init = b_init
        self.theta_init = {p: float(theta_full[p]) for p in ALL_PARAMS}
        self.theta = pd.Series({p: float(theta_full[p]) for p in self.free_params}, dtype=float)
        self.obj_value = float(obj_value)
        self.cov = None
        return self

    def has_solution(self):
        """True once the extensive form has been solved (parmest creates ef_instance in its first solve)."""
        return hasattr(self.pest, "ef_instance")

    def fitted_state(self):
        """
        (x, u, b) of the fitted point, each {condition: {key: value}}: the 9 balanced metabolites, all 7 cofactors and
        the 9 drains (fitted free drains, data b on the zero-drain rows).
        """
        if self.has_solution():
            return self.solved_state()
        x_init, u_init, b_init = self.anchor_primal
        fallback = self.setup.u_fallback
        x_by, u_by, b_by = {}, {}, {}
        for condition in self.conditions:
            measured_u = load_condition(condition)["u"]
            x_by[condition] = {k: float(x_init[condition][k]) for k in BALANCED_KEYS}
            cofactors = {k: float(measured_u.get(k, fallback[k])) for k in IMBALANCED_KEYS}
            for key, value in u_init.get(condition, {}).items():
                cofactors[key] = float(value)
            u_by[condition] = cofactors
            drains = data_drains(condition)
            for key, value in b_init.get(condition, {}).items():
                drains[key] = float(value)
            b_by[condition] = drains
        return x_by, u_by, b_by

    def solved_state(self):
        """(x, u, b) of the solved extensive form, in the format of ``fitted_state``."""
        x_by, u_by, b_by = {}, {}, {}
        for i, condition in enumerate(self.conditions):
            block = self.pest.ef_instance.exp_scenarios[i]
            x_by[condition] = {k: float(pyo.value(getattr(block, k))) for k in BALANCED_KEYS}
            u_by[condition] = {k: float(pyo.value(getattr(block, k))) for k in IMBALANCED_KEYS}
            drains = data_drains(condition)
            for key in FREE_DRAIN_KEYS:
                drains[key] = float(pyo.value(getattr(block, "b_" + key)))
            b_by[condition] = drains
        return x_by, u_by, b_by

    def primal_start(self):
        """(x_init, u_init) of the fitted point, {condition: {key: value}}, ready for the constructor."""
        if not self.has_solution():
            x_by, _u_by, _b_by = self.fitted_state()
            u_init = {c: dict(self.anchor_primal[1].get(c, {})) for c in self.conditions}
            return x_by, u_init
        x_by, u_by, _b_by = self.solved_state()
        return x_by, u_by

    def primal_b_free(self):
        """{condition: {key: value}} of the fitted free drains, ready for the ``b_init`` argument."""
        _x_by, _u_by, b_by = self.fitted_state()
        b_free = {}
        for condition, drains in b_by.items():
            b_free[condition] = {key: drains[key] for key in FREE_DRAIN_KEYS}
        return b_free

    def full_theta(self, theta=None):
        """{param: value} over ALL_PARAMS: ``theta`` if given, else the estimate, over theta_init and literature."""
        full = dict(LITERATURE_THETA)
        if self.theta_init:
            full.update(self.theta_init)
        if theta is not None:
            full.update(theta)
        elif self.theta is not None:
            for p in self.free_params:
                full[p] = float(self.theta[p])
        return full

    def estimate(self):
        """
        Weighted least-squares fit from the current start.  parmest raises on a non-optimal termination, so a normal
        return means IPOPT converged.  Returns theta (Series of the free parameters).
        """
        check_solver("ipopt")
        obj, theta = self.pest.theta_est()
        self.obj_value = float(obj)
        self.theta = pd.Series(dict(theta), dtype=float)
        return self.theta

    def solve_from(self, point, base_theta_init, base_options, extra_options):
        """
        Re-solve with the start ``point`` (theta, primal x / u, free drains b) and the options ``base_options`` +
        ``extra_options``.  Returns the solved point, or None if IPOPT did not converge (a normal outcome in a flat
        valley; the caller keeps its best point).
        """
        theta_init = dict(base_theta_init)
        for p in self.free_params:
            theta_init[p] = float(point["theta"][p])
        self.theta_init = theta_init
        self.x_init, self.u_init = point["primal"]
        self.b_init = point["b"]
        self.solver_options = {**base_options, **extra_options}
        try:
            self.exp_list = self.build_experiments()
            self.pest = self.build_parmest()
            theta = self.estimate()
        except Exception:   # noqa: BLE001  (parmest raises when IPOPT does not converge)
            return None
        finally:
            self.solver_options = dict(base_options)
        return self.current_point(theta)

    def current_point(self, theta):
        return {"obj": float(self.obj_value), "theta": theta.copy(), "primal": self.primal_start(),
                "b": self.primal_b_free(), "pest": self.pest, "exp_list": self.exp_list}

    def refine_anchor(self, max_iter=20, primary_options=None):
        """
        Re-optimize AT the current fit until it is a fixed point of the optimizer.

        ``estimate()`` started from theta alone re-initializes x and u from the data, so its solution is in general
        not an optimum of the extensive form: re-optimizing from the solved primal point lowers the objective
        further.  Each round re-solves from the best point with the estimator's options plus ``primary_options``
        and, if that fails or does not lower the objective, once more with the tight monotone warm start (which keeps
        descending where the adaptive barrier update stalls).  Only strict improvements are accepted; the rounds stop
        when the decrease is below 1e-6 of the objective.  Without a fit, ``estimate()`` is called first.  On return
        the estimator holds the best point (theta clipped into its bounds: IPOPT can return it ~1e-9 outside, which
        parmest then refuses).
        """
        if self.theta is None or not self.has_solution():
            self.estimate()
        base_theta_init = dict(self.theta_init or {})
        base_options = dict(self.solver_options)
        best = self.current_point(self.theta)
        attempts = [dict(primary_options or {}), dict(TIGHT_OPTIONS)]
        for _round in range(max_iter):
            new = None
            for extra_options in attempts:
                new = self.solve_from(best, base_theta_init, base_options, extra_options)
                if new is not None and new["obj"] < best["obj"]:
                    break
            if new is None or not new["obj"] < best["obj"]:
                break
            converged = (best["obj"] - new["obj"]) < 1e-6 * best["obj"]
            best = new
            if converged:
                break
        self.keep_point(best, base_theta_init, base_options)

    def keep_point(self, best, base_theta_init, base_options):
        """Leave the estimator at ``best`` (its extensive form is held by best["pest"]) with theta in its bounds."""
        theta_init = dict(base_theta_init)
        for p in self.free_params:
            theta_init[p] = float(best["theta"][p])
        self.theta_init = theta_init
        self.x_init, self.u_init = best["primal"]
        self.b_init = best["b"]
        self.solver_options = base_options
        self.exp_list = best["exp_list"]
        self.pest = best["pest"]
        clipped = {}
        for p in self.free_params:
            lower, upper = self.theta_bounds[p]
            clipped[p] = float(min(max(float(best["theta"][p]), lower), upper))
        for name in self.pest.ef_instance.parmest_theta:
            if str(name) in clipped:
                self.pest.ef_instance.parmest_theta[name].set_value(clipped[str(name)])
        for name in list(self.pest.estimated_theta):
            if name in clipped:
                self.pest.estimated_theta[name] = clipped[name]
        self.theta = pd.Series(clipped, dtype=float)
        self.theta_init = {**self.theta_init, **clipped}
        self.obj_value = float(best["obj"])
        self.cov = None

    def covariance(self):
        """
        Parameter covariance (DataFrame) from parmest's k_aug method (needs the k_aug and dot_sens binaries).  There
        is no fallback: a failure raises.  Sets ``cov_psd``.
        """
        if self.theta is None:
            self.estimate()
        self.cov = None
        self.cov_psd = None
        cov = self.pest.cov_est(method="automatic_differentiation_kaug", solver="ipopt", step=1e-3)
        self.cov = cov
        matrix = np.asarray(cov, dtype=float)
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        largest = max(abs(eigenvalues).max(), 1e-300)
        self.cov_psd = bool(np.all(np.isfinite(eigenvalues)) and eigenvalues.min() >= -1e-10 * largest)
        return self.cov

    def kinetic_model(self):
        """The CasADi kinetics of this estimator's bounds, built at the first use (compiling the NLP is slow)."""
        if self.kinetics is None:
            bounds_imbalanced = {k: self.u_bounds[k] for k in IMBALANCED_KEYS}
            self.kinetics = EcoliCarbonKinetics(bounds_imbalanced, self.x_bounds)
        return self.kinetics

    def predict_fitted(self):
        """
        In-sample prediction at the fitted point: the rates are evaluated at the fitted x, u and b of every condition
        (no steady-state re-solve, so the prediction is the fit itself; the residual |S v - b| is reported).
        """
        theta_full = self.full_theta()
        kinetics = self.kinetic_model()
        x_by, u_by, b_by = self.fitted_state()
        rows, diagnostics, drains = [], [], []
        for condition in self.conditions:
            concentrations = {**x_by[condition], **u_by[condition]}
            fluxes = []
            for flux in kinetics.compute_fluxes(concentrations, enzyme_levels(condition), theta_full):
                fluxes.append(float(flux))
            b_vector = np.array([b_by[condition][k] for k in BALANCED_KEYS])
            residual = kinetics.stoichiometric_matrix.values @ np.array(fluxes) - b_vector
            row = dict(x_by[condition])
            for key, flux in zip(FLUX_KEYS, fluxes):
                row[key] = flux
            rows.append(row)
            drains.append(dict(b_by[condition]))
            diagnostics.append({"return_status": "evaluated", "success": True, "ipopt_success": True,
                                "ss_residual": float(np.linalg.norm(residual)),
                                "max_residual": float(np.max(np.abs(residual)))})
        return self.prediction(rows, diagnostics, drains, self.conditions)

    def predict_held_out(self, condition, closure):
        """
        Out-of-sample prediction of ``condition`` with the "pts_fraction" closure (drains b_j = alpha_j v_pts, the
        non-unique steady state selected by the reference state; see EcoliCarbonKinetics.solve_closure).
        ``closure`` = ``closure_inputs()`` of an estimator fitted without this condition.
        """
        kinetics = self.kinetic_model()
        table, _residual = kinetics.solve_closure(enzyme_levels(condition), self.full_theta(),
                                                  closure["alpha_by_condition"], closure["u_ref"], closure["x_ref"])
        row = {}
        for key in OUTPUT_COLUMNS:
            row[key] = table.iloc[0][key]
        return self.prediction([row], [dict(kinetics.last_stats)], [dict(kinetics.last_closure_b)], [condition])

    def prediction(self, rows, diagnostics, drains, conditions):
        predicted = pd.DataFrame(rows, index=conditions)[OUTPUT_COLUMNS]
        diagnostics = pd.DataFrame(diagnostics, index=conditions, columns=DIAGNOSTIC_COLUMNS)
        drains = pd.DataFrame(drains, index=conditions)[BALANCED_KEYS]
        n_failed = int((~diagnostics["success"].astype(bool)).sum())
        return Prediction(predicted=predicted, real=measured_table(conditions), diagnostics=diagnostics, b=drains,
                          n_failed=n_failed)

    def closure_inputs(self):
        """
        Training-set summary of the "pts_fraction" closure from the fitted state: alpha = mean over the conditions of
        b_j / v_pts (0 on the zero-drain rows), u_ref / x_ref = geometric means of the fitted cofactors / balanced
        metabolites.  Use an estimator fitted on the training conditions only (no leak of the held-out data).
        """
        theta_full = self.full_theta()
        x_by, u_by, b_by = self.fitted_state()
        kinetics = self.kinetic_model()
        ratios, log_u, log_x = [], [], []
        for condition in self.conditions:
            concentrations = {**x_by[condition], **u_by[condition]}
            v_pts = float(kinetics.pts(theta_full, concentrations, enzyme_levels(condition)))
            ratios.append([b_by[condition][k] / v_pts for k in BALANCED_KEYS])
            log_u.append([np.log(u_by[condition][k]) for k in IMBALANCED_KEYS])
            log_x.append([np.log(x_by[condition][k]) for k in BALANCED_KEYS])
        alpha = np.mean(ratios, axis=0)
        mean_log_u = np.mean(log_u, axis=0)
        mean_log_x = np.mean(log_x, axis=0)
        alpha_by_key = {}
        for i, key in enumerate(BALANCED_KEYS):
            alpha_by_key[key] = float(alpha[i]) if key in FREE_DRAIN_KEYS else 0.0
        u_ref = {key: float(np.exp(mean_log_u[i])) for i, key in enumerate(IMBALANCED_KEYS)}
        x_ref = {key: float(np.exp(mean_log_x[i])) for i, key in enumerate(BALANCED_KEYS)}
        return {"alpha_by_condition": alpha_by_key, "u_ref": u_ref, "x_ref": x_ref}

    def objective_breakdown(self, theta=None):
        """
        Weighted SSE per condition at fixed kinetic parameters: every condition solves its inner problem over x, u
        and b with ALL parameters fixed at ``theta`` (default: the fit), started at x_init / u_init / b_init.
        parmest's 'SSE_weighted' is 0.5 * sum(((y_meas - y) / sigma)^2) per scenario and its objective is the MEAN
        over the scenarios, so the MEAN row (TOTAL / number of conditions) is the one comparable to obj_value.
        Columns flux_wsse, met_wsse, u_wsse (always 0: the cofactors are not measured outputs; the column keeps the
        layout of the earlier runs), total and termination; rows: conditions, TOTAL, MEAN.
        """
        check_solver("ipopt")
        theta_full = self.full_theta(theta)
        solver = pyo.SolverFactory("ipopt")
        for option, value in self.solver_options.items():
            solver.options[option] = value
        rows = []
        for condition in self.conditions:
            experiment = GlycolysisExperiment(condition, self.setup, theta_init=theta_full,
                                              x_init=self.x_init.get(condition), u_init=self.u_init.get(condition),
                                              b_init=self.b_init.get(condition))
            rows.append(condition_breakdown(experiment.get_labeled_model(), theta_full, solver))
        table = pd.DataFrame(rows, index=self.conditions)
        total = table[["flux_wsse", "met_wsse", "u_wsse", "total"]].sum(skipna=True)
        total["termination"] = "%d not optimal" % int((table["termination"] != "optimal").sum())
        table.loc["TOTAL"] = total
        mean = total[["flux_wsse", "met_wsse", "u_wsse", "total"]] / len(self.conditions)
        table.loc["MEAN"] = {**mean.to_dict(), "termination": ""}
        return table

def condition_breakdown(model, theta_full, solver):
    """Flux and metabolite weighted SSE of one labeled condition model solved at the fixed ``theta_full``."""
    for key in ALL_PARAMS:
        getattr(model, key).set_value(float(theta_full[key]))
        getattr(model, key).fix()
    flux_terms, met_terms = [], []
    for component, measured in model.experiment_outputs.items():
        term = ((float(measured) - component) / float(model.measurement_error[component])) ** 2
        if component.local_name in FLUX_KEYS:
            flux_terms.append(term)
        else:
            met_terms.append(term)
    model.wsse_obj = pyo.Objective(expr=0.5 * (sum(flux_terms) + sum(met_terms)))
    row = {"flux_wsse": np.nan, "met_wsse": np.nan, "u_wsse": np.nan, "total": np.nan, "termination": ""}
    try:
        result = solver.solve(model, tee=False)
    except Exception as exc:   # noqa: BLE001  (an IPOPT failure of one condition is reported, not raised)
        row["termination"] = ("%s: %s" % (type(exc).__name__, exc))[:200]
        return row
    row["termination"] = str(result.solver.termination_condition)
    row["flux_wsse"] = 0.5 * float(sum(pyo.value(t) for t in flux_terms))
    row["met_wsse"] = 0.5 * float(sum(pyo.value(t) for t in met_terms))
    row["u_wsse"] = 0.0
    row["total"] = row["flux_wsse"] + row["met_wsse"] + row["u_wsse"]
    return row
