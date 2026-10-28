"""
experiment.py -- one experimental condition as a labeled Pyomo model (the estimation problem of one scenario).

``GlycolysisExperiment`` builds, for ONE condition, a ConcreteModel with the balanced metabolites x, the cofactors u,
the free drains b and the kinetic parameters theta as variables, the 9 rate expressions (the rate laws of
model/kinetics.py), the steady-state constraint S v = b and the suffixes parmest needs (experiment_outputs,
measurement_error, unknown_parameters).  The cofactors are bounded decision variables that are not in the objective;
the drains of FREE_DRAIN_KEYS are decision variables, the four other rows keep b = 0, so pfk = fba = tpi, gap = pgk
and gpm = eno and the 5 linearly independent fluxes are model predictions fitted to the data.

Position in the pipeline: built by the estimator (estimation/estimator.py), once per condition; parmest sums the
scenarios into the extensive form.  Edit this file to change what is measured or how the variables enter the model.
"""
from dataclasses import dataclass

import numpy as np

import pyomo.environ as pyo
from pyomo.contrib.parmest.experiment import Experiment

from settings import THETA_BOUNDS
from model.kinetics import GlycolysisRateLaws
from model.parameters import (ALL_PARAMS, LITERATURE_THETA, LI_FLUX_KEYS, FREE_DRAIN_KEYS, BALANCED_KEYS,
                              IMBALANCED_KEYS, ENZYME_KEYS, FLUX_KEYS, RXN_ORDER)
from data import load_condition

@dataclass
class EstimationSetup:
    """What every condition of one estimator shares (picklable, so process pools can rebuild the models)."""
    keq: dict
    S: np.ndarray
    sigma_flux: dict
    sigma_met: dict
    u_fallback: dict
    x_bounds: dict
    u_bounds: dict
    drain_bounds: dict
    free_params: list

def metabolite_sigma(sigma_met, key, value):
    """Measurement sigma of a balanced metabolite; 20% of the value (at least 1e-3) where none is tabulated."""
    sigma = sigma_met.get(key)
    if sigma is None or sigma <= 0:
        return max(0.2 * abs(value), 1e-3)
    return sigma

def clip(value, lower, upper):
    return float(min(max(value, lower), upper))

class GlycolysisExperiment(Experiment):
    """
    One condition for parmest: x (9 balanced metabolites, measured values or the geometric mean of the bounds as
    start), u (7 cofactors, measured value or dataset fallback as start), theta (37 Vars, all fixed; parmest frees
    the ``unknown_parameters``), the 9 rate expressions, b (5 free drains) and S v = b.  Outputs: the measured LI
    fluxes and balanced metabolites with their 1-sigma errors.  ``x_init`` / ``u_init`` / ``b_init`` ({key: value})
    are primal warm starts (clipped into the bounds); ``theta_init`` updates the literature values.
    """

    def __init__(self, condition, setup, theta_init=None, x_init=None, u_init=None, b_init=None):
        super().__init__(model=None)
        self.condition = condition
        self.setup = setup
        self.data = load_condition(condition)
        self.theta_init = dict(LITERATURE_THETA)
        if theta_init:
            self.theta_init.update(theta_init)
        self.x_init = x_init or {}
        self.u_init = u_init or {}
        self.b_init = b_init or {}

    def create_model(self):
        # The order in which the components are added fixes the order of the variables in the NLP: keep it.
        setup = self.setup
        model = pyo.ConcreteModel(name="glycolysis_%s" % self.condition)
        for key in BALANCED_KEYS:
            lower, upper = setup.x_bounds[key]
            start = self.x_init.get(key, self.data["x_real"].get(key, np.sqrt(lower * upper)))
            model.add_component(key, pyo.Var(bounds=(lower, upper), initialize=clip(start, lower, upper)))
        for key in IMBALANCED_KEYS:
            lower, upper = setup.u_bounds[key]
            value = self.data["u"].get(key, setup.u_fallback[key])
            start = clip(self.u_init.get(key, value), lower, upper)
            model.add_component(key, pyo.Var(bounds=(lower, upper), initialize=start))
        for key in ALL_PARAMS:
            lower, upper = THETA_BOUNDS[key]
            value = clip(self.theta_init[key], lower, upper)
            model.add_component(key, pyo.Var(bounds=(lower, upper), initialize=value))
            getattr(model, key).fix(value)    # parmest unfixes the ones it estimates
        concentrations = {}
        for key in BALANCED_KEYS + IMBALANCED_KEYS:
            concentrations[key] = getattr(model, key)
        constants = {}
        for key in ALL_PARAMS:
            constants[key] = getattr(model, key)
        enzymes = {}
        for key in ENZYME_KEYS:
            enzymes[key] = float(self.data["e"][key])
        rate_laws = GlycolysisRateLaws(setup.keq)
        for flux_key, reaction in zip(FLUX_KEYS, RXN_ORDER):
            rate = getattr(rate_laws, reaction)(constants, concentrations, enzymes)
            model.add_component(flux_key, pyo.Expression(expr=rate))
        drains = self.data["b"]
        for key in FREE_DRAIN_KEYS:
            lower, upper = setup.drain_bounds[key]
            start = clip(float(self.b_init.get(key, drains.get(key, 0.0))), lower, upper)
            model.add_component("b_" + key, pyo.Var(bounds=(lower, upper), initialize=start))
        model.ss_constraint = pyo.Constraint(range(9), rule=self.steady_state_rule)
        self.model = model
        return model

    def steady_state_rule(self, model, i):
        """Row i of S v = b: a free drain variable or the data drain (0 for the four zero-drain rows)."""
        rate_sum = sum(self.setup.S[i, j] * getattr(model, FLUX_KEYS[j]) for j in range(9))
        if BALANCED_KEYS[i] in FREE_DRAIN_KEYS:
            return rate_sum == getattr(model, "b_" + BALANCED_KEYS[i])
        return rate_sum == float(self.data["b"].get(BALANCED_KEYS[i], 0.0))

    def label_model(self):
        model = self.model
        model.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        model.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # the order of the outputs is the order of the terms of the objective: fluxes first, then metabolites
        for key in FLUX_KEYS:
            if key in LI_FLUX_KEYS and key in self.data["v_real"]:
                component = getattr(model, key)
                model.experiment_outputs[component] = float(self.data["v_real"][key])
                model.measurement_error[component] = float(self.setup.sigma_flux[key])
        for key in BALANCED_KEYS:
            if key in self.data["x_real"]:
                component = getattr(model, key)
                value = float(self.data["x_real"][key])
                model.experiment_outputs[component] = value
                model.measurement_error[component] = float(metabolite_sigma(self.setup.sigma_met, key, value))
        model.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        for name in self.setup.free_params:
            component = getattr(model, name)
            model.unknown_parameters[component] = pyo.ComponentUID(component)
        return model

    def get_labeled_model(self):
        self.create_model()
        return self.label_model()
