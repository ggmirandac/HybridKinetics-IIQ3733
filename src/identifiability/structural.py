"""
structural.py -- structural identifiability of the rate laws (Stigter & Molenaar 2015).

Local structural identifiability after Stigter & Molenaar (2015, Automatica 58:118-124, "A fast algorithm to assess
local structural identifiability"), applied to the 9 reaction rates v(x, u, e; theta) at the fitted states of all
conditions:
  1. q random parameter vectors, every free parameter log-uniform within its bounds (the fixed ones at their values);
  2. the exact sensitivity matrix of the rates at each of them (CasADi Jacobian of the very same rate-law code the
     estimator uses), in log-parameters and scaled per reaction: S[i, j] = theta_j dv_i/dtheta_j / median |v_r|;
  3. the matrices of the q vectors stacked and decomposed by SVD: a singular value below structural_tol times the
     largest is a structural zero, a direction of the parameters that leaves every rate unchanged at all q points.
The right singular vector of the smallest singular value names the parameters of that direction; the rules of
identifiability/rules.py decide which one is fixed.  A direction that is null here is null even if all states and
rates were known without noise.

Position in the pipeline: the structural step at the start of the parameter selection and the structural check of
the final structure (selection.py).
"""
import numpy as np
import casadi as ca

from settings import THETA_BOUNDS
from model.kinetics import GlycolysisRateLaws
from model.parameters import ALL_PARAMS, FLUX_KEYS, BALANCED_KEYS, IMBALANCED_KEYS, ENZYME_KEYS
from model.stoichiometry import load_keq
from data import load_condition

class RateLaws:
    """CasADi functions of the 9 reaction rates v(C, e, k) and their exact Jacobian dv/dk (9 x 37)."""

    def __init__(self):
        rate_laws = GlycolysisRateLaws(load_keq())
        C = ca.SX.sym("C", len(BALANCED_KEYS) + len(IMBALANCED_KEYS))
        e = ca.SX.sym("e", len(ENZYME_KEYS))
        k = ca.SX.sym("k", len(ALL_PARAMS))
        concentrations = {key: C[i] for i, key in enumerate(BALANCED_KEYS + IMBALANCED_KEYS)}
        enzymes = {key: e[i] for i, key in enumerate(ENZYME_KEYS)}
        constants = {key: k[i] for i, key in enumerate(ALL_PARAMS)}
        v = ca.vertcat(*rate_laws.compute_fluxes(concentrations, enzymes, constants))
        self.f_v = ca.Function("rates", [C, e, k], [v])
        self.f_J = ca.Function("drates_dk", [C, e, k], [ca.jacobian(v, k)])

    def rates(self, C, e, theta):
        return self.f_v(C, e, theta).full().ravel()

    def jac(self, C, e, theta):
        """dv/dtheta (9 x 37)."""
        return self.f_J(C, e, theta).full()

def condition_states(est):
    """
    {condition: {"C": 16-vector (balanced + imbalanced keys), "e": 8 enzyme levels}} at the fitted operating point
    of the estimator ``est``.
    """
    x_by, u_by, _b_by = est.fitted_state()
    states = {}
    for condition in est.conditions:
        C = np.array([x_by[condition][k] for k in BALANCED_KEYS] + [u_by[condition][k] for k in IMBALANCED_KEYS],
                     dtype=float)
        enzymes = load_condition(condition)["e"]
        e = np.array([float(enzymes[k]) for k in ENZYME_KEYS], dtype=float)
        states[condition] = {"C": C, "e": e}
    return states

def random_parameter_vectors(theta, free, q, seed):
    """``q`` parameter dicts: every free parameter log-uniform within its bounds, the others at ``theta``."""
    rng = np.random.default_rng(seed)
    vectors = []
    for _ in range(q):
        vector = {p: float(theta[p]) for p in ALL_PARAMS}
        for p in free:
            lower, upper = THETA_BOUNDS[p]
            vector[p] = float(np.exp(rng.uniform(np.log(lower), np.log(upper))))
        vectors.append(vector)
    return vectors

def rate_sensitivity_matrix(rate_laws, theta, states, free):
    """
    Sensitivity matrix of the rates at one parameter vector: rows = (condition, reaction), columns = ``free``,
    S[i, j] = theta_j dv_i/dtheta_j / median over the conditions of |v_r| (r the reaction of row i).
    """
    theta_vector = np.array([theta[p] for p in ALL_PARAMS], dtype=float)
    columns = [ALL_PARAMS.index(p) for p in free]
    rates = []
    jacobians = []
    for state in states.values():
        rates.append(rate_laws.rates(state["C"], state["e"], theta_vector))
        jacobians.append(rate_laws.jac(state["C"], state["e"], theta_vector))
    scale = np.median(np.abs(np.array(rates)), axis=0)
    scale[scale == 0] = 1.0
    blocks = []
    for jacobian in jacobians:
        blocks.append(jacobian[:, columns] * theta_vector[columns][None, :] / scale[:, None])
    return np.vstack(blocks)

def stacked_sensitivity_matrix(rate_laws, theta, states, free, params):
    """The rate sensitivity matrices of params["structural_q"] random parameter vectors, stacked (columns = free)."""
    blocks = []
    for vector in random_parameter_vectors(theta, free, params["structural_q"], params["seed"]):
        blocks.append(rate_sensitivity_matrix(rate_laws, vector, states, free))
    return np.vstack(blocks)

def singular_spectrum(matrix):
    """(singular values / largest, right singular vectors as rows) of ``matrix``."""
    _, singular, Vt = np.linalg.svd(matrix, full_matrices=False)
    return singular / singular[0], Vt
