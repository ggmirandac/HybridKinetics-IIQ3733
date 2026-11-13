"""
anchors.py -- building, saving and restoring the anchored estimators (the fitted points of the pipeline).

``make_est`` builds the estimator of the pipeline (data-derived bounds of settings/bounds_rules.csv, theta bounds of
settings/theta_bounds.csv, IPOPT options of settings/solver_options.csv); ``save_anchor`` writes anchor_state.pkl
(theta, free set, x, u, b, objective) and theta_full.csv; ``load_point`` rebuilds an anchor-only estimator at a saved
point WITHOUT a solve; ``restore_anchor`` re-solves from the saved point with the warm-start options.
``check_bounds`` asserts that the bounds contain every measured value (called at the start of every run).

Position in the pipeline: shared by stage 0, the selection, the final analysis, bootstrap, profiles and LOOCV.
"""
import os
import pickle

import pandas as pd

from settings import SOLVER_OPTIONS, WARM_START_OPTIONS
from progress import log
from model.parameters import ALL_PARAMS
from data import variable_bounds, check_bounds_contain_data
from estimation.estimator import GlycolysisParameterEstimator

def check_bounds():
    """Assert that the bounds contain every measured value (the first check of every run)."""
    u_bounds, x_bounds, drain_bounds = variable_bounds()
    check_bounds_contain_data(u_bounds, x_bounds, drain_bounds)
    log("check_bounds_contain_data: OK (balanced, cofactors, drains); Pi bounds %s" % (u_bounds["C_pi"],))

def make_est(conditions, theta, free, solver_options=None, x_init=None, u_init=None, b_init=None):
    """
    The estimator of the pipeline at the starting theta ``theta`` with ``free`` free.  ``solver_options`` must be
    given HERE: parmest takes them at construction, so changing est.solver_options later does not reach estimate().
    """
    u_bounds, x_bounds, drain_bounds = variable_bounds()
    if solver_options is None:
        solver_options = SOLVER_OPTIONS
    return GlycolysisParameterEstimator(conditions=list(conditions), free_params=list(free), theta_init=dict(theta),
                                        solver_options=dict(solver_options), x_bounds=x_bounds, u_bounds=u_bounds,
                                        drain_bounds=drain_bounds, x_init=x_init, u_init=u_init, b_init=b_init)

def save_anchor(est, out_dir, extra=None):
    """anchor_state.pkl and theta_full.csv of the current anchor of ``est``; returns the saved state."""
    os.makedirs(out_dir, exist_ok=True)
    full = est.full_theta()
    theta_full = {p: float(full[p]) for p in ALL_PARAMS}
    x_init, u_init = est.primal_start()
    state = {"theta": theta_full, "free": list(est.free_params), "x_init": x_init, "u_init": u_init,
             "b_init": est.primal_b_free(), "obj": float(est.obj_value), "conditions": list(est.conditions)}
    state.update(extra or {})
    with open(os.path.join(out_dir, "anchor_state.pkl"), "wb") as fh:
        pickle.dump(state, fh)
    pd.Series(theta_full, name="value").rename_axis("param").to_csv(os.path.join(out_dir, "theta_full.csv"))
    return state

def read_anchor(anchor_dir):
    with open(os.path.join(anchor_dir, "anchor_state.pkl"), "rb") as fh:
        return pickle.load(fh)

def load_point(state):
    """
    Anchor-only estimator at a saved point (the dict of anchor_state.pkl) WITHOUT a solve: a warm-started solve from
    the point drifts by ~0.3% because IPOPT has no duals.  Supports reduced sensitivities, structural checks,
    predictions, objective_breakdown and LOOCV; the covariance is not available.
    """
    est = make_est(state["conditions"], state["theta"], state["free"])
    return est.anchor_only(state["x_init"], state["u_init"], state["b_init"], state["theta"], state["obj"])

def restore_anchor(anchor_dir):
    """
    Re-solve the anchor of ``anchor_dir/anchor_state.pkl`` from its theta and primal point with the warm-start
    options, then ``refine_anchor``.  Returns (estimator, saved state).
    """
    state = read_anchor(anchor_dir)
    est = make_est(state["conditions"], state["theta"], state["free"], solver_options=WARM_START_OPTIONS,
                   x_init=state["x_init"], u_init=state["u_init"], b_init=state["b_init"])
    est.estimate()
    obj_start = float(est.obj_value)
    est.refine_anchor()
    est.solver_options = dict(SOLVER_OPTIONS)
    log("anchor restored: obj %.6f (first warm-start solve %.6f, saved %.6f)" % (est.obj_value, obj_start,
                                                                                  state["obj"]))
    return est, state
