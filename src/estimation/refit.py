"""
refit.py -- refit of the estimator after parameters are fixed.

A fixed parameter is held at its estimated value, so the previous fit stays feasible with the SAME objective.  The
refit starts at that fit (theta and the primal point x, u, b) with the warm-start options and is kept only if its
objective is not higher; otherwise the previous fit itself is kept, with the fixed parameters removed from the free
set.

Position in the pipeline: after every fix of the parameter selection (selection.py).
"""
import copy
import dataclasses

import pandas as pd

from settings import WARM_START_OPTIONS
from model.parameters import ALL_PARAMS

def refit_with_fixed(est, new_free, log):
    """Refit with ``new_free`` free and everything else held at the values of ``est``; never worse than ``est``."""
    theta = {p: float(v) for p, v in est.full_theta().items()}
    x_init, u_init = est.primal_start()
    refit = est.clone(free_params=new_free, theta_init=theta, x_init=x_init, u_init=u_init,
                      b_init=est.primal_b_free(), solver_options=WARM_START_OPTIONS)
    try:
        refit.estimate()
        refit.refine_anchor()
    except Exception as exc:   # noqa: BLE001  (IPOPT not converging: the previous fit is kept)
        log("    refit failed (%s): previous fit kept" % str(exc)[:100])
        return keep_previous_fit(est, new_free, theta)
    refit.solver_options = dict(est.solver_options)
    if refit.obj_value > est.obj_value:
        log("    refit obj %.6f above the previous fit %.6f: previous fit kept" % (refit.obj_value, est.obj_value))
        return keep_previous_fit(est, new_free, theta)
    return refit

def keep_previous_fit(est, new_free, theta):
    """A copy of ``est`` restricted to ``new_free``: same point, same objective."""
    kept = copy.copy(est)
    kept.free_params = [p for p in ALL_PARAMS if p in set(new_free)]
    # the solved extensive form is shared with ``est``; only the description of the free set changes
    kept.setup = dataclasses.replace(est.setup, free_params=kept.free_params)
    kept.theta = pd.Series({p: theta[p] for p in kept.free_params}, dtype=float)
    kept.theta_init = dict(theta)
    return kept
