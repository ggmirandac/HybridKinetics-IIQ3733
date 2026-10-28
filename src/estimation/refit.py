"""
refit.py -- monotone refit of the estimator after parameters are fixed.

A fixed parameter is held at its ANCHORED value, so the anchor stays feasible with the SAME objective: a refit that
ends higher than the anchor has failed.  The attempts all start at the anchor primal point (theta, x, u, b), each
followed by ``refine_anchor``: the tight monotone warm start, the estimator's own options, a monotone barrier.  A
refit is accepted only if obj_after <= obj_anchor (1 + refit_rtol) + refit_atol; a start from the data is never
used; if every attempt fails the anchor itself is kept (refit_status "kept_anchor").

Position in the pipeline: after every fix of the parameter selection (selection.py).
"""
import copy
import dataclasses

import numpy as np
import pandas as pd

from settings import TIGHT_OPTIONS, REFIT_MONOTONE_OPTIONS
from model.parameters import ALL_PARAMS

def refit_with_fixed(est, new_free, params, log):
    """
    Refit with ``new_free`` free and everything else held at the anchored values of ``est``.  Always returns an
    estimator with the attributes refit_status ('tight' / 'default' / 'monotone' / 'kept_anchor'),
    refit_obj_before, refit_obj_after and refit_attempts ([{tag, obj, accepted, error}]).
    """
    theta_anchor = {p: float(v) for p, v in est.full_theta().items()}
    x_init, u_init = est.primal_start()
    b_init = est.primal_b_free()
    base_options = dict(est.solver_options)
    obj_anchor = float(est.obj_value)
    limit = obj_anchor * (1.0 + params["refit_rtol"]) + params["refit_atol"]
    attempts = [("tight", dict(TIGHT_OPTIONS)), ("default", {}), ("monotone", dict(REFIT_MONOTONE_OPTIONS))]
    records = []
    for tag, extra_options in attempts:
        refit = est.clone(free_params=new_free, theta_init=theta_anchor, x_init=x_init, u_init=u_init, b_init=b_init,
                          solver_options={**base_options, **extra_options})
        try:
            refit.estimate()
            refit.refine_anchor(primary_options=extra_options if tag == "tight" else None)
        except Exception as exc:   # noqa: BLE001  (IPOPT not converging is a normal outcome: try the next options)
            log("    refit attempt '%s' failed: %s" % (tag, str(exc)[:100]))
            records.append({"tag": tag, "obj": np.nan, "accepted": False, "error": str(exc)[:100]})
            continue
        refit.solver_options = dict(base_options)
        accepted = float(refit.obj_value) <= limit
        records.append({"tag": tag, "obj": float(refit.obj_value), "accepted": bool(accepted), "error": ""})
        log("    refit attempt '%s': obj %.6f (anchor %.6f) %s" % (
            tag, refit.obj_value, obj_anchor, "accepted" if accepted else "REJECTED (higher than the anchor)"))
        if accepted:
            refit.refit_status = tag
            refit.refit_obj_before = obj_anchor
            refit.refit_obj_after = float(refit.obj_value)
            refit.refit_attempts = records
            return refit
    kept = keep_anchor(est, new_free, theta_anchor)
    kept.refit_obj_before = obj_anchor
    kept.refit_obj_after = obj_anchor
    kept.refit_attempts = records
    log("    refit_status = kept_anchor (all attempts failed or rose above the anchor %.6f)" % obj_anchor)
    return kept

def keep_anchor(est, new_free, theta_anchor):
    """A copy of ``est`` restricted to ``new_free``: free and fixed parameters at the anchor, objective unchanged."""
    kept = copy.copy(est)
    kept.free_params = [p for p in ALL_PARAMS if p in set(new_free)]
    # the solved extensive form is shared with ``est``; only the description of the free set changes
    kept.setup = dataclasses.replace(est.setup, free_params=kept.free_params)
    kept.theta = pd.Series({p: theta_anchor[p] for p in kept.free_params}, dtype=float)
    kept.theta_init = dict(theta_anchor)
    kept.cov = None
    kept.refit_status = "kept_anchor"
    return kept
