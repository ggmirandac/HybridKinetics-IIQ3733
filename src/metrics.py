"""
metrics.py -- fit and prediction metrics, the likelihood-ratio threshold and the misfit scale.

``pred_metrics`` (RMSE and R2 over the measured cells, R2 against the pooled and the per-output means, the RMSE of the
mean predictor), ``b_vs_data`` (fitted drains against the data and their bounds), ``theta_at_bounds`` (free
parameters on a theta bound), ``lr_threshold`` (chi2(alpha, df) / (2 N), the likelihood-ratio threshold on the
parmest objective), ``n_nonzero_outputs`` and ``misfit_s2`` (residual chi2 per degree of freedom; it scales the LR
threshold when the model misses by more than the measurement error).

Position in the pipeline: reporting of every stage (stage 0, selection, final analysis, bootstrap, profiles, LOOCV).
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2

from model.parameters import LI_FLUX_KEYS, FREE_DRAIN_KEYS, BALANCED_KEYS, DUPLICATE_FLUX_GROUPS
from data import load_condition

def lr_threshold(alpha, n_conditions, s2=None):
    """
    Likelihood-ratio threshold on the parmest objective, chi2.ppf(alpha, 1) / (2 N), times ``s2`` if given.
    parmest's objective is (1/N) sum_j 0.5 chi2_j, so a delta-chi2 of c is a delta-obj of c / (2 N): a point is
    inside the region when obj < obj_hat + lr_threshold(...).  E.g. lr_threshold(0.95, 22) = 0.0873.
    """
    threshold = float(chi2.ppf(alpha, 1)) / (2.0 * float(n_conditions))
    if s2 is None:
        return threshold
    return threshold * float(s2)

def r_squared(y, y_hat, center):
    total = np.sum((y - center) ** 2)
    if total > 0:
        return float(1.0 - np.sum((y - y_hat) ** 2) / total)
    return np.nan

def pred_metrics(real, predicted, met_ref=None, flux_ref=None):
    """
    Metrics of the measured balanced metabolites ("met") and LI fluxes ("flux_li"): rmse_*, r2_*_pooled,
    r2_*_permean (against the per-column mean of ``real``, or ``met_ref`` / ``flux_ref`` {column: value}),
    rmse_*_meanbaseline (of that mean predictor) and n_*.
    """
    met_columns = [c for c in real.columns if c.startswith("C_") and real[c].notna().any()]
    flux_columns = [c for c in LI_FLUX_KEYS if c in real.columns and real[c].notna().any()]
    metrics = {}
    for name, columns, reference in (("met", met_columns, met_ref), ("flux_li", flux_columns, flux_ref)):
        real_values = real[columns].to_numpy(float)
        predicted_values = predicted[columns].to_numpy(float)
        measured = ~np.isnan(real_values)
        y = real_values[measured]
        y_hat = predicted_values[measured]
        if reference is None:
            column_mean = np.tile(np.nanmean(real_values, axis=0), (real_values.shape[0], 1))
        else:
            column_mean = np.tile(np.array([reference[c] for c in columns], float), (real_values.shape[0], 1))
        center = column_mean[measured]
        metrics["rmse_" + name] = float(np.sqrt(np.mean((y - y_hat) ** 2)))
        metrics["r2_" + name + "_pooled"] = r_squared(y, y_hat, y.mean())
        metrics["r2_" + name + "_permean"] = r_squared(y, y_hat, center)
        metrics["rmse_" + name + "_meanbaseline"] = float(np.sqrt(np.mean((y - center) ** 2)))
        metrics["n_" + name] = int(measured.sum())
    return metrics

def b_vs_data(est):
    """Fitted free drains of ``est`` against the data b and the drain bounds of the estimator."""
    _x_by, _u_by, fitted = est.fitted_state()
    rows = []
    for condition in est.conditions:
        data = load_condition(condition)["b"]
        for key in FREE_DRAIN_KEYS:
            lower, upper = est.drain_bounds[key]
            rows.append({"condition": condition, "key": key, "b_fit": fitted[condition][key],
                         "b_data": float(data.get(key, 0.0)), "lo": lower, "hi": upper})
    table = pd.DataFrame(rows)
    table["err"] = table["b_fit"] - table["b_data"]
    span = table["hi"] - table["lo"]
    table["at_bound"] = ((table["b_fit"] - table["lo"]) < 1e-4 * span) | ((table["hi"] - table["b_fit"]) < 1e-4 * span)
    return table

def theta_at_bounds(est, tol=0.01):
    """Free parameters within a log distance ``tol`` (1%) of a theta bound."""
    names = []
    full = est.full_theta()
    for p in est.free_params:
        value = float(full[p])
        lower, upper = est.theta_bounds[p]
        if abs(np.log(value / lower)) < tol or abs(np.log(upper / value)) < tol:
            names.append(p)
    return names

def n_nonzero_outputs(est):
    """
    Number of measured outputs (summed over the conditions) whose residual can be nonzero: the measured balanced
    metabolites and the measured LI fluxes, counting each group of duplicated flux rows (pfk = fba = tpi,
    gap = pgk, gpm = eno) once.
    """
    n = 0
    for condition in est.conditions:
        data = load_condition(condition)
        n += sum(1 for key in BALANCED_KEYS if key in data["x_real"])
        measured = {key for key in LI_FLUX_KEYS if key in data["v_real"]}
        grouped = set()
        for group in DUPLICATE_FLUX_GROUPS:
            n += int(any(key in measured for key in group))
            grouped.update(group)
        n += len(measured - grouped)
    return int(n)

def misfit_s2(est):
    """
    Misfit variance factor s2 = 2 N obj / (n_nonzero - p): the residual chi2 (summed over the conditions = 2 N obj)
    per degree of freedom, with p the number of free parameters.  s2 >> 1 means the model misses beyond the
    measurement error; the LR threshold is then multiplied by it.
    """
    dof = n_nonzero_outputs(est) - len(est.free_params)
    if dof <= 0:
        return float("nan")
    return 2.0 * len(est.conditions) * float(est.obj_value) / dof
