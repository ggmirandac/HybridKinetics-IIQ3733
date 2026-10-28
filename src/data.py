"""
data.py -- data loading, measurement sigmas and the bounds of the estimation variables.

Reads the CSV files of Data/ (balanced and imbalanced metabolites, fluxes, proteins, cellular needs) and the
measurement standard deviations, and builds the box bounds of the per-condition variables of the estimation problem:
balanced metabolites x (log-normal box exp(mu +/- n_std sigma)), cofactors u (data range +/- slack, C_pi from the
literature) and free drains b (data range widened by a fraction of the span).  The rules are read from
settings/bounds_rules.csv.  ``check_bounds_contain_data`` asserts that the box contains every measured value: with
n_std = 2 six measured values fell outside it and put a floor under the objective.

Position in the pipeline: first step; the experiment, the estimator, the selection and the uncertainty steps take
their data and bounds from here.
"""
import os

import numpy as np
import pandas as pd

from paths import DATA_DIR
from settings import BOUNDS_RULES
from model.parameters import FREE_DRAIN_KEYS, IMBALANCED_KEYS, BALANCED_KEYS

# Cofactors missing in a condition fall back to their dataset median (``imbalanced_fallbacks``); these literature
# values (mM) only cover species that have no measurement at all.
LITERATURE_U = {"C_pi": 10.0, "C_nadh": 0.083, "C_pyr": 0.9}

def read_data_csv(name):
    return pd.read_csv(os.path.join(DATA_DIR, name), index_col=0)

def load_data_frames():
    """(balanced_met_df, imbalanced_met_df, prot_df, flux_df, cell_needs_df) of Data/, indexed by the first column."""
    balanced = read_data_csv("balanced_metabolites.csv")
    imbalanced = read_data_csv("imbalanced_metabolites.csv")
    proteins = read_data_csv("important_proteins.csv")
    fluxes = read_data_csv("important_fluxes.csv")
    needs = read_data_csv("cellular_needs.csv")
    return balanced, imbalanced, proteins, fluxes, needs

def available_conditions():
    """Conditions present (as columns) in all four primary data files, sorted."""
    common = None
    for name in ("balanced_metabolites.csv", "important_fluxes.csv", "imbalanced_metabolites.csv",
                 "important_proteins.csv"):
        columns = set(read_data_csv(name).columns)
        common = columns if common is None else common & columns
    return sorted(common)

def measured_balanced_keys():
    """Balanced metabolites with at least one measured value, in BALANCED_KEYS order."""
    balanced = read_data_csv("balanced_metabolites.csv")
    return [key for key in BALANCED_KEYS if key in balanced.index and balanced.loc[key].notna().any()]

def load_condition(condition):
    """One experimental condition as {"v_real", "x_real", "u", "e", "b"}: {name: float} of the measured values."""
    tables = {"v_real": "important_fluxes.csv", "x_real": "balanced_metabolites.csv",
              "u": "imbalanced_metabolites.csv", "e": "important_proteins.csv", "b": "cellular_needs.csv"}
    data = {}
    for key, name in tables.items():
        table = read_data_csv(name)
        if condition not in table.columns:
            raise KeyError("Condition '%s' not in %s" % (condition, list(table.columns)))
        data[key] = table[condition].dropna().to_dict()
    return data

def load_measurement_sigmas():
    """
    Measurement standard deviations of the weighted objective: (sigma_flux, sigma_met) with sigma_flux = {flux_key:
    std} (important_fluxes_Q.csv) and sigma_met = {C_<met>: std or None} (balanced_metabolites_Q.csv).
    """
    flux_q = read_data_csv("important_fluxes_Q.csv")
    met_q = read_data_csv("balanced_metabolites_Q.csv")
    # the two files name their only column differently ("STD" / "SD")
    sigma_flux = {}
    for key, value in flux_q.iloc[:, 0].dropna().items():
        sigma_flux[key] = float(value)
    sigma_met = {}
    for key, value in met_q.iloc[:, 0].items():
        sigma_met[key] = None if pd.isna(value) else float(value)
    return sigma_flux, sigma_met

def imbalanced_fallbacks():
    """
    Values of the cofactors that are missing in some conditions: the dataset median where any measurement exists,
    otherwise the literature value (LITERATURE_U).
    """
    imbalanced = read_data_csv("imbalanced_metabolites.csv")
    fallback = {}
    for key in IMBALANCED_KEYS:
        if key in imbalanced.index and imbalanced.loc[key].notna().any():
            fallback[key] = float(imbalanced.loc[key].median(skipna=True))
    for key, value in LITERATURE_U.items():
        fallback.setdefault(key, value)
    return fallback

def default_drain_bounds(conditions=None):
    """
    Bounds {key: (lo, hi)} of the free drains: data min / max of Data/cellular_needs.csv over ``conditions`` (all by
    default), widened by drain_widen (settings/bounds_rules.csv) times the span on each side; the sign is free.
    """
    widen = BOUNDS_RULES["drain_widen"]
    needs = read_data_csv("cellular_needs.csv")
    if conditions is not None:
        needs = needs[[c for c in needs.columns if c in set(conditions)]]
    bounds = {}
    for key in FREE_DRAIN_KEYS:
        row = needs.loc[key].astype(float)
        lo = float(row.min())
        hi = float(row.max())
        span = hi - lo if hi > lo else max(abs(hi), 1e-3)
        bounds[key] = (lo - widen * span, hi + widen * span)
    return bounds

def positive_values(series):
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return values[values > 0]

def log_std_range(series, n_std):
    """[max(exp(mu - n_std sigma), floor), exp(mu + n_std sigma)] of ln C over the positive values; None if < 2."""
    values = positive_values(series)
    if values.size < 2:
        return None
    logs = np.log(values)
    mu = float(logs.mean())
    sigma = float(logs.std(ddof=1))
    return (max(float(np.exp(mu - n_std * sigma)), BOUNDS_RULES["floor"]), float(np.exp(mu + n_std * sigma)))

def data_range(series):
    """Positive data range widened by cofactor_slack on each side, floored; None without positive values."""
    values = positive_values(series)
    if values.size < 1:
        return None
    slack = BOUNDS_RULES["cofactor_slack"]
    return (max(float(values.min()) * (1.0 - slack), BOUNDS_RULES["floor"]), float(values.max()) * (1.0 + slack))

def group_bounds(keys, table, mode, n_std):
    """
    Bounds of one group of metabolites under ``mode`` ('log_std' or 'data_range').  Metabolites without usable data
    get (floor, largest upper bound of the measured ones in the group).
    """
    bounds = {}
    measured_upper = []
    for key in keys:
        if key not in table.index:
            continue
        if mode == "log_std":
            bound = log_std_range(table.loc[key], n_std)
        else:
            bound = data_range(table.loc[key])
        if bound is not None:
            bounds[key] = bound
            measured_upper.append(bound[1])
    global_max = max(measured_upper) if measured_upper else 10.0
    for key in keys:
        bounds.setdefault(key, (BOUNDS_RULES["floor"], global_max))
    return bounds

def metabolite_bounds(conditions=None, n_std=BOUNDS_RULES["n_std"]):
    """
    (u_bounds, x_bounds) of the cofactors and the balanced metabolites from the data of ``conditions`` (all by
    default), with the modes of settings/bounds_rules.csv (balanced: log_std, cofactors: data_range).  C_pi is never
    measured, so here it gets the pooled cofactor fallback; ``variable_bounds`` replaces it by the literature range.
    ``n_std = 2`` gives the bounds of the v2 runs.
    """
    balanced = read_data_csv("balanced_metabolites.csv")
    imbalanced = read_data_csv("imbalanced_metabolites.csv")
    if conditions is not None:
        balanced = balanced[[c for c in balanced.columns if c in set(conditions)]]
        imbalanced = imbalanced[[c for c in imbalanced.columns if c in set(conditions)]]
    u_bounds = group_bounds(IMBALANCED_KEYS, imbalanced, BOUNDS_RULES["cofactor_mode"], n_std)
    x_bounds = group_bounds(BALANCED_KEYS, balanced, BOUNDS_RULES["balanced_mode"], n_std)
    return u_bounds, x_bounds

def variable_bounds(conditions=None):
    """
    The bounds of the pipeline: (u_bounds, x_bounds, drain_bounds) from the data of ``conditions`` (all by default,
    the training conditions in a LOOCV fold) under the rules of settings/bounds_rules.csv, with C_pi set to the
    literature range (pi_lo, pi_hi).  ``check_bounds_contain_data`` checks the result.
    """
    u_bounds, x_bounds = metabolite_bounds(conditions)
    u_bounds["C_pi"] = (float(BOUNDS_RULES["pi_lo"]), float(BOUNDS_RULES["pi_hi"]))
    return u_bounds, x_bounds, default_drain_bounds(conditions)

def values_outside(kind, table, bounds, keys, conditions, skip_nonpositive):
    """Rows {kind, key, condition, value, lo, hi} of the measured values of ``table`` outside ``bounds``."""
    columns = [c for c in table.columns if conditions is None or c in set(conditions)]
    rows = []
    for key in keys:
        if key not in table.index or key not in bounds:
            continue
        lo, hi = bounds[key]
        for condition in columns:
            value = pd.to_numeric(pd.Series([table.loc[key, condition]]), errors="coerce").iloc[0]
            if not np.isfinite(value) or (skip_nonpositive and value <= 0):
                continue
            if value < lo - 1e-12 or value > hi + 1e-12:
                rows.append({"kind": kind, "key": key, "condition": condition, "value": float(value),
                             "lo": float(lo), "hi": float(hi)})
    return rows

def check_bounds_contain_data(u_bounds, x_bounds, drain_bounds, conditions=None, raise_on_fail=True):
    """
    Every measured value inside the bounds of the fit: the balanced metabolites against ``x_bounds``, the positive
    cofactor values against ``u_bounds`` (zero / missing values are skipped) and the drains of cellular_needs.csv
    (FREE_DRAIN_KEYS rows) against ``drain_bounds``, over ``conditions`` (all by default).  Returns the violations
    (kind, key, condition, value, lo, hi); raises AssertionError when there is one and ``raise_on_fail``.
    """
    balanced, imbalanced, _proteins, _fluxes, needs = load_data_frames()
    rows = values_outside("balanced", balanced, x_bounds, list(balanced.index), conditions, True)
    rows += values_outside("imbalanced", imbalanced, u_bounds, list(imbalanced.index), conditions, True)
    rows += values_outside("drain", needs, drain_bounds, list(FREE_DRAIN_KEYS), conditions, False)
    violations = pd.DataFrame(rows, columns=["kind", "key", "condition", "value", "lo", "hi"])
    if raise_on_fail and len(violations):
        raise AssertionError("bounds exclude %d measured values:\n%s" % (len(violations),
                                                                         violations.head(20).to_string()))
    return violations
