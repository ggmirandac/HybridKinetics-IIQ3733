"""
param_estimation_parmest.py
===========================
Pyomo + ``pyomo.contrib.parmest`` parameter estimation and *identifiability
analysis* for the E. coli glycolysis mechanistic model.

Mathematical problem (one scenario ``e`` per experimental condition)
--------------------------------------------------------------------
    min_{ {x_e}_e , theta }   sum_e sum_outputs  ((y_meas - y_model)/sigma)^2 / 2
    s.t.   S v(x_e, u_e, e_e; Keq, theta) = b_e      for all e
           x_lb <= x_e <= x_ub
           theta_lb <= theta <= theta_ub

where x_e (balanced metabolites) are *per-scenario* decision variables and
theta (kinetic parameters) are *shared* across all scenarios (parmest's
``unknown_parameters``).

Tested against the parmest API shipped with Pyomo >= 6.9.5 (Experiment-based
interface, ``obj_function='SSE_weighted'`` + ``measurement_error`` suffix,
``cov_est`` for covariance).

Quickstart
----------
See ``param_estimation_parmest_GUIDE.md`` for two full notebook recipes.
The following methods work without ipopt (CasADi only):

    est  = GlycolysisParameterEstimator(conditions=["KO02","KO03","KO05"])
    pred = est.predict()         # Prediction(predicted, real, rmse)
    r    = est.rmse()            # {met, flux, data_norm}
    G    = est.sensitivity_matrix()   # {condition: DataFrame 18x37}

The following methods require ipopt:

    theta = est.estimate()       # weighted least squares
    cov   = est.covariance()     # parameter covariance
    ms    = est.multistart()     # multistart or bootstrap
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import pyomo.environ as pyo
from pyomo.contrib.parmest.experiment import Experiment
import pyomo.contrib.parmest.parmest as parmest

# Import the authoritative kinetics from the same directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kinetics_noor import EcoliCarbonKinetics, ALL_PARAMS  # noqa: E402

# paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(_THIS_DIR, "..", "Data")


# ===========================================================================
# Solver guard
# ===========================================================================
def check_solver(name: str = "ipopt") -> bool:
    """
    Check that the named solver is available via Pyomo.  Raises RuntimeError
    with a clear install message if it is not.

    Returns True if the solver is available.

    Install note: conda install -n SaaLab -c conda-forge ipopt
    """
    from pyomo.environ import SolverFactory
    if not SolverFactory(name).available():
        raise RuntimeError(
            f"Solver '{name}' is not available in the current environment. "
            f"Install it with:\n"
            f"    conda install -n SaaLab -c conda-forge {name}\n"
            f"or, for IDAES-bundled solvers:\n"
            f"    idaes get-extensions"
        )
    return True


# ===========================================================================
# Prediction dataclass
# ===========================================================================
@dataclass
class Prediction:
    """
    Result of GlycolysisParameterEstimator.predict().

    Attributes
    ----------
    predicted : pd.DataFrame
        Model-predicted values (rows=conditions, cols=18 outputs: C_* then v_*).
        Cells where the corresponding measured value is NaN are also set to NaN
        so that predicted and real are aligned for plotting and scoring.
    real : pd.DataFrame
        Measured values (same shape).  NaN where not measured (or where
        fit_metabolites=False, which zeros out all C_* columns).
    rmse : dict
        Aggregate RMSE metrics over the measured (non-NaN) cells only:
          'met'       -- sqrt(mean((pred-real)^2)) over metabolite (C_*) cells, in mM.
                         NaN if no metabolite cells are measured.
          'flux'      -- sqrt(mean((pred-real)^2)) over flux (v_*) cells, in mmol/gDCW/h.
                         NaN if no flux cells are measured.
          'data_norm' -- normalized RMSE: for each output column, divide its residuals
                         by that column's mean measured value (nanmean of real[col]);
                         then sqrt(nanmean of all normalized squared residuals) over
                         ALL measured cells (metabolites + fluxes).  Columns whose mean
                         measured value is 0 or NaN are skipped.
          'weighted'  -- measurement-sigma-weighted RMSE: residuals divided by each
                         output's measurement std (the same sigmas the weighted-SSE
                         objective uses), then sqrt(nanmean of squared weighted
                         residuals) over ALL measured cells.  This is dimensionless and
                         directly interpretable: ~1 means the fit sits at the noise
                         floor, >>1 means the model misses beyond measurement error.
    per_output : pd.Series
        Per-output raw RMSE (one value per measured output, in that output's own
        units), indexed by output name (C_* then v_*).  Outputs with no measured
        cell are omitted.  Useful for spotting which single metabolite/flux fits
        worst and for bar plots.
    """
    predicted: pd.DataFrame = field(default_factory=pd.DataFrame)
    real: pd.DataFrame = field(default_factory=pd.DataFrame)
    rmse: dict = field(default_factory=dict)
    per_output: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def summary(self) -> str:
        """Short human-readable summary of the prediction result."""
        n_cond = len(self.predicted)
        n_meas = int(self.real.notna().sum().sum())
        n_outputs = len(self.predicted.columns)
        met_rmse = self.rmse.get("met", float("nan"))
        flux_rmse = self.rmse.get("flux", float("nan"))
        norm_rmse = self.rmse.get("data_norm", float("nan"))
        wgt_rmse = self.rmse.get("weighted", float("nan"))
        lines = [
            f"Prediction over {n_cond} condition(s), {n_outputs} outputs "
            f"({n_meas} measured cells):",
            f"  RMSE met       = {met_rmse:.4g} mM",
            f"  RMSE flux      = {flux_rmse:.4g} mmol/gDCW/h",
            f"  RMSE data_norm = {norm_rmse:.4g} (dimensionless, column-mean normalized)",
            f"  RMSE weighted  = {wgt_rmse:.4g} (sigma units; ~1 = at noise floor)",
        ]
        if not self.per_output.empty:
            worst = self.per_output.sort_values(ascending=False).head(3)
            worst_str = ", ".join(f"{k}={v:.3g}" for k, v in worst.items())
            lines.append(f"  worst-fit outputs: {worst_str}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Literature parameter values (initial guesses) -- from the __main__ block of
# kinetics_noor.py.  Units: kcat in 1/h, Ks/Kp in mM, v_max in mmol/gDW/h.
# ---------------------------------------------------------------------------
LITERATURE_THETA = {
    # PTS (empirical, Kadir et al. 2010)
    "v_max_1": 25.739, "Ka1_1": 1.0, "Ka2_1": 0.01, "Ka3_1": 1.0, "K_g6p_1": 0.5,
    # PGI
    "Ks_g6p_pgi": 0.48, "Kp_f6p_pgi": 0.19, "kcat_f_2": 1475.0,
    # PFK
    "Ks_f6p_3": 0.16, "Ks_atp_3": 0.12, "Kp_fbp_3": 0.50, "Kp_adp_3": 0.20, "kcat_f_3": 580.0,
    # FBA
    "Ks_fbp_4": 0.30, "Kp_g3p_4": 0.40, "Kp_dhap_4": 2.00, "kcat_f_4": 95.0,
    # TPI
    "kcat_f_5": 4300.0, "Ks_dhap_5": 0.61, "Kp_g3p_5": 1.20,
    # GAPDH
    "kcat_f_6": 118.0, "Ks_g3p_6": 0.21, "Ks_pi_6": 0.29, "Ks_nad_6": 0.09,
    "Kp_pgp_6": 0.01, "Kp_nadh_6": 0.06,
    # PGK
    "kcat_f_7": 1150.0, "Ks_pgp_7": 0.05, "Ks_adp_7": 0.10, "Ks_3pg_7": 0.53, "Ks_atp_7": 0.30,
    # GPM
    "kcat_f_8": 540.0, "Ks_3pg_8": 0.20, "Ks_2pg_8": 1.40,
    # ENO
    "kcat_f_9": 550.0, "Ks_2pg_9": 0.10, "Ks_pep_9": 0.50,
}

# Parameters the report fixed (sensitivity-matrix criterion, Table I -> 13 params).
# Provided as a convenience for staged / reduced-model estimation.  NOT applied
# by default (default estimates all 37 parameters jointly).
REPORT_FIXED_PARAMS = [
    "Ka2_1", "Ka3_1",                       # PTS
    "Kp_f6p_pgi",                            # PGI
    "Kp_fbp_3", "Kp_adp_3",                  # PFK
    "Kp_g3p_4", "Kp_dhap_4",                 # FBA
    "Ks_dhap_5",                             # TPI
    "Ks_g3p_6", "Ks_nad_6", "Ks_pi_6",       # GAPDH
    "Ks_2pg_8",                              # GPM
    "Ks_pep_9",                              # ENO
]


def resolve_free_params(free_params=None, fixed_params=None) -> list:
    """
    Resolve which kinetic parameters are *free* (estimated).

    Specify **either**:
      * ``free_params``  -- the parameters to estimate (the rest are held fixed), or
      * ``fixed_params`` -- the parameters to hold fixed (the rest are estimated).

    ``fixed_params`` is usually the convenient one (e.g.
    ``fixed_params=REPORT_FIXED_PARAMS``).  If both are given they must partition
    ``ALL_PARAMS`` (disjoint and covering).  If neither is given, all 37
    parameters are free.

    Returns the ordered list of free parameters (subset of ``ALL_PARAMS``).
    """
    for name, vals in (("free_params", free_params), ("fixed_params", fixed_params)):
        if vals is not None:
            bad = [p for p in vals if p not in ALL_PARAMS]
            if bad:
                raise ValueError(f"Unknown {name} (not in ALL_PARAMS): {bad}")

    if free_params is None and fixed_params is None:
        return list(ALL_PARAMS)
    if free_params is None:                       # fixed given -> free is the complement
        fixed = set(fixed_params)
        return [p for p in ALL_PARAMS if p not in fixed]
    if fixed_params is None:                       # free given directly
        return [p for p in ALL_PARAMS if p in set(free_params)]
    # both given: must be a clean partition
    if set(free_params) & set(fixed_params):
        raise ValueError("free_params and fixed_params overlap.")
    if set(free_params) | set(fixed_params) != set(ALL_PARAMS):
        raise ValueError("free_params and fixed_params together must cover ALL_PARAMS.")
    return [p for p in ALL_PARAMS if p in set(free_params)]

# ---------------------------------------------------------------------------
# Default bounds.  Tighten with literature where possible.
# ---------------------------------------------------------------------------
# Kinetic parameters: strictly positive, several orders of magnitude wide.
THETA_BOUNDS = {p: (1e-6, 1e5) for p in ALL_PARAMS}

# Balanced metabolites (mM).  Strictly-positive lower bound avoids division
# blow-ups in the thermodynamic gamma terms.  The three *unmeasured* metabolites
# (C_2pg, C_g3p, C_pgp) get physiological ranges; they are pinned only by the
# steady-state constraint.
X_BOUNDS = {
    "C_g6p":  (1e-4, 10.0),
    "C_f6p":  (1e-4, 10.0),
    "C_fbp":  (1e-4, 10.0),
    "C_dhap": (1e-4, 10.0),
    "C_g3p":  (1e-4, 2.0),    # unmeasured
    "C_pgp":  (1e-5, 1.0),    # unmeasured
    "C_3pg":  (1e-4, 10.0),
    "C_2pg":  (1e-5, 1.0),    # unmeasured
    "C_pep":  (1e-4, 10.0),
}

# Imbalanced metabolites (mM): fixed inputs.  Three are incompletely measured:
#   C_pi   -> absent from the dataset entirely (literature ~10 mM, Bennett 2009)
#   C_nadh -> missing in 16/22 conditions
#   C_pyr  -> missing in 10/22 conditions
# For the missing entries we fall back to each metabolite's own dataset median
# (data-driven, see ``imbalanced_fallbacks``); ``LITERATURE_U`` only covers
# species with no measurements at all.  Alternative (not used by default): treat
# C_pyr/C_nadh as bounded decision variables instead of fixed inputs.
LITERATURE_U = {"C_pi": 10.0, "C_nadh": 0.083, "C_pyr": 0.9}


def imbalanced_fallbacks(data_dir: str = DEFAULT_DATA_DIR) -> dict:
    """
    Per-metabolite fallback values for imbalanced metabolites that are missing
    in some conditions: the dataset median where any measurement exists,
    otherwise the literature value.
    """
    df = _read_csv("imbalanced_metabolites.csv", data_dir, "VarNames")
    fb = {}
    for key in EcoliCarbonKinetics.imbalanced_keys:
        if key in df.index and df.loc[key].notna().any():
            fb[key] = float(df.loc[key].median(skipna=True))
    for key, val in LITERATURE_U.items():
        fb.setdefault(key, val)
    return fb


# Literature ranges (mM) for imbalanced metabolites with no usable dataset spread
# (only C_pi has no measurements at all).
LITERATURE_U_BOUNDS = {"C_pi": (1.0, 20.0)}


def imbalanced_bounds(data_dir: str = DEFAULT_DATA_DIR, pad: float = 0.0) -> dict:
    """
    Bounds (lb, ub) for the imbalanced metabolites when they are treated as
    decision variables.  Following the report, bounds come from the experimental
    min/max of each metabolite across conditions.

    Parameters
    ----------
    pad : float
        Relative widening of the data range, e.g. ``pad=0.1`` expands [lo, hi]
        by 10% on each side.  Degenerate (single-value) ranges are always padded
        by at least 50%.  Lower bounds are floored at 1e-6 (strictly positive, so
        the thermodynamic/denominator terms stay finite).
    """
    df = _read_csv("imbalanced_metabolites.csv", data_dir, "VarNames")
    bounds = {}
    for key in EcoliCarbonKinetics.imbalanced_keys:
        if key in df.index and df.loc[key].notna().any():
            row = df.loc[key].dropna().astype(float)
            lo, hi = float(row.min()), float(row.max())
            if lo == hi:
                lo, hi = lo * 0.5, hi * 1.5
            elif pad:
                span = hi - lo
                lo, hi = lo - pad * span, hi + pad * span
            bounds[key] = (max(lo, 1e-6), hi)
    for key, rng in LITERATURE_U_BOUNDS.items():
        bounds.setdefault(key, rng)
    for key in EcoliCarbonKinetics.imbalanced_keys:        # final safety net
        bounds.setdefault(key, (1e-4, 50.0))
    return bounds


# ===========================================================================
# Data layer
# ===========================================================================
def load_keq(data_dir: str = DEFAULT_DATA_DIR) -> dict:
    """Load equilibrium constants from the pickle as a {reaction: float} dict."""
    with open(os.path.join(data_dir, "k_eq_values.pkl"), "rb") as fh:
        keq = pickle.load(fh)
    # Cast numpy scalars to plain floats so the kinetics' gamma terms stay numeric.
    return {k: float(v) for k, v in keq.items()}


def build_stoichiometric_matrix() -> np.ndarray:
    """
    Return the 9x9 stoichiometric matrix S, rows ordered by
    ``EcoliCarbonKinetics.balanced_keys`` and columns by ``flux_keys``.

    The values come from the *authoritative* ``_construct_stoichiometric_matrix``
    in ``kinetics_noor.py`` (which is alphabetically indexed); here we reindex it
    to the ``balanced_keys`` order used everywhere else in this module.  An
    assertion guards against silent drift.
    """
    inst = object.__new__(EcoliCarbonKinetics)        # bypass heavy __init__
    S_df = inst._construct_stoichiometric_matrix()    # rows: met names, cols: rxn names

    # Map "C_g6p" -> "g6p" etc. and reindex rows to balanced_keys order.
    met_order = [k[2:] for k in EcoliCarbonKinetics.balanced_keys]  # strip "C_"
    rxn_order = ["pts", "pgi", "pfk", "fba", "tpi", "gap", "pgk", "gpm", "eno"]
    S = S_df.loc[met_order, rxn_order].to_numpy(dtype=float)

    # Sanity check vs. the known glycolytic topology (degree of each metabolite).
    assert S.shape == (9, 9), "Stoichiometric matrix must be 9x9."
    assert np.all(np.abs(S.sum(axis=0)) <= 2 + 1e-9), "Unexpected column sums in S."
    return S


def _read_csv(name: str, data_dir: str, index_col) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, name), index_col=index_col)


def load_measurement_sigmas(data_dir: str = DEFAULT_DATA_DIR):
    """
    Load measurement standard deviations for the weighted objective / covariance.

    Returns
    -------
    (sigma_flux, sigma_met) : tuple of dict
        sigma_flux : {flux_key: std}            from important_fluxes_Q.csv
        sigma_met  : {C_<met>:  std or None}    from balanced_metabolites_Q.csv
    """
    fq = _read_csv("important_fluxes_Q.csv", data_dir, index_col=0)
    bq = _read_csv("balanced_metabolites_Q.csv", data_dir, index_col=0)
    # Column names differ ("STD" / "SD"); take the first (only) column.
    sigma_flux = {k: float(v) for k, v in fq.iloc[:, 0].dropna().items()}
    sigma_met = {}
    for k, v in bq.iloc[:, 0].items():
        sigma_met[k] = (None if pd.isna(v) else float(v))
    return sigma_flux, sigma_met


def available_conditions(data_dir: str = DEFAULT_DATA_DIR) -> list:
    """Conditions present (as columns) in all four primary data files."""
    files = ["balanced_metabolites.csv", "important_fluxes.csv",
             "imbalanced_metabolites.csv", "important_proteins.csv"]
    cols = [set(_read_csv(f, data_dir, index_col=0).columns) for f in files]
    return sorted(set.intersection(*cols))


def load_condition(condition: str, data_dir: str = DEFAULT_DATA_DIR) -> dict:
    """
    Assemble one experimental condition into a dict with keys
    ``v_real, x_real, u, e, b`` (each a {name: float} mapping).
    """
    bal = _read_csv("balanced_metabolites.csv", data_dir, "VarNames")
    flux = _read_csv("important_fluxes.csv", data_dir, "VarNames")
    imbal = _read_csv("imbalanced_metabolites.csv", data_dir, "VarNames")
    prot = _read_csv("important_proteins.csv", data_dir, "VarNames")
    needs = _read_csv("cellular_needs.csv", data_dir, 0)

    def col(df):
        if condition not in df.columns:
            raise KeyError(f"Condition '{condition}' not in {list(df.columns)}")
        return df[condition].dropna().to_dict()

    return {
        "v_real": col(flux),
        "x_real": col(bal),
        "u": col(imbal),
        "e": col(prot),
        "b": col(needs),
    }


# ===========================================================================
# parmest Experiment: one experimental condition -> one labeled Pyomo model
# ===========================================================================
class GlycolysisExperiment(Experiment):
    """
    A single experimental condition for parmest.

    Builds a ``ConcreteModel`` with:
      * balanced metabolites x_e        -> free Vars (per-scenario)
      * imbalanced metabolites u_e      -> free bounded Vars (default) or fixed Vars;
                                           either way they are NOT in the objective
      * kinetic parameters theta        -> Vars; all fixed, only the *free*
                                           ones are listed in ``unknown_parameters``
      * 9 flux Expressions v(x,u,e;k)
      * steady-state constraint  S @ v = b_e

    Suffixes (parmest contract):
      * experiment_outputs : measured fluxes (+ measured metabolites if fitted)
      * measurement_error  : 1-sigma std for each output (enables SSE_weighted/cov)
      * unknown_parameters : the free theta (ComponentUID values)

    Parameters
    ----------
    condition : str
    keq : dict
    S : np.ndarray (9x9)            reindexed stoichiometric matrix
    sigma_flux, sigma_met : dict    measurement std-devs
    theta_init : dict               initial parameter values (default literature)
    free_params : list[str] or None which parameters to estimate
    fixed_params : list[str] or None which parameters to hold fixed (the rest are
                                    estimated).  Give EITHER free_params OR
                                    fixed_params; default (both None) = all 37 free.
    fit_metabolites : bool          include measured metabolites as outputs
    x_init : dict or None           optional initial values for balanced metabolites
                                    (e.g. from a steady-state warm start); overrides
                                    the measured-value / geometric-mean default
    """

    def __init__(self, condition, keq, S, sigma_flux, sigma_met,
                 theta_init=None, free_params=None, fixed_params=None,
                 fit_metabolites=True, u_fallback=None, x_init=None,
                 free_imbalanced=True, u_bounds=None,
                 data_dir: str = DEFAULT_DATA_DIR):
        super().__init__(model=None)
        self.condition = condition
        self.data = load_condition(condition, data_dir)
        self.keq = keq
        self.S = S
        self.sigma_flux = sigma_flux
        self.sigma_met = sigma_met
        self.theta_init = dict(LITERATURE_THETA)
        if theta_init:
            self.theta_init.update(theta_init)
        self.free_params = resolve_free_params(free_params, fixed_params)
        self.fit_metabolites = fit_metabolites
        self.u_fallback = u_fallback if u_fallback is not None else imbalanced_fallbacks(data_dir)
        self.x_init = x_init or {}
        # When True, imbalanced metabolites are bounded decision variables (pinned
        # only by S v = b and their bounds, NOT by the objective).  When False they
        # are fixed to the measured / fallback values.
        self.free_imbalanced = free_imbalanced
        self.u_bounds = u_bounds if u_bounds is not None else imbalanced_bounds(data_dir)

    # -- relative-error fallback for a metabolite without a tabulated SD -------
    def _met_sigma(self, met_key, value):
        s = self.sigma_met.get(met_key)
        if s is None or s <= 0:
            return max(0.2 * abs(value), 1e-3)   # 20% relative, small floor
        return s

    # ------------------------------------------------------------------ model
    def create_model(self):
        m = pyo.ConcreteModel(name=f"glycolysis_{self.condition}")
        bal_keys = EcoliCarbonKinetics.balanced_keys
        imb_keys = EcoliCarbonKinetics.imbalanced_keys
        flux_keys = EcoliCarbonKinetics.flux_keys

        # --- balanced metabolites: free decision variables -------------------
        # Initialization priority: explicit x_init (warm start) > measured value
        # > geometric mean of bounds.
        for key in bal_keys:
            lb, ub = X_BOUNDS[key]
            x0 = self.x_init.get(key, self.data["x_real"].get(key, np.sqrt(lb * ub)))
            x0 = float(min(max(x0, lb), ub))
            m.add_component(key, pyo.Var(bounds=(lb, ub), initialize=x0))

        # --- imbalanced metabolites ------------------------------------------
        # free_imbalanced=True  -> bounded decision variables (NOT in objective),
        #                          pinned only by S v = b and their bounds.
        # free_imbalanced=False -> fixed to the measured / fallback value.
        for key in imb_keys:
            val = self.data["u"].get(key, self.u_fallback.get(key))
            if val is None:
                raise KeyError(
                    f"Imbalanced metabolite {key} missing for {self.condition} "
                    f"and no fallback defined."
                )
            if self.free_imbalanced:
                lb, ub = self.u_bounds[key]
                u0 = float(min(max(val, lb), ub))
                m.add_component(key, pyo.Var(bounds=(lb, ub), initialize=u0))
            else:
                m.add_component(key, pyo.Var(initialize=float(val)))
                getattr(m, key).fix(float(val))

        # --- kinetic parameters: Vars; fix all, free ones are 'unknowns' -----
        for key in ALL_PARAMS:
            lb, ub = THETA_BOUNDS[key]
            val = float(min(max(self.theta_init.get(key, 1.0), lb), ub))
            m.add_component(key, pyo.Var(bounds=(lb, ub), initialize=val))
            getattr(m, key).fix(val)   # parmest unfixes those it estimates

        # --- dictionaries that point at the Pyomo components -----------------
        C = {k: getattr(m, k) for k in bal_keys + imb_keys}
        constants = {k: getattr(m, k) for k in ALL_PARAMS}
        e = {k: float(self.data["e"][k]) for k in EcoliCarbonKinetics.enzymes_keys}

        # Lightweight kinetics object: only Keq is needed by the flux methods,
        # so we bypass the (heavy, CasADi-building) __init__.
        kin = object.__new__(EcoliCarbonKinetics)
        kin.Keq = self.keq

        # --- flux expressions ------------------------------------------------
        flux_fns = [kin.pts, kin.pgi, kin.pfk, kin.fba, kin.tpi,
                    kin.gap, kin.pgk, kin.gpm, kin.eno]
        for fname, fn in zip(flux_keys, flux_fns):
            m.add_component(fname, pyo.Expression(expr=fn(constants, C, e)))

        # --- steady-state constraint  S v = b_e ------------------------------
        b = self.data.get("b", {})

        def ss_rule(mm, i):
            sv = sum(self.S[i, j] * getattr(mm, flux_keys[j]) for j in range(9))
            return sv == float(b.get(bal_keys[i], 0.0))

        m.ss_constraint = pyo.Constraint(range(9), rule=ss_rule)

        self.model = m
        return m

    # ------------------------------------------------------------------ labels
    def label_model(self):
        m = self.model
        flux_keys = EcoliCarbonKinetics.flux_keys
        bal_keys = EcoliCarbonKinetics.balanced_keys

        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)

        # measured fluxes
        for fk in flux_keys:
            if fk in self.data["v_real"]:
                comp = getattr(m, fk)
                m.experiment_outputs[comp] = float(self.data["v_real"][fk])
                m.measurement_error[comp] = float(self.sigma_flux.get(fk, 1.0))

        # measured balanced metabolites
        if self.fit_metabolites:
            for mk in bal_keys:
                if mk in self.data["x_real"]:
                    comp = getattr(m, mk)
                    val = float(self.data["x_real"][mk])
                    m.experiment_outputs[comp] = val
                    m.measurement_error[comp] = float(self._met_sigma(mk, val))

        # free kinetic parameters (shared across scenarios)
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (getattr(m, p), pyo.ComponentUID(getattr(m, p))) for p in self.free_params
        )
        return m

    def get_labeled_model(self):
        self.create_model()
        return self.label_model()

# ===========================================================================
# Parallel workers for the Monte Carlo perturbation analysis.
# Defined at module level so they are picklable by ProcessPoolExecutor.  Each
# worker process builds the (heavy) CasADi kinetics ONCE and caches per-condition
# inputs; subsequent samples reuse it (and warm-start via condition_key).
# ===========================================================================
_PERT_STATE = {}


def _pert_worker_init(data_dir, conditions, u_bounds, nominal):
    """Process-pool initializer: build the kinetics and per-condition inputs once."""
    bounds_imb = {k: u_bounds[k] for k in EcoliCarbonKinetics.imbalanced_keys}
    kin = EcoliCarbonKinetics(bounds_imbalanced_mets=bounds_imb,
                              bounds_balanced_mets=X_BOUNDS)
    per_cond = {}
    for cond in conditions:
        data = load_condition(cond, data_dir)
        per_cond[cond] = (
            {k: float(data["e"][k]) for k in EcoliCarbonKinetics.enzymes_keys},
            {k: float(data["b"].get(k, 0.0)) for k in EcoliCarbonKinetics.balanced_keys},
        )
    _PERT_STATE.update(
        kin=kin, per_cond=per_cond, conditions=list(conditions), nominal=dict(nominal),
        out_names=list(EcoliCarbonKinetics.balanced_keys) + list(EcoliCarbonKinetics.flux_keys),
    )


def _pert_worker_eval(task):
    """One sample: task=(param, value) -> (param, value, mean-output vector | None)."""
    param, value = task
    st = _PERT_STATE
    th = dict(st["nominal"])
    th[param] = float(value)
    try:
        rows = []
        for cond in st["conditions"]:
            enzymes, cell_needs = st["per_cond"][cond]
            df, _ = st["kin"].solve_steady_state(enzymes, th, cell_needs, condition_key=cond)
            rows.append(df.iloc[0][st["out_names"]].to_numpy(dtype=float))
        return (param, float(value), np.mean(rows, axis=0))
    except Exception:   # noqa: BLE001
        return (param, float(value), None)


# ===========================================================================
# Estimator: parameter estimation + covariance + sensitivities
# ===========================================================================
class GlycolysisParameterEstimator:
    """
    parmest-based parameter estimation for the E. coli glycolysis model, kept
    deliberately small.  It does three things:

      1. estimate the kinetic parameters (weighted least squares),
      2. provide the parameter **covariance matrix** (and correlation / CIs),
      3. build the **sensitivity (G) matrices** d(outputs)/d(theta), which also
         support standalone sensitivity analysis and the Fisher Information Matrix.

    Typical usage
    -------------
    >>> est   = GlycolysisParameterEstimator()        # all conditions, all 37 params
    >>> theta = est.estimate()                         # weighted least squares
    >>> cov   = est.covariance()                       # parameter covariance Sigma
    >>> corr  = est.correlation_matrix()               # r_ij from Sigma
    >>> G     = est.sensitivity_matrix()               # {condition: dOutputs/dtheta}
    >>> fim   = est.fisher_information_matrix()         # sum_e G^T Q^-1 G

    Parameters
    ----------
    conditions : list[str] or None
    free_params / fixed_params : list[str] or None
        Give EITHER one; default = all 37 free.  ``fixed_params`` is the complement
        (e.g. ``fixed_params=REPORT_FIXED_PARAMS``).
    fit_metabolites : bool          include measured metabolites as outputs
    weighted : bool                 SSE_weighted (sigma-scaled) vs plain SSE
    theta_init : dict or None       parameter initial values
    free_imbalanced : bool          imbalanced u_e as bounded decision vars (default)
    data_dir : str
    solver_options : dict or None   passed to ipopt
    """

    def __init__(self, conditions=None, free_params=None, fixed_params=None,
                 fit_metabolites=True, weighted=True, theta_init=None,
                 free_imbalanced=True, data_dir: str = DEFAULT_DATA_DIR,
                 solver_options=None):
        self.data_dir = data_dir
        self.conditions = list(conditions) if conditions else available_conditions(data_dir)
        self.free_params = resolve_free_params(free_params, fixed_params)
        self.fit_metabolites = fit_metabolites
        self.weighted = weighted
        self.obj_function = "SSE_weighted" if weighted else "SSE"
        self.theta_init = theta_init
        self.free_imbalanced = free_imbalanced
        self.solver_options = solver_options or {"tol": 1e-6, "max_iter": 3000}

        self.keq = load_keq(data_dir)
        self.S = build_stoichiometric_matrix()
        self.sigma_flux, self.sigma_met = load_measurement_sigmas(data_dir)
        self.u_fallback = imbalanced_fallbacks(data_dir)
        self.u_bounds = imbalanced_bounds(data_dir)

        self._validate_inputs()
        self.exp_list = self._build_experiments()
        self.pest = self._make_estimator()

        self.obj_value = None
        self.theta = None        # pd.Series, set by estimate()
        self._cov = None         # pd.DataFrame, set by covariance()

    # -- setup ----------------------------------------------------------------
    def _validate_inputs(self):
        if not self.free_params:
            raise ValueError("free_params is empty -- nothing to estimate.")
        avail = set(available_conditions(self.data_dir))
        missing = [c for c in self.conditions if c not in avail]
        if missing:
            raise ValueError(f"Conditions not present in all data files: {missing}")

    def _make_estimator(self):
        return parmest.Estimator(
            self.exp_list, obj_function=self.obj_function, tee=False,
            solver_options=self.solver_options,
        )

    def _build_experiments(self):
        return [
            GlycolysisExperiment(
                cond, self.keq, self.S, self.sigma_flux, self.sigma_met,
                theta_init=self.theta_init, free_params=self.free_params,
                fit_metabolites=self.fit_metabolites, u_fallback=self.u_fallback,
                free_imbalanced=self.free_imbalanced, u_bounds=self.u_bounds,
                data_dir=self.data_dir,
            )
            for cond in self.conditions
        ]

    @property
    def n_data_points(self):
        """Total number of measured outputs across all scenarios."""
        return sum(len(e.get_labeled_model().experiment_outputs) for e in self.exp_list)

    @property
    def fixed_params(self):
        """Parameters held fixed (complement of ``free_params``)."""
        free = set(self.free_params)
        return [p for p in ALL_PARAMS if p not in free]

    def set_fixed_parameters(self, fixed_params=None, free_params=None):
        """
        Re-select which parameters are fixed/free and rebuild (clears results).
        Give EITHER ``fixed_params`` OR ``free_params``.  Returns the new free list.
        """
        self.free_params = resolve_free_params(free_params, fixed_params)
        self._validate_inputs()
        self.exp_list = self._build_experiments()
        self.pest = self._make_estimator()
        self.theta = self._cov = self.obj_value = None
        return self.free_params

    def preview_model(self, index: int = 0):
        """Return the labeled Pyomo model of one condition (call ``.pprint()`` on it)."""
        return self.exp_list[index].get_labeled_model()

    # -- estimation -----------------------------------------------------------
    def estimate(self):
        """Weighted least-squares estimation. Returns theta (pd.Series)."""
        check_solver("ipopt")
        self.obj_value, self.theta = self.pest.theta_est()
        return self.theta

    # -- covariance & correlation --------------------------------------------
    def covariance(self, method: str = "finite_difference"):
        """
        Parameter covariance matrix Sigma (pd.DataFrame), via parmest's ``cov_est``.
        ``method`` is 'finite_difference', 'reduced_hessian', or
        'automatic_differentiation_kaug'.  Falls back to the deprecated
        ``theta_est(calc_cov=...)`` on older parmest.
        """
        valid = {"finite_difference", "reduced_hessian", "automatic_differentiation_kaug"}
        if method not in valid:
            raise ValueError(f"Unknown covariance method '{method}'. Choose from {valid}.")
        if self.theta is None:
            self.estimate()
        if hasattr(self.pest, "cov_est"):
            self._cov = self.pest.cov_est(method=method, solver="ipopt")
        else:  # legacy parmest
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.obj_value, self.theta, self._cov = self.pest.theta_est(
                    calc_cov=True, cov_n=self.n_data_points)
        return self._cov

    def correlation_matrix(self):
        """Correlation matrix r_ij = Sigma_ij / (sigma_i sigma_j) from the covariance."""
        cov = self._cov if self._cov is not None else self.covariance()
        C = cov.to_numpy()
        d = np.sqrt(np.clip(np.diag(C), 0.0, None))
        denom = np.outer(d, d)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.where(denom > 0, C / denom, 0.0)
        return pd.DataFrame(np.clip(R, -1.0, 1.0), index=cov.index, columns=cov.columns)

    def confidence_intervals(self, alpha: float = 0.05):
        """Marginal CIs theta_i +/- t*sigma_i, with standard errors and CV%."""
        if self.theta is None:
            self.estimate()
        cov = self._cov if self._cov is not None else self.covariance()
        names = list(cov.columns)
        sigmas = np.sqrt(np.clip(np.diag(cov.to_numpy()), 0.0, None))
        theta = np.array([self.theta[n] for n in names])
        dof = max(self.n_data_points - len(names), 1)
        try:
            from scipy.stats import t as _t
            tcrit = float(_t.ppf(1 - alpha / 2, dof))
        except Exception:  # noqa: BLE001
            tcrit = 1.96
        return pd.DataFrame({
            "theta": theta, "std_err": sigmas,
            "ci_low": theta - tcrit * sigmas, "ci_high": theta + tcrit * sigmas,
            "cv_percent": np.where(theta != 0, 100 * sigmas / np.abs(theta), np.inf),
        }, index=names)

    # -- sensitivities (G matrix) --------------------------------------------
    def _full_theta(self, theta=None):
        """Complete {param: value} dict over ALL_PARAMS (arg > estimate > literature)."""
        full = dict(LITERATURE_THETA)
        if self.theta_init:
            full.update(self.theta_init)
        if theta is not None:
            full.update(theta)
        elif self.theta is not None:
            full.update({p: float(self.theta[p]) for p in self.free_params})
        return full

    def _kinetics(self):
        """Instantiate EcoliCarbonKinetics (builds its CasADi steady-state solver)."""
        bounds_imb = {k: self.u_bounds[k] for k in EcoliCarbonKinetics.imbalanced_keys}
        return EcoliCarbonKinetics(bounds_imbalanced_mets=bounds_imb,
                                   bounds_balanced_mets=X_BOUNDS)

    def sensitivity_matrix(self, theta=None, conditions=None, kind="both"):
        """
        Sensitivity (G) matrices  d(outputs)/d(theta)  at the given parameters, one
        per condition, computed analytically by the kinetics model
        (``EcoliCarbonKinetics.gen_sensitivity_matrix`` -- implicit function theorem,
        one steady-state solve per condition, NO per-parameter re-runs).

        Rows are the 9 balanced-metabolite concentrations and the 9 fluxes;
        columns are the 37 kinetic parameters.  This feeds both standalone
        sensitivity analysis and the Fisher Information Matrix.

        Parameters
        ----------
        theta : dict or None     evaluate at these parameters (default: the estimate
                                 if available, else the literature values)
        conditions : list or None
        kind : {'both','flux','concentration'}

        Returns
        -------
        dict[str, pd.DataFrame]   {condition: G}, rows indexed by output, columns=params
        """
        theta_full = self._full_theta(theta)
        params = list(EcoliCarbonKinetics.params_keys)
        bal = EcoliCarbonKinetics.balanced_keys
        flux = EcoliCarbonKinetics.flux_keys
        rows = [("concentration", k) for k in bal] + [("flux", k) for k in flux]
        idx = pd.MultiIndex.from_tuples(rows, names=["kind", "name"])

        kin = self._kinetics()
        out = {}
        for cond in (conditions or self.conditions):
            data = load_condition(cond, self.data_dir)
            enzymes = {k: float(data["e"][k]) for k in EcoliCarbonKinetics.enzymes_keys}
            cell_needs = {k: float(data["b"].get(k, 0.0)) for k in bal}
            G = kin.gen_sensitivity_matrix(enzymes, theta_full, cell_needs,
                                           condition_key=cond)              # (18 x 37)
            df = pd.DataFrame(np.asarray(G, dtype=float), index=idx, columns=params)
            if kind == "flux":
                df = df.loc["flux"]
            elif kind in ("concentration", "conc"):
                df = df.loc["concentration"]
            out[cond] = df
        return out

    def _output_sigma(self, kind, name, value):
        """1-sigma measurement error for one output (flux or metabolite)."""
        if kind == "flux":
            return float(self.sigma_flux.get(name, 1.0))
        s = self.sigma_met.get(name)
        return float(s) if (s is not None and s > 0) else max(0.2 * abs(value), 1e-3)

    def fisher_information_matrix(self, theta=None, conditions=None, free_only=True):
        """
        Fisher Information Matrix  F = sum_e G_e^T Q_e^{-1} G_e  over the MEASURED
        outputs (report Eq. 10), assembled from the sensitivity matrices.  ``Q_e``
        is the diagonal measurement-error covariance.  Its inverse approximates the
        parameter covariance, so this is the a-priori counterpart of ``covariance``.

        free_only : bool   restrict to the free parameters (default) or all 37.
        """
        params = list(self.free_params) if free_only else list(EcoliCarbonKinetics.params_keys)
        Gdict = self.sensitivity_matrix(theta=theta, conditions=conditions, kind="both")
        F = np.zeros((len(params), len(params)))
        for cond, G in Gdict.items():
            data = load_condition(cond, self.data_dir)
            sel, sig = [], []
            for fk in EcoliCarbonKinetics.flux_keys:
                if fk in data["v_real"]:
                    sel.append(("flux", fk))
                    sig.append(self._output_sigma("flux", fk, data["v_real"][fk]))
            if self.fit_metabolites:
                for mk in EcoliCarbonKinetics.balanced_keys:
                    if mk in data["x_real"]:
                        sel.append(("concentration", mk))
                        sig.append(self._output_sigma("conc", mk, data["x_real"][mk]))
            Gsub = G.loc[sel, params].to_numpy(dtype=float)
            Qinv = np.diag(1.0 / np.asarray(sig) ** 2)
            F += Gsub.T @ Qinv @ Gsub
        return pd.DataFrame(F, index=params, columns=params)

    # -- perturbation (Monte Carlo, one-at-a-time) ---------------------------
    def _steady_state_outputs(self, theta_full, conditions, kin, aggregate="mean"):
        """
        Solve the steady state at fixed parameters for each condition and return
        the model outputs (9 balanced concentrations + 9 fluxes).  ``aggregate``:
        'mean' averages across conditions (report's choice); 'none' returns the
        per-condition table.
        """
        out_names = list(EcoliCarbonKinetics.balanced_keys) + list(EcoliCarbonKinetics.flux_keys)
        rows = []
        for cond in conditions:
            data = load_condition(cond, self.data_dir)
            enzymes = {k: float(data["e"][k]) for k in EcoliCarbonKinetics.enzymes_keys}
            cell_needs = {k: float(data["b"].get(k, 0.0)) for k in EcoliCarbonKinetics.balanced_keys}
            df, _ = kin.solve_steady_state(enzymes, theta_full, cell_needs, condition_key=cond)
            rows.append(df.iloc[0][out_names])
        table = pd.DataFrame(rows, index=conditions)[out_names]
        return table.mean(axis=0) if aggregate == "mean" else table

    # -- predict / rmse -------------------------------------------------------
    def predict(self, theta=None, conditions=None) -> "Prediction":
        """
        Predict steady-state outputs (metabolites + fluxes) at given parameters
        and return a Prediction dataclass with predicted/real DataFrames and RMSE.

        Parameters
        ----------
        theta : dict or None
            Parameters to evaluate at.  Resolved via _full_theta (arg > estimate >
            literature).  Works without ipopt (CasADi only).
        conditions : list[str] or None
            Conditions to predict.  Default: self.conditions.

        Returns
        -------
        Prediction
            .predicted  -- (n_cond x 18) DataFrame; cells masked NaN where real is NaN.
            .real       -- (n_cond x 18) DataFrame; NaN where not measured.
            .rmse       -- {'met': float, 'flux': float, 'data_norm': float}.
                           See Prediction docstring for normalization details.

        Notes
        -----
        RMSE normalization for 'data_norm': each output column's residuals are
        divided by that column's nanmean of the real values, then all normalized
        squared residuals (metabolites + fluxes) are averaged and sqrt-taken.
        Columns with zero or NaN mean are skipped.  This makes 'data_norm' a
        dimensionless aggregate over outputs of very different magnitudes.
        """
        conditions = list(conditions) if conditions is not None else list(self.conditions)
        theta_full = self._full_theta(theta)
        kin = self._kinetics()
        out_cols = (list(EcoliCarbonKinetics.balanced_keys)
                    + list(EcoliCarbonKinetics.flux_keys))
        met_cols = [c for c in out_cols if c.startswith("C_")]
        flux_cols = [c for c in out_cols if c.startswith("v_")]
        pred_df = self._steady_state_outputs(theta_full, conditions, kin, aggregate="none")
        real_rows = {}
        for cond in conditions:
            data = load_condition(cond, self.data_dir)
            row = {}
            for col in flux_cols:
                row[col] = data["v_real"].get(col, float("nan"))
            for col in met_cols:
                if self.fit_metabolites:
                    row[col] = data["x_real"].get(col, float("nan"))
                else:
                    row[col] = float("nan")
            real_rows[cond] = row
        real_df = pd.DataFrame(real_rows, index=out_cols).T.astype(float)
        real_df = real_df[out_cols]
        pred_aligned = pred_df.copy().reindex(columns=out_cols)
        pred_aligned[real_df.isna()] = float("nan")
        residual = pred_aligned - real_df
        def _nanrmse(arr):
            vals = arr.to_numpy(dtype=float).ravel()
            finite = vals[~np.isnan(vals)]
            if len(finite) == 0:
                return float("nan")
            return float(np.sqrt(np.mean(finite ** 2)))
        rmse_met = _nanrmse(residual[met_cols])
        rmse_flux = _nanrmse(residual[flux_cols])
        norm_sq_vals = []
        for col in out_cols:
            col_real = real_df[col].to_numpy(dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                col_mean = float(np.nanmean(col_real))
            if np.isnan(col_mean) or col_mean == 0.0:
                continue
            col_res = residual[col].to_numpy(dtype=float)
            norm_res = col_res / col_mean
            finite_norm = norm_res[~np.isnan(norm_res)]
            norm_sq_vals.extend(finite_norm ** 2)
        if norm_sq_vals:
            rmse_norm = float(np.sqrt(np.mean(norm_sq_vals)))
        else:
            rmse_norm = float("nan")
        # weighted (reduced-chi2-like) RMSE: residual / measurement sigma, where the
        # sigma is the same per-output std the weighted-SSE objective uses.  ~1 means
        # the fit sits at the measurement noise floor.
        wgt_sq_vals = []
        for col in out_cols:
            col_real = real_df[col].to_numpy(dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                col_mean = float(np.nanmean(col_real))
            kind = "flux" if col.startswith("v_") else "conc"
            sigma = self._output_sigma(kind, col, col_mean)
            if not np.isfinite(sigma) or sigma <= 0.0:
                continue
            col_res = residual[col].to_numpy(dtype=float)
            wgt_res = col_res / sigma
            finite_wgt = wgt_res[~np.isnan(wgt_res)]
            wgt_sq_vals.extend(finite_wgt ** 2)
        if wgt_sq_vals:
            rmse_wgt = float(np.sqrt(np.mean(wgt_sq_vals)))
        else:
            rmse_wgt = float("nan")
        # per-output raw RMSE: one value per measured output column (own units).
        per_output = {}
        for col in out_cols:
            val = _nanrmse(residual[[col]])
            if not np.isnan(val):
                per_output[col] = val
        per_output = pd.Series(per_output, dtype=float)
        return Prediction(
            predicted=pred_aligned,
            real=real_df,
            rmse={"met": rmse_met, "flux": rmse_flux,
                  "data_norm": rmse_norm, "weighted": rmse_wgt},
            per_output=per_output,
        )

    def rmse(self, theta=None, conditions=None) -> dict:
        """
        Thin wrapper around predict() that returns only the rmse dict.

        Returns
        -------
        dict with keys 'met', 'flux', 'data_norm'.  See Prediction docstring.
        """
        return self.predict(theta=theta, conditions=conditions).rmse

    # -- multistart -----------------------------------------------------------
    def multistart(self, method: str = "multistart", n_starts: int = 20,
                   seed=None, engine: str = "sampling", **kwargs) -> dict:
        """
        Multi-start or bootstrap estimation to explore the objective landscape.

        Parameters
        ----------
        method : 'multistart' or 'bootstrap'
            'multistart' -- pyomo.contrib.multistart solves parmest's extensive-form
                (EF) model from multiple random starts (see note below); falls back
                to a looped theta_est if that solver path raises.
            'bootstrap'  -- parmest theta_est_bootstrap (scenario resampling).
        n_starts : int
            Number of restarts (multistart) or bootstrap samples.
        seed : int or None
            Random seed for reproducibility.
        engine : 'sampling' | 'solver' | 'auto'   (multistart only)
            'sampling' -- diverse seeded restarts of theta_est() from perturbed
                          initial points (DEFAULT; robust -- does not depend on the
                          EF re-solving).  Effectiveness varies with n_starts and
                          spread; wide starts can fail to converge, so watch
                          'n_converged' in the result.
            'solver'   -- pyo.SolverFactory('multistart') on parmest's extensive-form
                          (EF) model (the Pyomo meta-solver; see note below).
            'auto'     -- try 'solver', fall back to 'sampling' on error.
        Both engines are stochastic on this non-convex problem and neither uniformly
        dominates -- compare 'best_obj' and read the diagnostics ('all_runs',
        'n_converged', 'frac_within') rather than assuming one is always better.
        **kwargs : additional keyword arguments
            For 'multistart': 'strategy' (random sampling strategy for the Pyomo
                meta-solver, default 'rand'); 'spread' (lognormal width for the
                sampling engine, default 0.75); 'distribution' ('lognormal' or
                'loguniform' for the sampling engine); 'factor' (loguniform
                half-width, default 3.0).
            For 'bootstrap':  'return_samples' (bool, default False); 'samplesize'
                (scenarios per bootstrap sample, default = number of conditions).

        Returns
        -------
        dict with keys:
            'best_theta' (pd.Series)  -- best estimated parameters found.
            'best_obj'   (float)      -- best objective value (NaN for bootstrap).
            'all_runs'   (pd.DataFrame) -- one row per restart / bootstrap sample,
                         sorted ascending by 'obj_value' (multistart), with a
                         'converged' bool column.
            'method'     (str)        -- 'multistart' or 'bootstrap'.
            'engine'     (str, multistart only) -- 'solver' or 'sampling' (which ran).
            'n_converged'(int, multistart only) -- restarts that solved successfully.
            'obj_median' (float, multistart only) -- median converged objective.
            'frac_within'(float, multistart only) -- fraction of converged starts
                         whose objective is within 1% of 'best_obj' (a basin-of-
                         attraction signal: low -> many basins -> keep adding starts).
            'stats'      (pd.DataFrame, bootstrap only) -- mean/std/quantiles
                         of bootstrap theta samples (rows=params, cols=[mean,std,q05,q50,q95]).

        Notes on the Pyomo multistart meta-solver
        ------------------------------------------
        pyo.SolverFactory('multistart') is a meta-solver: it re-solves the SAME model
        from many randomized initial points with an inner local solver (ipopt) and
        keeps the best feasible objective.  Key options: 'iterations' (number of
        restarts), 'strategy' ('rand', 'rand_distributed', 'midpoint_guess_and_bound',
        'rand_guess_and_bound'), and 'solver' (the inner NLP solver).  Here it is
        applied to parmest's extensive-form model (self.pest.ef_instance, built by one
        theta_est() call); the fitted theta is then read off the EF first-stage vars.
        The 'sampling' engine instead restarts the whole theta_est() from perturbed
        initial parameter values -- more robust when the EF path does not re-solve.
        """
        check_solver("ipopt")
        if method == "bootstrap":
            return_samples = kwargs.get("return_samples", False)
            samplesize = kwargs.get("samplesize", None)
            # parmest builds each bootstrap sample by resampling whole scenarios
            # (conditions) and rejects samples whose number of DISTINCT scenarios
            # is <= the number of free parameters.  With samplesize defaulting to
            # the scenario count, bootstrap is only feasible when there are more
            # conditions than free parameters; otherwise parmest raises an opaque
            # "timeout constructing a sample" error.  Guard it with a clear message.
            eff_samplesize = samplesize if samplesize is not None else len(self.conditions)
            n_theta = len(self.free_params)
            if eff_samplesize <= n_theta:
                raise ValueError(
                    f"bootstrap needs more distinct scenarios per sample than free "
                    f"parameters: samplesize={eff_samplesize} but {n_theta} free "
                    f"parameters.  Use more conditions (currently {len(self.conditions)}) "
                    f"or fewer free parameters (e.g. fixed_params=REPORT_FIXED_PARAMS), "
                    f"or pass a larger samplesize."
                )
            boot_kwargs = dict(
                bootstrap_samples=n_starts,
                seed=seed,
                return_samples=return_samples,
            )
            if samplesize is not None:
                boot_kwargs["samplesize"] = samplesize
            bt = self.pest.theta_est_bootstrap(**boot_kwargs)
            best_theta = bt.drop(columns=["samples"], errors="ignore").mean()
            stats_cols = bt.drop(columns=["samples"], errors="ignore")
            stats = pd.DataFrame({
                "mean":  stats_cols.mean(),
                "std":   stats_cols.std(),
                "q05":   stats_cols.quantile(0.05),
                "q50":   stats_cols.quantile(0.50),
                "q95":   stats_cols.quantile(0.95),
            })
            return {
                "best_theta": best_theta,
                "best_obj":   float("nan"),
                "all_runs":   bt,
                "method":     "bootstrap",
                "stats":      stats,
            }
        if method != "multistart":
            raise ValueError(f"method must be 'multistart' or 'bootstrap', got '{method}'.")
        if engine not in ("auto", "solver", "sampling"):
            raise ValueError(f"engine must be 'auto', 'solver' or 'sampling', got '{engine}'.")
        strategy = kwargs.get("strategy", "rand")
        if engine == "sampling":
            return self._multistart_looped(n_starts, seed, **kwargs)
        if engine == "solver":
            return self._multistart_ef(n_starts, seed, strategy)
        # engine == "auto": try the Pyomo meta-solver, fall back to sampling.
        try:
            return self._multistart_ef(n_starts, seed, strategy)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"multistart: solver engine (pyo.SolverFactory('multistart')) failed "
                f"({exc}); falling back to the sampling engine.",
                stacklevel=2,
            )
        return self._multistart_looped(n_starts, seed, **kwargs)

    @staticmethod
    def _finalize_multistart(best_series, best_obj, all_runs, engine, within=0.01):
        """
        Attach global-search diagnostics to a multistart result dict: sort all_runs
        ascending by objective, flag converged restarts, and report how concentrated
        the converged objectives are near the best (a basin-of-attraction signal).
        """
        objs = np.array([], dtype=float)
        if "obj_value" in all_runs.columns:
            all_runs = all_runs.copy()
            all_runs["converged"] = all_runs["obj_value"].notna()
            all_runs = all_runs.sort_values("obj_value", na_position="last").reset_index(drop=True)
            objs = all_runs.loc[all_runs["converged"], "obj_value"].to_numpy(dtype=float)
        n_conv = int(len(objs))
        obj_median = float(np.median(objs)) if n_conv else float("nan")
        if n_conv and np.isfinite(best_obj) and best_obj > 0:
            frac_within = float(np.mean(objs <= best_obj * (1.0 + within)))
        elif n_conv:
            frac_within = float(np.mean(np.isclose(objs, best_obj)))
        else:
            frac_within = float("nan")
        return {
            "best_theta":  best_series,
            "best_obj":    best_obj,
            "all_runs":    all_runs,
            "method":      "multistart",
            "engine":      engine,
            "n_converged": n_conv,
            "obj_median":  obj_median,
            "frac_within": frac_within,
        }

    def _multistart_ef(self, n_starts, seed, strategy):
        """
        Solver engine: solve parmest's extensive-form (EF) model with
        pyo.SolverFactory('multistart') (n_starts random restarts via ipopt), then
        read the fitted theta off the EF's first-stage variables.
        """
        from pyomo.contrib.parmest.parmest import ef_nonants
        if seed is not None:
            np.random.seed(seed)
        free = list(self.free_params)

        def _read_ef_theta(ef):
            vals = {}
            for _nd_name, var, sol_val in ef_nonants(ef):
                var_name = var.name[var.name.find(".") + 1:]
                vals[var_name] = float(sol_val)
            return vals

        # One theta_est() builds and stores self.pest.ef_instance and leaves the EF
        # at the single-start solution.  Capture that theta as the incumbent.
        base_obj, _ = self.pest.theta_est()
        ef = self.pest.ef_instance
        base_theta = _read_ef_theta(ef)
        # pyomo.contrib.multistart restarts ipopt from random points.  Note: its
        # num_iter==1 bookkeeping can leave the EF at a WORSE point than the
        # incumbent, so we compare objectives explicitly and keep the better theta
        # (never return a best_theta inconsistent with best_obj).
        ms = pyo.SolverFactory("multistart")
        ms_opts = dict(iterations=n_starts, strategy=strategy, solver="ipopt")
        if self.solver_options:
            ms_opts["solver_args"] = {"options": dict(self.solver_options)}
        ms.solve(ef, **ms_opts)
        ms_theta = _read_ef_theta(ef)
        ms_obj = float(pyo.value(ef.EF_Obj))
        if ms_obj < float(base_obj):
            best_obj, best_theta = ms_obj, ms_theta
        else:
            best_obj, best_theta = float(base_obj), base_theta
        best_series = pd.Series(
            {p: float(best_theta[p]) for p in free}, name="best_theta"
        )
        all_runs = pd.DataFrame([
            {"start": "single", "obj_value": float(base_obj),
             **{p: float(base_theta[p]) for p in free}},
            {"start": "multistart", "obj_value": ms_obj,
             **{p: float(ms_theta[p]) for p in free}},
        ])
        return self._finalize_multistart(best_series, best_obj, all_runs, "solver")

    def _multistart_looped(self, n_starts, seed, **kwargs):
        """
        Sampling engine: call theta_est() n_starts times from perturbed initial
        values of the free parameters, keeping the best.  Wide, seeded sampling so
        the restarts genuinely explore the parameter space toward a global optimum.

        kwargs: 'spread' (lognormal width, default 0.75; alias 'rel_sigma');
        'distribution' ('lognormal' default, or 'loguniform'); 'factor' (loguniform
        half-width, default 3.0).
        """
        rng = np.random.default_rng(seed)
        spread = float(kwargs.get("spread", kwargs.get("rel_sigma", 0.75)))
        distribution = kwargs.get("distribution", "lognormal")
        factor = float(kwargs.get("factor", 3.0))
        if distribution not in ("lognormal", "loguniform"):
            raise ValueError("distribution must be 'lognormal' or 'loguniform'.")
        nominal = self._full_theta(None)
        free = list(self.free_params)
        best_obj = float("inf")
        best_theta = None
        run_records = []
        for i in range(n_starts):
            perturbed = {}
            for p in free:
                base = float(nominal.get(p, LITERATURE_THETA.get(p, 1.0)))
                lb, ub = THETA_BOUNDS[p]
                if distribution == "lognormal":
                    val = base * float(np.exp(rng.normal(0.0, spread)))
                else:
                    val = base * float(np.exp(rng.uniform(np.log(1.0 / factor),
                                                          np.log(factor))))
                perturbed[p] = float(min(max(val, lb), ub))
            orig_theta_init = self.theta_init
            self.theta_init = perturbed
            self.exp_list = self._build_experiments()
            self.pest = self._make_estimator()
            try:
                obj_i, theta_i = self.pest.theta_est()
                obj_i = float(obj_i)
                run_records.append({"start": i, "obj_value": obj_i,
                                    **{p: float(theta_i[p]) for p in free}})
                if obj_i < best_obj:
                    best_obj = obj_i
                    best_theta = theta_i
            except Exception as exc:  # noqa: BLE001
                run_records.append({"start": i, "obj_value": float("nan"),
                                    **{p: float("nan") for p in free}})
                warnings.warn(f"multistart restart {i} failed: {exc}", stacklevel=2)
            finally:
                self.theta_init = orig_theta_init
                self.exp_list = self._build_experiments()
                self.pest = self._make_estimator()
        all_runs = pd.DataFrame(run_records)
        if best_theta is not None:
            best_series = pd.Series(
                {p: float(best_theta[p]) for p in free}, name="best_theta"
            )
        else:
            best_series = pd.Series({p: float("nan") for p in free}, name="best_theta")
        best_obj_final = best_obj if best_theta is not None else float("nan")
        return self._finalize_multistart(best_series, best_obj_final, all_runs, "sampling")

    @staticmethod
    def _correlation(x, Y, rank=True):
        """(Rank) correlation of vector x (n,) with each column of Y (n, m)."""
        x = np.asarray(x, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if rank:                                   # Spearman = Pearson on ranks
            x = pd.Series(x).rank().to_numpy()
            Y = np.column_stack([pd.Series(Y[:, j]).rank().to_numpy()
                                 for j in range(Y.shape[1])])
        xc = x - x.mean()
        out = np.zeros(Y.shape[1])
        for j in range(Y.shape[1]):
            yc = Y[:, j] - Y[:, j].mean()
            denom = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
            out[j] = float(xc @ yc / denom) if denom > 0 else 0.0
        return out

    def perturbation_analysis(self, params=None, conditions=None, n_samples=200,
                              distribution="lognormal", rel_sigma=0.5, factor=2.0,
                              rank=True, theta=None, seed=None, n_jobs=1, verbose=False):
        """
        Monte Carlo one-at-a-time (OAT) perturbation sensitivity (the report's
        Monte Carlo / PRCC analysis), optionally parallelized.

        For each parameter, draw ``n_samples`` perturbed values **with all other
        parameters held at their nominal values**, re-solve the steady state for
        every condition, aggregate the outputs across conditions (mean), and
        compute the (rank) correlation between the perturbed parameter and each
        output.  Because only one parameter varies at a time, this rank
        correlation is the partial rank correlation coefficient (PRCC).

        Parameters
        ----------
        params : list[str] or None     parameters to perturb (default: free_params)
        conditions : list[str] or None default: all conditions
        n_samples : int                draws per parameter
        distribution : {'lognormal','loguniform'}
            'lognormal'  -> value = nominal * exp(N(0, rel_sigma))
            'loguniform' -> value in [nominal/factor, nominal*factor] (log-uniform)
        rel_sigma : float              log-normal spread (multiplicative)
        factor : float                 log-uniform half-width factor
        rank : bool                    Spearman/PRCC (True) vs Pearson (False)
        theta : dict or None           nominal parameters (default: estimate/literature)
        seed : int or None             reproducible sampling (drawn before dispatch)
        n_jobs : int                   1 = serial; >1 = that many processes;
                                       -1/0/None = all CPU cores.  Each sample is an
                                       independent steady-state solve, so this scales
                                       almost linearly.  Workers build the kinetics
                                       once and warm-start across samples.

        Returns
        -------
        pd.DataFrame                   rows = perturbed parameter, columns = output
                                       (C_* and v_*), values = (rank) correlation.

        Notes
        -----
        Cost is ``len(params) * n_samples * len(conditions)`` steady-state solves;
        parallelize with ``n_jobs`` and/or scope ``params``/``conditions``.
        """
        rng = np.random.default_rng(seed)
        nominal = self._full_theta(theta)
        params = list(params if params is not None else self.free_params)
        conditions = list(conditions if conditions is not None else self.conditions)
        out_names = list(EcoliCarbonKinetics.balanced_keys) + list(EcoliCarbonKinetics.flux_keys)

        # 1) draw all samples up front (reproducible regardless of n_jobs)
        samples = {}
        for p in params:
            base = float(nominal[p])
            if distribution == "lognormal":
                samples[p] = base * np.exp(rng.normal(0.0, rel_sigma, n_samples))
            elif distribution == "loguniform":
                samples[p] = base * np.exp(rng.uniform(np.log(1.0 / factor),
                                                       np.log(factor), n_samples))
            else:
                raise ValueError("distribution must be 'lognormal' or 'loguniform'.")

        evals = {p: [] for p in params}      # param -> list of (x_value, output_vector)

        # 2) evaluate the samples -- serially or across a process pool
        if n_jobs == 1:
            kin = self._kinetics()
            for p in params:
                for s in samples[p]:
                    th = dict(nominal)
                    th[p] = float(s)
                    try:
                        y = self._steady_state_outputs(th, conditions, kin).to_numpy(dtype=float)
                        evals[p].append((float(s), y))
                    except Exception as exc:   # noqa: BLE001
                        if verbose:
                            print(f"  [{p}] sample skipped: {exc}")
        else:
            from concurrent.futures import ProcessPoolExecutor
            workers = (os.cpu_count() or 1) if n_jobs in (-1, 0, None) else int(n_jobs)
            tasks = [(p, float(s)) for p in params for s in samples[p]]
            chunk = max(1, len(tasks) // (workers * 8))
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_pert_worker_init,
                initargs=(self.data_dir, conditions, self.u_bounds, nominal),
            ) as ex:
                for param, value, y in ex.map(_pert_worker_eval, tasks, chunksize=chunk):
                    if y is not None:
                        evals[param].append((value, y))

        # 3) assemble (rank) correlations
        result = pd.DataFrame(index=params, columns=out_names, dtype=float)
        for p in params:
            pairs = evals[p]
            if len(pairs) < 3:
                warnings.warn(f"Too few successful solves for {p}; correlations set to NaN.")
                result.loc[p] = np.nan
                continue
            xs = np.array([a for a, _ in pairs], dtype=float)
            ys = np.array([b for _, b in pairs], dtype=float)
            result.loc[p] = self._correlation(xs, ys, rank=rank)
            if verbose:
                print(f"  {p}: {len(pairs)}/{n_samples} solves OK")
        return result

    # -- notebook pipeline: correlation -> perturbation -> likelihood --------
    def _reconfigure(self, free_params=None, fixed_values=None):
        """
        Set the free parameters and the values of the fixed parameters, then
        rebuild the estimator.  Used by the pipeline methods so each analysis is
        self-contained in a notebook.

        free_params : list[str] or None     parameters to estimate / analyze
        fixed_values : dict or None          {fixed_param: value} for the rest

        Conventions: pass ``fixed_values`` only -> the rest become free; pass
        ``free_params`` only -> the rest are fixed at their literature values;
        pass both -> they must be disjoint.  Pass neither -> keep current config.
        """
        fixed_values = {k: float(v) for k, v in (fixed_values or {}).items()}
        bad = [k for k in fixed_values if k not in ALL_PARAMS]
        if bad:
            raise ValueError(f"Unknown fixed parameters: {bad}")
        if free_params is None and not fixed_values:
            return self.free_params                      # nothing to change
        if free_params is None and fixed_values:
            free_params = [p for p in ALL_PARAMS if p not in fixed_values]
        if free_params is not None and fixed_values:
            overlap = set(free_params) & set(fixed_values)
            if overlap:
                raise ValueError(f"Parameters listed as both free and fixed: {sorted(overlap)}")
        if fixed_values:
            ti = dict(self.theta_init or {})
            ti.update(fixed_values)
            self.theta_init = ti
        self.free_params = resolve_free_params(free_params=list(free_params))
        self._validate_inputs()
        self.exp_list = self._build_experiments()
        self.pest = self._make_estimator()
        self.theta = self._cov = self.obj_value = None
        return self.free_params

    def correlation(self, free_params=None, fixed_values=None,
                    cov_method="finite_difference"):
        """
        Pipeline step 1 -- parameter correlation matrix.

        Reconfigures (free / fixed parameters), estimates, computes the covariance,
        and returns the correlation matrix ``r_ij`` (report's |r_ij| analysis).

        >>> est.correlation(free_params=["kcat_f_2", "Ks_g6p_pgi"],
        ...                 fixed_values={"Ka2_1": 0.01, "Ka3_1": 1.0})
        """
        self._reconfigure(free_params, fixed_values)
        self.estimate()
        self.covariance(method=cov_method)
        return self.correlation_matrix()

    def perturbation(self, free_params=None, fixed_values=None, **kwargs):
        """
        Pipeline step 2 -- Monte Carlo one-at-a-time (PRCC) perturbation.

        Reconfigures, then perturbs each free parameter (others at their nominal /
        fixed values) and returns the rank-correlation matrix.  Extra keyword
        arguments are forwarded to :meth:`perturbation_analysis` (``n_samples``,
        ``n_jobs``, ``distribution``, ``rel_sigma``, ``seed`` ...).

        >>> est.perturbation(free_params=[...], fixed_values={...},
        ...                  n_samples=1000, n_jobs=-1)
        """
        self._reconfigure(free_params, fixed_values)
        return self.perturbation_analysis(**kwargs)

    def _profile_grid(self, theta, names, n_grid, span):
        """1-D scan of each free parameter (others held at the estimate)."""
        base = {p: float(theta[p]) for p in names}
        rows = []
        for p in names:
            lo = max(base[p] * (1.0 - span), 1e-9)
            hi = base[p] * (1.0 + span)
            for val in np.linspace(lo, hi, n_grid):
                row = dict(base)
                row[p] = float(val)
                rows.append(row)
        return pd.DataFrame(rows, columns=names)

    def profile_likelihood(self, free_params=None, fixed_values=None,
                           theta_values=None, n_grid=15, span=0.5, alphas=(0.95,)):
        """
        Pipeline step 3 -- profile likelihood / likelihood-ratio confidence region
        via parmest (``objective_at_theta`` + ``likelihood_ratio_test``).

        Reconfigures, estimates theta_hat and the optimal objective, then evaluates
        the objective over a grid of parameter values and runs the chi2 likelihood-
        ratio test.  By default the grid is a 1-D scan of each free parameter (others
        held at theta_hat); pass a custom ``theta_values`` DataFrame (columns = the
        free parameters) for a 2-D region or any custom scan.

        Returns
        -------
        dict with keys:
            'theta'          : theta_hat (pd.Series)
            'obj_value'      : objective at theta_hat
            'obj_at_theta'   : objective for each grid point (pd.DataFrame)
            'likelihood_ratio': obj_at_theta + a True/False column per alpha
                                (True = inside the confidence region)
            'thresholds'     : chi2 objective threshold per alpha

        >>> res = est.profile_likelihood(free_params=["kcat_f_2"],
        ...                              fixed_values={...}, n_grid=25, alphas=(0.9, 0.95))
        >>> res["likelihood_ratio"]      # scan + in/out of each region
        """
        self._reconfigure(free_params, fixed_values)
        self.estimate()
        names = list(self.free_params)
        if theta_values is None:
            theta_values = self._profile_grid(self.theta, names, n_grid, span)
        obj_at_theta = self.pest.objective_at_theta(theta_values)
        LR, thresholds = self.pest.likelihood_ratio_test(
            obj_at_theta, float(self.obj_value), list(alphas), return_thresholds=True)
        return {
            "theta": self.theta,
            "obj_value": float(self.obj_value),
            "obj_at_theta": obj_at_theta,
            "likelihood_ratio": LR,
            "thresholds": thresholds,
        }

    # -- reporting ------------------------------------------------------------
    def plot_correlation_heatmap(self, ax=None, threshold=None):
        """Heatmap of the parameter correlation matrix (matplotlib)."""
        import matplotlib.pyplot as plt
        R = self.correlation_matrix().copy()
        if threshold is not None:
            R = R.where(R.abs() > threshold, 0.0)
        if ax is None:
            _, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(R.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(R.columns)))
        ax.set_xticklabels(R.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(R.index)))
        ax.set_yticklabels(R.index, fontsize=6)
        ax.figure.colorbar(im, ax=ax, label=r"$r_{ij}$")
        ax.set_title("Parameter correlation matrix")
        return ax

    def export_results(self, path_prefix: str):
        """Write theta, covariance, correlation and CIs to CSV files."""
        if self.theta is not None:
            self.theta.to_csv(f"{path_prefix}_theta.csv")
        if self._cov is not None:
            self._cov.to_csv(f"{path_prefix}_covariance.csv")
            self.correlation_matrix().to_csv(f"{path_prefix}_correlation.csv")
            self.confidence_intervals().to_csv(f"{path_prefix}_confidence_intervals.csv")

    def summary(self):
        """Short textual summary of the estimator configuration/state."""
        return "\n".join([
            "GlycolysisParameterEstimator",
            f"  conditions       : {len(self.conditions)}",
            f"  free parameters  : {len(self.free_params)} / {len(ALL_PARAMS)}",
            f"  fixed parameters : {len(self.fixed_params)}",
            f"  data points      : {self.n_data_points}",
            f"  objective        : {self.obj_function}",
            f"  imbalanced u_e   : {'free decision vars' if self.free_imbalanced else 'fixed inputs'}",
            f"  estimated        : {self.theta is not None}",
        ])

    def __repr__(self):
        return (f"GlycolysisParameterEstimator(conditions={len(self.conditions)}, "
                f"free_params={len(self.free_params)}, estimated={self.theta is not None})")


# ===========================================================================
# Smoke test (no solve unless ipopt/casadi available)
# ===========================================================================
if __name__ == "__main__":
    print("Available conditions:", available_conditions())
    print(pd.DataFrame(build_stoichiometric_matrix(),
                       index=EcoliCarbonKinetics.balanced_keys,
                       columns=EcoliCarbonKinetics.flux_keys))

    exp = GlycolysisExperiment(available_conditions()[0], load_keq(),
                               build_stoichiometric_matrix(), *load_measurement_sigmas())
    m = exp.get_labeled_model()
    print(f"Model for {exp.condition}: {len(m.experiment_outputs)} outputs, "
          f"{len(m.unknown_parameters)} free parameters.")

    # Full workflow (requires ipopt + casadi):
    # est   = GlycolysisParameterEstimator(conditions=["KO02", "KO03", "KO05"])
    # theta = est.estimate()
    # cov   = est.covariance()
    # corr  = est.correlation_matrix()
    # G     = est.sensitivity_matrix()           # {condition: dOutputs/dtheta}
    # fim   = est.fisher_information_matrix()     # sum_e G^T Q^-1 G
