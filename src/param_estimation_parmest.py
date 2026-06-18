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
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings

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

        Reconfigures, estimates θ̂ and the optimal objective, then evaluates the
        objective over a grid of parameter values and runs the χ² likelihood-ratio
        test.  By default the grid is a 1-D scan of each free parameter (others held
        at θ̂); pass a custom ``theta_values`` DataFrame (columns = the free
        parameters) for a 2-D region or any custom scan.

        Returns
        -------
        dict with keys:
            'theta'          : θ̂ (pd.Series)
            'obj_value'      : objective at θ̂
            'obj_at_theta'   : objective for each grid point (pd.DataFrame)
            'likelihood_ratio': obj_at_theta + a True/False column per alpha
                                (True = inside the confidence region)
            'thresholds'     : χ² objective threshold per alpha

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
