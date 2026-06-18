"""
param_estimation_scaffold.py
============================
Suggested scaffold for the parmest-based parameter estimation of the
E. coli glycolysis mechanistic model (IIQ3733).

This file is a DESIGN PROPOSAL, not a finished implementation.
See param_estimation_notes.md for the full specification and open questions
that must be resolved before this can be run.

Mathematical problem being implemented:

    min_{x_e, u_e, theta}  sum_e [ w_v * ||v_real_e - v_e||^2
                                  + w_x * ||x_real_e - x_e||^2 ]
    s.t.
        S * v(x_e, u_e, e_e; Keq, theta) - b_e = 0   for all e
        x_lb <= x_e <= x_ub                           for all e
        u_lb <= u_e <= u_ub                           for all e
        theta_lb <= theta <= theta_ub

where:
    x_e    = balanced metabolite concentrations (decision variables per experiment)
    u_e    = imbalanced metabolite concentrations (inputs, fixed or bounded per experiment)
    e_e    = enzyme concentrations (fixed inputs from proteomics data)
    theta  = kinetic parameters shared across all experiments
    Keq    = thermodynamic equilibrium constants (fixed, from k_eq_values.pkl)
    b_e    = cellular drain vector per experiment (from cellular_needs.csv)
    v_e    = flux vector predicted by kinetics model
    v_real_e = measured flux vector (from important_fluxes.csv)
    x_real_e = measured metabolite concentrations (from balanced_metabolites.csv)
"""

import os
import pickle
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.parmest.experiment import Experiment
import pyomo.contrib.parmest.parmest as parmest

# ---------------------------------------------------------------------------
# Import the kinetics class from its file.
# ASSUMPTION: kinetics_noor.py is the authoritative kinetics source.
# param_estimation.py (the existing file) duplicates the class — decide which
# one to keep before running. See notes.md §1.
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.dirname(__file__))
from kinetics_noor import EcoliCarbonKinetics, ALL_PARAMS, PARAM_RXN_MAP

# ---------------------------------------------------------------------------
# Weights for the objective function.
# DECISION: choose values that appropriately scale fluxes vs. metabolite
# concentrations. Fluxes are in mmol/gdcw/h; metabolites in mmol/L.
# Suggested starting point: w_v=1, w_x=0 to fit only fluxes first,
# then introduce w_x once the flux fit is satisfactory.
# ---------------------------------------------------------------------------
W_V = 1.0   # weight on flux residuals
W_X = 0.0   # weight on metabolite concentration residuals (start at 0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_experiment_data(condition: str, data_dir: str = "Data") -> dict:
    """
    Assemble a per-experiment data dict from the four CSV files.

    Parameters
    ----------
    condition : str
        Column name in the CSVs, e.g. 'KO02', 'GR01'.
    data_dir : str
        Path to the Data directory.

    Returns
    -------
    dict with keys:
        'v_real' : dict {flux_name: float}   — measured fluxes (mmol/gdcw/h)
        'x_real' : dict {met_name: float}    — measured balanced metabolites (mmol/L)
        'u'      : dict {met_name: float}    — imbalanced metabolite values (mmol/L)
        'e'      : dict {enzyme_name: float} — enzyme concentrations (mmol/gdcw)
        'b'      : dict {met_name: float}    — cellular drain vector (mmol/gdcw/h)

    Notes
    -----
    - balanced_metabolites.csv covers only 6 of 9 balanced metabolites.
      C_2pg, C_g3p, C_pgp are NOT measured and will not appear in 'x_real'.
    - imbalanced_metabolites.csv does not contain C_pi or C_nadh, which are
      needed by the GAP kinetics. These must be handled separately (see notes.md §3).
    - The 'e' dict lacks 'PTS' enzyme; v_max_1 in theta absorbs that.
    """
    bal   = pd.read_csv(os.path.join(data_dir, "balanced_metabolites.csv"),  index_col="VarNames")
    flux  = pd.read_csv(os.path.join(data_dir, "important_fluxes.csv"),      index_col="VarNames")
    imbal = pd.read_csv(os.path.join(data_dir, "imbalanced_metabolites.csv"), index_col="VarNames")
    prot  = pd.read_csv(os.path.join(data_dir, "important_proteins.csv"),    index_col="VarNames")
    needs = pd.read_csv(os.path.join(data_dir, "cellular_needs.csv"),        index_col="Unnamed: 0")

    def col(df):
        if condition not in df.columns:
            raise KeyError(f"Condition '{condition}' not found in {df.columns.tolist()}")
        return df[condition].dropna().to_dict()

    return {
        "v_real": col(flux),
        "x_real": col(bal),
        "u":      col(imbal),
        "e":      col(prot),
        "b":      col(needs),
    }


def available_conditions(data_dir: str = "Data") -> list:
    """Return conditions that have data in all four CSV files."""
    dfs = [
        pd.read_csv(os.path.join(data_dir, f), index_col=0)
        for f in [
            "balanced_metabolites.csv",
            "important_fluxes.csv",
            "imbalanced_metabolites.csv",
            "important_proteins.csv",
        ]
    ]
    cols = [set(df.columns) for df in dfs]
    return sorted(set.intersection(*cols))


def load_keq(data_dir: str = "Data") -> dict:
    """Load equilibrium constants from pickle."""
    path = os.path.join(data_dir, "k_eq_values.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Stoichiometric matrix (standalone helper, avoids full class instantiation)
# ---------------------------------------------------------------------------

def build_stoichiometric_matrix() -> np.ndarray:
    """
    Return the 9x9 stoichiometric matrix S (balanced_metabolites × reactions)
    as a numpy array.

    Row order : EcoliCarbonKinetics.balanced_keys
                ['C_g6p','C_f6p','C_fbp','C_dhap','C_g3p','C_pgp','C_3pg','C_2pg','C_pep']
    Column order: EcoliCarbonKinetics.flux_keys
                  ['v_pts','v_pgi','v_pfkB','v_fbaA','v_tpiA','v_gapA','v_pgk','v_gpmA','v_eno']

    IMPORTANT: verify this matrix against the CLAUDE.md stoichiometric
    description before using. This is a hand-coded placeholder — the existing
    _construct_stoichiometric_matrix() method in EcoliCarbonKinetics should
    be the authoritative source.
    """
    # Metabolite rows: g6p, f6p, fbp, dhap, g3p, pgp, 3pg, 2pg, pep
    # Reaction cols:   pts, pgi, pfk, fba,  tpi, gap, pgk, gpm, eno
    S = np.array([
    #  pts  pgi  pfk  fba  tpi  gap  pgk  gpm  eno
    [   1,  -1,   0,   0,   0,   0,   0,   0,   0],  # g6p
    [   0,   1,  -1,   0,   0,   0,   0,   0,   0],  # f6p
    [   0,   0,   1,  -1,   0,   0,   0,   0,   0],  # fbp
    [   0,   0,   0,   1,  -1,   0,   0,   0,   0],  # dhap
    [   0,   0,   0,   1,   1,  -1,   0,   0,   0],  # g3p  (fba splits fbp -> g3p+dhap; tpi: dhap->g3p)
    [   0,   0,   0,   0,   0,   1,  -1,   0,   0],  # pgp
    [   0,   0,   0,   0,   0,   0,   1,  -1,   0],  # 3pg
    [   0,   0,   0,   0,   0,   0,   0,   1,  -1],  # 2pg
    [  -1,   0,   0,   0,   0,   0,   0,   0,   1],  # pep  (pts consumes pep; eno produces pep)
    ], dtype=float)
    # TODO: cross-check every sign with EcoliCarbonKinetics._construct_stoichiometric_matrix()
    return S


# ---------------------------------------------------------------------------
# Default parameter bounds
# ---------------------------------------------------------------------------

# DECISION: these bounds must be tightened using literature values.
# kcat_f values are in 1/s, Ks/Kp values in mmol/L, v_max in mmol/gdcw/h.
THETA_BOUNDS = {p: (1e-6, 1e4) for p in ALL_PARAMS}

# Balanced metabolite bounds (mmol/L)
# DECISION: set tighter bounds from physiological ranges.
X_BOUNDS = {k: (1e-5, 50.0) for k in EcoliCarbonKinetics.balanced_keys}

# Imbalanced metabolite bounds (mmol/L) — used when u_e is treated as a free variable
U_BOUNDS = {k: (1e-5, 20.0) for k in EcoliCarbonKinetics.imbalanced_keys}


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------

class GlycolysisExperiment(Experiment):
    """
    One experimental condition for parmest parameter estimation.

    Each instance wraps one column from the Ishii dataset and builds a
    Pyomo model representing:

        S * v(x_e, u_e, e_e; Keq, theta) = b_e        (steady-state)
        x_lb <= x_e <= x_ub                            (metabolite bounds)

    The kinetic parameters `theta` are `.fix()`-ed here; parmest unfixes
    them across all experiments simultaneously during estimation.

    Parameters
    ----------
    data : dict
        Output of load_experiment_data().
    keq : dict
        Equilibrium constants from load_keq().
    theta_init : dict, optional
        Initial guess for kinetic parameters. Defaults to 1.0 for all.
    fix_u : bool
        If True (default), imbalanced metabolites are fixed to experimental
        values. If False, they become bounded decision variables (x_e-like),
        which is needed when experimental values have missing entries.
    """

    def __init__(self, data: dict, keq: dict,
                 theta_init: dict = None, fix_u: bool = True):
        self.data       = data
        self.keq        = keq
        self.theta_init = theta_init or {p: 1.0 for p in ALL_PARAMS}
        self.fix_u      = fix_u
        self.model      = None

    # ------------------------------------------------------------------
    def create_model(self) -> None:
        m = pyo.ConcreteModel()

        # ---- Balanced metabolites (x_e): free decision variables -------
        for key in EcoliCarbonKinetics.balanced_keys:
            lb, ub = X_BOUNDS[key]
            val = self.data["x_real"].get(key, np.sqrt(lb * ub))
            m.add_component(key, pyo.Var(bounds=(lb, ub), initialize=val))

        # ---- Imbalanced metabolites (u_e) --------------------------------
        # DECISION: missing C_pi and C_nadh from the CSV (see notes.md §3).
        # Fallback values are placeholders — replace with literature estimates.
        U_FALLBACK = {
            "C_pi":   10.0,   # mmol/L — typical intracellular phosphate
            "C_nadh": 0.083,  # mmol/L — typical NADH
            "C_nad":  1.5,    # mmol/L — if also missing
        }
        for key in EcoliCarbonKinetics.imbalanced_keys:
            val = self.data["u"].get(key, U_FALLBACK.get(key, 1.0))
            lb, ub = U_BOUNDS[key]
            m.add_component(key, pyo.Var(bounds=(lb, ub), initialize=val))
            if self.fix_u:
                getattr(m, key).fix(val)

        # ---- Kinetic parameters (theta): fixed, parmest will unfix ------
        for key in ALL_PARAMS:
            lb, ub = THETA_BOUNDS[key]
            val = max(lb, min(ub, self.theta_init.get(key, 1.0)))
            m.add_component(key, pyo.Var(bounds=(lb, ub), initialize=val))
            getattr(m, key).fix(val)

        # ---- Build C and constants dicts pointing to Pyomo Vars ----------
        # The kinetics methods use only Python arithmetic (+, -, *, /), so
        # passing Pyomo Var objects builds symbolic Pyomo expressions correctly.
        # NOTE: no np.exp calls in the kinetics — pure arithmetic is safe.
        C = {k: getattr(m, k)
             for k in EcoliCarbonKinetics.balanced_keys
                      + EcoliCarbonKinetics.imbalanced_keys}
        constants = {k: getattr(m, k) for k in ALL_PARAMS}

        # Enzyme concentrations are plain floats (fixed inputs).
        # PTS enzyme (not in proteomics) is absorbed into v_max_1 (in theta).
        e = dict(self.data["e"])  # e.g. {'Pgi': 0.68, 'PfkB': 0.037, ...}

        # ---- Instantiate a lightweight kinetics object (for Keq access) --
        # ISSUE: EcoliCarbonKinetics.__init__ requires bounds dicts and calls
        # construct_steady_state_problem(). We bypass __init__ to avoid that.
        # Only self.Keq is needed by the kinetics methods.
        kin = object.__new__(EcoliCarbonKinetics)
        kin.Keq = self.keq

        # ---- Flux Expressions --------------------------------------------
        flux_fns = [kin.pts, kin.pgi, kin.pfk, kin.fba, kin.tpi,
                    kin.gap, kin.pgk, kin.gpm, kin.eno]
        for fname, fn in zip(EcoliCarbonKinetics.flux_keys, flux_fns):
            m.add_component(fname, pyo.Expression(expr=fn(constants, C, e)))

        # ---- Steady-state constraint: S * v = b_e -------------------------
        S = build_stoichiometric_matrix()
        b = self.data.get("b", {})
        bal_keys  = EcoliCarbonKinetics.balanced_keys
        flux_keys = EcoliCarbonKinetics.flux_keys

        def ss_rule(m, i):
            sv = sum(S[i, j] * getattr(m, flux_keys[j]) for j in range(9))
            drain = b.get(bal_keys[i], 0.0)
            return sv == drain

        m.ss_constraint = pyo.Constraint(range(9), rule=ss_rule)

        self.model = m

    # ------------------------------------------------------------------
    def label_model(self) -> None:
        m = self.model

        # experiment_outputs: what parmest computes SSE against.
        # Both measured fluxes AND measured metabolite concentrations are outputs.
        m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)

        for fname in EcoliCarbonKinetics.flux_keys:
            if fname in self.data["v_real"]:
                m.experiment_outputs[getattr(m, fname)] = self.data["v_real"][fname]

        for mname in EcoliCarbonKinetics.balanced_keys:
            if mname in self.data["x_real"]:
                m.experiment_outputs[getattr(m, mname)] = self.data["x_real"][mname]

        # unknown_parameters: theta shared across all experiments.
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (getattr(m, k), pyo.value(getattr(m, k))) for k in ALL_PARAMS
        )

        # measurement_error: needed for covariance estimation (cov_est()).
        # DECISION: populate from experimental standard deviations if available.
        # For now, left as a TODO (will cause cov_est() to fail until filled).
        # m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        # m.measurement_error.update([(getattr(m, fname), sigma_v[fname]) ...])

    # ------------------------------------------------------------------
    def get_labeled_model(self):
        if self.model is None:
            self.create_model()
            self.label_model()
        return self.model


# ---------------------------------------------------------------------------
# Weighted objective variant
# ---------------------------------------------------------------------------

class GlycolysisExperimentWeighted(GlycolysisExperiment):
    """
    Alternative experiment class that defines the weighted SSE objective
    explicitly in the model (required when W_V != W_X).

    Uses the simple_reaction_parmest_example.py pattern:
    define model.SecondStageCost and model.Total_Cost_Objective directly
    instead of relying on parmest's built-in 'SSE' aggregation.

    DECISION: use this class if w_v != w_x or if normalisation by sigma
    is needed. Use GlycolysisExperiment with obj_function='SSE' otherwise.
    """

    def label_model(self) -> None:
        m = self.model

        # Residual expressions
        v_real = self.data["v_real"]
        x_real = self.data["x_real"]

        flux_sse = sum(
            W_V * (v_real[fname] - getattr(m, fname)) ** 2
            for fname in EcoliCarbonKinetics.flux_keys
            if fname in v_real
        )
        met_sse = sum(
            W_X * (x_real[mname] - getattr(m, mname)) ** 2
            for mname in EcoliCarbonKinetics.balanced_keys
            if mname in x_real
        )

        m.FirstStageCost  = pyo.Expression(expr=0)
        m.SecondStageCost = pyo.Expression(expr=flux_sse + met_sse)
        m.Total_Cost_Objective = pyo.Objective(
            expr=m.FirstStageCost + m.SecondStageCost,
            sense=pyo.minimize,
        )

        # unknown_parameters suffix still required
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (getattr(m, k), pyo.value(getattr(m, k))) for k in ALL_PARAMS
        )


# ---------------------------------------------------------------------------
# Main estimation runner
# ---------------------------------------------------------------------------

def run_parameter_estimation(
    conditions: list = None,
    data_dir:   str  = "Data",
    theta_init: dict = None,
    fix_u:      bool = True,
    weighted:   bool = False,
) -> tuple:
    """
    Run parameter estimation over all (or a subset of) experimental conditions.

    Parameters
    ----------
    conditions : list of str, optional
        Subset of condition IDs to use. Defaults to all available.
    data_dir : str
        Path to the Data directory.
    theta_init : dict, optional
        Initial parameter values. Defaults to 1.0 for all.
    fix_u : bool
        Whether to fix imbalanced metabolites to experimental values.
    weighted : bool
        Use GlycolysisExperimentWeighted (explicit W_V/W_X) instead of
        GlycolysisExperiment with parmest SSE.

    Returns
    -------
    (obj, theta) or (obj, theta, cov)
    """
    keq = load_keq(data_dir)

    if conditions is None:
        conditions = available_conditions(data_dir)

    ExperimentClass = GlycolysisExperimentWeighted if weighted else GlycolysisExperiment
    obj_fn = None if weighted else "SSE"

    exp_list = []
    for cond in conditions:
        try:
            data = load_experiment_data(cond, data_dir)
            exp_list.append(ExperimentClass(data, keq, theta_init, fix_u))
        except Exception as exc:
            print(f"[WARNING] Skipping condition {cond}: {exc}")

    if not exp_list:
        raise RuntimeError("No valid experiments loaded.")

    pest = parmest.Estimator(exp_list, obj_function=obj_fn)
    obj, theta = pest.theta_est()
    return obj, theta


# ---------------------------------------------------------------------------
# Entry point (quick smoke-test on a single condition)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    keq  = load_keq()
    data = load_experiment_data("KO02")

    # Build and inspect a single experiment model
    exp = GlycolysisExperiment(data, keq)
    m   = exp.get_labeled_model()
    m.pprint()

    # Uncomment to run full estimation (slow — 38 parameters, 22 conditions):
    # obj, theta = run_parameter_estimation(conditions=["KO02", "KO03"])
    # print("Objective:", obj)
    # print("Theta:", theta)
