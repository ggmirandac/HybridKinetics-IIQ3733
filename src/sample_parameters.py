"""
sample_parameters.py
====================
Canonical Bar-Even / Table IV parameter sampler for the E. coli glycolysis
mechanistic model (IIQ3733).

Exposes
-------
build_theta_init(seed=0) -> dict
    Sample a full 37-parameter initial guess using literature values for
    Table IV entries and log-normal draws (Bar-Even 2011) for the rest.

build_theta_sources_df(theta) -> pd.DataFrame
    Tag every parameter with its origin (literature, sampled_kcat,
    sampled_km_direct, sampled_km_linked) and return a DataFrame indexed by
    param in ALL_PARAMS order.

load_theta_init(path="Data/theta_init_sampled.csv") -> dict
    Load a frozen CSV written by main() and return {param: float}.

CLI
---
Run from the repo root to write the frozen CSV:

    python src/sample_parameters.py [--seed 0] [--out Data/theta_init_sampled.csv]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Intra-src import -- match the convention used in param_estimation_parmest.py
# and sentitivity.py: insert the directory of THIS file so that sibling
# modules resolve correctly both when run as a script and when imported as
# src.sample_parameters.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from param_estimation_parmest import THETA_BOUNDS  # noqa: E402
from kinetics_noor import ALL_PARAMS  # noqa: E402

# ---------------------------------------------------------------------------
# Literature parameters (Table IV, 21 entries)
# ---------------------------------------------------------------------------
TABLE_IV_THETA = {
    "v_max_1": 25.739, "Ka1_1": 1.0, "Ka2_1": 0.01, "Ka3_1": 1.0, "K_g6p_1": 0.5,
    "Ks_g6p_pgi": 1.018, "Kp_f6p_pgi": 0.078,
    "Ks_f6p_3": 0.013, "Ks_atp_3": 0.02, "Kp_fbp_3": 0.14,
    "Ks_fbp_4": 0.24, "kcat_f_4": 0.95,
    "Ks_dhap_5": 1.03, "kcat_f_5": 1201.2,
    "Ks_g3p_6": 0.89, "Ks_nad_6": 0.045, "Ks_pi_6": 0.53,
    "Ks_3pg_8": 0.2, "Ks_2pg_8": 0.19, "kcat_f_8": 41.602,
    "Ks_2pg_9": 0.1,
}
LITERATURE_PARAMS = list(TABLE_IV_THETA.keys())

# ---------------------------------------------------------------------------
# Parameters sampled from Bar-Even 2011 distributions
# ---------------------------------------------------------------------------
SAMPLED_KCAT = {"kcat_f_2": "Pgi", "kcat_f_3": "PfkB", "kcat_f_6": "GapA",
                "kcat_f_7": "Pgk", "kcat_f_9": "Eno"}
SAMPLED_KM = ["Kp_adp_3", "Kp_g3p_4", "Kp_dhap_4", "Kp_g3p_5",
              "Kp_pgp_6", "Kp_nadh_6", "Ks_pgp_7", "Ks_adp_7", "Ks_3pg_7", "Ks_atp_7", "Ks_pep_9"]
SAMPLED_PARAMS = list(SAMPLED_KCAT.keys()) + SAMPLED_KM

# ---------------------------------------------------------------------------
# Enzyme molecular weights (kDa) for kcat unit conversion
# ---------------------------------------------------------------------------
ENZYME_MW_KDA = {"Pgi": 59.0, "PfkB": 36.0, "GapA": 35.0, "Pgk": 43.7, "Eno": 46.0}


def kcat_convert(kcat_s, enzyme):
    """Convert kcat from 1/s to 1/h, normalized by enzyme MW in g/mol."""
    mw_g_per_mol = ENZYME_MW_KDA[enzyme] * 1000
    return kcat_s * 3600 / mw_g_per_mol


# ---------------------------------------------------------------------------
# Bar-Even 2011 distribution parameters
# ---------------------------------------------------------------------------
KCAT_MEDIAN_S = 79
KCAT_RANGE_S = [50, 90]
KCATKM_MEDIAN = 410
KCATKM_RANGE = [300, 500]
KM_MEDIAN_MM = 0.500
KM_RANGE_MM = [0.010, 0.700]

# ---------------------------------------------------------------------------
# Linkage: each kcat enzyme drives its paired Km parameters
# ---------------------------------------------------------------------------
LINKED_KMS = {"Pgi": [], "PfkB": ["Kp_adp_3"], "GapA": ["Kp_pgp_6", "Kp_nadh_6"],
              "Pgk": ["Ks_pgp_7", "Ks_adp_7", "Ks_3pg_7", "Ks_atp_7"], "Eno": ["Ks_pep_9"]}
DIRECT_KMS = ["Kp_g3p_4", "Kp_dhap_4", "Kp_g3p_5"]


# ---------------------------------------------------------------------------
# Core sampling helpers
# ---------------------------------------------------------------------------
def _sample_lognormal(rng, median, lo, hi):
    mu = np.log(median)
    sigma = (np.log(hi) - np.log(lo)) / (2 * 1.96)
    return float(np.exp(rng.normal(mu, sigma)))


def build_theta_init(seed):
    """
    Build a full 37-parameter initial guess.

    Literature (Table IV) values are used as-is.  The remaining parameters
    are drawn from log-normal distributions (Bar-Even 2011) and clipped to
    THETA_BOUNDS.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        {param_name: float} for all parameters in ALL_PARAMS.
    """
    rng = np.random.default_rng(seed)
    theta = dict(TABLE_IV_THETA)
    for kcat_param, enzyme in SAMPLED_KCAT.items():
        kcat_s = _sample_lognormal(rng, KCAT_MEDIAN_S, *KCAT_RANGE_S)
        lb, ub = THETA_BOUNDS[kcat_param]
        theta[kcat_param] = float(min(max(kcat_convert(kcat_s, enzyme), lb), ub))
        for km_param in LINKED_KMS[enzyme]:
            kcat_km = _sample_lognormal(rng, KCATKM_MEDIAN, *KCATKM_RANGE)
            lbk, ubk = THETA_BOUNDS[km_param]
            theta[km_param] = float(min(max(kcat_s / kcat_km, lbk), ubk))
    for km_param in DIRECT_KMS:
        lbk, ubk = THETA_BOUNDS[km_param]
        val = _sample_lognormal(rng, KM_MEDIAN_MM, *KM_RANGE_MM)
        theta[km_param] = float(min(max(val, lbk), ubk))
    return theta


def build_theta_sources_df(theta):
    """
    Build a DataFrame tagging each parameter with its sampling origin.

    Parameters
    ----------
    theta : dict
        {param_name: float} as returned by build_theta_init().

    Returns
    -------
    pd.DataFrame
        Indexed by param (in ALL_PARAMS order), columns: value, source.
        source is one of: literature, sampled_kcat, sampled_km_direct,
        sampled_km_linked.
    """
    _direct_set = set(DIRECT_KMS)
    _linked_set = set(SAMPLED_KM) - _direct_set
    records = []
    for param in ALL_PARAMS:
        value = theta[param]
        if param in TABLE_IV_THETA:
            source = "literature"
        elif param in SAMPLED_KCAT:
            source = "sampled_kcat"
        elif param in _direct_set:
            source = "sampled_km_direct"
        else:
            source = "sampled_km_linked"
        records.append({"param": param, "value": value, "source": source})
    df = pd.DataFrame(records).set_index("param")
    return df


def load_theta_init(path="Data/theta_init_sampled.csv"):
    """
    Load a frozen theta CSV written by main() and return a plain dict.

    Parameters
    ----------
    path : str
        Path to the CSV file (index column must be named 'param').

    Returns
    -------
    dict
        {param_name: float(value)}.
    """
    df = pd.read_csv(path, index_col="param")
    return {p: float(df.loc[p, "value"]) for p in df.index}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":

    seed = 0
    out_path = "../Data/theta_init_sampled.csv"
    theta = build_theta_init(seed=seed)
    df = build_theta_sources_df(theta)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index_label="param")

    counts = df["source"].value_counts()
    print(f"Written {len(df)} parameters to {out_path}")
    print("Counts by source:")
    for src, n in counts.items():
        print(f"  {src}: {n}")
    print(f"Total: {len(df)}")
