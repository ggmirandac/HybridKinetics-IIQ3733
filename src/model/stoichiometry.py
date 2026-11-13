"""
stoichiometry.py -- stoichiometric matrix and equilibrium constants of the glycolysis model.

``stoichiometric_table`` is the 9x9 matrix S (balanced metabolites x reactions) with the rows in BALANCED_KEYS order,
the same order as the drain vector b and the Pyomo model, so S v - b is consistent by construction.
``load_keq`` reads the equilibrium constants computed by gen_K_eq.ipynb (Data/k_eq_values.pkl).

Position in the pipeline: model definition, used by kinetics.py and by the estimation modules.
"""
import os
import pickle

import numpy as np
import pandas as pd

from paths import DATA_DIR
from model.parameters import BALANCED_KEYS, BALANCED_NAMES, RXN_ORDER, ALPHABETICAL_ROW_ORDER

def load_keq():
    """Equilibrium constants {reaction: float} (eQuilibrator, pH 7, ionic strength 0.1 M, 37 C)."""
    with open(os.path.join(DATA_DIR, "k_eq_values.pkl"), "rb") as fh:
        stored = pickle.load(fh)
    keq = {}
    for reaction, value in stored.items():
        keq[reaction] = float(value)
    return keq

def stoichiometric_table(balanced_keys=BALANCED_KEYS):
    """
    The 9x9 stoichiometric matrix (metabolites x reactions) as a DataFrame.  Rows: ``balanced_keys`` order (default
    g6p, f6p, fbp, dhap, g3p, pgp, 3pg, 2pg, pep); columns: RXN_ORDER.  The table is written below in the
    alphabetical row order in which the stoichiometric table is built (ALPHABETICAL_ROW_ORDER) and reindexed at the end.
    """
    matrix = np.array([
        # pts  pgi  pfk  fba  tpi  gap  pgk  gpm  eno
        [0,    0,   0,   0,   0,   0,   0,   1,  -1],  # 2pg
        [0,    0,   0,   0,   0,   0,   1,  -1,   0],  # 3pg
        [0,    0,   0,   1,  -1,   0,   0,   0,   0],  # dhap
        [0,    1,  -1,   0,   0,   0,   0,   0,   0],  # f6p
        [0,    0,   1,  -1,   0,   0,   0,   0,   0],  # fbp
        [0,    0,   0,   1,   1,  -1,   0,   0,   0],  # g3p
        [1,   -1,   0,   0,   0,   0,   0,   0,   0],  # g6p
        [-1,   0,   0,   0,   0,   0,   0,   0,   1],  # pep
        [0,    0,   0,   0,   0,   1,  -1,   0,   0],  # pgp
    ])
    table = pd.DataFrame(matrix, index=ALPHABETICAL_ROW_ORDER, columns=RXN_ORDER)
    rows = [key[2:] for key in balanced_keys]
    return table.loc[rows]

def build_stoichiometric_matrix():
    """The 9x9 stoichiometric matrix S as a float array, rows in BALANCED_KEYS order and columns in RXN_ORDER."""
    table = stoichiometric_table()
    return table.loc[BALANCED_NAMES, RXN_ORDER].to_numpy(dtype=float)
