"""
parameters.py -- the parameters and the structure of the glycolysis model (names only, no numbers to fit).

The 37 kinetic parameters (ALL_PARAMS) in model order, the parameters of every reaction (PARAM_RXN_MAP), the
reaction order (RXN_ORDER), the metabolite orders (BALANCED_KEYS, BALANCED_NAMES, ALPHABETICAL_ROW_ORDER), the
cofactors, enzymes and flux keys, the literature starting values and the conventions of the measured outputs and the
free drains (LI_FLUX_KEYS, DUPLICATE_FLUX_GROUPS, FREE_DRAIN_KEYS).  ``reaction_groups()`` returns one parameter
group per reaction: the selection keeps at least one free parameter in each group, because an equation without a
free parameter cannot be adjusted.

Position in the pipeline: the model is defined here, in kinetics.py and in stoichiometry.py; every other module
takes it from the model package.  Edit this file to change which parameters the model has.
"""

ALL_PARAMS = [
    # PTS (empirical formulation)
    "v_max_1",
    "Ka1_1",
    "Ka2_1",
    "Ka3_1",
    "K_g6p_1",
    # PGI  (Noor formulation)
    "Ks_g6p_pgi",
    "Kp_f6p_pgi",
    "kcat_f_2",
    # PFK  (Noor formulation)
    "Ks_f6p_3",
    "Ks_atp_3",
    "Kp_fbp_3",
    "Kp_adp_3",
    "kcat_f_3",
    # FBA  (Noor formulation)
    "Ks_fbp_4",
    "Kp_g3p_4",
    "Kp_dhap_4",
    "kcat_f_4",
    # TPI  (Noor formulation)
    "kcat_f_5",
    "Ks_dhap_5",
    "Kp_g3p_5",
    # GAPDH (Noor formulation)
    "kcat_f_6",
    "Ks_g3p_6",
    "Ks_pi_6",
    "Ks_nad_6",
    "Kp_pgp_6",
    "Kp_nadh_6",
    # PGK  (Noor formulation)
    "kcat_f_7",
    "Ks_pgp_7",
    "Ks_adp_7",
    "Ks_3pg_7",
    "Ks_atp_7",
    # GPM  (Noor formulation)
    "kcat_f_8",
    "Ks_3pg_8",
    "Ks_2pg_8",
    # ENO  (Noor formulation)
    "kcat_f_9",
    "Ks_2pg_9",
    "Ks_pep_9",
]

PARAM_RXN_MAP = {
    "pts": [
        "v_max_1",
        "Ka1_1",
        "Ka2_1",
        "Ka3_1",
        "K_g6p_1",
    ],
    "pgi": ["Ks_g6p_pgi", "Kp_f6p_pgi", "kcat_f_2"],
    "pfk": ["Ks_f6p_3", "Ks_atp_3", "Kp_fbp_3", "Kp_adp_3", "kcat_f_3"],
    "fba": ["Ks_fbp_4", "Kp_g3p_4", "Kp_dhap_4", "kcat_f_4"],
    "tpi": ["kcat_f_5", "Ks_dhap_5", "Kp_g3p_5"],
    "gap": ["kcat_f_6", "Ks_g3p_6", "Ks_pi_6", "Ks_nad_6", "Kp_pgp_6", "Kp_nadh_6"],
    "pgk": ["kcat_f_7", "Ks_pgp_7", "Ks_adp_7", "Ks_3pg_7", "Ks_atp_7"],
    "gpm": ["kcat_f_8", "Ks_3pg_8", "Ks_2pg_8"],
    "eno": ["kcat_f_9", "Ks_2pg_9", "Ks_pep_9"],
}

# Balanced metabolites (steady-state states x), imbalanced metabolites (cofactors u), enzymes and fluxes.
BALANCED_KEYS = ["C_g6p", "C_f6p", "C_fbp", "C_dhap", "C_g3p",
                 "C_pgp", "C_3pg", "C_2pg", "C_pep"]
IMBALANCED_KEYS = ["C_atp", "C_adp", "C_nad", "C_nadh", "C_pi", "C_pyr", "C_glc"]
ENZYME_KEYS = ["Pgi", "PfkB", "FbaA", "TpiA", "GapA", "Pgk", "GpmA", "Eno"]
FLUX_KEYS = ["v_pts", "v_pgi", "v_pfkB", "v_fbaA", "v_tpiA",
             "v_gapA", "v_pgk", "v_gpmA", "v_eno"]

# Reactions in model order (columns of S, order of the flux vector; the rate-law methods of EcoliCarbonKinetics carry
# these names).  The ONE definition: stoichiometry.py and kinetics.py take the order from here.
RXN_ORDER = ["pts", "pgi", "pfk", "fba", "tpi", "gap", "pgk", "gpm", "eno"]

# Balanced metabolites without the "C_" prefix, in BALANCED_KEYS order (the row order of S and of the drain vector b).
BALANCED_NAMES = [k[2:] for k in BALANCED_KEYS]

# Alphabetical metabolite order (the row order in which the stoichiometric table is built, and the row order of
# Data/cellular_needs.csv).  stoichiometry.py writes the table of S in this order and reindexes it to BALANCED_KEYS.
ALPHABETICAL_ROW_ORDER = ["2pg", "3pg", "dhap", "f6p", "fbp", "g3p", "g6p", "pep", "pgp"]

def reaction_of_parameters():
    """{parameter: reaction} from PARAM_RXN_MAP."""
    reaction_of = {}
    for reaction, members in PARAM_RXN_MAP.items():
        for name in members:
            reaction_of[name] = reaction
    return reaction_of

RXN_OF_PARAM = reaction_of_parameters()

def reaction_groups():
    """{reaction: [parameters]}, one group per rate equation in model order (built from PARAM_RXN_MAP)."""
    groups = {}
    for reaction in RXN_ORDER:
        groups[reaction] = list(PARAM_RXN_MAP[reaction])
    return groups

# ---------------------------------------------------------------------------
# Literature parameter values (starting values of the fits).  Units: kcat in 1/h, Ks / Kp in mM, v_max in mmol/gDW/h.
# ---------------------------------------------------------------------------
LITERATURE_THETA = {
    # PTS (empirical, Kadir et al. 2010)
    # K_g6p_1 = K (mM) in the inhibition 1/(1 + (C_g6p/K)^4)
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

# Linearly independent measured fluxes.  The data rows are duplicated:
# pfkB = fbaA = tpiA, gapA = pgk, gpmA = eno (identical in every condition).
LI_FLUX_KEYS = ["v_pts", "v_pgi", "v_pfkB", "v_gapA", "v_gpmA"]

# Measured flux rows that the four zero-drain balances force equal (identical in every condition); the number of
# independent measured outputs counts each group once (metrics.n_nonzero_outputs).
DUPLICATE_FLUX_GROUPS = [("v_pfkB", "v_fbaA", "v_tpiA"), ("v_gapA", "v_pgk"), ("v_gpmA", "v_eno")]

# Balanced metabolites with nonzero drains in Data/cellular_needs.csv; the other four
# rows (fbp, dhap, pgp, 2pg) have b = 0.
FREE_DRAIN_KEYS = ["C_g6p", "C_f6p", "C_g3p", "C_3pg", "C_pep"]
