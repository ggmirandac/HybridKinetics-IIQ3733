"""
kinetics.py - Mechanistic kinetic model of E. coli glycolysis (IIQ3733).

Models 9 enzymatic reactions from glucose uptake (PTS) to PEP production (Eno),
covering the core of E. coli central carbon metabolism.
Each reaction is modeled with convenient kinetic rate laws that include substrate saturation and allosteric regulation, based on literature data.
"""

# %%
import time

import numpy as np
import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt
import scipy
import pyomo.environ as pyo

ALL_PARAMS = [
    # PTS
    "kI_pyr_1",
    "kA_pep_1",
    "v_max_1",
    "Ka1_1",
    "Ka2_1",
    "Ka3_1",
    # "n_g6p_1", HARD CODED AS HILL COEFFICIENT
    "K_g6p_1",
    # PGI
    "kI_pep_2",
    "Km_g6p_2",
    "Km_f6p_2",
    "kcat_f_2",
    "kcat_r_2",
    # PFK
    "kI_f6p_3",
    "kI_fbp_3",
    "kI_gtp_3",
    "kI_pep_3",
    "kI_pi_3",
    "kA_adp_3",
    "kA_gdp_3",
    "Km_f6p_3",
    "Km_atp_3",
    "Km_fbp_3",
    "Km_adp_3",
    "kcat_f_3",
    "kcat_r_3",
    # FBA
    "kI_3pg_4",
    "kI_dhap_4",
    "kI_g3p_4",
    "kA_pep_4",
    "kcat_f_4",
    "kcat_r_4",
    "Km_fbp_4",
    "Km_g3p_4",
    "Km_dhap_4",
    # TPI
    "kcat_f_5",
    "kcat_r_5",
    "Km_dhap_5",
    "Km_g3p_5",
    # GAPDH
    "kI_adp_6",
    "kI_amp_6",
    "kI_atp_6",
    "kcat_f_6",
    "kcat_r_6",
    "Km_g3p_6",
    "Km_pi_6",
    "Km_nad_6",
    "Km_pgp_6",
    "Km_nadh_6",
    # PGK
    "kA_3pg_7",
    "kA_atp_7",
    "kcat_f_7",
    "kcat_r_7",
    "Km_pgp_7",
    "Km_adp_7",
    "Km_3pg_7",
    "Km_atp_7",
    # GPM
    "kI_pi_8",
    "kcat_f_8",
    "kcat_r_8",
    "Km_3pg_8",
    "Km_2pg_8",
    # ENO
    "kcat_f_9",
    "kcat_r_9",
    "Km_2pg_9",
    "Km_pep_9",
]
PARAM_RXN_MAP = {
    "pts": [
        "kI_pyr_1",
        "kA_pep_1",
        "v_max_1",
        "Ka1_1",
        "Ka2_1",
        "Ka3_1",
        # "n_g6p_1", # HARD CODED AS IT IS STRUCTURAL
        "K_g6p_1",
    ],
    "pgi": ["kI_pep_2", "Km_g6p_2", "Km_f6p_2", "kcat_f_2", "kcat_r_2"],
    "pfk": [
        "kI_f6p_3",
        "kI_fbp_3",
        "kI_gtp_3",
        "kI_pep_3",
        "kI_pi_3",
        "kA_adp_3",
        "kA_gdp_3",
        "Km_f6p_3",
        "Km_atp_3",
        "Km_fbp_3",
        "Km_adp_3",
        "kcat_f_3",
        "kcat_r_3",
    ],
    "fba": [
        "kI_3pg_4",
        "kI_dhap_4",
        "kI_g3p_4",
        "kA_pep_4",
        "kcat_f_4",
        "kcat_r_4",
        "Km_fbp_4",
        "Km_g3p_4",
        "Km_dhap_4",
    ],
    "tpi": ["kcat_f_5", "kcat_r_5", "Km_dhap_5", "Km_g3p_5"],
    "gap": [
        "kI_adp_6",
        "kI_amp_6",
        "kI_atp_6",
        "kcat_f_6",
        "kcat_r_6",
        "Km_g3p_6",
        "Km_pi_6",
        "Km_nad_6",
        "Km_pgp_6",
        "Km_nadh_6",
    ],
    "pgk": [
        "kA_3pg_7",
        "kA_atp_7",
        "kcat_f_7",
        "kcat_r_7",
        "Km_pgp_7",
        "Km_adp_7",
        "Km_3pg_7",
        "Km_atp_7",
    ],
    "gpm": ["kI_pi_8", "kcat_f_8", "kcat_r_8", "Km_3pg_8", "Km_2pg_8"],
    "eno": ["kcat_f_9", "kcat_r_9", "Km_2pg_9", "Km_pep_9"],
}

SUBSTRATE_MAP = {
    "D-Glucose-6-phosphate": "g6p",
    "D-Glucose 6-phosphate": "g6p",
    "ATP": "atp",
    "D-fructose 6-phosphate": "f6p",
    "fructose 6-phosphate": "f6p",
    "D-fructose 1,6-bisphosphate": "fbp",
    "D-glyceraldehyde 3-phosphate": "g3p",
    "NAD+": "nad",
    "phosphate": "pi",
}
SUBSTRATE_MAP = {
    k.lower().replace(" ", "_").replace("-", "_"): v for k, v in SUBSTRATE_MAP.items()
}
REACTION_MAP = {
    "1": "pts",
    "2": "pgi",
    "3": "pfk",
    "4": "fba",
    "5": "tpi",
    "6": "gap",
    "7": "pgk",
    "8": "gpm",
    "9": "eno",
}
REVERSE_REACTION_MAP = {v: k for k, v in REACTION_MAP.items()}


class EcoliCarbonKinetics:
    """
    Mechanistic kinetic model of E. coli central carbon metabolism.

    Each method computes the reaction flux for one glycolytic enzyme.
    All methods accept (constants, C, e) dicts so the call site in
    _ode_system is uniform across all 9 reactions.

    Attributes
    ----------
    metabolites : dict
        Metabolite concentrations keyed as 'C_<name>' (e.g. 'C_pep').
    enzymes : dict
        Enzyme concentrations keyed by gene name (e.g. 'PfkB', 'GapA').
    constants : dict
        Kinetic constants (Km, kcat, kI, kA, etc.).
    stoichiometric_matrix : pd.DataFrame
        9x9 stoichiometric matrix N (metabolites x reactions).
    """
    balanced_keys = ["C_g6p", "C_f6p", "C_fbp", "C_dhap", "C_g3p", "C_pgp", "C_3pg", "C_2pg", "C_pep"]
    imbalanced_keys = ["C_atp", "C_adp", "C_amp", "C_gtp", "C_gdp", "C_nad", "C_nadh", "C_pi", "C_pyr", "C_glc"]
    enzymes_keys = ["Pgi", "PfkB", "FbaA", "TpiA", "GapA", "Pgk", "GpmA", "Eno"]
    flux_keys = ['v_pts', 'v_pgi', 'v_pfkB', 'v_fbaA', 'v_tpiA', 'v_gapA', 'v_pgk',
       'v_gpmA', 'v_eno']
    params_keys = [
    # PTS
    "kI_pyr_1",
    "kA_pep_1",
    "v_max_1",
    "Ka1_1",
    "Ka2_1",
    "Ka3_1",
    # "n_g6p_1", HARD CODED AS HILL COEFFICIENT
    "K_g6p_1",
    # PGI
    "kI_pep_2",
    "Km_g6p_2",
    "Km_f6p_2",
    "kcat_f_2",
    "kcat_r_2",
    # PFK
    "kI_f6p_3",
    "kI_fbp_3",
    "kI_gtp_3",
    "kI_pep_3",
    "kI_pi_3",
    "kA_adp_3",
    "kA_gdp_3",
    "Km_f6p_3",
    "Km_atp_3",
    "Km_fbp_3",
    "Km_adp_3",
    "kcat_f_3",
    "kcat_r_3",
    # FBA
    "kI_3pg_4",
    "kI_dhap_4",
    "kI_g3p_4",
    "kA_pep_4",
    "kcat_f_4",
    "kcat_r_4",
    "Km_fbp_4",
    "Km_g3p_4",
    "Km_dhap_4",
    # TPI
    "kcat_f_5",
    "kcat_r_5",
    "Km_dhap_5",
    "Km_g3p_5",
    # GAPDH
    "kI_adp_6",
    "kI_amp_6",
    "kI_atp_6",
    "kcat_f_6",
    "kcat_r_6",
    "Km_g3p_6",
    "Km_pi_6",
    "Km_nad_6",
    "Km_pgp_6",
    "Km_nadh_6",
    # PGK
    "kA_3pg_7",
    "kA_atp_7",
    "kcat_f_7",
    "kcat_r_7",
    "Km_pgp_7",
    "Km_adp_7",
    "Km_3pg_7",
    "Km_atp_7",
    # GPM
    "kI_pi_8",
    "kcat_f_8",
    "kcat_r_8",
    "Km_3pg_8",
    "Km_2pg_8",
    # ENO
    "kcat_f_9",
    "kcat_r_9",
    "Km_2pg_9",
    "Km_pep_9",
]
    def __init__(self, bounds_imbalanced_mets: dict, bounds_balanced_mets: dict, 
                options: dict = {"ipopt": {
                 "print_level": 0,
                 "sb": "yes",
                 "mu_strategy": "adaptive",
                 "tol": 1e-6,
                 "warm_start_init_point": "yes",
                 "warm_start_bound_push": 1e-6,
                 "warm_start_mult_bound_push": 1e-6,
             }, "print_time": 0},
                ss_tolerance: float = 1e-6):
        self.stoichiometric_matrix = self._construct_stoichiometric_matrix()
        self.bounds_imbalanced_mets = bounds_imbalanced_mets
        self.bounds_balanced_mets = bounds_balanced_mets
        self.opts = options # options for the ipopt solver, e.g. {"ipopt": {"print_level": 0}} to suppress output
        self.ss_tolerance = ss_tolerance # tolerance for steady-state constraints (can be relaxed to allow near-steady-state solutions)
        # Pre-compute bounds and default x0 once
        self._lbx = np.array(
            [bounds_balanced_mets[k][0]   for k in self.balanced_keys] +
            [bounds_imbalanced_mets[k][0] for k in self.imbalanced_keys]
        )
        self._ubx = np.array(
            [bounds_balanced_mets[k][1]   for k in self.balanced_keys] +
            [bounds_imbalanced_mets[k][1] for k in self.imbalanced_keys]
        )
        self._x0_default = np.sqrt(self._lbx * self._ubx)  # geometric mean

        # Warm-start cache: one entry per experimental condition
        self._warm_start_cache: dict[str, np.ndarray] = {}

        
        self.construct_steady_state_problem()
    def pts(self, constants: dict, C: dict, e: dict) -> float:
        """
        Phosphotransferase system (PTS).

        Reaction : glc + pep -> g6p + pyr
        Enzymes : PtsI, PtsH

        Rate is the product of three multiplicative terms:
            v = inhibition(pyr) * activation(pep) * kinetic(glc, g6p, pep/pyr)

        constants keys : kI_pyr_1, kA_pep_1, v_max_1, Ka1_1, Ka2_1, Ka3_1, n_g6p_1, K_g6p_1
        C keys : C_pyr, C_pep, C_glc, C_g6p
        e keys : (none)
        """
        kI_pyr_1 = constants["kI_pyr_1"]
        kA_pep_1 = constants["kA_pep_1"]
        v_max_1 = constants["v_max_1"]
        Ka1_1 = constants["Ka1_1"]
        Ka2_1 = constants["Ka2_1"]
        Ka3_1 = constants["Ka3_1"]
        # n_g6p_1 = constants["n_g6p_1"] 
        K_g6p_1 = constants["K_g6p_1"]

        C_pyr = C["C_pyr"]
        C_pep = C["C_pep"]
        C_glc = C["C_glc"]

        inhibition = kI_pyr_1 / (kI_pyr_1 + C_pyr)
        activation = C_pep / (C_pep + kA_pep_1)

        C_g6p = C["C_g6p"]
        num = v_max_1 * C_glc * (C_pep / C_pyr)
        den1 = Ka1_1 + Ka2_1 * (C_pep / C_pyr) + Ka3_1 * C_glc + C_glc * (C_pep / C_pyr)
        den2 = 1 + (C_g6p**4) / K_g6p_1
        kinetic = num / (den1 * den2)

        return inhibition * activation * kinetic

    def pgi(self, constants: dict, C: dict, e: dict) -> float:
        """
        Glucose-6-phosphate isomerase (PGI).
        EC: EC:5.3.1.9

        Reaction : g6p <=> f6p
        Enzyme : Pgi
        Inhibited by PEP.

        constants keys : kI_pep_2, Km_g6p_2, Km_f6p_2, kcat_f_2, kcat_r_2
        C keys : C_g6p, C_f6p, C_pep
        e keys : Pgi
        """
        kI_pep_2 = constants["kI_pep_2"]
        Km_g6p_2 = constants["Km_g6p_2"]
        Km_f6p_2 = constants["Km_f6p_2"]
        kcat_f_2 = constants["kcat_f_2"]
        kcat_r_2 = constants["kcat_r_2"]

        C_g6p = C["C_g6p"]
        C_f6p = C["C_f6p"]
        C_pep = C["C_pep"]

        h = kI_pep_2 / (kI_pep_2 + C_pep)
        num = kcat_f_2 * (C_g6p / Km_g6p_2) - kcat_r_2 * (C_f6p / Km_f6p_2)
        den = (C_g6p / Km_g6p_2 + 1) + (C_f6p / Km_f6p_2 + 1) - 1

        return e["Pgi"] * h * num / den

    def pfk(self, constants: dict, C: dict, e: dict) -> float:
        """
        Phosphofructokinase B (PFK).
        EC:2.7.1.11
        Reaction : f6p + ATP <=> fbp + ADP
        Enzyme : PfkB
        Inhibited by f6p, fbp, GTP, PEP, Pi.
        Activated by ADP, GDP.

        constants keys : kI_f6p_3, kI_fbp_3, kI_gtp_3, kI_pep_3, kI_pi_3,
                         kA_adp_3, kA_gdp_3,
                         Km_f6p_3, Km_atp_3, Km_fbp_3, Km_adp_3,
                         kcat_f_3, kcat_r_3
        C keys : C_f6p, C_atp, C_fbp, C_adp, C_gtp, C_gdp, C_pep, C_pi
        e keys : PfkB
        """
        kI_f6p_3 = constants["kI_f6p_3"]
        kI_fbp_3 = constants["kI_fbp_3"]
        kI_gtp_3 = constants["kI_gtp_3"]
        kI_pep_3 = constants["kI_pep_3"]
        kI_pi_3 = constants["kI_pi_3"]
        kA_adp_3 = constants["kA_adp_3"]
        kA_gdp_3 = constants["kA_gdp_3"]
        Km_f6p_3 = constants["Km_f6p_3"]
        Km_atp_3 = constants["Km_atp_3"]
        Km_fbp_3 = constants["Km_fbp_3"]
        Km_adp_3 = constants["Km_adp_3"]
        kcat_f_3 = constants["kcat_f_3"]
        kcat_r_3 = constants["kcat_r_3"]

        C_f6p = C["C_f6p"]
        C_atp = C["C_atp"]
        C_fbp = C["C_fbp"]
        C_adp = C["C_adp"]
        C_gtp = C["C_gtp"]
        C_gdp = C["C_gdp"]
        C_pep = C["C_pep"]
        C_pi = C["C_pi"]

        h = 1
        for conc, kI in [
            (C_f6p, kI_f6p_3),
            (C_fbp, kI_fbp_3),
            (C_gtp, kI_gtp_3),
            (C_pep, kI_pep_3),
            (C_pi, kI_pi_3),
        ]:
            h *= kI / (kI + conc)
        # for conc, kA in [(C_adp, kA_adp_3), (C_gdp, kA_gdp_3)]:
        #     h *= conc / (kA + conc)

        num = kcat_f_3 * (C_f6p / Km_f6p_3) * (C_atp / Km_atp_3) - kcat_r_3 * (
            C_fbp / Km_fbp_3
        ) * (C_adp / Km_adp_3)
        den = (
            (C_f6p / Km_f6p_3 + 1) * (C_atp / Km_atp_3 + 1)
            + (C_fbp / Km_fbp_3 + 1) * (C_adp / Km_adp_3 + 1)
            - 1
        )

        return e["PfkB"] * h * num / den

    def fba(self, constants: dict, C: dict, e: dict) -> float:
        """
        Fructose-bisphosphate aldolase A (FBA).
        EC: 4.1.2.13
        Reaction : fbp <=> g3p + dhap
        Enzyme : FbaA
        Inhibited by 3PG, DHAP, G3P.
        Activated by PEP.

        constants keys : kI_3pg_4, kI_dhap_4, kI_g3p_4, kA_pep_4,
                         kcat_f_4, kcat_r_4, Km_fbp_4, Km_g3p_4, Km_dhap_4
        C keys : C_3pg, C_dhap, C_g3p, C_pep, C_fbp
        e keys : FbaA
        """
        kI_3pg_4 = constants["kI_3pg_4"]
        kI_dhap_4 = constants["kI_dhap_4"]
        kI_g3p_4 = constants["kI_g3p_4"]
        kA_pep_4 = constants["kA_pep_4"]
        kcat_f_4 = constants["kcat_f_4"]
        kcat_r_4 = constants["kcat_r_4"]
        Km_fbp_4 = constants["Km_fbp_4"]
        Km_g3p_4 = constants["Km_g3p_4"]
        Km_dhap_4 = constants["Km_dhap_4"]

        C_3pg = C["C_3pg"]
        C_dhap = C["C_dhap"]
        C_g3p = C["C_g3p"]
        C_pep = C["C_pep"]
        C_fbp = C["C_fbp"]

        h = 1
        for conc, kI in [(C_3pg, kI_3pg_4), (C_dhap, kI_dhap_4), (C_g3p, kI_g3p_4)]:
            h *= kI / (kI + conc)
        h *= C_pep / (kA_pep_4 + C_pep)

        num = kcat_f_4 * (C_fbp / Km_fbp_4) - kcat_r_4 * (C_g3p / Km_g3p_4) * (
            C_dhap / Km_dhap_4
        )
        den = (
            (C_fbp / Km_fbp_4 + 1)
            + (C_g3p / Km_g3p_4 + 1) * (C_dhap / Km_dhap_4 + 1)
            - 1
        )

        return e["FbaA"] * h * num / den

    def tpi(self, constants: dict, C: dict, e: dict) -> float:
        """
        Triosephosphate isomerase (TPI).
        EC: 5.3.1.1
        Reaction : dhap <=> g3p
        Enzyme : TpiA
        No allosteric regulation.

        constants keys : kcat_f_5, kcat_r_5, Km_dhap_5, Km_g3p_5
        C keys : C_dhap, C_g3p
        e keys : TpiA
        """
        kcat_f_5 = constants["kcat_f_5"]
        kcat_r_5 = constants["kcat_r_5"]
        Km_dhap_5 = constants["Km_dhap_5"]
        Km_g3p_5 = constants["Km_g3p_5"]

        C_dhap = C["C_dhap"]
        C_g3p = C["C_g3p"]

        num = kcat_f_5 * (C_dhap / Km_dhap_5) - kcat_r_5 * (C_g3p / Km_g3p_5)
        den = (C_dhap / Km_dhap_5 + 1) + (C_g3p / Km_g3p_5 + 1) - 1

        return e["TpiA"] * num / den

    def gap(self, constants: dict, C: dict, e: dict) -> float:
        """
        Glyceraldehyde-3-phosphate dehydrogenase A (GAP).
        EC: 1.2.1.12
        Reaction : g3p + Pi + NAD+ <=> pgp + NADH # pgp -> 1,3 biphosphoglycerate
        Enzyme : GapA
        Inhibited by ADP, AMP, ATP.

        constants keys : kI_adp_6, kI_amp_6, kI_atp_6,
                         kcat_f_6, kcat_r_6,
                         Km_g3p_6, Km_pi_6, Km_nad_6, Km_pgp_6, Km_nadh_6
        C keys : C_adp, C_amp, C_atp, C_g3p, C_pi, C_nad, C_pgp, C_nadh
        e keys : GapA
        """
        kI_adp_6 = constants["kI_adp_6"]
        kI_amp_6 = constants["kI_amp_6"]
        kI_atp_6 = constants["kI_atp_6"]
        kcat_f_6 = constants["kcat_f_6"]
        kcat_r_6 = constants["kcat_r_6"]
        Km_g3p_6 = constants["Km_g3p_6"]
        Km_pi_6 = constants["Km_pi_6"]
        Km_nad_6 = constants["Km_nad_6"]
        Km_pgp_6 = constants["Km_pgp_6"]
        Km_nadh_6 = constants["Km_nadh_6"]

        C_adp = C["C_adp"]
        C_amp = C["C_amp"]
        C_atp = C["C_atp"]
        C_g3p = C["C_g3p"]
        C_pi = C["C_pi"]
        C_nad = C["C_nad"]
        C_pgp = C["C_pgp"]
        C_nadh = C["C_nadh"]

        h = 1
        for conc, kI in [(C_adp, kI_adp_6), (C_amp, kI_amp_6), (C_atp, kI_atp_6)]:
            h *= kI / (kI + conc)

        num = kcat_f_6 * (C_g3p / Km_g3p_6) * (C_pi / Km_pi_6) * (
            C_nad / Km_nad_6
        ) - kcat_r_6 * (C_pgp / Km_pgp_6) * (C_nadh / Km_nadh_6)
        den = (
            (C_g3p / Km_g3p_6 + 1) * (C_pi / Km_pi_6 + 1) * (C_nad / Km_nad_6 + 1)
            + (C_pgp / Km_pgp_6 + 1) * (C_nadh / Km_nadh_6 + 1)
            - 1
        )

        return e["GapA"] * h * num / den

    def pgk(self, constants: dict, C: dict, e: dict) -> float:
        """
        Phosphoglycerate kinase (PGK).
        EC: 2.7.2.3
        Reaction : pgp + ADP <=> 3pg + ATP
        Enzyme : Pgk
        Activated by 3PG and ATP.

        constants keys : kA_3pg_7, kA_atp_7,
                         kcat_f_7, kcat_r_7,
                         Km_pgp_7, Km_adp_7, Km_3pg_7, Km_atp_7
        C keys : C_pgp, C_adp, C_3pg, C_atp
        e keys : Pgk
        """
        kA_3pg_7 = constants["kA_3pg_7"]
        kA_atp_7 = constants["kA_atp_7"]
        kcat_f_7 = constants["kcat_f_7"]
        kcat_r_7 = constants["kcat_r_7"]
        Km_pgp_7 = constants["Km_pgp_7"]
        Km_adp_7 = constants["Km_adp_7"]
        Km_3pg_7 = constants["Km_3pg_7"]
        Km_atp_7 = constants["Km_atp_7"]

        C_pgp = C["C_pgp"]
        C_adp = C["C_adp"]
        C_3pg = C["C_3pg"]
        C_atp = C["C_atp"]

        h = 1
        for conc, kA in [(C_3pg, kA_3pg_7), (C_atp, kA_atp_7)]:
            h *= conc / (kA + conc)

        num = kcat_f_7 * (C_pgp / Km_pgp_7) * (C_adp / Km_adp_7) - kcat_r_7 * (
            C_3pg / Km_3pg_7
        ) * (C_atp / Km_atp_7)
        den = (
            (C_pgp / Km_pgp_7 + 1) * (C_adp / Km_adp_7 + 1)
            + (C_3pg / Km_3pg_7 + 1) * (C_atp / Km_atp_7 + 1)
            - 1
        )

        return e["Pgk"] * h * num / den

    def gpm(self, constants: dict, C: dict, e: dict) -> float:
        """
        Phosphoglycerate mutase A (GPM).
        EC: 5.4.2.11
        Reaction : 3pg <=> 2pg
        Enzyme : GpmA
        Inhibited by Pi.

        constants keys : kI_pi_8, kcat_f_8, kcat_r_8, Km_3pg_8, Km_2pg_8
        C keys : C_3pg, C_2pg, C_pi
        e keys : GpmA
        """
        kI_pi_8 = constants["kI_pi_8"]
        kcat_f_8 = constants["kcat_f_8"]
        kcat_r_8 = constants["kcat_r_8"]
        Km_3pg_8 = constants["Km_3pg_8"]
        Km_2pg_8 = constants["Km_2pg_8"]

        C_3pg = C["C_3pg"]
        C_2pg = C["C_2pg"]
        C_pi = C["C_pi"]

        h = kI_pi_8 / (kI_pi_8 + C_pi)
        num = kcat_f_8 * (C_3pg / Km_3pg_8) - kcat_r_8 * (C_2pg / Km_2pg_8)
        den = (C_3pg / Km_3pg_8 + 1) + (C_2pg / Km_2pg_8 + 1) - 1

        return e["GpmA"] * h * num / den

    def eno(self, constants: dict, C: dict, e: dict) -> float:
        """
        Enolase (ENO).
        EC: 4.2.1.11
        Reaction : 2pg <=> pep
        Enzyme : Eno
        No allosteric regulation.

        constants keys : kcat_f_9, kcat_r_9, Km_2pg_9, Km_pep_9
        C keys : C_2pg, C_pep
        e keys : Eno
        """
        kcat_f_9 = constants["kcat_f_9"]
        kcat_r_9 = constants["kcat_r_9"]
        Km_2pg_9 = constants["Km_2pg_9"]
        Km_pep_9 = constants["Km_pep_9"]

        C_2pg = C["C_2pg"]
        C_pep = C["C_pep"]

        num = kcat_f_9 * (C_2pg / Km_2pg_9) - kcat_r_9 * (C_pep / Km_pep_9)
        den = (C_2pg / Km_2pg_9 + 1) + (C_pep / Km_pep_9 + 1) - 1

        return e["Eno"] * num / den


    def compute_fluxes(self, C, e, constants) -> np.ndarray:
        """
        Evaluate all 9 reaction fluxes at metabolite concentrations C.

        Parameters
        ----------
        C : dict
            Full concentration dict (balanced + imbalanced metabolites).
        e : dict
            Enzyme concentrations.
        constants : dict
            Kinetic parameters.

        Returns
        -------
        list, shape (9,)
            Flux vector [v1, ..., v9].
        """
        
        return [
            self.pts(constants, C, e),
            self.pgi(constants, C, e),
            self.pfk(constants, C, e),
            self.fba(constants, C, e),
            self.tpi(constants, C, e),
            self.gap(constants, C, e),
            self.pgk(constants, C, e),
            self.gpm(constants, C, e),
            self.eno(constants, C, e),
        ]
        
        
    def build_parameter_estimation_nlp(self,
                                    conditions_enzymes : list[dict],
                                    conditions_obs     : list[pd.Series],
                                    bounds_params      : dict,
                                    ipopt_opts      : dict
                                    ):
        """
        Extends the solve_steady_state objective (||S@v||²) to also cover
        parameter estimation, by adding theta as decision variables and a
        data-fitting term. Pure box-constrained NLP — no equality constraints.

        Objective = Σ_k [ ss_weight * ||S @ v_k(C_k, θ, e_k)||²
                        +             ||predicted_k - observed_k||² ]

        Parameters
        ----------
        ss_weight : weight on the steady-state residual term relative to
                    the data-fitting term. Start with 1.0 and increase if
                    the solver finds solutions far from steady state.
        """
        N     = len(conditions_enzymes)
        n_par = len(self.params_keys)
        S     = ca.DM(self.stoichiometric_matrix.values)

        # Decision variables: kinetic parameters + concentrations per condition
        theta = ca.SX.sym("theta", n_par)
        C_all = [ca.SX.sym(f"C_{k}", len(self.balanced_keys) + len(self.imbalanced_keys))
                for k in range(N)]

        constants = {key: theta[i] for i, key in enumerate(self.params_keys)}
        obj = ca.SX(0)
        ss_list = [] # steady-state constraints
        for k in range(N):
            C_k     = C_all[k]
            C_bal   = {key: C_k[i]                           for i, key in enumerate(self.balanced_keys)}
            C_imbal = {key: C_k[i + len(self.balanced_keys)] for i, key in enumerate(self.imbalanced_keys)}
            enzymes_k = {key: float(conditions_enzymes[k][key]) for key in self.enzymes_keys}

            fluxes_k = self.compute_fluxes({**C_bal, **C_imbal}, enzymes_k, constants)

            # steady-state residual term
            ss_list.append(S @ ca.vertcat(*fluxes_k))

            # Term 2: fit to observed concentrations and/or fluxes
            pred = {**C_bal,
                    **{self.flux_keys[i]: fluxes_k[i] for i in range(len(fluxes_k))}}
            for var_name in conditions_obs[k].index:
                obs_val = conditions_obs[k][var_name]
                if not np.isnan(float(obs_val)) and var_name in pred:
                    obj += ((pred[var_name] - float(obs_val)) / (abs(float(obs_val)) + 1e-12)) ** 2

        # Assemble variable vector and bounds
        x        = ca.vertcat(theta, *C_all)
        lb_theta = np.array([bounds_params[k][0] for k in self.params_keys])
        ub_theta = np.array([bounds_params[k][1] for k in self.params_keys])
        lb_x     = np.concatenate([lb_theta] + [self._lbx] * N)
        ub_x     = np.concatenate([ub_theta] + [self._ubx] * N)

        # Auto-size bounds from the actual constraint vector — avoids any mismatch
        g    = ca.vertcat(*ss_list)
        n_g  = g.shape[0]
        lbg  = - np.ones(n_g) * self.ss_tolerance
        ubg  = np.ones(n_g) * self.ss_tolerance
        print(f"[DEBUG] {N} conditions {n_g // N} constraints/condition = {n_g} total")

        if ipopt_opts is None:
            ipopt_opts = {
                "ipopt": {
                    "print_level": 0, "sb": "yes",
                    "mu_strategy": "adaptive",
                    "tol": 1e-4, "max_iter": 3000,
                },
                "print_time": 0,
            }
        else: 
            ipopt_opts = ipopt_opts

        solver = ca.nlpsol("pe_solver", "ipopt",
                        {"x": x, "f": obj, "g": g}, ipopt_opts)
        return solver, lb_x, ub_x, lbg, ubg


    def construct_steady_state_problem(self):
        """
        Solves the feasibility problem:
            min || S @ v(C_balanced, C_imbalanced, e; params) - b ||_2^2
            s.t.
                lb_balanced <= C_balanced <= ub_balanced
                lb_imbalanced <= C_imbalanced <= ub_imbalanced
        where b is the cell needs.
        """
        n_variables = len(self.balanced_keys) + len(self.imbalanced_keys)

        C_sym = ca.SX.sym("C", n_variables)
        k_sym = ca.SX.sym("k", len(self.params_keys))
        e_sym = ca.SX.sym("e", len(self.enzymes_keys))
        b_sym = ca.SX.sym("b", self.stoichiometric_matrix.shape[0])  # = 9

        C_balanced   = {key: C_sym[i] for i, key in enumerate(self.balanced_keys)}
        C_imbalanced = {key: C_sym[i + len(self.balanced_keys)] for i, key in enumerate(self.imbalanced_keys)}
        constants    = {key: k_sym[i] for i, key in enumerate(self.params_keys)}
        enzymes      = {key: e_sym[i] for i, key in enumerate(self.enzymes_keys)}

        fluxes = self.compute_fluxes({**C_balanced, **C_imbalanced}, enzymes, constants)
        S = ca.DM(self.stoichiometric_matrix.values)

        f = ca.sumsqr(S @ ca.vertcat(*fluxes) - b_sym)
        nlp = {
            "x": C_sym,
            "f": f,                      # feasibility — no objective
            "p": ca.vertcat(k_sym, e_sym, b_sym),
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp, self.opts)
        
    
        
        
    def solve_steady_state(self, 
                           enzymes            : dict, # dictionary of enzyme concentrations
                           cell_needs         : dict, # dictionary of cell needs (b vector)
                           kinetic_params     : dict, # dictionary of kinetic parameters
                           condition_key  : str = None
                           ):
        """
        Simulate the kinetic system to steady state using CasADi/CVODES.

        Solves the algebraic equation system:
            S @ v(C, e; params) - b = 0
        where S is the stoichiometric matrix and v is the flux vector computed by
        compute_fluxes().
        For this the following nlp problem is solved:
        min || S @ v(C_balanced, C_imbalanced, e; params) - b ||_2^2
        s.t. 
            xlb <= C_balanced <= xub
            ulb <= C_imbalanced <= uub
        
        then we return the C_balanced that minimizes the norm and the corresponding fluxes v.
        
        
        Parameters
        ----------
        params : dict
            Kinetic parameters (Km, kcat, kI, kA, etc.).
        enzymes : dict
            Enzyme concentrations keyed by gene name (e.g. 'PfkB', 'GapA').
        metabolites : dict
            Initial metabolite concentrations keyed as 'C_<name>' (e.g. 'C_pep').
        opts : dict, optional
            Simulation options (e.g. tolerances, max iterations).

        Returns
        -------
        pd.DataFrame
            Final metabolite concentrations keyed by name.
        """
        
        # 
        p = np.array(
            [kinetic_params[key] for key in self.params_keys] +
            [enzymes[key]        for key in self.enzymes_keys] +
            [cell_needs[key]     for key in self.balanced_keys]
        )
        
        # Use cached warm-start for this condition, or fall back to geometric mean
        x0 = self._warm_start_cache.get(condition_key, self._x0_default)

        sol = self.solver(
                    x0=x0,
                    lbx=self._lbx, ubx=self._ubx,
                    p=p,
                )
        

        C_opt = sol["x"].full().flatten()
        self._warm_start_cache[condition_key] = C_opt  # save for next call

        C_balanced_opt   = {key: C_opt[i] for i, key in enumerate(self.balanced_keys)}
        C_imbalanced_opt = {key: C_opt[i + len(self.balanced_keys)]
                            for i, key in enumerate(self.imbalanced_keys)}

        fluxes_opt = self.compute_fluxes(
            {**C_balanced_opt, **C_imbalanced_opt}, enzymes, kinetic_params
        )
        dict_v_opt = {self.flux_keys[i]: fluxes_opt[i] for i in range(len(fluxes_opt))}
        return pd.DataFrame({**C_balanced_opt, **dict_v_opt}, index=[0]), sol['f'].full().item()  # return both concentrations and final objective value (residual norm)
    
    def _construct_stoichiometric_matrix(self) -> pd.DataFrame:
        """
        Build the 9x9 stoichiometric matrix N (metabolites x reactions).

        Rows : 2pg, 3pg, dhap, f6p, fbp, g3p, g6p, pep, pgp
        Columns: pts, pgi, pfk, fba, tpi, gap, pgk, gpm, eno

        Returns
        -------
        pd.DataFrame
            N with metabolite names as index and reaction names as columns.
        """
        N = np.array(
            [
                # pts  pgi  pfk  fba  tpi  gap  pgk  gpm  eno
                [0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , -1],  # 2pg
                [0 , 0 , 0 , 0 , 0 , 0 , 1 , -1, 0 ],  # 3pg
                [0 , 0 , 0 , 1 , -1, 0 , 0 , 0 , 0 ],  # dhap
                [0 , 1 , -1, 0 , 0 , 0 , 0 , 0 , 0 ],  # f6p
                [0 , 0 , 1 , -1, 0 , 0 , 0 , 0 , 0 ],  # fbp
                [0 , 0 , 0 , 1 , 1 , -1, 0 , 0 , 0 ],  # g3p
                [1 , -1, 0 , 0 , 0 , 0 , 0 , 0 , 0 ],  # g6p
                [-1, 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 ],  # pep
                [0 , 0 , 0 , 0 , 0 , 1 , -1, 0 , 0 ],  # pgp
            ]
        )

        metabolites = ["2pg", "3pg", "dhap", "f6p", "fbp", "g3p", "g6p", "pep", "pgp"]
        reactions = ["pts", "pgi", "pfk", "fba", "tpi", "gap", "pgk", "gpm", "eno"]

        return pd.DataFrame(N, index=metabolites, columns=reactions)

def load_params(csv_path: str) -> dict:
    """
    Load kinetic parameters from an csv file and return as a dictionary.
    """

    param_df = pd.read_csv(csv_path)
    param_dict = dict.fromkeys(ALL_PARAMS, None)

    # hardcoded parameters from literature
    # (Kadir et al., 2010)
    param_dict["v_max_1"] = 25.739  # mmol/gDW/h
    param_dict["Ka1_1"] = 1  # mM
    param_dict["Ka2_1"] = 0.01  # mM
    param_dict["Ka3_1"] = 1  # mM
    # param_dict["n_g6p_1"] = 4  # unitless HARD CODED AS HILL COEFCIINET
    param_dict["K_g6p_1"] = 0.5  # mM
    # data from Brenda
    for _, row in param_df.iterrows():

        parameter_name = row["parameter"]
        reaction = row["reaction"]
        substrate = row["substrate"]
        approximate_name = substrate.lower().replace(" ", "_").replace("-", "_")
        if approximate_name in SUBSTRATE_MAP:
            substrate = approximate_name
        value = row["value"]
        if (
            parameter_name == "Km"
            and substrate in SUBSTRATE_MAP
            and reaction in REVERSE_REACTION_MAP
        ):
            param_key = (
                "Km_" + SUBSTRATE_MAP[substrate] + "_" + REVERSE_REACTION_MAP[reaction]
            )
        if param_dict[param_key] == None:
            param_dict[param_key] = [value]
        else:
            param_dict[param_key].append(value)

    return param_dict



    
# %%
if __name__ == "__main__":
    # Example usage
    constants = {
        # v1: PTS (Sistema de fosfotransferasa)
        "kI_pyr_1": 0.5,  # Inhibición por piruvato
        "kA_pep_1": 0.3,  # Afinidad/Activación por PEP
        "v_max_1": 25.739,  # Flujo máximo típico en fase exponencial
        "Ka1_1": 1.0,  # Parámetros de la ecuación compleja del PTS
        "Ka2_1": 0.01,
        "Ka3_1": 1.0,
        "n_g6p_1": 4,  # Cooperatividad/Regulación por G6P
        "K_g6p_1": 0.5,
        # v2: PGI (Glucosa-6-fosfato isomerasa)
        "kI_pep_2": 0.12,  # Fuerte inhibición por PEP
        "Km_g6p_2": 0.48,
        "Km_f6p_2": 0.19,
        "kcat_f_2": 1475.0,  # Reacción muy rápida
        "kcat_r_2": 1000.0,
        # v3: PFK (Fosfofructoquinasa) - Enzima regulatoria clave
        "kI_f6p_3": 0.9,
        "kI_fbp_3": 2.0,
        "kI_gtp_3": 0.8,
        "kI_pep_3": 0.5,  # Inhibidor alostérico principal
        "kI_pi_3": 1.5,
        "kA_adp_3": 0.12,  # Activador alostérico
        "kA_gdp_3": 0.15,
        "Km_f6p_3": 0.16,
        "Km_atp_3": 0.12,
        "Km_fbp_3": 0.5,
        "Km_adp_3": 0.2,
        "kcat_f_3": 580.0,
        "kcat_r_3": 100.0,
        # v4: FBA (Fructosa-bisfosfato aldolasa)
        "kI_3pg_4": 2.0,
        "kI_dhap_4": 0.08,
        "kI_g3p_4": 0.1,
        "kA_pep_4": 1.5,
        "kcat_f_4": 95.0,
        "kcat_r_4": 150.0,
        "Km_fbp_4": 0.3,
        "Km_g3p_4": 0.4,
        "Km_dhap_4": 2.0,
        # v5: TPI (Triosafosfato isomerasa)
        "kcat_f_5": 4300.0,  # Cercana a la perfección catalítica
        "kcat_r_5": 2400.0,
        "Km_dhap_5": 0.61,
        "Km_g3p_5": 1.2,
        # v6: GAPDH (Gliceraldehído-3-fosfato deshidrogenasa)
        "kI_adp_6": 0.8,
        "kI_amp_6": 1.0,
        "kI_atp_6": 0.2,
        "kcat_f_6": 118.0,
        "kcat_r_6": 10.0,
        "Km_g3p_6": 0.21,
        "Km_pi_6": 0.29,
        "Km_nad_6": 0.09,
        "Km_pgp_6": 0.01,
        "Km_nadh_6": 0.06,
        # v7: PGK (Fosfoglicerato quinasa)
        "kA_3pg_7": 0.5,
        "kA_atp_7": 0.3,
        "kcat_f_7": 1150.0,
        "kcat_r_7": 40.0,
        "Km_pgp_7": 0.05,
        "Km_adp_7": 0.1,
        "Km_3pg_7": 0.53,
        "Km_atp_7": 0.3,
        # v8: GPM (Fosfoglicerato mutasa)
        "kI_pi_8": 10.0,  # Inhibición débil por Pi
        "kcat_f_8": 540.0,
        "kcat_r_8": 120.0,
        "Km_3pg_8": 0.2,
        "Km_2pg_8": 1.4,
        # v9: ENO (Enolasa)
        "kcat_f_9": 550.0,
        "kcat_r_9": 210.0,
        "Km_2pg_9": 0.1,
        "Km_pep_9": 0.5,
    }
    metabolites_balanced = {
        # balanced
        "C_2pg": 2.84e-5,
        "C_3pg": 4.52e-4,
        "C_dhap": 4.48e-6,
        "C_f6p": 3.53e-5,
        "C_fbp": 3.42e-5,
        "C_g3p": 4.23e-7,
        "C_g6p": 2.45e-4,
        "C_pep": 1.02e-4,
        "C_pgp": 2.56e-6
    }
    metabolites_imbalanced = {
        "C_atp": 0.1,
        "C_adp": 0.01,
        "C_amp": 0.01,
        "C_gdp": 0.01,
        "C_glc": 0.01,
        "C_gtp": 0.01,
        "C_nad": 0.01,
        "C_nadh": 0.01,
        "C_pi": 0.01,
        "C_pyr": 0.01,
    }
    enzymes = {
        "Pgi": 0.01,
        "PfkB": 0.1,
        "FbaA": 0.1,
        "TpiA": 0.1,
        "GapA": 0.1,
        "Pgk": 0.01,
        "GpmA": 0.1,
        "Eno": 0.1,
    }
    model = EcoliCarbonKinetics(
        bounds_imbalanced_mets = {
            "C_atp": (0.01, 10),
            "C_adp": (0.001, 10),
            "C_amp": (0.001, 10),
            "C_gdp": (0.001, 10),
            "C_glc": (0.001, 10.0),
            "C_gtp": (0.001, 10),
            "C_nad": (0.001, 10),
            "C_nadh": (0.001, 10),
            "C_pi": (0.001, 10.0),
            "C_pyr": (0.001, 10.0),
        },
        bounds_balanced_mets= {
            "C_2pg": (1e-6, 1),
            "C_3pg": (1e-6, 1e-3),
            "C_dhap": (1e-6, 1e-3),
            "C_f6p": (1e-6, 1e-3),
            "C_fbp": (1e-6, 1e-3),
            "C_g3p": (1e-6, 1e-3),
            "C_g6p": (1e-6, 1e-3),
            "C_pep": (1e-6, 1e-3),
            "C_pgp": (1e-6, 1e-3)
        },
    )
    ti = time.time()

    b = {
        "C_2pg": 0.0,
        "C_3pg": 0.0,
        "C_dhap": 0.0,
        "C_f6p": 0.0,
        "C_fbp": 0.0,
        "C_g3p": 0.0,
        "C_g6p": 0.0,
        "C_pep": 0.0,
        "C_pgp": 0.0
    }
    solved_concentrations = model.solve_steady_state(
        enzymes=enzymes,
        kinetic_params=constants,   
        cell_needs=b,     
    )
    print(time.time() - ti)
    # %%