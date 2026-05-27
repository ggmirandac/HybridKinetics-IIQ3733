"""
kinetics.py - Mechanistic kinetic model of E. coli glycolysis (IIQ3733).

Models 9 enzymatic reactions from glucose uptake (PTS) to PEP production (Eno),
covering the core of E. coli central carbon metabolism.
Each reaction is modeled with Noor et al. thermodynamic kinetics:
    v = E * kcat_f * kappa * gamma
where kappa is the saturation term and gamma = 1 - exp(-dGr/RT) is the
thermodynamic driving force, except for the PTS which retains its original
empirical formulation.
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
    # PTS (empirical formulation — unchanged)
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

    Each method computes the reaction flux for one glycolytic enzyme using
    the Noor et al. thermodynamic kinetics formulation:

        v = E * kcat_f * kappa * gamma

    where kappa is the substrate-saturation term and
    gamma = 1 - exp(-dGr / RT) is the thermodynamic driving force.
    The PTS reaction retains its original empirical form.

    Attributes
    ----------
    metabolites : dict
        Metabolite concentrations keyed as 'C_<name>' (e.g. 'C_pep').
    enzymes : dict
        Enzyme concentrations keyed by gene name (e.g. 'PfkB', 'GapA').
    constants : dict
        Kinetic constants (Ks, Kp, kcat_f, etc.).
    stoichiometric_matrix : pd.DataFrame
        9x9 stoichiometric matrix N (metabolites x reactions).
    """
    balanced_keys   = ["C_g6p", "C_f6p", "C_fbp", "C_dhap", "C_g3p",
                       "C_pgp", "C_3pg", "C_2pg", "C_pep"]
    imbalanced_keys = ["C_atp", "C_adp", "C_amp", "C_gtp", "C_gdp",
                       "C_nad", "C_nadh", "C_pi", "C_pyr", "C_glc"]
    enzymes_keys    = ["Pgi", "PfkB", "FbaA", "TpiA", "GapA", "Pgk", "GpmA", "Eno"]
    flux_keys       = ["v_pts", "v_pgi", "v_pfkA", "v_fbaA", "v_tpiA",
                       "v_gapA", "v_pgk", "v_gpmA", "v_eno"]
    params_keys = [
        # PTS
        "v_max_1",
        "Ka1_1",
        "Ka2_1",
        "Ka3_1",
        "K_g6p_1",
        # PGI
        "Ks_g6p_pgi",
        "Kp_f6p_pgi",
        "kcat_f_2",
        # PFK
        "Ks_f6p_3",
        "Ks_atp_3",
        "Kp_fbp_3",
        "Kp_adp_3",
        "kcat_f_3",
        # FBA
        "Ks_fbp_4",
        "Kp_g3p_4",
        "Kp_dhap_4",
        "kcat_f_4",
        # TPI
        "kcat_f_5",
        "Ks_dhap_5",
        "Kp_g3p_5",
        # GAPDH
        "kcat_f_6",
        "Ks_g3p_6",
        "Ks_pi_6",
        "Ks_nad_6",
        "Kp_pgp_6",
        "Kp_nadh_6",
        # PGK
        "kcat_f_7",
        "Ks_pgp_7",
        "Ks_adp_7",
        "Ks_3pg_7",
        "Ks_atp_7",
        # GPM
        "kcat_f_8",
        "Ks_3pg_8",
        "Ks_2pg_8",
        # ENO
        "kcat_f_9",
        "Ks_2pg_9",
        "Ks_pep_9",
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
                 ss_tolerance: float = 1e-6,
                 R_ct: float = 8.314,   # J/(mol·K)
                 T: float = 310.15,     # K  (37 °C)
                 ):
        self.stoichiometric_matrix = self._construct_stoichiometric_matrix()
        self.bounds_imbalanced_mets = bounds_imbalanced_mets
        self.bounds_balanced_mets   = bounds_balanced_mets
        self.construct_dGR()
        self.R = R_ct
        self.T = T

        self.opts         = options
        self.ss_tolerance = ss_tolerance

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

    def construct_dGR(self):
        # Standard Gibbs free energy changes for each reaction (kJ/mol)
        # Source: https://equilibrator.weizmann.ac.il/
        dG0_prime = {
            "pgi": 2.6,    # kJ/mol
            "pfk": -17.8,  # kJ/mol
            "fba": 23.2,   # kJ/mol
            "tpi": -5.6,   # kJ/mol
            "gap": 1.2,    # kJ/mol
            "pgk": 19.5,   # kJ/mol
            "gpm": -4.5,   # kJ/mol
            "eno": -3.8,   # kJ/mol
        }
        # Convert to J/mol
        self.D_gR_circ = {rxn: dG0_prime[rxn] * 1000 for rxn in dG0_prime}

    # ------------------------------------------------------------------
    # Flux methods
    # ------------------------------------------------------------------

    def pts(self, constants: dict, C: dict, e: dict) -> float:
        """
        Phosphotransferase system (PTS).

        Reaction : glc + pep -> g6p + pyr
        Enzymes  : PtsI, PtsH  (v_max_1 absorbs enzyme concentration)

        Empirical formulation — no thermodynamic gamma term.

        constants keys : v_max_1, Ka1_1, Ka2_1, Ka3_1, K_g6p_1
        C keys         : C_pyr, C_pep, C_glc, C_g6p
        e keys         : (none — v_max already incorporates enzyme level)
        """
        v_max_1  = constants["v_max_1"]
        Ka1_1    = constants["Ka1_1"]
        Ka2_1    = constants["Ka2_1"]
        Ka3_1    = constants["Ka3_1"]
        K_g6p_1  = constants["K_g6p_1"]

        C_pyr = C["C_pyr"]
        C_pep = C["C_pep"]
        C_glc = C["C_glc"]
        C_g6p = C["C_g6p"]

        num     = v_max_1 * C_glc * (C_pep / C_pyr)
        den1    = Ka1_1 + Ka2_1 * (C_pep / C_pyr) + Ka3_1 * C_glc + C_glc * (C_pep / C_pyr)
        den2    = 1 + (C_g6p ** 4) / K_g6p_1   # Hill inhibition by g6p, n=4 hard-coded
        kinetic = num / (den1 * den2)

        return kinetic

    def pgi(self, constants: dict, C: dict, e: dict) -> float:
        """
        Glucose-6-phosphate isomerase (PGI).
        EC: 5.3.1.9

        Reaction : g6p <=> f6p
        Enzyme   : Pgi

        Noor formulation — thermodynamic driving force via gamma.

        constants keys : Ks_g6p_pgi, Kp_f6p_pgi, kcat_f_2
        C keys         : C_g6p, C_f6p
        e keys         : Pgi
        """
        Ks_g6p_pgi = constants["Ks_g6p_pgi"]
        Kp_f6p_pgi = constants["Kp_f6p_pgi"]
        kcat_f_2   = constants["kcat_f_2"]

        C_g6p = C["C_g6p"]
        C_f6p = C["C_f6p"]

        kappa = (C_g6p / Ks_g6p_pgi) / (1 + C_g6p / Ks_g6p_pgi + C_f6p / Kp_f6p_pgi)
        dGr   = self.D_gR_circ["pgi"] + self.R * self.T * np.log(C_f6p / C_g6p)
        gamma = 1 - np.exp(-dGr / (self.R * self.T))

        return e["Pgi"] * kcat_f_2 * kappa * gamma

    def pfk(self, constants: dict, C: dict, e: dict) -> float:
        """
        Phosphofructokinase B (PFK).
        EC: 2.7.1.11

        Reaction : f6p + atp <=> fbp + adp
        Enzyme   : PfkB

        Noor formulation — thermodynamic driving force via gamma.
        Allosteric modifiers (original formulation) removed; directionality
        is now encoded entirely in dGr.

        constants keys : Ks_f6p_3, Ks_atp_3, Kp_fbp_3, Kp_adp_3, kcat_f_3
        C keys         : C_f6p, C_atp, C_fbp, C_adp
        e keys         : PfkB
        """
        Ks_f6p_3 = constants["Ks_f6p_3"]
        Ks_atp_3 = constants["Ks_atp_3"]
        Kp_fbp_3 = constants["Kp_fbp_3"]
        Kp_adp_3 = constants["Kp_adp_3"]
        kcat_f_3 = constants["kcat_f_3"]

        C_f6p = C["C_f6p"]
        C_atp = C["C_atp"]
        C_fbp = C["C_fbp"]
        C_adp = C["C_adp"]

        prod_subs = (C_f6p / Ks_f6p_3) * (C_atp / Ks_atp_3)
        prod_prods = (C_fbp / Kp_fbp_3) * (C_adp / Kp_adp_3)
        kappa = prod_subs / (1 + prod_subs + prod_prods)

        dGr   = self.D_gR_circ["pfk"] + self.R * self.T * np.log(
            (C_fbp * C_adp) / (C_f6p * C_atp)
        )
        gamma = 1 - np.exp(-dGr / (self.R * self.T))

        return e["PfkB"] * kcat_f_3 * kappa * gamma

    def fba(self, constants: dict, C: dict, e: dict) -> float:
        """
        Fructose-bisphosphate aldolase A (FBA).
        EC: 4.1.2.13

        Reaction : fbp <=> g3p + dhap
        Enzyme   : FbaA

        Noor formulation — thermodynamic driving force via gamma.
        Allosteric modifiers (original formulation) removed.

        constants keys : Ks_fbp_4, Kp_g3p_4, Kp_dhap_4, kcat_f_4
        C keys         : C_fbp, C_g3p, C_dhap
        e keys         : FbaA
        """
        Ks_fbp_4  = constants["Ks_fbp_4"]
        Kp_g3p_4  = constants["Kp_g3p_4"]
        Kp_dhap_4 = constants["Kp_dhap_4"]
        kcat_f_4  = constants["kcat_f_4"]

        C_dhap = C["C_dhap"]
        C_g3p  = C["C_g3p"]
        C_fbp  = C["C_fbp"]

        prod_subs  = C_fbp / Ks_fbp_4
        prod_prods = (C_g3p / Kp_g3p_4) * (C_dhap / Kp_dhap_4)
        kappa = prod_subs / (1 + prod_subs + prod_prods)

        dGr   = self.D_gR_circ["fba"] + self.R * self.T * np.log(
            (C_g3p * C_dhap) / C_fbp
        )
        gamma = 1 - np.exp(-dGr / (self.R * self.T))

        return e["FbaA"] * kcat_f_4 * kappa * gamma

    def tpi(self, constants: dict, C: dict, e: dict) -> float:
        """
        Triosephosphate isomerase (TPI).
        EC: 5.3.1.1

        Reaction : dhap <=> g3p
        Enzyme   : TpiA

        Noor formulation — thermodynamic driving force via gamma.

        constants keys : kcat_f_5, Ks_dhap_5, Kp_g3p_5
        C keys         : C_dhap, C_g3p
        e keys         : TpiA
        """
        kcat_f_5  = constants["kcat_f_5"]
        Ks_dhap_5 = constants["Ks_dhap_5"]
        Kp_g3p_5  = constants["Kp_g3p_5"]

        C_dhap = C["C_dhap"]
        C_g3p  = C["C_g3p"]

        prod_subs  = C_dhap / Ks_dhap_5
        prod_prods = C_g3p  / Kp_g3p_5
        kappa = prod_subs / (1 + prod_subs + prod_prods)

        dGr   = self.D_gR_circ["tpi"] + self.R * self.T * np.log(C_g3p / C_dhap)
        gamma = 1 - np.exp(-dGr / (self.R * self.T))

        return e["TpiA"] * kcat_f_5 * kappa * gamma

    def gap(self, constants: dict, C: dict, e: dict) -> float:
        """
        Glyceraldehyde-3-phosphate dehydrogenase A (GAP/GAPDH).
        EC: 1.2.1.12

        Reaction : g3p + pi + nad <=> pgp + nadh
        Enzyme   : GapA

        Noor formulation — thermodynamic driving force via gamma.
        Allosteric inhibitors (ADP, AMP, ATP) removed from original formulation.

        constants keys : kcat_f_6, Ks_g3p_6, Ks_pi_6, Ks_nad_6, Kp_pgp_6, Kp_nadh_6
        C keys         : C_g3p, C_pi, C_nad, C_pgp, C_nadh
        e keys         : GapA
        """
        kcat_f_6  = constants["kcat_f_6"]
        Ks_g3p_6  = constants["Ks_g3p_6"]
        Ks_pi_6   = constants["Ks_pi_6"]
        Ks_nad_6  = constants["Ks_nad_6"]
        Kp_pgp_6  = constants["Kp_pgp_6"]
        Kp_nadh_6 = constants["Kp_nadh_6"]

        C_g3p  = C["C_g3p"]
        C_pi   = C["C_pi"]
        C_nad  = C["C_nad"]
        C_pgp  = C["C_pgp"]
        C_nadh = C["C_nadh"]

        prod_subs  = (C_g3p / Ks_g3p_6) * (C_pi / Ks_pi_6) * (C_nad / Ks_nad_6)
        prod_prods = (C_pgp / Kp_pgp_6) * (C_nadh / Kp_nadh_6)
        kappa = prod_subs / (1 + prod_subs + prod_prods)

        dGr   = self.D_gR_circ["gap"] + self.R * self.T * np.log(
            (C_pgp * C_nadh) / (C_g3p * C_pi * C_nad)
        )
        gamma = 1 - np.exp(-dGr / (self.R * self.T))

        return e["GapA"] * kcat_f_6 * kappa * gamma

    def pgk(self, constants: dict, C: dict, e: dict) -> float:
        """
        Phosphoglycerate kinase (PGK).
        EC: 2.7.2.3

        Reaction : pgp + adp <=> 3pg + atp
        Enzyme   : Pgk

        Noor formulation — thermodynamic driving force via gamma.
        Product-activation terms (original formulation) removed; directionality
        is now encoded entirely in dGr.

        constants keys : kcat_f_7, Ks_pgp_7, Ks_adp_7, Ks_3pg_7, Ks_atp_7
        C keys         : C_pgp, C_adp, C_3pg, C_atp
        e keys         : Pgk
        """
        kcat_f_7 = constants["kcat_f_7"]
        Ks_pgp_7 = constants["Ks_pgp_7"]
        Ks_adp_7 = constants["Ks_adp_7"]
        Ks_3pg_7 = constants["Ks_3pg_7"]
        Ks_atp_7 = constants["Ks_atp_7"]

        C_pgp = C["C_pgp"]
        C_adp = C["C_adp"]
        C_3pg = C["C_3pg"]
        C_atp = C["C_atp"]

        prod_subs  = (C_pgp / Ks_pgp_7) * (C_adp / Ks_adp_7)
        prod_prods = (C_3pg / Ks_3pg_7) * (C_atp / Ks_atp_7)
        kappa = prod_subs / (1 + prod_subs + prod_prods)

        dGr   = self.D_gR_circ["pgk"] + self.R * self.T * np.log(
            (C_3pg * C_atp) / (C_pgp * C_adp)
        )
        gamma = 1 - np.exp(-dGr / (self.R * self.T))

        return e["Pgk"] * kcat_f_7 * kappa * gamma

    def gpm(self, constants: dict, C: dict, e: dict) -> float:
        """
        Phosphoglycerate mutase A (GPM).
        EC: 5.4.2.11

        Reaction : 3pg <=> 2pg
        Enzyme   : GpmA

        Noor formulation — thermodynamic driving force via gamma.
        Phosphate inhibition (original formulation) removed.

        constants keys : kcat_f_8, Ks_3pg_8, Ks_2pg_8
        C keys         : C_3pg, C_2pg
        e keys         : GpmA
        """
        kcat_f_8 = constants["kcat_f_8"]
        Ks_3pg_8 = constants["Ks_3pg_8"]
        Ks_2pg_8 = constants["Ks_2pg_8"]

        C_3pg = C["C_3pg"]
        C_2pg = C["C_2pg"]

        prod_subs  = C_3pg / Ks_3pg_8
        prod_prods = C_2pg / Ks_2pg_8
        kappa = prod_subs / (1 + prod_subs + prod_prods)

        dGr   = self.D_gR_circ["gpm"] + self.R * self.T * np.log(C_2pg / C_3pg)
        gamma = 1 - np.exp(-dGr / (self.R * self.T))

        return e["GpmA"] * kcat_f_8 * kappa * gamma

    def eno(self, constants: dict, C: dict, e: dict) -> float:
        """
        Enolase (ENO).
        EC: 4.2.1.11

        Reaction : 2pg <=> pep  (+ h2o, excluded — constant activity)
        Enzyme   : Eno

        Noor formulation — thermodynamic driving force via gamma.

        constants keys : kcat_f_9, Ks_2pg_9, Ks_pep_9
        C keys         : C_2pg, C_pep
        e keys         : Eno
        """
        kcat_f_9 = constants["kcat_f_9"]
        Ks_2pg_9 = constants["Ks_2pg_9"]
        Ks_pep_9 = constants["Ks_pep_9"]

        C_2pg = C["C_2pg"]
        C_pep = C["C_pep"]

        prod_subs  = C_2pg / Ks_2pg_9
        prod_prods = C_pep / Ks_pep_9
        kappa = prod_subs / (1 + prod_subs + prod_prods)

        dGr   = self.D_gR_circ["eno"] + self.R * self.T * np.log(C_pep / C_2pg)
        gamma = 1 - np.exp(-dGr / (self.R * self.T))

        return e["Eno"] * kcat_f_9 * kappa * gamma

    # ------------------------------------------------------------------
    # Flux aggregation
    # ------------------------------------------------------------------

    def compute_fluxes(self, C, e, constants) -> list:
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
            Flux vector [v_pts, v_pgi, v_pfk, v_fba, v_tpi,
                         v_gap, v_pgk, v_gpm, v_eno].
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

    # ------------------------------------------------------------------
    # NLP construction
    # ------------------------------------------------------------------

    def build_parameter_estimation_nlp(self,
                                       conditions_enzymes: list[dict],
                                       conditions_obs:     list[pd.Series],
                                       bounds_params:      dict,
                                       ipopt_opts:         dict):
        """
        Extends the solve_steady_state objective (||S@v - b||²) to also
        cover parameter estimation, by adding theta as decision variables
        and a data-fitting term. Pure box-constrained NLP.

        Objective = Σ_k [ ss_weight * ||S @ v_k(C_k, θ, e_k)||²
                        +             ||predicted_k - observed_k||² ]
        """
        N     = len(conditions_enzymes)
        n_par = len(self.params_keys)
        S     = ca.DM(self.stoichiometric_matrix.values)

        theta = ca.SX.sym("theta", n_par)
        C_all = [ca.SX.sym(f"C_{k}", len(self.balanced_keys) + len(self.imbalanced_keys))
                 for k in range(N)]

        constants = {key: theta[i] for i, key in enumerate(self.params_keys)}
        obj    = ca.SX(0)
        ss_list = []
        for k in range(N):
            C_k       = C_all[k]
            C_bal     = {key: C_k[i]                           for i, key in enumerate(self.balanced_keys)}
            C_imbal   = {key: C_k[i + len(self.balanced_keys)] for i, key in enumerate(self.imbalanced_keys)}
            enzymes_k = {key: float(conditions_enzymes[k][key]) for key in self.enzymes_keys}

            fluxes_k = self.compute_fluxes({**C_bal, **C_imbal}, enzymes_k, constants)
            ss_list.append(S @ ca.vertcat(*fluxes_k))

            pred = {**C_bal,
                    **{self.flux_keys[i]: fluxes_k[i] for i in range(len(fluxes_k))}}
            for var_name in conditions_obs[k].index:
                obs_val = conditions_obs[k][var_name]
                if not np.isnan(float(obs_val)) and var_name in pred:
                    obj += ((pred[var_name] - float(obs_val)) / (abs(float(obs_val)) + 1e-12)) ** 2

        x        = ca.vertcat(theta, *C_all)
        lb_theta = np.array([bounds_params[k][0] for k in self.params_keys])
        ub_theta = np.array([bounds_params[k][1] for k in self.params_keys])
        lb_x     = np.concatenate([lb_theta] + [self._lbx] * N)
        ub_x     = np.concatenate([ub_theta] + [self._ubx] * N)

        g   = ca.vertcat(*ss_list)
        n_g = g.shape[0]
        lbg = -np.ones(n_g) * self.ss_tolerance
        ubg =  np.ones(n_g) * self.ss_tolerance
        print(f"[DEBUG] {N} conditions  {n_g // N} constraints/condition = {n_g} total")

        if ipopt_opts is None:
            ipopt_opts = {
                "ipopt": {
                    "print_level": 0, "sb": "yes",
                    "mu_strategy": "adaptive",
                    "tol": 1e-4, "max_iter": 3000,
                },
                "print_time": 0,
            }

        solver = ca.nlpsol("pe_solver", "ipopt",
                           {"x": x, "f": obj, "g": g}, ipopt_opts)
        return solver, lb_x, ub_x, lbg, ubg

    def construct_steady_state_problem(self):
        """
        Builds the CasADi NLP:
            min  || S @ v(C, e; params) - b ||_2^2
            s.t. lb_balanced   <= C_balanced   <= ub_balanced
                 lb_imbalanced <= C_imbalanced <= ub_imbalanced

        where b is the 'cell needs' vector (net production/consumption rates
        for the balanced metabolites imposed by external demands).
        Set b = 0 for a pure steady-state computation.
        """
        n_variables = len(self.balanced_keys) + len(self.imbalanced_keys)

        C_sym = ca.SX.sym("C", n_variables)
        k_sym = ca.SX.sym("k", len(self.params_keys))
        e_sym = ca.SX.sym("e", len(self.enzymes_keys))
        b_sym = ca.SX.sym("b", self.stoichiometric_matrix.shape[0])  # = 9

        C_balanced   = {key: C_sym[i]                           for i, key in enumerate(self.balanced_keys)}
        C_imbalanced = {key: C_sym[i + len(self.balanced_keys)] for i, key in enumerate(self.imbalanced_keys)}
        constants    = {key: k_sym[i] for i, key in enumerate(self.params_keys)}
        enzymes      = {key: e_sym[i] for i, key in enumerate(self.enzymes_keys)}

        fluxes = self.compute_fluxes({**C_balanced, **C_imbalanced}, enzymes, constants)
        S      = ca.DM(self.stoichiometric_matrix.values)

        f   = ca.sumsqr(S @ ca.vertcat(*fluxes) - b_sym)
        nlp = {
            "x": C_sym,
            "f": f,
            "p": ca.vertcat(k_sym, e_sym, b_sym),
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp, self.opts)

    def solve_steady_state(self,
                           enzymes:        dict,
                           kinetic_params: dict,
                           cell_needs:     dict | None = None,
                           condition_key:  str  | None = None):
        """
        Find metabolite concentrations that minimise || S @ v - b ||_2^2.

        Parameters
        ----------
        enzymes : dict
            Enzyme concentrations keyed by gene name (e.g. 'PfkB').
        kinetic_params : dict
            Kinetic parameters matching params_keys.
        cell_needs : dict or None
            Net production demanded from each balanced metabolite (the b
            vector).  Keys must match balanced_keys.  Defaults to all-zero
            (pure steady-state, no external drain).
        condition_key : str or None
            Cache key for warm-starting across repeated calls.

        Returns
        -------
        pd.DataFrame
            Optimal balanced concentrations and fluxes.
        """
        if cell_needs is None:
            cell_needs = {key: 0.0 for key in self.balanced_keys}

        p = np.array(
            [kinetic_params[key] for key in self.params_keys] +
            [enzymes[key]        for key in self.enzymes_keys] +
            [cell_needs[key]     for key in self.balanced_keys]
        )

        x0  = self._warm_start_cache.get(condition_key, self._x0_default)
        sol = self.solver(x0=x0, lbx=self._lbx, ubx=self._ubx, p=p)

        C_opt = sol["x"].full().flatten()
        self._warm_start_cache[condition_key] = C_opt

        C_balanced_opt   = {key: C_opt[i] for i, key in enumerate(self.balanced_keys)}
        C_imbalanced_opt = {key: C_opt[i + len(self.balanced_keys)]
                            for i, key in enumerate(self.imbalanced_keys)}

        fluxes_opt = self.compute_fluxes(
            {**C_balanced_opt, **C_imbalanced_opt}, enzymes, kinetic_params
        )
        dict_v_opt = {self.flux_keys[i]: fluxes_opt[i] for i in range(len(fluxes_opt))}
        return pd.DataFrame({**C_balanced_opt, **dict_v_opt}, index=[0]), sol["f"].full().item()

    def _construct_stoichiometric_matrix(self) -> pd.DataFrame:
        """
        Build the 9×9 stoichiometric matrix N (metabolites × reactions).

        Rows    : 2pg, 3pg, dhap, f6p, fbp, g3p, g6p, pep, pgp
        Columns : pts, pgi, pfk, fba, tpi, gap, pgk, gpm, eno
        """
        N = np.array([
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

        metabolites = ["2pg", "3pg", "dhap", "f6p", "fbp", "g3p", "g6p", "pep", "pgp"]
        reactions   = ["pts", "pgi", "pfk", "fba", "tpi", "gap", "pgk", "gpm", "eno"]

        return pd.DataFrame(N, index=metabolites, columns=reactions)


def load_params(csv_path: str) -> dict:
    """
    Load kinetic parameters from a CSV file and return as a dictionary.

    Expected CSV columns: parameter, reaction, substrate, value.
    Only Ks/Kp values are loaded from the CSV; PTS parameters are
    hardcoded from literature (Kadir et al., 2010).
    """
    param_df   = pd.read_csv(csv_path)
    param_dict = dict.fromkeys(ALL_PARAMS, None)

    # PTS — hardcoded from literature (Kadir et al., 2010)
    param_dict["v_max_1"] = 25.739  # mmol/gDW/h
    param_dict["Ka1_1"]   = 1.0     # mM
    param_dict["Ka2_1"]   = 0.01    # mM
    param_dict["Ka3_1"]   = 1.0     # mM
    param_dict["K_g6p_1"] = 0.5     # mM  (represents K_g6p^4 in Hill term)

    # Noor Ks/Kp values from BRENDA (column header: "Ks" or "Kp")
    for _, row in param_df.iterrows():
        param_type = row["parameter"]   # e.g. "Ks" or "Kp"
        reaction   = row["reaction"]    # e.g. "pgi"
        substrate  = row["substrate"]   # e.g. "D-Glucose-6-phosphate"
        value      = row["value"]

        approx = substrate.lower().replace(" ", "_").replace("-", "_")
        met    = SUBSTRATE_MAP.get(approx)
        if met is None or reaction not in REVERSE_REACTION_MAP:
            continue

        rxn_num = REVERSE_REACTION_MAP[reaction]
        if param_type in ("Ks", "Kp"):
            param_key = f"{param_type}_{met}_{rxn_num}"
            if param_key in param_dict:
                if param_dict[param_key] is None:
                    param_dict[param_key] = [value]
                else:
                    param_dict[param_key].append(value)

    return param_dict


# %%
if __name__ == "__main__":
    # Example usage with placeholder parameter values.
    constants = {
        # v1: PTS — Kadir formulation
        "v_max_1": 25.739,  # mmol/gDW/h
        "Ka1_1":   1.0,
        "Ka2_1":   0.01,
        "Ka3_1":   1.0,
        "K_g6p_1": 0.5,     
        # K_g6p^4 (Hill inhibition, n=4 hard-coded)

        # v2: PGI — Noor formulation
        "Ks_g6p_pgi": 0.48,   # mM
        "Kp_f6p_pgi": 0.19,   # mM
        "kcat_f_2":   1475.0, # 1/h

        # v3: PFK — Noor formulation
        "Ks_f6p_3": 0.16,   # mM
        "Ks_atp_3": 0.12,   # mM
        "Kp_fbp_3": 0.50,   # mM
        "Kp_adp_3": 0.20,   # mM
        "kcat_f_3": 580.0,  # 1/h

        # v4: FBA — Noor formulation
        "Ks_fbp_4":  0.30,  # mM
        "Kp_g3p_4":  0.40,  # mM
        "Kp_dhap_4": 2.00,  # mM
        "kcat_f_4":  95.0,  # 1/h

        # v5: TPI — Noor formulation
        "kcat_f_5":  4300.0, # 1/h
        "Ks_dhap_5": 0.61,   # mM
        "Kp_g3p_5":  1.20,   # mM

        # v6: GAPDH — Noor formulation
        "kcat_f_6":  118.0, # 1/h
        "Ks_g3p_6":  0.21,  # mM
        "Ks_pi_6":   0.29,  # mM
        "Ks_nad_6":  0.09,  # mM
        "Kp_pgp_6":  0.01,  # mM
        "Kp_nadh_6": 0.06,  # mM

        # v7: PGK — Noor formulation
        "kcat_f_7": 1150.0, # 1/h
        "Ks_pgp_7": 0.05,   # mM
        "Ks_adp_7": 0.10,   # mM
        "Ks_3pg_7": 0.53,   # mM
        "Ks_atp_7": 0.30,   # mM

        # v8: GPM — Noor formulation
        "kcat_f_8": 540.0,  # 1/h
        "Ks_3pg_8": 0.20,   # mM
        "Ks_2pg_8": 1.40,   # mM

        # v9: ENO — Noor formulation
        "kcat_f_9": 550.0,  # 1/h
        "Ks_2pg_9": 0.10,   # mM
        "Ks_pep_9": 0.50,   # mM
    }

    enzymes = {
        "Pgi":  0.01,
        "PfkB": 0.10,
        "FbaA": 0.10,
        "TpiA": 0.10,
        "GapA": 0.10,
        "Pgk":  0.01,
        "GpmA": 0.10,
        "Eno":  0.10,
    }

    # b = 0  →  pure steady state (no external metabolite drain)
    cell_needs = {
        "C_g6p": 0.0, "C_f6p": 0.0, "C_fbp": 0.0,
        "C_dhap": 0.0, "C_g3p": 0.0, "C_pgp": 0.0,
        "C_3pg": 0.0, "C_2pg": 0.0, "C_pep": 0.0,
    }

    model = EcoliCarbonKinetics(
        bounds_imbalanced_mets={
            "C_atp":  (0.01,  1.0),
            "C_adp":  (0.001, 0.1),
            "C_amp":  (0.001, 0.1),
            "C_gdp":  (0.001, 0.1),
            "C_glc":  (0.001, 10.0),
            "C_gtp":  (0.001, 0.1),
            "C_nad":  (0.001, 0.1),
            "C_nadh": (0.001, 0.1),
            "C_pi":   (0.001, 10.0),
            "C_pyr":  (0.001, 10.0),
        },
        bounds_balanced_mets={
            "C_2pg": (1e-6, 10),
            "C_3pg": (1e-6, 10),
            "C_dhap": (1e-6, 10),
            "C_f6p": (1e-6, 10),
            "C_fbp": (1e-6, 10),
            "C_g3p": (1e-6, 10),
            "C_g6p": (1e-6, 10),
            "C_pep": (1e-6, 10),
            "C_pgp": (1e-6, 10),
        },
    )

    ti = time.time()
    solved = model.solve_steady_state(
        enzymes=enzymes,
        kinetic_params=constants,
        cell_needs=cell_needs,
    )
    print(f"Solved in {time.time() - ti:.3f} s")
    print(solved[0].T, solved[1])
# %%