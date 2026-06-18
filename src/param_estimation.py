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
import pickle
import os
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
    imbalanced_keys = ["C_atp", "C_adp", "C_nad", "C_nadh", "C_pi", "C_pyr", "C_glc"]
    enzymes_keys    = ["Pgi", "PfkB", "FbaA", "TpiA", "GapA", "Pgk", "GpmA", "Eno"]
    flux_keys       = ["v_pts", "v_pgi", "v_pfkB", "v_fbaA", "v_tpiA",
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
                 k_db: str = os.path.join('Data', 'k_eq_values.pkl')
                 ):
        self.stoichiometric_matrix = self._construct_stoichiometric_matrix()
        self.bounds_imbalanced_mets = bounds_imbalanced_mets
        self.bounds_balanced_mets   = bounds_balanced_mets
        self.R = R_ct
        self.T = T
        self.construct_Keq()

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

    def construct_Keq(self):
        # Standard Gibbs free energy changes for each reaction (kJ/mol)
        # Source: https://equilibrator.weizmann.ac.il/ through ComponentContribution API,
        # used pH 7. and ionic strength 0.1 M
        # NOTE: The data should be pre-computed and stored as a pickle file to avoid the overhead of querying 
        # the API every time the model is initialized. 
        # The code for generating this data is in gen_K_eq.ipynb, 
        # and the resulting pickle file is included in the repository.
        # setting the conditions
        k_eq = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Data', 'k_eq_values.pkl')
        with open(k_eq, 'rb') as f:
            k_eq_values = pickle.load(f)
        self.Keq = k_eq_values
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
        EcoCyc: https://ecocyc.org/gene?orgid=ECOLI&id=EG10702
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
        frac_prod_subs = C_f6p / C_g6p
        gamma = 1 - frac_prod_subs/self.Keq["pgi"]

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

        frac_prod_subs = (C_fbp * C_adp) / (C_f6p * C_atp)
        gamma = 1 - frac_prod_subs/self.Keq["pfk"]

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

        frac_prod_subs = (C_g3p * C_dhap) / C_fbp
        
        gamma = 1 - frac_prod_subs/self.Keq["fba"]

        return e["FbaA"] * kcat_f_4 * kappa * gamma

    def tpi(self, constants: dict, C: dict, e: dict) -> float:
        """
        Triose-phosphate isomerase (TPI).
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

        frac_prod_subs = C_g3p / C_dhap
        gamma = 1 - frac_prod_subs/self.Keq["tpi"]

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

        frac_prod_subs = (C_pgp * C_nadh) / (C_g3p * C_pi * C_nad)
        gamma = 1 - frac_prod_subs/self.Keq["gap"]

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

        frac_prod_subs = (C_3pg * C_atp) / (C_pgp * C_adp)
        gamma = 1 - frac_prod_subs/self.Keq["pgk"]

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

        frac_prod_subs = C_2pg / C_3pg
        gamma = 1 - frac_prod_subs/self.Keq["gpm"]

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

        frac_prod_subs = C_pep / C_2pg
        gamma = 1 - frac_prod_subs/self.Keq["eno"]

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



    def create_param_estimation_problem(
        data_condition: str,
        C_exp: dict,
        e_exp: dict
    ):
        model = pyo.ConcreteModel()
        # Define sets
        model.METABOLITES = pyo.Set(initialize=EcoliCarbonKinetics.balanced_keys + EcoliCarbonKinetics.imbalanced_keys)
        model.REACTIONS = pyo.Set(initialize=EcoliCarbonKinetics.flux_keys)
        model.PARAMS = pyo.Set(initialize=EcoliCarbonKinetics.params_keys)
        # Define variables with add components
        for met in model.METABOLITES:
            model.add_component(met, pyo.Var(bounds=(0, None), initialize=1.0))  # Adjust bounds as needed
        for rxn in model.REACTIONS:
            model.add_component(rxn, pyo.Var(bounds=(None, None), initialize=0.0))  # Fluxes can be positive or negative
        for param in model.PARAMS:
            model.add_component(param, pyo.Var(bounds=(0, None), initialize=1.0))  # Adjust bounds as needed
        # Define objective function (e.g., minimize squared error between predicted and experimental fluxes)    
        
        