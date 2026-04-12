"""
kinetics.py - Mechanistic kinetic model of E. coli glycolysis (IIQ3733).

Models 9 enzymatic reactions from glucose uptake (PTS) to PEP production (Eno),
covering the core of E. coli central carbon metabolism.
Each reaction is modeled with convenient kinetic rate laws that include substrate saturation and allosteric regulation, based on literature data.
"""

import numpy as np
import pandas as pd
import casadi as ca

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

    def __init__(self, metabolites: dict, enzymes: dict, constants: dict):
        self.metabolites = metabolites
        self.enzymes = enzymes
        self.constants = constants
        self.stoichiometric_matrix = self._construct_stoichiometric_matrix()

    def pts(self, constants: dict, C: dict, e: dict) -> float:
        """
        Phosphotransferase system (PTS).

        Reaction : glc + pep -> g6p + pyr
        Enzymes : PtsI, PtsH

        Rate is the sum of three additive terms:
            v = inhibition(pyr) + activation(pep) + kinetic(glc, pep/pyr)

        constants keys : k_pyr1_I, k_pep1_A, v_pts_max, K_a1, K_a2, K_a3, n_g6p, K_g6p
        C keys : C_pyr, C_pep, C_glc
        e keys : (none)
        """
        k_pyr1_I = constants["k_pyr1_I"]
        k_pep1_A = constants["k_pep1_A"]
        v_pts_max = constants["v_pts_max"]
        K_a1 = constants["K_a1"]
        K_a2 = constants["K_a2"]
        K_a3 = constants["K_a3"]
        n_g6p = constants["n_g6p"]
        K_g6p = constants["K_g6p"]

        C_pyr = C["C_pyr"]
        C_pep = C["C_pep"]
        C_glc = C["C_glc"]

        inhibition = k_pyr1_I / (k_pyr1_I + C_pyr)
        activation = C_pep / (C_pep + k_pep1_A)

        num = v_pts_max * C_glc * (C_pep / C_pyr)
        den1 = K_a1 + K_a2 * (C_pep / C_pyr) + K_a3 * C_glc + C_glc * (C_pep / C_pyr)
        den2 = 1 + (C_glc**n_g6p) / K_g6p
        kinetic = num / (den1 * den2)

        return inhibition + activation + kinetic

    def pgi(self, constants: dict, C: dict, e: dict) -> float:
        """
        Glucose-6-phosphate isomerase (PGI).

        Reaction : g6p <=> f6p
        Enzyme : Pgi
        Inhibited by PEP.

        constants keys : k_pep2_I, Km_g6p_2, Km_f6p_2, kcat_f2, kcat_r2
        C keys : C_g6p, C_f6p, C_pep
        e keys : Pgi
        """
        k_pep2_I = constants["k_pep2_I"]
        Km_g6p_2 = constants["Km_g6p_2"]
        Km_f6p_2 = constants["Km_f6p_2"]
        kcat_f2 = constants["kcat_f2"]
        kcat_r2 = constants["kcat_r2"]

        C_g6p = C["C_g6p"]
        C_f6p = C["C_f6p"]
        C_pep = C["C_pep"]

        h = k_pep2_I / (k_pep2_I + C_pep)
        num = kcat_f2 * (C_g6p / Km_g6p_2) - kcat_r2 * (C_f6p / Km_f6p_2)
        den = (C_g6p / Km_g6p_2 + 1) + (C_f6p / Km_f6p_2 + 1) - 1

        return e["Pgi"] * h * num / den

    def pfk(self, constants: dict, C: dict, e: dict) -> float:
        """
        Phosphofructokinase B (PFK).

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
        for conc, kA in [(C_adp, kA_adp_3), (C_gdp, kA_gdp_3)]:
            h *= conc / (kA + conc)

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

        Reaction : g3p + Pi + NAD+ <=> pgp + NADH
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

    def _ode_system(self, t, x, e, constants):
        """
        dx/dt = N @ v(x, e, constants)

        Parameters
        ----------
        t : float
            Current time (required by scipy ODE solvers).
        x : array-like, shape (9,)
            State vector: [C_2pg, C_3pg, C_dhap, C_f6p, C_fbp,
                           C_g3p, C_g6p, C_pep, C_pgp]
        e : dict
            Enzyme concentrations.
        constants : dict
            Kinetic constants.

        Returns
        -------
        numpy.ndarray, shape (9,)
            Time derivatives of all metabolite concentrations.
        """
        met_names = [
            "C_2pg",
            "C_3pg",
            "C_dhap",
            "C_f6p",
            "C_fbp",
            "C_g3p",
            "C_g6p",
            "C_pep",
            "C_pgp",
        ]
        C = dict(zip(met_names, x))

        v = np.array(
            [
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
        )

        return self.stoichiometric_matrix.values @ v

    def simulate_system(self, x0):
        """
        Simulate the kinetic system over time given initial conditions.

        Parameters
        ----------
        x0 : dict
            Initial concentrations of metabolites, keyed as 'C_<name>'.

        Returns
        -------
        pd.DataFrame
            Time series of metabolite concentrations.
        """
        # solve ODE with casadi.
        
        pass

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
                [0, 0, 0, 0, 0, 0, 0, 1, -1],  # 2pg
                [0, 0, 0, 0, 0, 0, 1, -1, 0],  # 3pg
                [0, 0, 0, 1, -1, 0, 0, 0, 0],  # dhap
                [0, 1, -1, 0, 0, 0, 0, 0, 0],  # f6p
                [0, 0, 1, -1, 0, 0, 0, 0, 0],  # fbp
                [0, 0, 0, 1, 1, -1, 0, 0, 0],  # g3p
                [1, -1, 0, 0, 0, 0, 0, 0, 0],  # g6p
                [-1, 0, 0, 0, 0, 0, 0, 0, 1],  # pep
                [0, 0, 0, 0, 0, 1, -1, 0, 0],  # pgp
            ]
        )

        metabolites = ["2pg", "3pg", "dhap", "f6p", "fbp", "g3p", "g6p", "pep", "pgp"]
        reactions = ["pts", "pgi", "pfk", "fba", "tpi", "gap", "pgk", "gpm", "eno"]

        return pd.DataFrame(N, index=metabolites, columns=reactions)
