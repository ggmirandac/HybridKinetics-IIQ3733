"""
kinetics.py -- rate laws of the E. coli glycolysis model and the closure solver of the held-out predictions.

Nine reactions from glucose uptake (PTS) to PEP production (enolase).  Every reaction except the PTS follows the
Noor et al. (2013) form with the thermodynamic driving force written out:
    v = E * kcat_f * kappa * gamma
where kappa is the saturation term and gamma = 1 - Q / Keq vanishes at equilibrium.  The PTS keeps its empirical
formulation.  ``GlycolysisRateLaws`` holds the 9 rate laws and nothing else, so the Pyomo estimation model
(estimation/experiment.py), the CasADi Jacobians of the structural check and the sympy check all evaluate the very
same code.  ``EcoliCarbonKinetics`` adds the metabolite bounds and the CasADi closure solve used for held-out
predictions (LOOCV).

Position in the pipeline: model definition (edit the rate laws here).
"""
import solver  # noqa: F401  (pins BLAS / OpenMP to one thread before numpy is loaded)

import numpy as np
import pandas as pd
import casadi as ca

from settings import CASADI_OPTIONS
from model.parameters import ALL_PARAMS, BALANCED_KEYS, IMBALANCED_KEYS, ENZYME_KEYS, FLUX_KEYS, RXN_ORDER
from model.stoichiometry import load_keq, stoichiometric_table

# Allowed steady-state residual |S v - alpha v_pts| of the closure solve
SS_TOLERANCE = 1e-6
# Weight of the balanced metabolites in the reference objective of the closure (the cofactors have weight 1)
CLOSURE_EPS = 1e-4

class GlycolysisRateLaws:
    """
    The 9 rate laws.  ``keq`` is {reaction: equilibrium constant}: numbers for the model, sympy symbols for the
    symbolic structural check.  Every method takes (constants, C, e) = (kinetic parameters, concentrations keyed
    'C_<name>', enzyme levels keyed by gene name) and works with floats, Pyomo, CasADi or sympy objects alike.
    """

    def __init__(self, keq):
        self.Keq = keq

    def pts(self, constants: dict, C: dict, e: dict) -> float:
        """
        Phosphotransferase system (PTS).

        Reaction : glc + pep -> g6p + pyr
        Enzymes  : PtsI, PtsH  (v_max_1 absorbs enzyme concentration)

        Empirical formulation -- no thermodynamic gamma term.

        Inhibition by g6p: 1 / (1 + (C_g6p / K_g6p_1)^4) (Hill coefficient 4, K_g6p_1 is K).

        constants keys : v_max_1, Ka1_1, Ka2_1, Ka3_1, K_g6p_1
        C keys         : C_pyr, C_pep, C_glc, C_g6p
        e keys         : (none -- v_max already incorporates enzyme level)
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
        den2 = 1 + (C_g6p / K_g6p_1) ** 4
        kinetic = num / (den1 * den2)

        return kinetic

    def pgi(self, constants: dict, C: dict, e: dict) -> float:
        """
        Glucose-6-phosphate isomerase (PGI).
        EC: 5.3.1.9

        Reaction : g6p <=> f6p
        Enzyme   : Pgi

        Noor formulation -- thermodynamic driving force via gamma.
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

        Noor formulation -- thermodynamic driving force via gamma.
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

        Noor formulation -- thermodynamic driving force via gamma.
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

        Noor formulation -- thermodynamic driving force via gamma.

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

        Noor formulation -- thermodynamic driving force via gamma.
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

        Noor formulation -- thermodynamic driving force via gamma.
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

        Noor formulation -- thermodynamic driving force via gamma.
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

        Reaction : 2pg <=> pep  (+ h2o, excluded -- constant activity)
        Enzyme   : Eno

        Noor formulation -- thermodynamic driving force via gamma.

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

    def compute_fluxes(self, C, e, constants):
        """The 9 reaction rates [v_pts, v_pgi, ..., v_eno] in RXN_ORDER."""
        fluxes = []
        for reaction in RXN_ORDER:
            rate_law = getattr(self, reaction)
            fluxes.append(rate_law(constants, C, e))
        return fluxes

class EcoliCarbonKinetics(GlycolysisRateLaws):
    """
    The rate laws with the equilibrium constants of Data/k_eq_values.pkl, the stoichiometric matrix, the metabolite
    bounds and the CasADi closure solve of a held-out condition (``solve_closure``).
    """

    def __init__(self, bounds_imbalanced_mets, bounds_balanced_mets):
        super().__init__(load_keq())
        self.stoichiometric_matrix = stoichiometric_table()
        lower = []
        upper = []
        for key in BALANCED_KEYS:
            lower.append(bounds_balanced_mets[key][0])
            upper.append(bounds_balanced_mets[key][1])
        for key in IMBALANCED_KEYS:
            lower.append(bounds_imbalanced_mets[key][0])
            upper.append(bounds_imbalanced_mets[key][1])
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.casadi_options = {"ipopt": dict(CASADI_OPTIONS), "print_time": 0}
        # Built at the first closure solve: compiling the NLP is the slow part and most callers never need it
        self.closure_solver = None
        self.closure_rates = None
        self.last_stats = None
        self.last_closure_b = None

    def build_closure_solver(self):
        """
        CasADi NLP of the out-of-sample "pts_fraction" closure:
            min  sum_k (ln u_k - ln u_ref_k)^2 + eps * sum_j (ln x_j - ln x_ref_j)^2
            s.t. S v(C, e, k) - alpha * v_pts(C, k) = 0
        p = [k (37), e (8), alpha (9), u_ref (7), x_ref (9), eps].  The drains are a fixed fraction alpha_j of the
        PTS flux.  The steady state is not unique (16 unknowns, 9 equations); the reference objective selects one.
        """
        n_balanced = len(BALANCED_KEYS)
        n_imbalanced = len(IMBALANCED_KEYS)
        C_sym = ca.SX.sym("C", n_balanced + n_imbalanced)
        k_sym = ca.SX.sym("k", len(ALL_PARAMS))
        e_sym = ca.SX.sym("e", len(ENZYME_KEYS))
        alpha_sym = ca.SX.sym("alpha", n_balanced)
        u_ref_sym = ca.SX.sym("u_ref", n_imbalanced)
        x_ref_sym = ca.SX.sym("x_ref", n_balanced)
        eps_sym = ca.SX.sym("eps")
        concentrations = {}
        for i, key in enumerate(BALANCED_KEYS):
            concentrations[key] = C_sym[i]
        for i, key in enumerate(IMBALANCED_KEYS):
            concentrations[key] = C_sym[i + n_balanced]
        constants = {}
        for i, key in enumerate(ALL_PARAMS):
            constants[key] = k_sym[i]
        enzymes = {}
        for i, key in enumerate(ENZYME_KEYS):
            enzymes[key] = e_sym[i]
        v_sym = ca.vertcat(*self.compute_fluxes(concentrations, enzymes, constants))
        S = ca.DM(self.stoichiometric_matrix.values)
        g = S @ v_sym - alpha_sym * v_sym[0]
        f = (ca.sumsqr(ca.log(C_sym[n_balanced:]) - ca.log(u_ref_sym))
             + eps_sym * ca.sumsqr(ca.log(C_sym[:n_balanced]) - ca.log(x_ref_sym)))
        # a cold start: the warm-start options of the casadi set do not apply here
        ipopt_options = {}
        for name, value in self.casadi_options["ipopt"].items():
            if not name.startswith("warm_start"):
                ipopt_options[name] = value
        options = dict(self.casadi_options)
        options["ipopt"] = ipopt_options
        nlp = {"x": C_sym, "f": f, "g": g, "p": ca.vertcat(k_sym, e_sym, alpha_sym, u_ref_sym, x_ref_sym, eps_sym)}
        self.closure_solver = ca.nlpsol("closure_solver", "ipopt", nlp, options)
        self.closure_rates = ca.Function("closure_v", [C_sym, ca.vertcat(k_sym, e_sym)], [v_sym])

    def record_stats(self, solution):
        """
        Status of the last closure solve.  ``success`` is strict: ipopt reported success AND max |g| <= 10 times the
        tolerance (ipopt's own constr_viol_tol accepts residuals 10-400 times larger); ``ipopt_success`` is the raw
        flag.
        """
        stats = self.closure_solver.stats()
        g = np.asarray(solution["g"]).flatten()
        max_residual = float(np.max(np.abs(g)))
        ipopt_success = bool(stats["success"])
        self.last_stats = {
            "return_status": str(stats["return_status"]),
            "success": bool(ipopt_success and max_residual <= 10.0 * SS_TOLERANCE),
            "ipopt_success": ipopt_success,
            "ss_residual": float(np.linalg.norm(g, ord=2)),
            "max_residual": max_residual,
        }

    def solve_closure(self, enzymes, kinetic_params, alpha, u_ref, x_ref):
        """
        Out-of-sample steady state with the "pts_fraction" closure S v = alpha * v_pts: every drain is a fixed
        fraction of the PTS flux (alpha, u_ref, x_ref: dicts over the balanced, imbalanced and balanced keys).  The
        start point is (x_ref, u_ref) clipped into the bounds.  Returns (one-row DataFrame of the balanced
        concentrations and the fluxes, ||S v - alpha v_pts||); ``last_stats`` holds the solver status and
        ``last_closure_b`` the implied drains {balanced_key: alpha_j * v_pts}.
        """
        if self.closure_solver is None:
            self.build_closure_solver()
        n_balanced = len(BALANCED_KEYS)
        alpha_vector = np.array([float(alpha[key]) for key in BALANCED_KEYS])
        u_ref_vector = np.array([float(u_ref[key]) for key in IMBALANCED_KEYS])
        x_ref_vector = np.array([float(x_ref[key]) for key in BALANCED_KEYS])
        u_ref_vector = np.clip(u_ref_vector, self.lower[n_balanced:], self.upper[n_balanced:])
        x_ref_vector = np.clip(x_ref_vector, self.lower[:n_balanced], self.upper[:n_balanced])
        theta_values = [kinetic_params[key] for key in ALL_PARAMS]
        enzyme_values = [enzymes[key] for key in ENZYME_KEYS]
        p = np.array(theta_values + enzyme_values + list(alpha_vector) + list(u_ref_vector) + list(x_ref_vector)
                     + [float(CLOSURE_EPS)])
        x0 = np.clip(np.concatenate([x_ref_vector, u_ref_vector]), self.lower, self.upper)
        solution = self.closure_solver(x0=x0, lbx=self.lower.copy(), ubx=self.upper.copy(), lbg=-SS_TOLERANCE,
                                       ubg=SS_TOLERANCE, p=p)
        self.record_stats(solution)
        C_opt = solution["x"].full().flatten()
        v = self.closure_rates(C_opt, ca.DM(theta_values + enzyme_values)).full().flatten()
        residual = self.stoichiometric_matrix.values @ v - alpha_vector * v[0]
        self.last_closure_b = {}
        for i, key in enumerate(BALANCED_KEYS):
            self.last_closure_b[key] = float(alpha_vector[i] * v[0])
        row = {}
        for i, key in enumerate(BALANCED_KEYS):
            row[key] = C_opt[i]
        for i, key in enumerate(FLUX_KEYS):
            row[key] = v[i]
        return pd.DataFrame(row, index=[0]), float(np.linalg.norm(residual))
