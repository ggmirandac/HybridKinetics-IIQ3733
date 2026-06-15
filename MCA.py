"""
mca_analysis.py - Metabolic Control Analysis (MCA) for the E. coli
glycolysis model (EcoliCarbonKinetics, kinetics_noor.py).

Computes, at a given steady-state operating point:
    1. Elasticity matrices:
         eps_v_C   (9 x 9)   d ln v / d ln C_bal   (unscaled -> scaled)
         eps_v_Cimb (9 x 7)  d ln v / d ln C_imbalanced  (informational)
    2. Concentration control coefficients:
         C_C = -(N * eps_v_C)^(-1) * N           (9 x 9)
    3. Flux control coefficients:
         C_J = I + eps_v_C * C_C                 (9 x 9)
    4. Summation theorem checks:
         sum_j C_J[i,j] == 1   for every flux i
         sum_j C_C[i,j] == 0   for every metabolite i
    5. Connectivity theorem checks:
         sum_j C_J[:,j] * eps_v_C[j,k] == 0   for each metabolite k  (flux side)
         C_C @ eps_v_C == -I                  (concentration side)

All derivatives are obtained from CasADi (exact, not finite differences),
reusing the same symbolic machinery as gen_sensitivity_matrix.

Usage
-----
    from mca_analysis import run_mca

    G_total, results = run_mca(model, enzymes, kinetic_params, cell_needs,
                                condition_key="ref")
"""

import numpy as np
import pandas as pd
import casadi as ca


def compute_elasticities(model, enzymes, kinetic_params, cell_needs,
                          condition_key=None):
    """
    Compute scaled (logarithmic) elasticity matrices at the steady state.

    Returns
    -------
    eps_v_Cbal : ndarray (n_rxn, n_bal)
        Scaled elasticity of each flux wrt each balanced metabolite,
        eps[i,k] = (C_k / v_i) * dv_i/dC_k
    eps_v_Cimb : ndarray (n_rxn, n_imb)
        Same, wrt imbalanced ("external"/cofactor) metabolites.
    C_opt, v_opt : operating point concentrations and fluxes (for scaling)
    """
    p = np.array(
        [kinetic_params[key] for key in model.params_keys] +
        [enzymes[key]        for key in model.enzymes_keys] +
        [cell_needs[key]     for key in model.balanced_keys]
    )

    # --- Solve steady state to get operating point C* ---
    x0 = model._warm_start_cache.get(condition_key, model._x0_default)
    sol = model.solver(x0=x0, lbx=model._lbx, ubx=model._ubx,
                        lbg=-model.ss_tolerance, ubg=model.ss_tolerance, p=p)
    C_opt = sol["x"].full().flatten()
    model._warm_start_cache[condition_key] = C_opt

    n_bal = len(model.balanced_keys)
    n_imb = len(model.imbalanced_keys)
    n_var = n_bal + n_imb
    n_par = len(model.params_keys)
    n_enz = len(model.enzymes_keys)

    # --- Build fresh symbols and rebuild flux expressions ---
    C_s = ca.SX.sym("C", n_var)
    k_s = ca.SX.sym("k", n_par)
    e_s = ca.SX.sym("e", n_enz)
    b_s = ca.SX.sym("b", n_bal)
    p_s = ca.vertcat(k_s, e_s, b_s)

    C_dict = {**{key: C_s[i]         for i, key in enumerate(model.balanced_keys)},
              **{key: C_s[i + n_bal] for i, key in enumerate(model.imbalanced_keys)}}
    k_dict = {key: k_s[i] for i, key in enumerate(model.params_keys)}
    e_dict = {key: e_s[i] for i, key in enumerate(model.enzymes_keys)}

    v_s = ca.vertcat(*model.compute_fluxes(C_dict, e_dict, k_dict))   # (n_rxn,)

    # --- Unscaled elasticities: dv/dC ---
    dv_dC_fn = ca.Function("dv_dC", [C_s, p_s], [ca.jacobian(v_s, C_s)])
    v_fn     = ca.Function("v_fn",  [C_s, p_s], [v_s])

    dv_dC = dv_dC_fn(C_opt, p).full()        # (n_rxn, n_var)
    v_opt = v_fn(C_opt, p).full().flatten()  # (n_rxn,)

    if not np.isfinite(dv_dC).all() or not np.isfinite(v_opt).all():
        raise FloatingPointError(
            f"[compute_elasticities] Non-finite entries for condition {condition_key!r}"
        )

    # --- Scale to logarithmic (dimensionless) elasticities ---
    # eps[i,k] = (C_k / v_i) * dv_i/dC_k
    v_safe = np.where(np.abs(v_opt) < 1e-12, 1e-12, v_opt)
    eps_full = dv_dC * (C_opt[np.newaxis, :] / v_safe[:, np.newaxis])

    eps_v_Cbal = eps_full[:, :n_bal]
    eps_v_Cimb = eps_full[:, n_bal:]

    return eps_v_Cbal, eps_v_Cimb, C_opt, v_opt


def compute_control_coefficients(model, eps_v_Cbal, v_opt, C_opt):
    """
    Compute concentration (C_C) and flux (C_J) control coefficient matrices
    from the scaled elasticity matrix eps_v_Cbal (n_rxn x n_bal), where
    n_rxn == n_bal == 9 here (square stoichiometric matrix N).

    C_C = -(N @ eps_v_Cbal)^-1 @ N      (n_bal x n_rxn)
    C_J = I + eps_v_Cbal @ C_C          (n_rxn x n_rxn)
    """
    N = model.stoichiometric_matrix.values  # (9,9) metabolites x reactions

    M = N @ eps_v_Cbal   # (n_bal, n_rxn) -- here square (9x9)
    if M.shape[0] != M.shape[1]:
        raise ValueError(
            f"N @ eps_v_Cbal is {M.shape}; control coefficient formula "
            f"as implemented assumes a square (balanced x reaction) system."
        )

    # Check conditioning before inversion
    cond_num = np.linalg.cond(M)
    if cond_num > 1e10:
        print(f"  [compute_control_coefficients] WARNING: N @ eps is ill-conditioned "
              f"(cond={cond_num:.2e}); control coefficients may be unreliable.")

    M_inv = np.linalg.pinv(M)
    C_C = -M_inv @ N                         # (n_bal, n_rxn)

    n_rxn = eps_v_Cbal.shape[0]
    C_J = np.eye(n_rxn) + eps_v_Cbal @ C_C   # (n_rxn, n_rxn)

    return C_C, C_J


def check_theorems(model, eps_v_Cbal, C_C, C_J, tol=1e-6):
    """
    Verify the MCA summation and connectivity theorems.

    Summation:
        sum_j C_J[i,j] == 1   for each flux i
        sum_j C_C[i,j] == 0   for each metabolite i

    Connectivity:
        C_C @ eps_v_Cbal == -I            (concentration connectivity)
        C_J @ eps_v_Cbal == 0             (flux connectivity)
    """
    n_rxn = C_J.shape[0]
    n_bal = C_C.shape[0]

    sum_CJ = C_J.sum(axis=1)
    sum_CC = C_C.sum(axis=1)

    flux_summation_ok = np.allclose(sum_CJ, 1.0, atol=tol)
    conc_summation_ok = np.allclose(sum_CC, 0.0, atol=tol)

    conc_connectivity   = C_C @ eps_v_Cbal       # should be -I
    flux_connectivity   = C_J @ eps_v_Cbal       # should be 0

    conc_connectivity_ok = np.allclose(conc_connectivity, -np.eye(n_bal), atol=tol)
    flux_connectivity_ok = np.allclose(flux_connectivity, 0.0, atol=tol)

    results = {
        "flux_summation":       pd.Series(sum_CJ, index=model.flux_keys, name="sum_j C_J[i,j] (should be 1)"),
        "flux_summation_ok":    flux_summation_ok,
        "conc_summation":       pd.Series(sum_CC, index=model.balanced_keys, name="sum_j C_C[i,j] (should be 0)"),
        "conc_summation_ok":    conc_summation_ok,
        "conc_connectivity_max_err": np.max(np.abs(conc_connectivity + np.eye(n_bal))),
        "conc_connectivity_ok": conc_connectivity_ok,
        "flux_connectivity_max_err": np.max(np.abs(flux_connectivity)),
        "flux_connectivity_ok": flux_connectivity_ok,
    }
    return results


def run_mca(model, enzymes, kinetic_params, cell_needs, condition_key=None, tol=1e-6, verbose=True):
    """
    Full MCA pipeline for one steady-state condition.

    Returns
    -------
    dict with keys:
        eps_v_Cbal, eps_v_Cimb : elasticity matrices
        C_C, C_J               : control coefficient matrices (as DataFrames)
        C_opt, v_opt           : operating point
        theorems                : dict of theorem-check results
    """
    eps_v_Cbal, eps_v_Cimb, C_opt, v_opt = compute_elasticities(
        model, enzymes, kinetic_params, cell_needs, condition_key=condition_key
    )

    C_C, C_J = compute_control_coefficients(model, eps_v_Cbal, v_opt, C_opt)

    theorems = check_theorems(model, eps_v_Cbal, C_C, C_J, tol=tol)

    eps_v_Cbal_df = pd.DataFrame(eps_v_Cbal, index=model.flux_keys, columns=model.balanced_keys)
    eps_v_Cimb_df = pd.DataFrame(eps_v_Cimb, index=model.flux_keys, columns=model.imbalanced_keys)
    C_C_df = pd.DataFrame(C_C, index=model.balanced_keys, columns=model.flux_keys)
    C_J_df = pd.DataFrame(C_J, index=model.flux_keys, columns=model.flux_keys)

    if verbose:
        print(f"\n=== MCA results for condition {condition_key!r} ===")
        print("\nFlux control coefficients C_J (rows=flux i, cols=controlling enzyme j):")
        print(C_J_df.round(3))
        print("\nConcentration control coefficients C_C (rows=metabolite, cols=controlling enzyme):")
        print(C_C_df.round(3))
        print("\nSummation theorem (flux):")
        print(theorems["flux_summation"].round(4))
        print(f"  -> OK: {theorems['flux_summation_ok']}")
        print("\nSummation theorem (concentration):")
        print(theorems["conc_summation"].round(4))
        print(f"  -> OK: {theorems['conc_summation_ok']}")
        print(f"\nConnectivity theorem (concentration), max |C_C @ eps + I| = "
              f"{theorems['conc_connectivity_max_err']:.2e}  -> OK: {theorems['conc_connectivity_ok']}")
        print(f"Connectivity theorem (flux), max |C_J @ eps| = "
              f"{theorems['flux_connectivity_max_err']:.2e}  -> OK: {theorems['flux_connectivity_ok']}")

    return {
        "eps_v_Cbal": eps_v_Cbal_df,
        "eps_v_Cimb": eps_v_Cimb_df,
        "C_C": C_C_df,
        "C_J": C_J_df,
        "C_opt": C_opt,
        "v_opt": pd.Series(v_opt, index=model.flux_keys),
        "theorems": theorems,
    }


if __name__ == "__main__":
    print("Import run_mca(model, enzymes, kinetic_params, cell_needs, condition_key=...) "
          "and call it with your fitted model/parameters/condition.")
