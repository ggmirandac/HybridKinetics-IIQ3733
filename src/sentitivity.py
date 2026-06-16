import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import casadi as ca


def compute_sensitivity(
    model,
    params_estimate: dict,
    input_enzyme: pd.DataFrame,
    input_cell_needs: pd.DataFrame,
    conditions: list,
    measurement_error: dict,
):
    """
    Sensitivity and parameter-correlation analysis across multiple steady-state conditions.

    Parameters
    ----------
    model : EcoliCarbonKinetics
    params_estimate : dict  kinetic parameters keyed by model.params_keys
    input_enzyme : pd.DataFrame  shape (n_conditions, n_enzymes), conditions as rows
    input_cell_needs : pd.DataFrame  shape (n_conditions, n_balanced), conditions as rows
    conditions : list of dicts  e.g. [{'condition': 'KO02'}, ...]
    measurement_error : dict  output_key -> measurement SD (np.nan means unmeasured)
                         Q is shared across all conditions (property of the instrument).

    Returns
    -------
    G_total : ndarray (n_cond * n_meas, n_par)  stacked relative sensitivity matrices
    corr : ndarray (n_par, n_par)  parameter correlation matrix (Cramer-Rao bound)
    diag : dict  FIM, rank, n_par, G_list, measured
    """
    theta   = np.array([params_estimate[k] for k in model.params_keys])
    n_par   = len(theta)
    out_keys = model.balanced_keys + model.flux_keys          # 18 outputs (rows of G)
    measured = [k for k in out_keys if np.isfinite(measurement_error.get(k, np.nan))]
    meas_idx = [out_keys.index(k) for k in measured]

    # Q is unique across conditions: build Q_inv once from absolute SDs
    q_diag = np.array([measurement_error[k] ** 2 for k in measured])
    Q_inv  = np.diag(1.0 / q_diag)

    G_list   = []
    FIM      = np.zeros((n_par, n_par))
    skipped  = []

    for cond in conditions:
        cond_key = cond['condition']
        try:
            enzymes    = input_enzyme.loc[cond_key, :].to_dict()
            cell_needs = input_cell_needs.loc[cond_key, :].to_dict()
            df_opt, _  = model.solve_steady_state(enzymes, params_estimate, cell_needs,
                                                  condition_key=cond_key)

            y      = df_opt.iloc[0][out_keys].values.astype(float)
            y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)

            G_abs  = np.asarray(model.gen_sensitivity_matrix(enzymes, params_estimate,
                                                             cell_needs, cond_key))
            # relative (log) sensitivity: d ln y / d ln theta
            G_rel  = G_abs * (theta[np.newaxis, :] / y_safe[:, np.newaxis])

            G_cond = G_rel[meas_idx, :]
            G_list.append(G_cond)
            FIM   += G_cond.T @ Q_inv @ G_cond

        except Exception as exc:
            print(f"  [compute_sensitivity] Skipping condition {cond_key!r}: {exc}")
            skipped.append(cond_key)


    G_total = np.vstack(G_list)
    print(G_total.shape, "total measurements across all conditions,", n_par, "parameters.")

    # Diagnose FIM conditioning before attempting inversion
    FIM = 0.5 * (FIM + FIM.T)  # enforce symmetry before eigendecomposition because
                               # floating-point noise in the sum of G.T @ Q_inv @ G can make FIM 
                               # non-symmetric enough to produce small imaginary parts in eigvals.
    fim_eigvals = np.linalg.eigvalsh(FIM)
    cond_number = fim_eigvals.max() / max(fim_eigvals[fim_eigvals > 0].min(), np.finfo(float).tiny)
    print(f"  FIM condition number: {cond_number:.2e}  (rank-deficient if >> 1e12)")
    near_zero = fim_eigvals < fim_eigvals.max() * 1e-10
    if near_zero.any():
        print(f"  Near-zero eigenvalues: {near_zero.sum()} -- FIM is numerically rank-deficient.")

    # Use pseudoinverse: handles near-singular FIM via SVD truncation of small eigenvalues.
    # For well-determined parameters pinv == inv; for near-zero eigenvalue directions
    # it returns zero variance instead of the numerical garbage that inv would produce.
    cov  = np.linalg.pinv(FIM)
    var  = np.diag(cov).clip(0)    # clip tiny negatives from floating-point noise
    stds = np.sqrt(var)

    # Mark parameters whose variance is negligible: they are not estimable from this data
    max_std = stds.max()
    if max_std > 0:
        unidentifiable = stds < max_std * 1e-8
    else:
        unidentifiable = np.ones(n_par, dtype=bool)

    if unidentifiable.any():
        unident_names = [model.params_keys[i] for i in np.where(unidentifiable)[0]]
        print(f"  Non-estimable parameters (near-zero FIM contribution): {unident_names}")

    stds_safe = stds.copy()
    stds_safe[unidentifiable] = np.nan

    outer_stds = np.outer(stds_safe, stds_safe)
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = cov / outer_stds
    np.fill_diagonal(corr, np.where(~unidentifiable, 1.0, np.nan))

    # Enforce exact symmetry: floating-point error in pinv of an ill-conditioned FIM
    # can make corr[i,j] and corr[j,i] differ enough to break threshold-based masks.
    corr = (corr + corr.T) / 2

    return G_total, corr, {
        "FIM": FIM, "cov": cov, "stds": stds_safe, "n_par": n_par,
        "G_list": G_list, "measured": measured, "skipped": skipped,
        "unidentifiable": np.where(unidentifiable)[0].tolist(),
    }
