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
    per_condition_diag = []

    for cond in conditions:
        cond_key = cond['condition']
        try:
            enzymes    = input_enzyme.loc[cond_key, :].to_dict()
            cell_needs = input_cell_needs.loc[cond_key, :].to_dict()
            df_opt, _  = model.solve_steady_state(enzymes, params_estimate, cell_needs,
                                                  condition_key=cond_key)

            y      = df_opt.iloc[0][out_keys].values.astype(float)
            y_safe = np.where(np.abs(y) < 1e-12, 1e-12, y)

            G_df, diag = model.gen_sensitivity_matrix(enzymes, params_estimate,
                                                      cell_needs, cond_key,
                                                      return_diagnostics=True)
            per_condition_diag.append(diag)
            G_abs  = np.asarray(G_df)
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

    report = build_structural_report(
        per_condition_diag,
        pd.DataFrame(FIM, index=model.params_keys, columns=model.params_keys),
        {k: float(theta[i]) for i, k in enumerate(model.params_keys)},
        list(model.params_keys),
        influence=pd.Series(np.abs(G_total).max(axis=0), index=model.params_keys),
        fim_is_relative=True,
    )

    return G_total, corr, {
        "FIM": FIM, "cov": cov, "stds": stds_safe, "n_par": n_par,
        "G_list": G_list, "measured": measured, "skipped": skipped,
        "unidentifiable": np.where(unidentifiable)[0].tolist(),
        "report": report,
        "per_condition_diag": per_condition_diag,
    }


def build_structural_report(per_condition_diag, fim_df, theta, params,
                            influence=None, corr_threshold=0.9, ident_tol=1e-8,
                            fim_is_relative=False):
    """
    Assemble the a-priori structural-analysis report as three tidy DataFrames.

    Identifiability is judged in the scale-free (relative / log) metric: the FIM is
    scaled by the parameter magnitudes, F_rel = diag(theta) F diag(theta) (skipped if
    the FIM is already relative).  Estimability is then decided by SUBSPACE PROJECTION
    onto the identifiable eigen-directions (eigenvalue > tol * max), so the
    per-parameter count is consistent with the FIM rank -- unlike a raw pinv-variance
    threshold, which can label rank-deficient directions as "precise".

    Parameters
    ----------
    per_condition_diag : list[dict]
        Per-condition operating-point diagnostics from
        EcoliCarbonKinetics.gen_sensitivity_matrix(..., return_diagnostics=True).
    fim_df : pd.DataFrame   Fisher Information Matrix (params x params), labeled.
    theta : dict or array   Parameter values (scale), aligned with ``params``.
    params : list[str]      Parameter names (FIM row/col order).
    influence : pd.Series or None   optional per-parameter influence (max |sensitivity|).
    corr_threshold : float  threshold for counting strongly correlated parameter pairs.
    ident_tol : float       relative eigenvalue tolerance for FIM rank / estimability.
    fim_is_relative : bool  True if ``fim_df`` is already in the relative metric.

    Returns
    -------
    dict[str, pd.DataFrame]
        'per_condition', 'identifiability', 'per_parameter'.
    """
    cols = ["ss_residual", "rank_A", "cond_A", "n_bound_active", "rank_G"]
    if per_condition_diag:
        pc = pd.DataFrame(per_condition_diag)
        if "condition" in pc.columns:
            pc = pc.set_index("condition")
        per_condition = pc[[c for c in cols if c in pc.columns]]
    else:
        per_condition = pd.DataFrame(columns=cols)

    if isinstance(theta, dict):
        theta_vals = np.array([float(theta[p]) for p in params])
    else:
        theta_vals = np.asarray(theta, dtype=float)
    scale = np.abs(theta_vals)

    F = np.asarray(fim_df, dtype=float)
    n = F.shape[0]
    F_rel = F if fim_is_relative else F * np.outer(scale, scale)
    F_rel = 0.5 * (F_rel + F_rel.T)

    if n:
        eigvals, eigvecs = np.linalg.eigh(F_rel)
        max_eig = float(eigvals.max())
        keep = eigvals > max_eig * ident_tol if max_eig > 0 else np.zeros(n, dtype=bool)
        fim_rank = int(keep.sum())
        pos = eigvals[keep]
        fim_cond = float(max_eig / pos.min()) if fim_rank else float("inf")
        # fraction of each parameter axis captured by the identifiable eigen-subspace
        subspace_proj = (eigvecs[:, keep] ** 2).sum(axis=1) if fim_rank else np.zeros(n)
        cov_rel = np.linalg.pinv(F_rel)
        rel_std = np.sqrt(np.clip(np.diag(cov_rel), 0.0, None))   # relative std = CV fraction
    else:
        fim_rank, fim_cond = 0, float("inf")
        subspace_proj = np.zeros(0)
        cov_rel = np.zeros((0, 0))
        rel_std = np.zeros(0)

    identifiable = subspace_proj > 0.5

    rel_std_safe = rel_std.copy()
    rel_std_safe[~identifiable] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov_rel / np.outer(rel_std_safe, rel_std_safe) if n else np.zeros((0, 0))
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (identifiable[i] and identifiable[j]
                    and np.isfinite(corr[i, j]) and abs(corr[i, j]) >= corr_threshold):
                n_pairs += 1

    identifiability = pd.DataFrame([{
        "n_free":          n,
        "FIM_rank":        fim_rank,
        "FIM_cond":        fim_cond,
        "n_identifiable":  int(identifiable.sum()),
        "n_non_estimable": int((~identifiable).sum()),
        f"n_corr_pairs_ge_{corr_threshold}": n_pairs,
    }])

    cv_percent = np.where(identifiable, 100.0 * rel_std, np.nan)
    abs_std = np.where(identifiable, rel_std * scale, np.nan)
    per_parameter = pd.DataFrame({
        "std":           abs_std,
        "cv_percent":    cv_percent,
        "identifiable":  identifiable,
        "subspace_proj": subspace_proj,
    }, index=params)
    if influence is not None:
        per_parameter["max_abs_sens"] = pd.Series(influence).reindex(params).to_numpy()
    return {
        "per_condition":   per_condition,
        "identifiability": identifiability,
        "per_parameter":   per_parameter,
    }
