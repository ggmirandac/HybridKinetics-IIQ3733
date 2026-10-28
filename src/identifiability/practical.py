"""
practical.py -- practical identifiability at the current anchor (reduced-sensitivity FIM and its Schur complement).

J = Q^-1/2 G D, stacked over conditions and measured outputs (G = dy*/dtheta from the reduced sensitivities,
Q = diag(sigma^2), D = diag(theta)), so F = J^T J is the Fisher information in the log-parameters.  The criteria
(rank at the eigenvalue cut, singular spectrum, correlations from the pseudo-inverse, influence, flags of unreliable
finite differences) are computed on the Schur complement F_aa - F_ab pinv(F_bb) F_ba, where b are the nuisance
parameters ("kept_free_by_rule": output-level non-identifiable but protected from fixing) and a the other free
parameters.  Profiling the nuisance parameters out keeps them from making the others look (non-)identifiable.

Position in the pipeline: input of the SVD / correlation / flag rules of every iteration of the parameter selection
(selection.py) and of the final analysis (final_analysis.py).
"""
import time

import numpy as np
import pandas as pd

from sensitivity import reduced_sensitivity

def pinv_psd(F, rel_cut):
    """Pseudo-inverse of a symmetric PSD matrix; eigenvalues <= rel_cut * max are dropped."""
    eigenvalues, vectors = np.linalg.eigh(0.5 * (F + F.T))
    keep = eigenvalues > rel_cut * max(eigenvalues.max(), 1e-300)
    inverse = np.zeros_like(eigenvalues)
    inverse[keep] = 1.0 / eigenvalues[keep]
    return (vectors * inverse[None, :]) @ vectors.T

def weighted_jacobian(sensitivity, theta):
    """
    (J, R): J = sigma-weighted sensitivities times theta (rows = conditions x measured outputs, columns = the free
    parameters), R = d ln y / d ln theta (rows with |y| < 1e-8 set to 0).  NaN entries enter as 0 only because J
    cannot hold NaN; they are counted and flagged by the reduced sensitivities.
    """
    j_blocks, r_blocks = [], []
    for condition, columns in sensitivity["sensitivities"].items():
        values = columns.to_numpy(dtype=float)
        # C order: the column means below sum in memory order, and C order reproduces the published numbers
        values = np.ascontiguousarray(np.where(np.isnan(values), 0.0, values))
        weight = 1.0 / sensitivity["sigma"][condition].to_numpy(dtype=float)
        y = sensitivity["y0"][condition].to_numpy(dtype=float)
        safe = np.abs(y) >= 1e-8
        j_blocks.append(values * theta[None, :] * weight[:, None])
        r_blocks.append(np.where(safe[:, None], values * theta[None, :] / np.where(safe, y, 1.0)[:, None], 0.0))
    return np.vstack(j_blocks), np.vstack(r_blocks)

def schur_columns(J, analysed_columns, nuisance_columns, eig_cut):
    """J_S = (I - P_b) J_a: the analysed columns projected off the column space of the nuisance columns."""
    J_a = J[:, analysed_columns]
    if not nuisance_columns:
        return J_a
    U_b, singular_b, _ = np.linalg.svd(J[:, nuisance_columns], full_matrices=False)
    keep = singular_b > np.sqrt(eig_cut) * max(singular_b[0], 1e-300)
    U_b = U_b[:, keep]
    return J_a - U_b @ (U_b.T @ J_a)

def correlation_from_fim(F, names, eig_cut):
    """Correlation matrix (DataFrame) from the pseudo-inverse of F; entries of a vanishing variance are 0."""
    P = pinv_psd(F, eig_cut)
    variance = np.diag(P)
    usable = variance > 1e-14 * max(variance.max(), 1e-300)
    sd = np.sqrt(np.where(usable, variance, 1.0))
    R = P / np.outer(sd, sd)
    R[~usable, :] = 0.0
    R[:, ~usable] = 0.0
    np.fill_diagonal(R, 1.0)
    return pd.DataFrame(np.clip(R, -1.0, 1.0), index=names, columns=names)

def size_of_r(pair):
    return abs(pair[2])

def correlated_pairs(corr, threshold):
    """[(p1, p2, r)] with |r| > threshold, by decreasing |r| (pairs with equal |r| keep their order)."""
    names = list(corr.index)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = float(corr.iloc[i, j])
            if abs(r) > threshold:
                pairs.append((names[i], names[j], r))
    pairs.sort(key=size_of_r, reverse=True)
    return pairs

def practical_analysis(est, nuisance, params):
    """
    Practical identifiability analysis at the current anchor of ``est`` with the nuisance set ``nuisance``.  The
    settings are corr_threshold, zero_cut, eig_cut (the ONE rank cut, an eigenvalue ratio of F: 1e-8, i.e. singular
    ratio 1e-4) and those of the reduced sensitivities (sens_step, sens_step_check, fd_consistency_*, n_jobs).

    Returns a dict: fp (free), fa (analysed = free minus nuisance), nuisance, theta, J, JS (Schur-projected columns
    of fa), infl (RMS of the columns of J: sigma-weighted relative influence, all free), infl_S (the same on JS),
    infl_dlny (RMS of d ln y / d ln theta), zero (parameters of fa with influence < zero_cut * max), nz (the others
    of fa), s (singular values of JS), s_nz / Vt_nz (SVD of JS restricted to nz), ranks {cut: rank of F_S at that
    eigenvalue ratio}, full_rank, sigma_ratio, sigma_ratio_nz, cond, corr (over nz, from the pseudo-inverse), pairs
    [(p1, p2, r)] with |r| > corr_threshold (meaningful only at full rank), cov_abs (over fp: the a block
    D pinv(F_S) D, the nuisance block D pinv(F_bb) D, no cross terms), cov_abs_nuis, flags {param: [at_bound,
    fd_unstable, fd_failed]}, flagged (fa with a flag), flagged_nuisance, report (of the reduced sensitivities),
    eig_cut, wall_s.
    """
    started = time.time()
    eig_cut = params["eig_cut"]
    sensitivity = reduced_sensitivity(est, params, check=True)
    report = dict(sensitivity["report"])
    free = list(est.free_params)
    nuisance = [p for p in free if p in set(nuisance)]
    analysed = [p for p in free if p not in set(nuisance)]
    analysed_columns = [free.index(p) for p in analysed]
    nuisance_columns = [free.index(p) for p in nuisance]
    theta_full = est.full_theta()
    theta = np.array([float(theta_full[p]) for p in free])
    J, R = weighted_jacobian(sensitivity, theta)
    influence = pd.Series(np.sqrt(np.mean(J ** 2, axis=0)), index=free)
    influence_dlny = pd.Series(np.sqrt(np.mean(R ** 2, axis=0)), index=free)
    JS = schur_columns(J, analysed_columns, nuisance_columns, eig_cut)
    influence_schur = pd.Series(np.sqrt(np.mean(JS ** 2, axis=0)), index=analysed)
    zero = [p for p in analysed if influence[p] < params["zero_cut"] * influence.max()]
    nonzero = [p for p in analysed if p not in zero]
    nonzero_columns = [analysed.index(p) for p in nonzero]
    singular = np.linalg.svd(JS, compute_uv=False)
    _, singular_nz, Vt_nz = np.linalg.svd(JS[:, nonzero_columns], full_matrices=False)
    ranks = {}
    for cut in sorted({1e-6, 1e-8, 1e-10, eig_cut}):
        ranks[cut] = int(np.sum(singular ** 2 / singular[0] ** 2 > cut))
    corr = correlation_from_fim(JS[:, nonzero_columns].T @ JS[:, nonzero_columns], nonzero, eig_cut)
    covariance = np.zeros((len(free), len(free)))
    covariance[np.ix_(analysed_columns, analysed_columns)] = (
        pinv_psd(JS.T @ JS, eig_cut) * np.outer(theta[analysed_columns], theta[analysed_columns]))
    covariance_nuisance = pd.DataFrame(np.zeros((len(nuisance), len(nuisance))), index=nuisance, columns=nuisance)
    if nuisance_columns:
        J_b = J[:, nuisance_columns]
        block = pinv_psd(J_b.T @ J_b, eig_cut) * np.outer(theta[nuisance_columns], theta[nuisance_columns])
        covariance[np.ix_(nuisance_columns, nuisance_columns)] = block
        covariance_nuisance = pd.DataFrame(block, index=nuisance, columns=nuisance)
    flags = {p: list(report["flags"].get(p, [])) for p in free}
    return {"fp": free, "fa": analysed, "nuisance": nuisance, "theta": {p: float(theta_full[p]) for p in free},
            "J": J, "JS": JS, "infl": influence, "infl_S": influence_schur, "infl_dlny": influence_dlny,
            "zero": zero, "nz": nonzero, "s": singular, "s_nz": singular_nz, "Vt_nz": Vt_nz, "ranks": ranks,
            "full_rank": bool(ranks[eig_cut] == len(analysed)), "sigma_ratio": float(singular[-1] / singular[0]),
            "sigma_ratio_nz": float(singular_nz[-1] / singular_nz[0]),
            "cond": float((singular[0] / max(singular[-1], 1e-300)) ** 2), "corr": corr,
            "pairs": correlated_pairs(corr, params["corr_threshold"]),
            "cov_abs": pd.DataFrame(covariance, index=free, columns=free), "cov_abs_nuis": covariance_nuisance,
            "flags": flags, "flagged": [p for p in analysed if flags[p]],
            "flagged_nuisance": [p for p in nuisance if flags[p]], "report": report, "eig_cut": float(eig_cut),
            "wall_s": time.time() - started}
