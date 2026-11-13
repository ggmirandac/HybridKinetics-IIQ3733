"""
practical.py -- practical identifiability at the current fit (Fisher information matrix).

J = Q^-1/2 G D, stacked over conditions and measured outputs (G = dy*/dtheta, the reduced sensitivities of
sensitivity.py, Q = diag(sigma^2), D = diag(theta)), so F = J^T J is the Fisher information in the log-parameters.
The criteria:
  - rank: F is rank deficient if the smallest singular value of J is below rank_tol times the largest;
  - correlation: r = C_ij / sqrt(C_ii C_jj) with C = F^-1 (full rank only);
  - CV: the standard deviation of ln theta, sqrt(C_ii), which is the relative standard deviation of theta;
  - sensitivity: the mean of the normalized sensitivity coefficients |d ln y*/d ln theta| over the conditions and
    measured outputs (how much the model outputs change, in relative terms, per relative change of the parameter).
    Entries of outputs below NEAR_ZERO_FRACTION of their median over the conditions are left out, because the
    relative sensitivity diverges as an output approaches zero.

Position in the pipeline: the practical step of every iteration of the parameter selection and the final structure
(selection.py).
"""
import numpy as np
import pandas as pd

from sensitivity import reduced_sensitivity

# An output below this fraction of its median over the conditions is left out of the sensitivity
NEAR_ZERO_FRACTION = 0.01

def output_medians(y_by_condition):
    """Median |y*| of every measured output over the conditions (conditions without a solution are skipped)."""
    table = pd.DataFrame(y_by_condition).T
    return table.abs().median()

def sensitivity_matrices(est):
    """
    (J, relative, failed) at the current fit of ``est``; rows = conditions x measured outputs, columns = the free
    parameters:
      J = sigma-weighted sensitivities times theta (the rows of the conditions without sensitivities are zero);
      relative = |d ln y* / d ln theta| (NaN for the conditions without sensitivities and for the outputs below
      NEAR_ZERO_FRACTION of their median);
      failed = the conditions without sensitivities.
    """
    sensitivity = reduced_sensitivity(est)
    theta_full = est.full_theta()
    theta = np.array([float(theta_full[p]) for p in est.free_params])
    medians = output_medians(sensitivity["y0"])
    weighted_blocks = []
    relative_blocks = []
    for condition, columns in sensitivity["sensitivities"].items():
        values = columns.to_numpy(dtype=float)
        weight = 1.0 / sensitivity["sigma"][condition].to_numpy(dtype=float)
        weighted_blocks.append(np.nan_to_num(values, nan=0.0) * theta[None, :] * weight[:, None])
        y = sensitivity["y0"][condition]
        relative = np.abs(values * theta[None, :] / y.to_numpy(dtype=float)[:, None])
        near_zero = (y.abs() < NEAR_ZERO_FRACTION * medians[y.index]).to_numpy()
        relative[near_zero, :] = np.nan
        relative_blocks.append(relative)
    J = pd.DataFrame(np.vstack(weighted_blocks), columns=list(est.free_params))
    relative = pd.DataFrame(np.vstack(relative_blocks), columns=list(est.free_params))
    return J, relative, sensitivity["failed"]

def size_of_r(pair):
    return abs(pair[2])

def correlated_pairs(corr, threshold):
    """[(p1, p2, r)] with |r| > threshold, by decreasing |r|."""
    names = list(corr.index)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = float(corr.iloc[i, j])
            if abs(r) > threshold:
                pairs.append((names[i], names[j], r))
    pairs.sort(key=size_of_r, reverse=True)
    return pairs

def mean_sensitivity(relative, names):
    """The sensitivity of every parameter in ``names``: the mean of its column of ``relative`` (NaN entries skipped)."""
    return pd.Series(np.nanmean(relative[names].to_numpy(), axis=0), index=names)

def fim_analysis(J, relative, names, params):
    """
    The practical criteria of the columns ``names`` of J and ``relative``.  Returns a dict: names, sensitivity
    (Series), sigma_ratio (singular values / largest), null_vector (right singular vector of the smallest singular
    value), full_rank, and at full rank corr (DataFrame), cv_percent (Series), pairs [(p1, p2, r)] with
    |r| > corr_threshold, max_abs_r.
    """
    columns = J[names].to_numpy()
    sensitivity = mean_sensitivity(relative, names)
    _, singular, Vt = np.linalg.svd(columns, full_matrices=False)
    ratio = singular / singular[0]
    analysis = {"names": list(names), "sensitivity": sensitivity, "sigma_ratio": ratio, "null_vector": Vt[-1],
                "full_rank": bool(ratio[-1] >= params["rank_tol"]), "corr": None, "cv_percent": None, "pairs": [],
                "max_abs_r": np.nan}
    if not analysis["full_rank"]:
        return analysis
    covariance = np.linalg.inv(columns.T @ columns)
    sd = np.sqrt(np.diag(covariance))
    corr = pd.DataFrame(covariance / np.outer(sd, sd), index=names, columns=names)
    analysis["corr"] = corr
    analysis["cv_percent"] = pd.Series(100.0 * sd, index=names)
    analysis["pairs"] = correlated_pairs(corr, params["corr_threshold"])
    if len(names) > 1:
        off_diagonal = np.abs(corr.to_numpy() - np.eye(len(names)))
        analysis["max_abs_r"] = float(off_diagonal.max())
    return analysis
