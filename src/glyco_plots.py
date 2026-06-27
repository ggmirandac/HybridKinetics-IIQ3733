"""
glyco_plots.py
==============
Reusable matplotlib plotting functions for the E. coli glycolysis parameter-
estimation pipeline.

All functions are pure-matplotlib: they accept DataFrames, Series, or a results
directory path as inputs and return a (fig, ax) or (fig, axes) tuple so callers
can further customize before saving.  There are NO imports from the estimator
module -- this module can be used independently of any CasADi or Pyomo dependency,
making it suitable for post-processing and report generation from saved CSV/JSON
artifacts alone.

Usage example (in analysis.ipynb)
----------------------------------
    import sys; sys.path.insert(0, "src")
    import glyco_plots as gp

    res = gp.load_results("results/first_estimation")
    fig, ax = gp.plot_theta_init_vs_fitted(
        res["theta_init"], res.get("theta_fitted"),
        savepath="results/first_estimation/figures/theta_comparison.png"
    )
"""

import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Result loader
# ---------------------------------------------------------------------------
def load_results(results_dir):
    """
    Load all artifacts from a results directory produced by first_estimation.ipynb.

    Returns a dict with keys:
        theta_init         pd.Series (37 params)
        theta_fitted       pd.Series or None
        theta_init_sources pd.DataFrame [param, value, source]
        covariance         pd.DataFrame or None
        correlation        pd.DataFrame or None
        confidence_intervals pd.DataFrame or None
        fim                pd.DataFrame or None
        predictions_init   pd.DataFrame or None
        predictions_fitted pd.DataFrame or None
        real               pd.DataFrame or None
        rmse_init          dict or None
        rmse_fitted        dict or None
        sensitivity        dict {condition: DataFrame} or None
        manifest           dict or None
    """
    def _csv(name, index_col=0):
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            return pd.read_csv(path, index_col=index_col)
        return None

    def _json(name):
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
        return None

    def _pkl(name):
        path = os.path.join(results_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as fh:
                return pickle.load(fh)
        return None

    def _series_from_csv(df):
        if df is None:
            return None
        if "value" in df.columns:
            return pd.Series(df["value"].values, index=df.index)
        if df.shape[1] == 1:
            return df.iloc[:, 0]
        return df.iloc[:, 0]

    theta_init_df = _csv("theta_init.csv")
    theta_init = _series_from_csv(theta_init_df)

    theta_fitted_df = _csv("theta_fitted.csv")
    theta_fitted = _series_from_csv(theta_fitted_df)

    sensitivity = _pkl("sensitivity.pkl")
    if sensitivity is None:
        sens_dir = os.path.join(results_dir, "sensitivity")
        if os.path.isdir(sens_dir):
            sensitivity = {}
            for fname in sorted(os.listdir(sens_dir)):
                if fname.endswith(".csv"):
                    cond = fname.replace(".csv", "")
                    sensitivity[cond] = pd.read_csv(
                        os.path.join(sens_dir, fname), index_col=[0, 1]
                    )

    return {
        "theta_init":           theta_init,
        "theta_fitted":         theta_fitted,
        "theta_init_sources":   _csv("theta_init_sources.csv"),
        "covariance":           _csv("covariance.csv"),
        "correlation":          _csv("correlation.csv"),
        "confidence_intervals": _csv("confidence_intervals.csv"),
        "fim":                  _csv("fim.csv"),
        "predictions_init":     _csv("predictions_init.csv"),
        "predictions_fitted":   _csv("predictions_fitted.csv"),
        "real":                 _csv("real.csv"),
        "rmse_init":            _json("rmse_init.json"),
        "rmse_fitted":          _json("rmse_fitted.json"),
        "sensitivity":          sensitivity,
        "manifest":             _json("manifest.json"),
    }

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _save_or_show(fig, savepath):
    """Save figure to savepath if given, otherwise call plt.show()."""
    if savepath is not None:
        os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight", dpi=150)
        plt.close(fig)
    return fig

# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------
def plot_theta_init_vs_fitted(theta_init, theta_fitted=None, savepath=None):
    """
    Scatter / bar comparison of initial vs. fitted parameter values.

    Parameters
    ----------
    theta_init   : pd.Series indexed by parameter name
    theta_fitted : pd.Series indexed by parameter name, or None
    savepath     : str or None

    Returns
    -------
    (fig, ax)
    """
    params = list(theta_init.index)
    x = np.arange(len(params))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(x, theta_init.values, s=30, label="init", color="steelblue", zorder=3)
    if theta_fitted is not None:
        common = [p for p in params if p in theta_fitted.index]
        xi = [i for i, p in enumerate(params) if p in theta_fitted.index]
        ax.scatter(xi, [theta_fitted[p] for p in common],
                   s=30, marker="x", label="fitted", color="tomato", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=90, fontsize=6)
    ax.set_yscale("log")
    ax.set_ylabel("parameter value (log scale)")
    ax.set_title("Parameter values: init vs. fitted")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save_or_show(fig, savepath), ax

def plot_correlation_heatmap(corr_df, title="Parameter correlation", savepath=None):
    """
    Heatmap of the parameter correlation matrix.

    Parameters
    ----------
    corr_df  : pd.DataFrame (n_params x n_params), values in [-1, 1]
    title    : str
    savepath : str or None

    Returns
    -------
    (fig, ax)
    """
    n = len(corr_df)
    figsize = max(6, n * 0.32)
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.85))
    im = ax.imshow(corr_df.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr_df.index, fontsize=5)
    fig.colorbar(im, ax=ax, label="r_ij", fraction=0.03, pad=0.02)
    ax.set_title(title)
    fig.tight_layout()
    return _save_or_show(fig, savepath), ax

def plot_ci_errorbars(ci_df, savepath=None):
    """
    Confidence-interval error-bar plot (one bar per free parameter).

    Parameters
    ----------
    ci_df    : pd.DataFrame with columns [theta, ci_low, ci_high]
               (output of GlycolysisParameterEstimator.confidence_intervals())
    savepath : str or None

    Returns
    -------
    (fig, ax)
    """
    params = list(ci_df.index)
    theta = ci_df["theta"].values
    lo = theta - ci_df["ci_low"].values
    hi = ci_df["ci_high"].values - theta
    x = np.arange(len(params))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.errorbar(x, theta, yerr=[lo, hi], fmt="o", ms=4, capsize=3,
                color="steelblue", ecolor="gray", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(params, rotation=90, fontsize=6)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_ylabel("parameter value (symlog)")
    ax.set_title("95% confidence intervals")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save_or_show(fig, savepath), ax

def plot_fim_spectrum(fim_df, savepath=None):
    """
    Log-scale eigenvalue spectrum of the Fisher Information Matrix.

    Eigenvalues sorted descending.  A steep drop reveals near-unidentifiable
    parameter combinations (the rank deficit visualized).

    Parameters
    ----------
    fim_df   : pd.DataFrame (n x n), symmetric positive semi-definite
    savepath : str or None

    Returns
    -------
    (fig, ax)
    """
    F = fim_df.to_numpy(dtype=float)
    eigvals = np.sort(np.linalg.eigvalsh(F))[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(range(1, len(eigvals) + 1), np.clip(eigvals, 1e-30, None),
                "o-", ms=5, color="steelblue")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue (log scale)")
    ax.set_title("FIM eigenvalue spectrum")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save_or_show(fig, savepath), ax

def plot_pred_vs_meas(pred_df, real_df, title="Predicted vs. measured", savepath=None):
    """
    Parity (predicted vs. measured) scatter plot over all conditions and outputs.

    Parameters
    ----------
    pred_df  : pd.DataFrame (conditions x outputs), predicted
    real_df  : pd.DataFrame (conditions x outputs), measured (NaN where missing)
    title    : str
    savepath : str or None

    Returns
    -------
    (fig, ax)
    """
    pv = pred_df.to_numpy(dtype=float).ravel()
    rv = real_df.to_numpy(dtype=float).ravel()
    mask = ~(np.isnan(pv) | np.isnan(rv))
    pv, rv = pv[mask], rv[mask]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(rv, pv, s=20, alpha=0.7, color="steelblue", edgecolors="none")
    lim = [min(rv.min(), pv.min()) * 0.9, max(rv.max(), pv.max()) * 1.1]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("measured")
    ax.set_ylabel("predicted")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    return _save_or_show(fig, savepath), ax

def plot_residuals(pred_df, real_df, title="Residuals (pred - meas)", savepath=None):
    """
    Bar chart of per-output mean residual (predicted - measured) and +/- std.

    Parameters
    ----------
    pred_df  : pd.DataFrame (conditions x outputs)
    real_df  : pd.DataFrame (conditions x outputs)
    title    : str
    savepath : str or None

    Returns
    -------
    (fig, ax)
    """
    residual = (pred_df - real_df)
    out_cols = list(pred_df.columns)
    means, stds = [], []
    for col in out_cols:
        vals = residual[col].dropna().to_numpy(dtype=float)
        means.append(float(np.mean(vals)) if len(vals) > 0 else float("nan"))
        stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
    x = np.arange(len(out_cols))
    colors = ["tab:blue" if c.startswith("C_") else "tab:red" for c in out_cols]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(x, means, color=colors, alpha=0.7, label="mean residual")
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", capsize=3, lw=1)
    ax.axhline(0, color="k", lw=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(out_cols, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("pred - meas (own units)")
    ax.set_title(title)
    fig.tight_layout()
    return _save_or_show(fig, savepath), ax

def plot_sensitivity_heatmap(sens_dict, condition=None, log_scale=True, savepath=None):
    """
    Heatmap of the sensitivity matrix G = d(outputs)/d(params) for one condition.

    Parameters
    ----------
    sens_dict  : dict {condition: pd.DataFrame (18 x 37) with MultiIndex rows}
                 Output of GlycolysisParameterEstimator.sensitivity_matrix().
                 The DataFrame may also be flat-indexed (from CSV load).
    condition  : str or None; if None uses the first condition in the dict
    log_scale  : bool; if True plots |G| in log10 scale
    savepath   : str or None

    Returns
    -------
    (fig, ax)
    """
    if condition is None:
        condition = next(iter(sens_dict))
    G_df = sens_dict[condition]
    G = G_df.to_numpy(dtype=float)
    if log_scale:
        with np.errstate(divide="ignore", invalid="ignore"):
            data = np.log10(np.abs(G) + 1e-30)
        cmap = "viridis"
        label = "log10(|G|)"
    else:
        data = G
        cmap = "RdBu_r"
        label = "G value"
    row_labels = [str(r) for r in G_df.index]
    col_labels = list(G_df.columns)
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=5)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6)
    fig.colorbar(im, ax=ax, label=label, fraction=0.02, pad=0.02)
    ax.set_title(f"Sensitivity matrix G  (condition: {condition})")
    fig.tight_layout()
    return _save_or_show(fig, savepath), ax

def plot_rmse_summary(rmse_init, rmse_fitted=None, savepath=None):
    """
    Bar chart comparing RMSE metrics before and after fitting.

    Parameters
    ----------
    rmse_init   : dict with keys 'met', 'flux', 'data_norm', 'weighted'
    rmse_fitted : dict or None; same structure
    savepath    : str or None

    Returns
    -------
    (fig, ax)
    """
    keys = [k for k in ("met", "flux", "data_norm", "weighted")
            if k in rmse_init and np.isfinite(rmse_init[k])]
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, [rmse_init[k] for k in keys], width,
           label="init", color="steelblue", alpha=0.8)
    if rmse_fitted is not None:
        vals_fit = [rmse_fitted.get(k, float("nan")) for k in keys]
        ax.bar(x + width / 2, vals_fit, width,
               label="fitted", color="tomato", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_yscale("log")
    ax.set_ylabel("RMSE (log scale)")
    ax.set_title("RMSE summary: init vs. fitted")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _save_or_show(fig, savepath), ax
