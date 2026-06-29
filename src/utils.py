"""Cross-cutting constants, data-loading helpers, and metabolite-bounds functions
for the E. coli glycolysis model (IIQ3733).

Provides shared constants (REACTION_GROUPS, PARAM_LATEX_LABELS, TYPE_COLOR,
KCAT0, PTS0), canonical key lists derived from EcoliCarbonKinetics, re-exports
from sample_parameters (kcat_convert, ENZYME_MW_KDA), a lightweight CSV loader
(load_data_frames), and the three pipeline functions moved verbatim from
sensitivity_pipeline (metabolite_bounds, build_analysis_model,
save_per_condition_sensitivity). Plot functions (styled_matrix_heatmap, etc.)
intentionally live inline in the plot notebooks and are not included here.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from kinetics_noor import EcoliCarbonKinetics, ALL_PARAMS
from sample_parameters import kcat_convert, ENZYME_MW_KDA

# ---------------------------------------------------------------------------
# Canonical key lists (convenience wrappers around class attributes)
# ---------------------------------------------------------------------------
BALANCED_KEYS = list(EcoliCarbonKinetics.balanced_keys)
FLUX_KEYS = list(EcoliCarbonKinetics.flux_keys)
IMBALANCED_KEYS = list(EcoliCarbonKinetics.imbalanced_keys)
ENZYMES_KEYS = list(EcoliCarbonKinetics.enzymes_keys)
PARAMS_KEYS = list(EcoliCarbonKinetics.params_keys)

# ---------------------------------------------------------------------------
# Plotting constants (extracted verbatim from plots_first_estimation.ipynb
# cell e6e123dc; plots_fixed_estimation.ipynb cell f9a6fbb6 is identical)
# ---------------------------------------------------------------------------
REACTION_GROUPS = [
    ('PTS', 0, 4, '#d73027'), ('PGI', 5, 7, '#4575b4'),
    ('PFK', 8, 12, '#1a9850'), ('FBA', 13, 16, '#762a83'),
    ('TPI', 17, 19, '#e08214'), ('GAP', 20, 25, '#8c510a'),
    ('PGK', 26, 30, '#de77ae'), ('GPM', 31, 33, '#80cdc1'),
    ('ENO', 34, 36, '#74add1'),
]

# LaTeX labels in ALL_PARAMS order (37 entries)
PARAM_LATEX_LABELS = [
    r'$v^{\max}_{PTS}$', r'$K_{a1}$', r'$K_{a2}$', r'$K_{a3}$', r'$K_{g6p}$',
    r'$K_{s,g6p}^{2}$', r'$K_{p,f6p}^{2}$', r'$k^{+}_{cat,2}$',
    r'$K_{s,f6p}^{3}$', r'$K_{s,atp}^{3}$', r'$K_{p,fbp}^{3}$', r'$K_{p,adp}^{3}$', r'$k^{+}_{cat,3}$',
    r'$K_{s,fbp}^{4}$', r'$K_{p,g3p}^{4}$', r'$K_{p,dhap}^{4}$', r'$k^{+}_{cat,4}$',
    r'$k^{+}_{cat,5}$', r'$K_{s,dhap}^{5}$', r'$K_{p,g3p}^{5}$',
    r'$k^{+}_{cat,6}$', r'$K_{s,g3p}^{6}$', r'$K_{s,pi}^{6}$', r'$K_{s,nad}^{6}$', r'$K_{p,pgp}^{6}$', r'$K_{p,nadh}^{6}$',
    r'$k^{+}_{cat,7}$', r'$K_{s,pgp}^{7}$', r'$K_{s,adp}^{7}$', r'$K_{p,3pg}^{7}$', r'$K_{p,atp}^{7}$',
    r'$k^{+}_{cat,8}$', r'$K_{s,3pg}^{8}$', r'$K_{p,2pg}^{8}$',
    r'$k^{+}_{cat,9}$', r'$K_{s,2pg}^{9}$', r'$K_{p,pep}^{9}$',
]

TYPE_COLOR = {'kcat': '#117733', 'km': '#332288', 'pts': '#882255'}

# Index sets in ALL_PARAMS / PARAM_LATEX_LABELS ordering
KCAT0 = {7, 12, 16, 17, 20, 26, 31, 34}
PTS0 = {0, 1, 2, 3, 4}

# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------
def load_data_frames(data_dir="Data"):
    """Load the five standard CSVs and return them as DataFrames.

    Returns (balanced_met_df, imbalanced_met_df, prot_df, flux_df, cell_needs_df),
    all read with index_col=0.
    """
    d = str(data_dir)
    balanced_met_df = pd.read_csv("%s/balanced_metabolites.csv" % d, index_col=0)
    imbalanced_met_df = pd.read_csv("%s/imbalanced_metabolites.csv" % d, index_col=0)
    prot_df = pd.read_csv("%s/important_proteins.csv" % d, index_col=0)
    flux_df = pd.read_csv("%s/important_fluxes.csv" % d, index_col=0)
    cell_needs_df = pd.read_csv("%s/cellular_needs.csv" % d, index_col=0)
    return balanced_met_df, imbalanced_met_df, prot_df, flux_df, cell_needs_df

# ---------------------------------------------------------------------------
# Pipeline functions moved verbatim from sensitivity_pipeline.py
# ---------------------------------------------------------------------------
def metabolite_bounds(data_dir="Data", n_std=3.0, floor=1e-6):
    """Per-metabolite concentration bounds from the across-condition data.

    This is a MODELING CHOICE. For each metabolite, `mu` and `sigma` are the mean
    and std of ln(C) over the conditions with positive measurements, and the bound
    is `[max(exp(mu - n_std*sigma), floor), exp(mu + n_std*sigma)]` -- log-space
    (geometric), so positivity is guaranteed and the spread matches the typically
    log-normal distribution of metabolite levels. `n_std` is the user knob: how
    much room the steady state has to move around the observed regime (default 3).

    Metabolites with fewer than 2 positive measurements (e.g. the unmeasured
    C_g3p / C_pgp / C_2pg) fall back to a wide default `(floor, global_max)`, where
    `global_max` is the largest upper bound among that group's measured metabolites.

    Computes BOTH the balanced (X) and imbalanced/cofactor (U) bound sets with the
    same rule. Returns `(u_bounds, x_bounds)` -- imbalanced first, then balanced.
    """
    bal_df = pd.read_csv("%s/balanced_metabolites.csv" % data_dir, index_col=0)
    imb_df = pd.read_csv("%s/imbalanced_metabolites.csv" % data_dir, index_col=0)

    def _log_range(series):
        v = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        v = v[v > 0]
        if v.size < 2:
            return None
        ln = np.log(v)
        mu = float(ln.mean())
        sigma = float(ln.std(ddof=1))
        return (max(float(np.exp(mu - n_std * sigma)), floor),
                float(np.exp(mu + n_std * sigma)))

    def _group_bounds(keys, df):
        bounds = {}
        measured_hi = []
        for k in keys:
            rng = _log_range(df.loc[k]) if k in df.index else None
            if rng is not None:
                bounds[k] = rng
                measured_hi.append(rng[1])
        global_max = max(measured_hi) if measured_hi else 10.0
        for k in keys:
            bounds.setdefault(k, (floor, global_max))
        return bounds

    x_bounds = _group_bounds(EcoliCarbonKinetics.balanced_keys, bal_df)
    u_bounds = _group_bounds(EcoliCarbonKinetics.imbalanced_keys, imb_df)
    return u_bounds, x_bounds


def build_analysis_model(data_dir="Data", n_std=3.0):
    """Build the canonical analysis footing shared by both pipelines.

    Returns (model, conditions, input_enzyme, input_cell_needs, measurement_error):
    - model: EcoliCarbonKinetics with the data-derived bounds.
    - conditions: the SORTED list of valid condition names (enzymes present and
      shared with cell_needs).
    - input_enzyme / input_cell_needs: condition-indexed DataFrames, aligned to
      `conditions`.
    - measurement_error: {output_key: SD}, np.nan where the output is unmeasured.

    `n_std` is forwarded to `metabolite_bounds` so the model here uses the EXACT
    same data-derived bounds that first_estimation passes to its fit.
    """
    u_bounds, x_bounds = metabolite_bounds(data_dir, n_std=n_std)
    model = EcoliCarbonKinetics(bounds_imbalanced_mets=u_bounds,
                                bounds_balanced_mets=x_bounds)

    prot_df = pd.read_csv("%s/important_proteins.csv" % data_dir, index_col=0)
    cell_needs_df = pd.read_csv("%s/cellular_needs.csv" % data_dir, index_col=0)
    input_enzyme = prot_df.T.dropna()
    input_cell_needs = cell_needs_df.T
    conditions = sorted(input_enzyme.index.intersection(input_cell_needs.index))
    input_enzyme = input_enzyme.loc[conditions]
    input_cell_needs = input_cell_needs.loc[conditions]

    Q_bal = pd.read_csv("%s/balanced_metabolites_Q.csv" % data_dir, index_col=0).squeeze()
    Q_flux = pd.read_csv("%s/important_fluxes_Q.csv" % data_dir, index_col=0).squeeze()
    measurement_error = {k: Q_bal.get(k, np.nan) for k in model.balanced_keys}
    measurement_error.update({k: Q_flux.get(k, np.nan) for k in model.flux_keys})

    return model, conditions, input_enzyme, input_cell_needs, measurement_error


def save_per_condition_sensitivity(model, theta, conditions, input_enzyme,
                                   input_cell_needs, out_dir, data_dir="Data"):
    """Write the per-condition ABSOLUTE sensitivity CSVs at `theta`, plus the
    full model predictions and the measured real values.

    Each <out_dir>/sensitivity/<cond>.csv has the canonical schema
    `kind,name,<37 params>` with 18 rows (9 concentration + 9 flux), saved with
    index=False so the header is byte-identical across both pipelines. Returns
    (pred_df, real_df) indexed by condition over the 18 outputs (the caller saves
    them under its own filename: predictions_fitted.csv vs predictions.csv).
    """
    out_dir = Path(out_dir)
    sens_dir = out_dir / "sensitivity"
    sens_dir.mkdir(parents=True, exist_ok=True)

    balanced_met_df = pd.read_csv("%s/balanced_metabolites.csv" % data_dir, index_col=0)
    flux_df = pd.read_csv("%s/important_fluxes.csv" % data_dir, index_col=0)
    out_keys = list(model.balanced_keys) + list(model.flux_keys)
    n_conc = len(model.balanced_keys)
    n_flux = len(model.flux_keys)

    pred_rows = {}
    real_rows = {}
    for c in conditions:
        enz = input_enzyme.loc[c].to_dict()
        need = input_cell_needs.loc[c].to_dict()

        df_pred, _ = model.solve_steady_state(enz, theta, need, condition_key=c)
        pred_rows[c] = df_pred.iloc[0][out_keys].to_dict()

        real_row = {}
        for k in out_keys:
            src = balanced_met_df if k.startswith("C_") else flux_df
            if k in src.index and c in src.columns:
                real_row[k] = float(src.loc[k, c])
            else:
                real_row[k] = np.nan
        real_rows[c] = real_row

        G_df = model.gen_sensitivity_matrix(enz, theta, need, condition_key=c)
        sens_out = pd.DataFrame({
            "kind": ["concentration"] * n_conc + ["flux"] * n_flux,
            "name": list(G_df.index),
        })
        for pk in model.params_keys:
            sens_out[pk] = G_df[pk].values
        sens_out.to_csv(sens_dir / (c + ".csv"), index=False)

    pred_df = pd.DataFrame(pred_rows).T
    pred_df.index.name = "condition"
    real_df = pd.DataFrame(real_rows).T
    real_df.index.name = "condition"
    return pred_df, real_df
