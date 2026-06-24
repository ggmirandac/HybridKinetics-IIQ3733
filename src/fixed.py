from sentitivity import compute_sensitivity
import pandas as pd

# --- measurement errors (from data_analysis.ipynb output) ---
met_Q  = pd.read_csv('Data/balanced_metabolites_Q.csv', index_col=0)
flux_Q = pd.read_csv('Data/important_fluxes_Q.csv',     index_col=0)

measurement_error = {}
for met in met_Q.index:
    measurement_error[met] = float(met_Q.loc[met].mean())
for flux in flux_Q.index:
    measurement_error[flux] = float(flux_Q.loc[flux, 'STD'])
# unmeasured outputs get NaN → skipped in FIM
for key in model.balanced_keys + model.flux_keys:
    if key not in measurement_error:
        measurement_error[key] = np.nan

# --- conditions list (all columns in your data) ---
conditions = [{'condition': col} for col in input_enzyme.columns]

# --- initial parameters from Table V (before CMA-ES) ---
# use kinetic_params if you want post-optimization identifiability,
# or replace with your Table V dict for pre-optimization analysis
params_estimate = kinetic_params   # dict: {param_name: value}

# --- run ---
G_total, corr, diag = compute_sensitivity(
    model          = model,
    params_estimate= params_estimate,
    input_enzyme   = input_enzyme.T,       # rows=conditions, cols=enzymes
    input_cell_needs = input_cell_needs.T, # rows=conditions, cols=metabolites
    conditions     = conditions,
    measurement_error = measurement_error,
)

print(diag["identifiability"].to_string())