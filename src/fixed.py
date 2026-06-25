from sentitivity import compute_sensitivity
import pandas as pd
import numpy as np
import pickle
from src.kinetics_noor import EcoliCarbonKinetics

with open('Data/k_eq_values.pkl', 'rb') as f:
    k_eq = pickle.load(f)

balanced_met_df   = pd.read_csv('Data/balanced_metabolites.csv', index_col=0)
imbalanced_met_df = pd.read_csv('Data/imbalanced_metabolites.csv', index_col=0)

imbalanced_bounds = {}
total_min = 1e12
for met in imbalanced_met_df.index:
    imbalanced_bounds[met] = (imbalanced_met_df.loc[met].min() * 0.8,
                              imbalanced_met_df.loc[met].max() * 1.2)
    if imbalanced_bounds[met][0] < total_min:
        total_min = imbalanced_bounds[met][0]
imbalanced_bounds["C_pi"] = (total_min, 10.0)

max_max_balanced = balanced_met_df.max().max() * 1.2
min_min_balanced = 1e-6
balanced_bounds = {k: (min_min_balanced, max_max_balanced) for k in [
    "C_g6p","C_f6p","C_fbp","C_dhap","C_g3p","C_pgp","C_3pg","C_2pg","C_pep"
]}

model = EcoliCarbonKinetics(imbalanced_bounds, balanced_bounds)

input_enzyme     = pd.read_csv('Data/important_proteins.csv', index_col=0)
input_cell_needs = pd.read_csv('Data/cellular_needs.csv', index_col=0)

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