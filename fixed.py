from src.sentitivity import compute_sensitivity
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
    # CORRECT - matches the notebook exactly
    imbalanced_bounds[met] = (
    float(np.where(imbalanced_met_df.loc[met].min() > 0,
                   imbalanced_met_df.loc[met].min() * 0.8, 1e-6)),
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
kinetic_params = {
    "v_max_1"  : 25.739,
    "Ka1_1"        : 1.000,
    "Ka2_1"        : 0.010,
    "Ka3_1"        : 1.000,
    "K_g6p_1"       : 0.500,
    "Ks_g6p_pgi"   : 1.018,
    "Kp_f6p_pgi"   : 0.139,
    "kcat_f_2"     : 5.193,
    "Ks_f6p_3"   : 0.030,
    "Ks_atp_3"   : 0.060,
    "Kp_fbp_3"   : 0.173,
    "Kp_adp_3"   : 0.155,
    "kcat_f_3"     : 7.738,
    "Ks_fbp_4"   : 0.240,
    "Kp_g3p_4"   : 0.077,
    "Kp_dhap_4"  : 0.272,
    "kcat_f_4"     : 0.930,
    "kcat_f_5"     : 1158.545,
    "Ks_dhap_5"  : 1.030,
    "Kp_g3p_5"   : 0.167,
    "kcat_f_6"     : 7.845,
    "Ks_g3p_6"   : 0.890,
    "Ks_pi_6"    : 0.530,
    "Ks_nad_6"   : 0.045,
    "Kp_pgp_6"   : 0.192,
    "Kp_nadh_6"  : 0.151,
    "kcat_f_7"     : 7.302,
    "Ks_pgp_7"   : 0.230,
    "Ks_adp_7"   : 0.201,
    "Ks_3pg_7"   : 0.230,
    "Ks_atp_7"   : 0.230,
    "kcat_f_8"     : 29.333,
    "Ks_3pg_8"   : 0.200,
    "Ks_2pg_8"   : 0.190,
    "kcat_f_9"     : 6.411,
    "Ks_2pg_9"   : 0.100,
    "Ks_pep_9"   : 0.256,
}
print(input_enzyme.columns.tolist())
print(input_cell_needs.columns.tolist())
print("input_enzyme shape:", input_enzyme.shape)
print("input_cell_needs shape:", input_cell_needs.shape)
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
import numpy as np

masamolar = {
    'pgi': 59.0, 'pfk': 36.0, 'fba': 40.0, 'tpi': 27.5,
    'gap': 35.0, 'pgk': 43.7, 'gpm': 27.0, 'eno': 46.0
}

def kcat_convert(kcat_s, enzyme):
    return kcat_s / (masamolar[enzyme] * 1000) * 3600

params_from_literature = {
    "v_max_1": 25.739, "Ka1_1": 1.0, "Ka2_1": 0.01, "Ka3_1": 1.0,
    "K_g6p_1": 0.5, "kcat_f_2": None, "Ks_g6p_pgi": 1.018,
    "Kp_f6p_pgi": np.mean([0.200, 0.078]), "kcat_f_3": None,
    "Ks_f6p_3": 0.030, "Ks_atp_3": 0.060, "Kp_fbp_3": None,
    "Kp_adp_3": None, "kcat_f_4": kcat_convert(10.33, 'fba'),
    "Ks_fbp_4": 0.240, "Kp_g3p_4": None, "Kp_dhap_4": None,
    "kcat_f_5": kcat_convert(np.mean([8700, 9000]), 'tpi'),
    "Ks_dhap_5": 1.030, "Kp_g3p_5": None, "kcat_f_6": None,
    "Ks_g3p_6": 0.890, "Ks_pi_6": 0.530, "Ks_nad_6": 0.045,
    "Kp_pgp_6": None, "Kp_nadh_6": None, "kcat_f_7": None,
    "Ks_pgp_7": None, "Ks_adp_7": None, "Ks_3pg_7": None,
    "Ks_atp_7": None, "kcat_f_8": kcat_convert(220, 'gpm'),
    "Ks_3pg_8": 0.200, "Ks_2pg_8": 0.190, "kcat_f_9": None,
    "Ks_2pg_9": 0.100, "Ks_pep_9": None,
}

params_to_set     = ['kcat_f_2', 'kcat_f_3', 'kcat_f_6', 'kcat_f_7', 'kcat_f_9']
params_kcat_fixed = ['kcat_f_4', 'kcat_f_5']
missing_kms = {
    'pgi': [], 'pfk': ['Kp_fbp_3', 'Kp_adp_3'],
    'fba': ['Kp_g3p_4', 'Kp_dhap_4'], 'tpi': ['Kp_g3p_5'],
    'gap': ['Kp_pgp_6', 'Kp_nadh_6'],
    'pgk': ['Ks_pgp_7', 'Ks_adp_7', 'Ks_3pg_7', 'Ks_atp_7'],
    'gpm': [], 'eno': ['Ks_pep_9'],
}
suffix_to_enzyme = {
    '2': 'pgi', '3': 'pfk', '4': 'fba', '5': 'tpi',
    '6': 'gap', '7': 'pgk', '8': 'gpm', '9': 'eno'
}

def sample_lognormal(median, lo, hi):
    mu    = np.log(median)
    sigma = (np.log(hi) - np.log(lo)) / (2 * 1.96)
    return np.exp(np.random.normal(mu, sigma))

np.random.seed(42)
sampled_params = params_from_literature.copy()
for kcat_key in params_to_set:
    enzyme = suffix_to_enzyme[kcat_key.split('_')[-1]]
    kcat_s = sample_lognormal(79, 50, 90)
    sampled_params[kcat_key] = kcat_convert(kcat_s, enzyme)
    for km_key in missing_kms[enzyme]:
        kcat_kM = sample_lognormal(410, 300, 500)
        sampled_params[km_key] = kcat_s / kcat_kM
for kcat_key in params_kcat_fixed:
    enzyme = suffix_to_enzyme[kcat_key.split('_')[-1]]
    for km_key in missing_kms[enzyme]:
        sampled_params[km_key] = sample_lognormal(0.5, 0.01, 0.7)

import numpy as np

masamolar = {
    'pgi': 59.0, 'pfk': 36.0, 'fba': 40.0, 'tpi': 27.5,
    'gap': 35.0, 'pgk': 43.7, 'gpm': 27.0, 'eno': 46.0
}

def kcat_convert(kcat_s, enzyme):
    return kcat_s / (masamolar[enzyme] * 1000) * 3600

params_from_literature = {
    "v_max_1": 25.739, "Ka1_1": 1.0, "Ka2_1": 0.01, "Ka3_1": 1.0,
    "K_g6p_1": 0.5, "kcat_f_2": None, "Ks_g6p_pgi": 1.018,
    "Kp_f6p_pgi": np.mean([0.200, 0.078]), "kcat_f_3": None,
    "Ks_f6p_3": 0.030, "Ks_atp_3": 0.060, "Kp_fbp_3": None,
    "Kp_adp_3": None, "kcat_f_4": kcat_convert(10.33, 'fba'),
    "Ks_fbp_4": 0.240, "Kp_g3p_4": None, "Kp_dhap_4": None,
    "kcat_f_5": kcat_convert(np.mean([8700, 9000]), 'tpi'),
    "Ks_dhap_5": 1.030, "Kp_g3p_5": None, "kcat_f_6": None,
    "Ks_g3p_6": 0.890, "Ks_pi_6": 0.530, "Ks_nad_6": 0.045,
    "Kp_pgp_6": None, "Kp_nadh_6": None, "kcat_f_7": None,
    "Ks_pgp_7": None, "Ks_adp_7": None, "Ks_3pg_7": None,
    "Ks_atp_7": None, "kcat_f_8": kcat_convert(220, 'gpm'),
    "Ks_3pg_8": 0.200, "Ks_2pg_8": 0.190, "kcat_f_9": None,
    "Ks_2pg_9": 0.100, "Ks_pep_9": None,
}

params_to_set     = ['kcat_f_2', 'kcat_f_3', 'kcat_f_6', 'kcat_f_7', 'kcat_f_9']
params_kcat_fixed = ['kcat_f_4', 'kcat_f_5']
missing_kms = {
    'pgi': [], 'pfk': ['Kp_fbp_3', 'Kp_adp_3'],
    'fba': ['Kp_g3p_4', 'Kp_dhap_4'], 'tpi': ['Kp_g3p_5'],
    'gap': ['Kp_pgp_6', 'Kp_nadh_6'],
    'pgk': ['Ks_pgp_7', 'Ks_adp_7', 'Ks_3pg_7', 'Ks_atp_7'],
    'gpm': [], 'eno': ['Ks_pep_9'],
}
suffix_to_enzyme = {
    '2': 'pgi', '3': 'pfk', '4': 'fba', '5': 'tpi',
    '6': 'gap', '7': 'pgk', '8': 'gpm', '9': 'eno'
}

def sample_lognormal(median, lo, hi):
    mu    = np.log(median)
    sigma = (np.log(hi) - np.log(lo)) / (2 * 1.96)
    return np.exp(np.random.normal(mu, sigma))

np.random.seed(42)
sampled_params = params_from_literature.copy()
for kcat_key in params_to_set:
    enzyme = suffix_to_enzyme[kcat_key.split('_')[-1]]
    kcat_s = sample_lognormal(79, 50, 90)
    sampled_params[kcat_key] = kcat_convert(kcat_s, enzyme)
    for km_key in missing_kms[enzyme]:
        kcat_kM = sample_lognormal(410, 300, 500)
        sampled_params[km_key] = kcat_s / kcat_kM
for kcat_key in params_kcat_fixed:
    enzyme = suffix_to_enzyme[kcat_key.split('_')[-1]]
    for km_key in missing_kms[enzyme]:
        sampled_params[km_key] = sample_lognormal(0.5, 0.01, 0.7)

params_estimate = sampled_params  # use this as params_estimate
# --- run ---
G_total, corr, diag = compute_sensitivity(
    model          = model,
    params_estimate = params_estimate,
    input_enzyme   = input_enzyme.T,       # rows=conditions, cols=enzymes
    input_cell_needs = input_cell_needs.T, # rows=conditions, cols=metabolites
    conditions     = conditions,
    measurement_error = measurement_error,
)

print(diag["identifiability"].to_string())