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
    "Kp_f6p_pgi": 0.078, "kcat_f_3": None,
    "Ks_f6p_3": 0.013, "Ks_atp_3": 0.020, "Kp_fbp_3": 0.14,
    "Kp_adp_3": None, "kcat_f_4": kcat_convert(10.33, 'fba'),
    "Ks_fbp_4": 0.240, "Kp_g3p_4": None, "Kp_dhap_4": None,
    "kcat_f_5": kcat_convert(9000, 'tpi'),
    "Ks_dhap_5": 1.030, "Kp_g3p_5": None, "kcat_f_6": None,
    "Ks_g3p_6": 0.890, "Ks_pi_6": 0.530, "Ks_nad_6": 0.045,
    "Kp_pgp_6": None, "Kp_nadh_6": None, "kcat_f_7": None,
    "Ks_pgp_7": None, "Ks_adp_7": None, "Ks_3pg_7": None,
    "Ks_atp_7": None, "kcat_f_8": kcat_convert(330, 'gpm'),
    "Ks_3pg_8": 0.200, "Ks_2pg_8": 0.190, "kcat_f_9": None,
    "Ks_2pg_9": 0.100, "Ks_pep_9": None,
}

params_to_set     = ['kcat_f_2', 'kcat_f_3', 'kcat_f_6', 'kcat_f_7', 'kcat_f_9']
params_kcat_fixed = ['kcat_f_4', 'kcat_f_5']
missing_kms = {
    'pgi': [], 'pfk': ['Kp_adp_3'],
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

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle as MplRect
import os

os.makedirs('figures', exist_ok=True)

# ── 1. Mean sensitivity heatmap (Cell 7) ─────────────────────────────────
G_mean = np.mean([np.abs(G) for G in diag['G_list']], axis=0)

meas_bal  = [k for k in diag['measured'] if k.startswith('C_')]
meas_flux = [k for k in diag['measured'] if k.startswith('v_')]
n_meas_bal  = len(meas_bal)
n_meas_flux = len(meas_flux)

top_n   = 15
top_idx = np.argsort(G_mean.max(axis=0))[::-1][:top_n]
xlabels = [model.params_keys[i] for i in top_idx]

fig, (ax_c, ax_f) = plt.subplots(2, 1, figsize=(14, 7),
                                  gridspec_kw={'height_ratios': [n_meas_bal, n_meas_flux],
                                               'hspace': 0.05})
im_c = ax_c.imshow(G_mean[:n_meas_bal, :][:, top_idx], aspect='auto', cmap='viridis')
ax_c.set_xticks([])
ax_c.set_yticks(range(n_meas_bal))
ax_c.set_yticklabels(meas_bal, fontsize=8)
ax_c.set_title("Mean |relative sensitivity| -- top 15 parameters", fontsize=10)
plt.colorbar(im_c, ax=ax_c, label="|d ln C / d ln θ|")

im_f = ax_f.imshow(G_mean[n_meas_bal:, :][:, top_idx], aspect='auto', cmap='viridis')
ax_f.set_xticks(range(top_n))
ax_f.set_xticklabels(xlabels, rotation=55, ha='right', fontsize=8)
ax_f.set_yticks(range(n_meas_flux))
ax_f.set_yticklabels(meas_flux, fontsize=8)
plt.colorbar(im_f, ax=ax_f, label="|d ln v / d ln θ|")
plt.tight_layout()
plt.savefig('figures/sensitivity_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.show()

# ── 2. Correlation matrix — unmasked (Cell 8) ────────────────────────────
reaction_groups = [
    ("PTS",  0,  4,  '#d73027'), ("PGI",  5,  7,  '#4575b4'),
    ("PFK",  8, 12,  '#1a9850'), ("FBA", 13, 16,  '#762a83'),
    ("TPI", 17, 19,  '#e08214'), ("GAP", 20, 25,  '#8c510a'),
    ("PGK", 26, 30,  '#de77ae'), ("GPM", 31, 33,  '#80cdc1'),
    ("ENO", 34, 36,  '#74add1'),
]
param_labels = [
    r"$v^{\max}_{PTS}$", r"$K_{a1}$", r"$K_{a2}$", r"$K_{a3}$", r"$K_{g6p}$",
    r"$K_{s,g6p}^{2}$", r"$K_{p,f6p}^{2}$", r"$k^{+}_{cat,2}$",
    r"$K_{s,f6p}^{3}$", r"$K_{s,atp}^{3}$", r"$K_{p,fbp}^{3}$", r"$K_{p,adp}^{3}$", r"$k^{+}_{cat,3}$",
    r"$K_{s,fbp}^{4}$", r"$K_{p,g3p}^{4}$", r"$K_{p,dhap}^{4}$", r"$k^{+}_{cat,4}$",
    r"$k^{+}_{cat,5}$", r"$K_{s,dhap}^{5}$", r"$K_{p,g3p}^{5}$",
    r"$k^{+}_{cat,6}$", r"$K_{s,g3p}^{6}$", r"$K_{s,pi}^{6}$", r"$K_{s,nad}^{6}$", r"$K_{p,pgp}^{6}$", r"$K_{p,nadh}^{6}$",
    r"$k^{+}_{cat,7}$", r"$K_{s,pgp}^{7}$", r"$K_{s,adp}^{7}$", r"$K_{p,3pg}^{7}$", r"$K_{p,atp}^{7}$",
    r"$k^{+}_{cat,8}$", r"$K_{s,3pg}^{8}$", r"$K_{p,2pg}^{8}$",
    r"$k^{+}_{cat,9}$", r"$K_{s,2pg}^{9}$", r"$K_{p,pep}^{9}$",
]

def make_corr_plot(corr_mat, labels, reaction_groups, thresh, filename):
    # reorder: kcat first within each reaction group
    _KCAT0 = {7, 12, 16, 17, 20, 26, 31, 34}
    _PTS0  = {0, 1, 2, 3, 4}
    def _t0(i): return 'pts' if i in _PTS0 else ('kcat' if i in _KCAT0 else 'km')
    _rank = {'kcat': 0, 'km': 1, 'pts': 0}
    _perm = []
    for _nm, _is, _ie, _c in reaction_groups:
        _perm += sorted(range(_is, _ie + 1), key=lambda i: (_rank[_t0(i)], i))
    
    labels_p = [labels[i] for i in _perm]
    corr_p   = corr_mat[np.ix_(_perm, _perm)]
    KCAT_IDX = {p for p, o in enumerate(_perm) if o in _KCAT0}
    PTS_IDX  = {p for p, o in enumerate(_perm) if o in _PTS0}

    corr_masked = np.where(np.abs(corr_p) >= thresh, corr_p, np.nan)
    np.fill_diagonal(corr_masked, np.nan)
    n = len(labels_p)

    mpl.rcParams.update({'font.family': 'sans-serif', 'font.size': 9,
                         'axes.linewidth': 0.6, 'mathtext.fontset': 'dejavusans'})
    cmap = mpl.colormaps['RdBu_r'].copy()
    cmap.set_bad('#e9e9ec')

    GAP, STRIP, TICK_FS, STRIP_FS = 0.35, 1.4, 11, 9
    fig, ax = plt.subplots(figsize=(10.5, 10.5), dpi=200)
    im = ax.imshow(corr_masked, cmap=cmap, vmin=-1, vmax=1,
                   interpolation='none', aspect='equal')
    ax.set_xlim(-0.5, n - 0.5 + GAP + STRIP)
    ax.set_ylim(n - 0.5, -0.5 - GAP - STRIP)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels_p, rotation=90, fontsize=TICK_FS, ha='center')
    ax.set_yticklabels(labels_p, fontsize=TICK_FS)

    TYPE_COLOR = {'kcat': '#117733', 'km': '#332288', 'pts': '#882255'}
    def _type_color(i):
        if i in PTS_IDX:  return TYPE_COLOR['pts']
        if i in KCAT_IDX: return TYPE_COLOR['kcat']
        return TYPE_COLOR['km']
    for i, t in enumerate(ax.get_xticklabels()):
        t.set_color(_type_color(i)); t.set_fontweight('semibold')
    for i, t in enumerate(ax.get_yticklabels()):
        t.set_color(_type_color(i)); t.set_fontweight('semibold')
    ax.tick_params(length=2, width=0.5, pad=2)

    for k in np.arange(-0.5, n, 1):
        ax.plot([-0.5, n-0.5], [k, k], color='white', lw=0.5, zorder=1)
        ax.plot([k, k], [-0.5, n-0.5], color='white', lw=0.5, zorder=1)

    for name, i_s, i_e, col in reaction_groups:
        if i_s != 0:
            ax.plot([i_s-0.5, i_s-0.5], [-0.5, n-0.5], color='#2b2b2b', lw=1.1, zorder=3)
            ax.plot([-0.5, n-0.5], [i_s-0.5, i_s-0.5], color='#2b2b2b', lw=1.1, zorder=3)
        ax.add_patch(MplRect((i_s-0.5, i_s-0.5), i_e-i_s+1, i_e-i_s+1,
                             fill=False, edgecolor=col, lw=1.6, zorder=5))
    ax.add_patch(MplRect((-0.5, -0.5), n, n, fill=False, edgecolor='#2b2b2b', lw=1.1, zorder=6))

    top_y, right_x = -0.5 - GAP - STRIP, n - 0.5 + GAP
    for name, i_s, i_e, col in reaction_groups:
        w = i_e - i_s + 1
        ax.add_patch(MplRect((i_s-0.5, top_y), w, STRIP,
                             facecolor=col, edgecolor='white', lw=1.0, zorder=4))
        ax.text((i_s+i_e)/2, top_y+STRIP/2, name, ha='center', va='center',
                fontsize=STRIP_FS, fontweight='bold', color='white', zorder=5)
        ax.add_patch(MplRect((right_x, i_s-0.5), STRIP, w,
                             facecolor=col, edgecolor='white', lw=1.0, zorder=4))
        ax.text(right_x+STRIP/2, (i_s+i_e)/2, name, ha='center', va='center',
                fontsize=STRIP_FS, fontweight='bold', color='white', rotation=270, zorder=5)

    for s in ('top', 'right', 'bottom', 'left'): ax.spines[s].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, shrink=0.6)
    cbar.set_label(r'Correlation  $r_{ij}$', fontsize=11, labelpad=4)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=9, length=3, width=0.5)
    cbar.outline.set_linewidth(0.6)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    mpl.rcParams.update(mpl.rcParamsDefault)

# unmasked (all correlations)
make_corr_plot(corr, param_labels, reaction_groups, thresh=0.0,
               filename='figures/corr_not_masked.pdf')

# masked (|r| >= 0.9 only)
make_corr_plot(corr, param_labels, reaction_groups, thresh=0.9,
               filename='figures/corr_masked.pdf')

# ── 3. Print correlated pairs ─────────────────────────────────────────────
df_corr = pd.DataFrame(corr, index=model.params_keys, columns=model.params_keys)
list_corrs = []
for i in range(len(df_corr)):
    for j in range(i+1, len(df_corr)):
        if abs(df_corr.iloc[i, j]) >= 0.9:
            list_corrs.append((df_corr.index[i], df_corr.columns[j],
                               round(df_corr.iloc[i, j], 3)))
df_corrs = pd.DataFrame(list_corrs, columns=['Parameter 1', 'Parameter 2', 'Correlation'])
print(df_corrs.to_string())
print(f"\nTotal correlated pairs (|r| >= 0.9): {len(df_corrs)}")