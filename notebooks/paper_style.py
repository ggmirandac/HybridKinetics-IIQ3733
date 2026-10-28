"""
paper_style.py -- shared style of the figure notebooks (figures_main.ipynb, figures_si.ipynb).

Only constants and two text helpers live here: the matplotlib rcParams of the paper-style figures (RC), the colours
(reactions, parameter classes, stages), the parameter labels in the paper notation (superscript = reaction number)
and ``classes_of_free`` (class of each free parameter, used for the colours).  There are NO plotting functions:
every figure is written out in full in its own notebook cell.  The reaction order, the reaction of each parameter and
the metabolite / flux keys come from the model (src/model/parameters.py); only the display names are written here.
"""
import re

from model.parameters import (RXN_ORDER, RXN_OF_PARAM, BALANCED_KEYS, LI_FLUX_KEYS, FLUX_KEYS, DUPLICATE_FLUX_GROUPS)

RC = {
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888888",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "axes.labelcolor": "#333333",
    "axes.titlelocation": "left",
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
}

# display names of the reactions of the model, in model order (RXN_ORDER); the default is the upper-case name
DISPLAY_NAME = {"gap": "GAPDH"}
REACTIONS = [DISPLAY_NAME.get(r, r.upper()) for r in RXN_ORDER]
REACTION_NUMBER = {r: i + 1 for i, r in enumerate(RXN_ORDER)}
REACTION_COLOR = ["#7a5c99", "#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#eb6834", "#6250d6", "#008300",
                  "#e34948"]

CLASSES = ["identifiable", "poorly determined", "flagged", "nuisance (kept by rule)"]
CLASS_COLOR = {"identifiable": "#2a78d6", "poorly determined": "#eb6834", "flagged": "#1baf7a",
               "nuisance (kept by rule)": "#eda100"}

STAGE_COLOR = {"literature": "#9a9a9a", "stage0": "#86b6ef", "final": "#2a78d6", "loocv": "#eb6834"}
INK = "#333333"
MUTED = "#777777"
GRID = "#e6e6e6"

METABOLITE_LABEL = {k: k[2:] for k in BALANCED_KEYS}

def flux_label(key):
    """Reaction name of a measured flux row, followed by the rows that equal it (DUPLICATE_FLUX_GROUPS)."""
    name = REACTIONS[FLUX_KEYS.index(key)]
    same = []
    for group in DUPLICATE_FLUX_GROUPS:
        if key in group:
            same += [REACTIONS[FLUX_KEYS.index(k)] for k in group if k != key]
    if same:
        return name + " (= %s)" % " = ".join(same)
    return name

FLUX_LABEL = {k: flux_label(k) for k in LI_FLUX_KEYS}

def reaction_of(param):
    """Reaction number (1-9, position in the model order) of a parameter name; 0 if it is not a model parameter."""
    return REACTION_NUMBER.get(RXN_OF_PARAM.get(param), 0)

def label(param):
    """Paper notation: kcat_f_4 -> k_cat^4, Ks_g6p_pgi -> K_s,g6p^2, Kp_adp_3 -> K_p,adp^3, Ka2_1 -> K_a2^1."""
    r = reaction_of(param)
    if param.startswith("kcat_f_"):
        return r"$k_{\mathrm{cat}}^{%d}$" % r
    if param == "v_max_1":
        return r"$v_{\max}^{1}$"
    m = re.match(r"K([sp])_([a-z0-9]+)_", param)
    if m:
        return r"$K_{%s,\mathrm{%s}}^{%d}$" % (m.group(1), m.group(2), r)
    m = re.match(r"Ka(\d)_1", param)
    if m:
        return r"$K_{a%s}^{1}$" % m.group(1)
    if param == "K_g6p_1":
        return r"$K_{\mathrm{g6p}}^{1}$"
    return param

def classes_of_free(final_structure, profile_summary=None):
    """
    {free parameter: class} for the colours of the figures, from final/final_structure.csv (ident_class) and, if
    given, profiles/profile_summary.csv (class_lr):
      "nuisance (kept by rule)"  ident_class == kept_free_by_rule
      "flagged"                  ident_class == flagged (at a bound / unstable finite differences)
      "identifiable"             the profile likelihood crosses the threshold on both sides
                                 (without a profile table: ident_class == identifiable)
      "poorly determined"        all the others
    """
    fs = final_structure.set_index("param") if "param" in final_structure.columns else final_structure
    out = {}
    for p in fs.index[fs["status"] == "free"]:
        cls = fs.loc[p, "ident_class"]
        if cls == "kept_free_by_rule":
            out[p] = "nuisance (kept by rule)"
        elif cls == "flagged":
            out[p] = "flagged"
        elif profile_summary is not None and p in profile_summary.index:
            out[p] = "identifiable" if profile_summary.loc[p, "class_lr"] == "identifiable" else "poorly determined"
        else:
            out[p] = "identifiable" if cls in ("identifiable", "identifiable_by_fim") else "poorly determined"
    return out
