"""
settings.py -- the run settings, read from the CSV files of settings/.

Every number that steers the pipeline lives in a CSV file: bound rules, theta bounds, thresholds of the parameter
selection, protected parameters, IPOPT option sets and the defaults of the uncertainty steps.  Edit the CSV, not the
code.  This module is the only reader.  ``default_params(step)`` collects the settings one step of the method uses
into a plain dict; the notebooks show that dict in a cell and a user can change an entry before running the step.

Position in the pipeline: imported by every module; it has no dependency on the model.
"""
import os
import shutil

import solver  # noqa: F401  (pins BLAS / OpenMP to one thread before numpy is loaded)

import pandas as pd

from paths import SETTINGS_DIR
from progress import log

TYPES = {"int": int, "float": float, "str": str}

def cast(value, type_name):
    """Convert the text of a CSV cell to the Python type named in its ``type`` column (int, float, str, bool)."""
    value = str(value).strip()
    type_name = str(type_name).strip()
    if type_name == "bool":
        if value.lower() not in ("true", "false"):
            raise ValueError("bool setting must be True or False, got %r" % value)
        return value.lower() == "true"
    return TYPES[type_name](value)

def read_table(file_name, settings_dir=SETTINGS_DIR):
    return pd.read_csv(os.path.join(settings_dir, file_name), dtype=str, keep_default_na=False)

def read_name_value(file_name, settings_dir=SETTINGS_DIR):
    """{name: value} of a settings file with the columns name, value, type, description."""
    table = read_table(file_name, settings_dir)
    values = {}
    for _, row in table.iterrows():
        values[row["name"]] = cast(row["value"], row["type"])
    return values

def thresholds(settings_dir=SETTINGS_DIR):
    """Thresholds of the parameter selection and of the structural and sensitivity checks (thresholds.csv)."""
    return read_name_value("thresholds.csv", settings_dir)

def bounds_rules(settings_dir=SETTINGS_DIR):
    """Rules that turn the data into concentration and drain bounds (bounds_rules.csv)."""
    return read_name_value("bounds_rules.csv", settings_dir)

def run_settings(settings_dir=SETTINGS_DIR):
    """Workers, stage 0 seed and warm start, default output directory, reference run of the report (run.csv)."""
    return read_name_value("run.csv", settings_dir)

def uncertainty_settings(settings_dir=SETTINGS_DIR):
    """Defaults of the bootstrap, profile likelihood and LOOCV steps (uncertainty.csv)."""
    return read_name_value("uncertainty.csv", settings_dir)

def theta_bounds(settings_dir=SETTINGS_DIR):
    """{param: [lower, upper]} of every kinetic parameter (theta_bounds.csv), in file order."""
    table = read_table("theta_bounds.csv", settings_dir)
    bounds = {}
    for _, row in table.iterrows():
        bounds[row["param"]] = [float(row["lower"]), float(row["upper"])]
    return bounds

def protected_prefixes(settings_dir=SETTINGS_DIR):
    """Prefixes of the parameters that are never fixed (protected_params.csv)."""
    return tuple(read_table("protected_params.csv", settings_dir)["prefix"].tolist())

def solver_options(option_set, settings_dir=SETTINGS_DIR):
    """IPOPT options of one named set of solver_options.csv ('default', 'tight', 'retry_monotone', ...)."""
    table = read_table("solver_options.csv", settings_dir)
    table = table[table["set"] == option_set]
    if table.empty:
        raise KeyError("no solver option set %r in solver_options.csv" % option_set)
    options = {}
    for _, row in table.iterrows():
        options[row["option"]] = cast(row["value"], row["type"])
    return options

# Read once at import.  Treat them as read-only; copy before changing an entry.
THRESHOLDS = thresholds()
BOUNDS_RULES = bounds_rules()
RUN = run_settings()
UNCERTAINTY = uncertainty_settings()
THETA_BOUNDS = theta_bounds()
PROTECTED_PREFIXES = protected_prefixes()
SOLVER_OPTIONS = solver_options("default")
TIGHT_OPTIONS = solver_options("tight")
RETRY_MONOTONE_OPTIONS = solver_options("retry_monotone")
REFIT_MONOTONE_OPTIONS = solver_options("refit_monotone")
CASADI_OPTIONS = solver_options("casadi")

# The settings each step of the method reads, by file.  default_params(step) returns them as one dict.
SENSITIVITY_SETTINGS = ["sens_step", "sens_step_check", "fd_consistency_rtol", "fd_consistency_atol"]
PRACTICAL_SETTINGS = SENSITIVITY_SETTINGS + ["corr_threshold", "zero_cut", "eig_cut"]
STRUCTURAL_SETTINGS = ["structural_q", "structural_q_out", "seed_base", "final_seed_base", "perturb_seed_offset",
                       "structural_sv_tol", "structural_guard_rel", "structural_load_min", "tie_rtol", "realization_lo",
                       "realization_hi", "perturb_lo", "perturb_hi", "output_realization_lo", "output_realization_hi",
                       "output_perturb_lo", "output_perturb_hi", "active_rtol", "rank_rcond"]
REDUCTION_SETTINGS = ["max_iter", "max_structural_rounds", "svd_tie_tol", "svd_min_loading", "multistart_every",
                      "n_starts", "multistart_top_n", "multistart_max_ratio", "end_multistart_rtol", "refit_rtol",
                      "refit_atol"]
STEP_SETTINGS = {
    "stage0": {"run.csv": ["n_jobs", "stage0_seed", "stage0_warm_start"],
               "thresholds.csv": ["n_starts_stage0", "theta_at_bound_tol"]},
    "sensitivity": {"run.csv": ["n_jobs"], "thresholds.csv": SENSITIVITY_SETTINGS},
    "practical": {"run.csv": ["n_jobs"], "thresholds.csv": PRACTICAL_SETTINGS},
    "structural": {"thresholds.csv": STRUCTURAL_SETTINGS},
    "selection": {"run.csv": ["n_jobs"],
                  "thresholds.csv": REDUCTION_SETTINGS + PRACTICAL_SETTINGS + STRUCTURAL_SETTINGS},
    "final": {"run.csv": ["n_jobs"],
              "thresholds.csv": ["lr_alpha", "final_accept_lr_fraction", "theta_at_bound_tol"]
              + PRACTICAL_SETTINGS + STRUCTURAL_SETTINGS},
    "bootstrap": {"run.csv": ["n_jobs"], "thresholds.csv": ["theta_at_bound_tol"],
                  "uncertainty.csv": ["bootstrap_n", "bootstrap_chunk", "bootstrap_seed", "bootstrap_ci_ratio_max",
                                      "bootstrap_frac_at_bound_max", "bootstrap_ci_span_max",
                                      "bootstrap_parked_rtol"]},
    "profiles": {"run.csv": ["n_jobs"], "thresholds.csv": ["lr_alpha"],
                 "uncertainty.csv": ["profile_half_points", "profile_span"]},
    "loocv": {"run.csv": ["n_jobs"], "thresholds.csv": ["loocv_bound_rtol"], "uncertainty.csv": ["loocv_n_starts"]},
    "isolation": {"run.csv": ["n_jobs", "isolation_n_starts"]},
}

def default_params(step, settings_dir=SETTINGS_DIR):
    """
    The settings of one step of the method as a dict {name: value} (names as in the CSV files), read from the CSV
    files of ``settings_dir``.  The defaults reproduce the published run.  Steps: stage0, selection, structural,
    practical, sensitivity, final, bootstrap, profiles, loocv, isolation.
    """
    if step not in STEP_SETTINGS:
        raise KeyError("unknown step %r; steps: %s" % (step, ", ".join(STEP_SETTINGS)))
    params = {}
    for file_name, names in STEP_SETTINGS[step].items():
        values = read_name_value(file_name, settings_dir)
        for name in names:
            params[name] = values[name]
    return params

def snapshot_settings(root, settings_dir=SETTINGS_DIR):
    """
    Copy the settings CSV files into ``<root>/settings_used`` (existing files are overwritten, so the folder holds
    the settings of the latest run).  A resumed run with edited settings would mix two settings, so the files that
    differ from the existing copy are logged first.  Returns the names of the differing files.
    """
    target = os.path.join(root, "settings_used")
    names = sorted(name for name in os.listdir(settings_dir) if name.endswith(".csv"))
    differing = []
    if os.path.isdir(target):
        old_names = [name for name in os.listdir(target) if name.endswith(".csv")]
        for name in sorted(set(names) | set(old_names)):
            new_file = os.path.join(settings_dir, name)
            old_file = os.path.join(target, name)
            if not (os.path.exists(new_file) and os.path.exists(old_file)):
                differing.append(name)
            elif open(new_file, "rb").read() != open(old_file, "rb").read():
                differing.append(name)
        if differing:
            log("WARNING: settings_used of %s differs from settings/ in %s; the files are overwritten with the "
                "settings of this run" % (root, ", ".join(differing)))
    os.makedirs(target, exist_ok=True)
    shutil.copytree(settings_dir, target, dirs_exist_ok=True)
    return differing
