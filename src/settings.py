"""
settings.py -- the run settings, read from the CSV files of settings/.

Every number that steers the pipeline lives in a CSV file: the method settings (method.csv), the bound rules of the
data (bounds_rules.csv), the parameter bounds (theta_bounds.csv), the protected parameters (protected_params.csv) and
the IPOPT options (solver_options.csv).  Edit the CSV, not the code.  This module is the only reader.
``default_params()`` returns the method settings as a plain dict; the notebooks show that dict in a cell and a user
can change an entry before running a step.

Position in the pipeline: imported by every module; it has no dependency on the model.
"""
import os
import shutil

import solver  # noqa: F401  (pins BLAS / OpenMP to one thread before numpy is loaded)

import pandas as pd

from paths import SETTINGS_DIR

TYPES = {"int": int, "float": float, "str": str}

def cast(value, type_name):
    """Convert the text of a CSV cell to the Python type named in its ``type`` column (int, float, str)."""
    return TYPES[str(type_name).strip()](str(value).strip())

def read_table(file_name, settings_dir=SETTINGS_DIR):
    return pd.read_csv(os.path.join(settings_dir, file_name), dtype=str, keep_default_na=False)

def read_name_value(file_name, settings_dir=SETTINGS_DIR):
    """{name: value} of a settings file with the columns name, value, type, description."""
    table = read_table(file_name, settings_dir)
    values = {}
    for _, row in table.iterrows():
        values[row["name"]] = cast(row["value"], row["type"])
    return values

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
    """IPOPT options of one named set of solver_options.csv ('default', 'warm_start', 'closure')."""
    table = read_table("solver_options.csv", settings_dir)
    table = table[table["set"] == option_set]
    if table.empty:
        raise KeyError("no solver option set %r in solver_options.csv" % option_set)
    options = {}
    for _, row in table.iterrows():
        options[row["option"]] = cast(row["value"], row["type"])
    return options

def default_params(settings_dir=SETTINGS_DIR):
    """The method settings of method.csv as a dict {name: value}.  The defaults reproduce the published run."""
    return read_name_value("method.csv", settings_dir)

# Read once at import.  Treat them as read-only; copy before changing an entry.
METHOD = default_params()
BOUNDS_RULES = read_name_value("bounds_rules.csv")
THETA_BOUNDS = theta_bounds()
PROTECTED_PREFIXES = protected_prefixes()
SOLVER_OPTIONS = solver_options("default")
WARM_START_OPTIONS = {**SOLVER_OPTIONS, **solver_options("warm_start")}
CLOSURE_OPTIONS = solver_options("closure")

def snapshot_settings(root, settings_dir=SETTINGS_DIR):
    """Copy the settings CSV files into ``<root>/settings_used`` (the settings of the latest run)."""
    target = os.path.join(root, "settings_used")
    os.makedirs(target, exist_ok=True)
    shutil.copytree(settings_dir, target, dirs_exist_ok=True)
