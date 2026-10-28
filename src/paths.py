"""
paths.py -- repository root, data and settings folders, output directories and the guard of the frozen results.

``resolve_out`` turns an output directory into an absolute path; a relative path is taken relative to the repo root,
never to the working directory.  ``check_out_dir`` refuses every directory inside the frozen result folders
(results/, results_v2/, results_v3/), also when they do not exist in the checkout.  The comparison follows symbolic
links and ignores the case on case-insensitive file systems (macOS, Windows), so ``Results_v3/x`` or a link that
points into ``results_v3`` is refused as well.

Position in the pipeline: imported by settings.py, by every step that writes results and by the notebooks.
"""
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "Data")
SETTINGS_DIR = os.path.join(REPO_ROOT, "settings")
FORBIDDEN_OUT = ("results", "results_v2", "results_v3")

def resolve_out(out):
    """Absolute path of the output directory ``out``; a relative path is relative to the repo root."""
    out = os.path.expanduser(str(out))
    if os.path.isabs(out):
        return os.path.abspath(out)
    return os.path.abspath(os.path.join(REPO_ROOT, out))

def canonical(path):
    """Real path, lower case on case-insensitive platforms (darwin, win32), for comparing two paths."""
    real = os.path.normcase(os.path.realpath(path))
    if sys.platform in ("darwin", "win32"):
        return real.lower()
    return real

def check_out_dir(out):
    """Refuse an output directory inside the frozen result folders; returns the absolute output path."""
    out_abs = resolve_out(out)
    out_real = canonical(out_abs)
    for name in FORBIDDEN_OUT:
        root = os.path.join(REPO_ROOT, name)
        root_real = canonical(root)
        if out_real == root_real or out_real.startswith(root_real + os.sep):
            raise ValueError("refusing to write into %s: new runs go to results_v4/" % root)
    return out_abs
