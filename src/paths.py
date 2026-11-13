"""
paths.py -- repository root, data and settings folders and output directories.

``resolve_out`` turns an output directory into an absolute path; a relative path is taken relative to the repo root,
never to the working directory.

Position in the pipeline: imported by settings.py, by every step that writes results and by the notebooks.
"""
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "Data")
SETTINGS_DIR = os.path.join(REPO_ROOT, "settings")

def resolve_out(out):
    """Absolute path of the output directory ``out``; a relative path is relative to the repo root."""
    out = os.path.expanduser(str(out))
    if os.path.isabs(out):
        return os.path.abspath(out)
    return os.path.abspath(os.path.join(REPO_ROOT, out))
