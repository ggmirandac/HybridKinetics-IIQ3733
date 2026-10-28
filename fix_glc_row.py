"""
Rebuild the C_glc row of Data/imbalanced_metabolites.csv BY LABEL from sheet 'Extracelular' of
Data/data_important.xlsx (review item B1: the old row came from sheet 'ExtracelularRead', whose labels
are shifted against its values).

Condition labels are taken from the 'Sample ID' row of 'Extracelular'; KO05 is the column labelled
KO05 (not KO05x), the same replicate the 'Metabolites' sheet uses; RF03 is the column labelled RF03.

The old (shifted) row is saved once to Data/imbalanced_metabolites_glc_shifted_backup.csv (an existing
backup is never overwritten).  Only the C_glc row of the CSV is rewritten; other rows are untouched.

Usage (repo root):  python fix_glc_row.py [--check]
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.dirname(__file__))
XLSX = os.path.join(REPO, "Data", "data_important.xlsx")
CSV = os.path.join(REPO, "Data", "imbalanced_metabolites.csv")
BACKUP = os.path.join(REPO, "Data", "imbalanced_metabolites_glc_shifted_backup.csv")

def labelled_glucose():
    """{condition label: glucose value} from sheet Extracelular, by the Sample ID labels."""
    x = pd.read_excel(XLSX, sheet_name="Extracelular", header=None)
    labels = x.iloc[0].tolist()
    rows = x.iloc[:, 0].tolist()
    r = rows.index("Glucose")
    out = {}
    for j, lab in enumerate(labels):
        if isinstance(lab, str) and lab != "Sample ID":
            out[lab] = float(x.iloc[r, j])
    return out

def main():
    gl = labelled_glucose()
    imb = pd.read_csv(CSV, index_col=0)
    missing = [c for c in imb.columns if c not in gl]
    if missing:
        raise SystemExit("conditions without a labelled glucose value: %s" % missing)
    new = pd.Series({c: gl[c] for c in imb.columns}, name="C_glc")

    if "--check" in sys.argv:
        err = (imb.loc["C_glc"].astype(float) - new).abs().max()
        print("max |csv - labelled sheet| = %.3g" % err)
        return

    if not os.path.exists(BACKUP):
        imb.loc[["C_glc"]].to_csv(BACKUP)
        print("backup written: %s" % BACKUP)
    else:
        print("backup exists, kept: %s" % BACKUP)

    imb.loc["C_glc"] = new.to_numpy(dtype=float)
    imb.to_csv(CSV)
    chk = pd.read_csv(CSV, index_col=0).loc["C_glc"].astype(float)
    print("rows written: %d, max |csv - sheet| = %.3g" % (len(chk), (chk - new).abs().max()))
    for c in ("KO16", "KO05", "RF03"):
        print("  %s = %.5f" % (c, chk[c]))

if __name__ == "__main__":
    main()
