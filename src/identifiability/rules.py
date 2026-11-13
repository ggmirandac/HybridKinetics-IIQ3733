"""
rules.py -- which parameter a rule acts on, and whether it may be fixed.

Protections: the parameters with a prefix of settings/protected_params.csv are never fixed, and the LAST free member
of a reaction group is kept, so every rate equation keeps at least one adjustable parameter (one group per reaction,
model.parameters.reaction_groups).

One rule picks the parameter of a non-identifiable direction (structural or practical): the one with the largest
absolute loading in the right singular vector of the smallest singular value (the first in model order if two are
equal to 6 decimals).  If the protections allow it, the parameter is fixed at its estimated value; otherwise it stays
free, is reported as not identifiable ("kept free by design") and is left out of the later criteria.

Position in the pipeline: used by the structural and the practical steps of the selection (selection.py).
"""
import numpy as np

from settings import PROTECTED_PREFIXES
from model.parameters import reaction_groups

REACTION_GROUPS = reaction_groups()

def is_last_free_member(name, free_now):
    """True if ``name`` belongs to a reaction group in which no OTHER member is free."""
    for members in REACTION_GROUPS.values():
        if name not in members:
            continue
        others_free = [p for p in members if p in free_now and p != name]
        if not others_free:
            return True
    return False

def is_protected(name, free_now):
    """A protected prefix or the last free member of a reaction group: the rules cannot fix it."""
    if name.startswith(PROTECTED_PREFIXES):
        return True
    return is_last_free_member(name, free_now)

def can_fix(name, free_now):
    return not is_protected(name, free_now)

def largest_loading(names, null_vector):
    """(name, loading) of the largest |loading| of ``null_vector`` (the first in ``names`` order on a tie)."""
    sizes = np.round(np.abs(null_vector), 6)
    i = int(np.argmax(sizes))
    return names[i], float(null_vector[i])
