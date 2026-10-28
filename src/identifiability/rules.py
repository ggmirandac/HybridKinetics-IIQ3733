"""
rules.py -- which parameters may be fixed, and which members of a structural null group are fixed.

Protections: the parameters with a prefix of settings/protected_params.csv (kcat_f_*) are never fixed, and the LAST
free member of a reaction group is kept, so every rate equation keeps at least one adjustable parameter (one group
per reaction, model.parameters.reaction_groups).  ``choose_structural_fixes`` picks, for one null group, the
parameters to fix: candidates in ascending practical influence (the highest-influence parameter stays free); a set
of fixes removes the null space iff the columns of the group basis indexed by the set form an invertible matrix.

Position in the pipeline: used by the structural check (identifiability/check.py) and by the practical rules of the
selection loop (selection.py).
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
    """kcat_f_* (never fixed) or the last free member of a reaction group: the rules cannot fix it."""
    if name.startswith(PROTECTED_PREFIXES):
        return True
    return is_last_free_member(name, free_now)

def can_fix(name, free_now):
    return not is_protected(name, free_now)

def room_to_bounds(name, values, bounds):
    """Log distance of the value to its nearest theta bound."""
    lower, upper = bounds[name]
    return min(np.log(values[name] / lower), np.log(upper / values[name]))

def influence_ties(members, influence, tie_rtol):
    """
    The members in ascending influence, split into runs of ties: a member whose influence is within ``tie_rtol``
    (relative) of the previous member of the run joins the run.
    """
    ordered = sorted(members, key=influence.get)
    runs = [[ordered[0]]]
    for name in ordered[1:]:
        previous = runs[-1][-1]
        if abs(influence[name] - influence[previous]) <= tie_rtol * max(abs(influence[previous]), 1e-300):
            runs[-1].append(name)
        else:
            runs.append([name])
    return runs

def choose_structural_fixes(influence, null_basis, members, values, bounds, free_now, tie_rtol):
    """
    Parameters of one null group to fix.  ``null_basis`` is the (d x m) basis of the group's null space over
    ``members``.  Candidates come in ascending practical influence, so the parameter with the highest influence
    stays free.  The members of an exact null group have IDENTICAL influence in theory (their log-sensitivity
    columns coincide up to sign); influences within ``tie_rtol`` count as tied (the finite-difference noise of
    low-influence columns is ~10%), and among tied members the one with the most room to its bounds is kept free, so
    the identified combination (e.g. the product of a pair) is not clipped by a bound.  Protected parameters are
    skipped.  Returns (fix list, kept list, {member: tied flag}).
    """
    d = null_basis.shape[0]
    column_of = {name: i for i, name in enumerate(members)}
    candidates = []
    tied = {}
    room = {name: room_to_bounds(name, values, bounds) for name in members}
    for run in influence_ties(members, influence, tie_rtol):
        for name in sorted(run, key=room.get):
            candidates.append(name)
            tied[name] = len(run) > 1
    fix = []
    free_set = set(free_now)
    for name in candidates:
        if len(fix) == d:
            break
        if name.startswith(PROTECTED_PREFIXES) or is_last_free_member(name, free_set):
            continue
        trial = fix + [name]
        sub_basis = null_basis[:, [column_of[q] for q in trial]]
        if np.linalg.matrix_rank(sub_basis, tol=1e-8) == len(trial):
            fix = trial
            free_set.discard(name)
    kept = [name for name in members if name not in fix]
    return fix, kept, {name: tied[name] for name in members}
