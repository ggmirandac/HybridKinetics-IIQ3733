"""
check.py -- the structural check at the current anchor (rate level + output level) and the plan of structural fixes.

``StructuralCheck.run`` runs the Stigter-Molenaar rate-level test (structural.py) and the output-level bound-free
test (output_level.py: free cofactors, drains and unmeasured states absorb rate changes) at the CURRENT anchor of an
estimator, restricted to its free parameters.  ``plan_structural_fixes`` then picks the fixes with the practical
influence (rules.choose_structural_fixes: the highest-influence parameter of a null group stays free, kcat_f_* are
never fixed, every reaction keeps a free parameter).  Members of an unresolved group that the protections forbid to
fix become "kept_free_by_rule": nuisance parameters of the practical criteria.

Realizations: theta_i = anchor * exp(U(ln realization_lo, ln realization_hi)) of the free parameters, with seed
seed_base + iteration (+ 100 * round), final_seed_base + iteration for the final check; the states are the fitted
x, u and b of every condition.  The perturbed-states variant (perturb_lo / perturb_hi, seed + perturb_seed_offset)
is a check only and never flags.

Position in the pipeline: at the start of every iteration of the parameter selection and once more at the final
anchor (selection.py, final_analysis.py).
"""
import time

import numpy as np

from settings import THETA_BOUNDS
from model.parameters import ALL_PARAMS
from identifiability.structural import (RateLaws, condition_states, analyse_free_set, sigma_gap, spectra_frame,
                                        symbolic_null_vectors, exact_vectors_free, groups_from_projector,
                                        group_basis, basis_strings, match_exact)
from identifiability.output_level import output_level_test
from identifiability.rules import choose_structural_fixes, is_protected

class StructuralCheck:
    """
    The structural check of the anchors of one run.  The exact null vectors of the rate laws are computed once here
    (sympy, about a minute); ``run`` evaluates both tests at the anchor of an estimator.
    """

    def __init__(self, params, log):
        self.params = params
        self.rate_laws = RateLaws()
        started = time.time()
        self.symbolic = symbolic_null_vectors(print_line)
        log("StructuralCheck: symbolic null vectors of all rate laws in %.0fs" % (time.time() - started))

    def seed(self, iteration, round_label):
        if round_label == "final":
            return self.params["final_seed_base"] + int(iteration)
        return self.params["seed_base"] + int(iteration) + 100 * int(round_label)

    def run(self, est, iteration, round_label):
        """
        Both tests at the anchor of ``est``.  Returns a dict: flagged (a consensus null space at the rate or the
        output level), rate_flagged, out_flagged, row (one row of structural_per_iteration.csv), spectra and
        spectra_perturbed (k x runs of the singular value ratios), groups (rate-level groups first, then
        output-level ones: members, reaction, dim, cls, d_exact, R, R_exact), output_level (the dict of
        output_level_test), fixes / unresolved / kept_free_by_rule (filled by plan_structural_fixes).
        """
        params = self.params
        free = [p for p in ALL_PARAMS if p in est.free_params]
        full = est.full_theta()
        anchor = {p: float(full[p]) for p in ALL_PARAMS}
        states = condition_states(est)
        seed = self.seed(iteration, round_label)
        rate = analyse_free_set(self.rate_laws, anchor, THETA_BOUNDS, states, free, params, seed)
        rate_perturbed = analyse_free_set(self.rate_laws, anchor, THETA_BOUNDS, states, free, params, seed,
                                          perturbed=True, state_seed=seed + params["perturb_seed_offset"])
        output = output_level_test(est, free, self.rate_laws, states, THETA_BOUNDS, params, seed)
        row = summary_row(iteration, round_label, free, seed, rate, rate_perturbed, output, params)
        k_rate = int(rate["N"].shape[1])
        k_output = int(output["k_i"])
        check = {"flagged": (k_rate > 0) or (k_output > 0), "row": row, "spectra": spectra_frame(rate),
                 "spectra_perturbed": spectra_frame(rate_perturbed), "groups": [], "fixes": [], "unresolved": [],
                 "kept_free_by_rule": [], "output_level": output, "rate_flagged": k_rate > 0,
                 "out_flagged": k_output > 0, "anchor": anchor, "free": free}
        if not check["flagged"]:
            row.update({"n_groups_flagged": 0, "n_exact": 0, "n_mixed": 0, "n_point_dependent": 0, "groups": "",
                        "n_flagged_params": 0, "n_kept_free_by_rule": 0, "kept_free_by_rule": ""})
            return check
        rate_groups = []
        if k_rate > 0:
            exact = exact_vectors_free(self.symbolic, free)
            exact_projector = exact.T @ exact if exact.shape[0] else None
            rate_groups = groups_from_projector(rate["P"], free, params["structural_load_min"], exact_projector)
            for group in rate_groups:
                group["R"], group["pivots"] = group_basis(rate["P"], group["idx"])
                group["vectors"] = basis_strings(group["R"], group["members"])
            match_exact(rate_groups, exact, free)
        output_groups = []
        if k_output > 0:
            output_groups = output["groups"]
            for group in output_groups:
                group["cls"] = "output-level"
                group["d_exact"] = 0
                group["R_exact"] = group["R"]
        check["rate_groups"] = rate_groups
        check["output_groups"] = output_groups
        check["groups"] = rate_groups + output_groups
        row.update(group_counts(check["groups"], len(output_groups)))
        return check

def summary_row(iteration, round_label, free, seed, rate, rate_perturbed, output, params):
    """The row of structural_per_iteration.csv (columns in the order of the earlier runs)."""
    tol = params["structural_sv_tol"]
    smallest = np.array([float(ratio[-1]) for ratio in rate["ratios"]])
    second = np.array([float(ratio[-2]) if len(ratio) > 1 else np.nan for ratio in rate["ratios"]])
    smallest_perturbed = np.array([float(ratio[-1]) for ratio in rate_perturbed["ratios"]])
    largest_zero, smallest_nonzero = sigma_gap(rate)
    k_rate = int(rate["N"].shape[1])
    class_counts = {}
    for name in ("identifiable", "bound_identified", "non_identifiable"):
        class_counts[name] = sum(1 for value in output["classes"].values() if value == name)
    output_groups = " | ".join("%s[dim%d]" % ("+".join(g["members"]), g["dim"]) for g in output["groups"])
    return {"iter": iteration, "round": round_label, "n_free": len(free), "seed": seed,
            "n_runs": len(rate["ratios"]),
            "sigma_ratio_anchor": float(smallest[0]), "sigma_ratio_min_over_runs": float(smallest.min()),
            "sigma_ratio_max_over_runs": float(smallest.max()),
            "sigma_2nd_ratio_anchor": float(second[0]),
            "gap_decades_to_next_anchor": float(np.log10(second[0] / max(smallest[0], 1e-300))),
            "gap_decades_to_next_min_over_runs": float(np.log10(np.nanmin(second) / max(smallest.min(), 1e-300))),
            "n_runs_with_ratio_below_cut": int((smallest < tol).sum()),
            "null_count_per_run": " ".join(str(count) for count in rate["counts"]),
            "null_dim_consensus": k_rate,
            "zero_ratio_max": largest_zero, "first_nonzero_ratio_min": smallest_nonzero,
            "gap_decades_null": float(np.log10(smallest_nonzero / max(largest_zero, 1e-300))) if k_rate else np.nan,
            "null_dim_perturbed_states": int(rate_perturbed["N"].shape[1]),
            "sigma_ratio_min_perturbed": float(smallest_perturbed.min()),
            "n_guarded_rows_max": int(max(rate["n_guard"])), "sv_cut": tol,
            "out_null_dim": int(output["k_i"]), "out_null_dim_active_set": int(output["k_ii"]),
            "out_null_dim_generic_states": int(output["k_generic"]),
            "out_sigma_ratio_anchor": float(output["ratio_anchor_i"][-1]),
            "out_sigma_ratio_min_over_runs": float(min(ratio[-1] for ratio in output["ratios_i"])),
            "out_n_active_w": int(output["n_active"]),
            "out_n_identifiable": class_counts["identifiable"],
            "out_n_bound_identified": class_counts["bound_identified"],
            "out_n_non_identifiable": class_counts["non_identifiable"],
            "out_groups": output_groups}

def group_counts(groups, n_output_groups):
    """Columns of a flagged check: counts by class and the groups as reaction:members[class, dim, exact dim]."""
    flagged_params = set()
    for group in groups:
        flagged_params.update(group["members"])
    described = []
    for group in groups:
        described.append("%s:%s[%s,dim%d,exact%d]" % (group["reaction"], "+".join(group["members"]), group["cls"],
                                                       group["dim"], group["d_exact"]))
    return {"n_groups_flagged": len(groups),
            "n_exact": sum(group["cls"] == "exact" for group in groups),
            "n_mixed": sum(group["cls"] == "mixed" for group in groups),
            "n_point_dependent": sum(group["cls"] == "point-dependent" for group in groups),
            "n_output_level": n_output_groups,
            "n_flagged_params": len(flagged_params),
            "groups": " | ".join(described)}

def plan_structural_fixes(check, influence, params):
    """
    The structural fixes of a flagged check, from the practical ``influence`` of the free parameters.  The
    rate-level groups (exact nulls) come first; the output-level groups contain the rate-level null space and are
    only used when the rate-level groups plan no fix, so nothing is fixed twice.  Exact and mixed groups use the
    exact null vectors.  Fills check["fixes"], check["unresolved"] and check["kept_free_by_rule"] (members of a
    group that cannot be resolved because the protections forbid fixing them) and the matching row columns.
    """
    free_now = list(check["free"])
    kept_by_rule = []
    group_number = 0
    for group_list in (check["rate_groups"], check["output_groups"]):
        if group_list is check["output_groups"] and check["fixes"]:
            break
        for group in group_list:
            group_number += 1
            null_basis = group["R_exact"] if group["cls"] in ("exact", "mixed") else group["R"]
            fix, kept, tied = choose_structural_fixes(influence, null_basis, group["members"], check["anchor"],
                                                      THETA_BOUNDS, free_now, params["tie_rtol"])
            for name in fix:
                free_now.remove(name)
                check["fixes"].append({"param": name, "group": group_number, "reaction": group["reaction"],
                                       "cls": group["cls"], "members": ";".join(group["members"]),
                                       "kept": ";".join(kept), "tied": bool(tied[name]),
                                       "influence": float(influence[name]), "value": check["anchor"][name],
                                       "output_class": check["output_level"]["classes"].get(name, ""),
                                       "null_dim": int(null_basis.shape[0]),
                                       "vectors": " | ".join(basis_strings(null_basis, group["members"]))})
            if len(fix) >= null_basis.shape[0]:
                continue
            protected = [p for p in group["members"] if p not in fix and is_protected(p, free_now)]
            if protected:
                for name in protected:
                    if name not in kept_by_rule:
                        kept_by_rule.append(name)
            else:
                check["unresolved"].append("structural group %s [%s] needs %d fixes, only %d fixable" % (
                    "+".join(group["members"]), group["cls"], null_basis.shape[0], len(fix)))
    check["kept_free_by_rule"] = kept_by_rule
    check["row"].update({"n_kept_free_by_rule": len(kept_by_rule), "kept_free_by_rule": ";".join(kept_by_rule)})

def print_line(message):
    print(message, flush=True)
