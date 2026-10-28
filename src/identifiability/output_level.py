"""
output_level.py -- output-level structural identifiability (free per-condition variables absorb rate changes).

The rate-level test of structural.py is only a NECESSARY condition.  The estimation problem has per-condition free
variables w = (x [9], u [7 cofactors], b [5 free drains]) with the steady-state constraint F(w, theta) =
S v(x, u, e; theta) - b = 0 and the measured outputs h(w, theta) (measured balanced metabolites and LI fluxes).  A
theta direction is output-level non-identifiable if in every condition the change of the rates and outputs can be
absorbed by moving w: B_c dtheta in range(A_c) with A_c = [F_w; h_w], B_c = [F_theta; h_theta].  With
Q_c = I - A_c pinv(A_c) these directions are the null space of the stacked Q_c B_c (theta columns relative, i.e. times
theta; rows scaled by 1/sigma and 1/(mean |v|)).  Version (i) ignores the bounds (structural verdict: null at all
q + 1 parameter points); version (ii) pins every w that sits on a bound at the anchor.  Classes per parameter:
identifiable / bound_identified (identifiable only through a bound) / non_identifiable.

Position in the pipeline: second half of the structural check of every iteration of the parameter selection
(through identifiability/check.py); its classes are reported in the final structure.
"""
import time

import numpy as np

from model.parameters import ALL_PARAMS, FREE_DRAIN_KEYS, FLUX_KEYS, BALANCED_KEYS, IMBALANCED_KEYS
from identifiability.structural import (realizations, perturb_states, svd_ratio, consensus_null, null_projector,
                                        groups_from_projector, group_basis, basis_strings)

def output_layout(est):
    """
    Per condition: the measured outputs [(name, sigma)] (read off the labeled Pyomo model, so they follow the
    estimation problem exactly), the free cofactors and drains, and the bounds of every w name (balanced keys,
    cofactor keys, "b_" + drain key).
    """
    layout = {}
    for i, condition in enumerate(est.conditions):
        experiment = est.exp_list[i]
        model = experiment.get_labeled_model()
        outputs = []
        for component in model.experiment_outputs.keys():
            outputs.append((component.local_name, float(model.measurement_error[component])))
        u_free = [k for k in IMBALANCED_KEYS if not getattr(model, k).fixed]
        lower, upper = {}, {}
        for key in BALANCED_KEYS:
            lower[key], upper[key] = est.x_bounds[key]
        for key in u_free:
            lower[key], upper[key] = est.u_bounds[key]
        for key in FREE_DRAIN_KEYS:
            lower["b_" + key], upper["b_" + key] = experiment.setup.drain_bounds[key]
        layout[condition] = {"outputs": outputs, "u_free": u_free, "b_free": list(FREE_DRAIN_KEYS), "lo": lower,
                             "hi": upper}
    return layout

def on_bound(value, lower, upper, active_rtol):
    return (value - lower) <= active_rtol * max(abs(lower), 1e-9) + 1e-9 or (upper - value) <= active_rtol * abs(upper)

def condition_block(rate_laws, S, lay, state, theta_vector, columns, active_set, params):
    """Q_c B_c of one condition and the number of its w variables on a bound (only counted with ``active_set``)."""
    n_balanced = len(BALANCED_KEYS)
    C = state["C"]
    e = state["e"]
    v = rate_laws.rates(C, e, theta_vector)
    JC = rate_laws.jac_state(C, e, theta_vector)
    Jk = rate_laws.jac(C, e, theta_vector)[:, columns]
    u_columns = [n_balanced + IMBALANCED_KEYS.index(k) for k in lay["u_free"]]
    w_names = list(BALANCED_KEYS) + list(lay["u_free"]) + ["b_" + k for k in lay["b_free"]]
    n_w = len(w_names)
    # F = S v - b, rows scaled by the mean |v| of the condition
    flux_scale = max(float(np.mean(np.abs(v))), 1e-12)
    F_w = np.zeros((len(FLUX_KEYS), n_w))
    F_w[:, :n_balanced] = S @ JC[:, :n_balanced]
    for j, u_column in enumerate(u_columns):
        F_w[:, n_balanced + j] = S @ JC[:, u_column]
    for j, key in enumerate(lay["b_free"]):
        F_w[BALANCED_KEYS.index(key), n_balanced + len(u_columns) + j] = -1.0
    rows_w = [F_w / flux_scale]
    rows_theta = [(S @ Jk) / flux_scale]
    for name, sigma in lay["outputs"]:
        row_w = np.zeros(n_w)
        row_theta = np.zeros(len(columns))
        if name in FLUX_KEYS:
            r = FLUX_KEYS.index(name)
            row_w[:n_balanced] = JC[r, :n_balanced]
            for j, u_column in enumerate(u_columns):
                row_w[n_balanced + j] = JC[r, u_column]
            row_theta = Jk[r]
        elif name in BALANCED_KEYS:
            row_w[BALANCED_KEYS.index(name)] = 1.0
        rows_w.append(row_w[None, :] / sigma)
        rows_theta.append(row_theta[None, :] / sigma)
    A = np.vstack(rows_w)
    # log-scaled w columns (x, u: times the state value; b: times the flux scale)
    w_scale = np.array([C[i] for i in range(n_balanced)] + [C[c] for c in u_columns]
                       + [flux_scale] * len(lay["b_free"]), dtype=float)
    A = A * w_scale[None, :]
    B = np.vstack(rows_theta) * theta_vector[columns][None, :]
    n_active = 0
    if active_set:
        values = {}
        for i, key in enumerate(BALANCED_KEYS):
            values[key] = C[i]
        for i, key in enumerate(IMBALANCED_KEYS):
            values[key] = C[n_balanced + i]
        keep = []
        for j, name in enumerate(w_names):
            value = float(state["b"][name[2:]]) if name.startswith("b_") else float(values[name])
            if on_bound(value, lay["lo"][name], lay["hi"][name], params["active_rtol"]):
                n_active += 1
            else:
                keep.append(j)
        A = A[:, keep]
    U, singular, _ = np.linalg.svd(A, full_matrices=False)
    rank = int(np.sum(singular > params["rank_rcond"] * max(singular[0], 1e-300))) if singular.size else 0
    U_range = U[:, :rank]
    return B - U_range @ (U_range.T @ B), n_active

def output_level_matrix(rate_laws, S, layout, theta, states, free, active_set, params):
    """
    Stacked Q_c B_c (rows = sum over the conditions of 9 + n_outputs, columns = ``free``, relative) at the parameter
    dict ``theta`` and the states {condition: {"C", "e", "b"}}.  ``active_set`` removes the w columns whose variable
    sits on a bound.  Returns (D, number of w on a bound).
    """
    theta_vector = np.array([theta[p] for p in ALL_PARAMS], dtype=float)
    columns = [ALL_PARAMS.index(p) for p in free]
    blocks = []
    n_active = 0
    for condition, lay in layout.items():
        block, n_on_bound = condition_block(rate_laws, S, lay, states[condition], theta_vector, columns, active_set,
                                            params)
        blocks.append(block)
        n_active += n_on_bound
    return np.vstack(blocks), n_active

def output_level_test(est, free, rate_laws, states, bounds, params, seed):
    """
    Output-level structural test at the anchor of ``est`` for the free parameters ``free``.

    Version (i) (bound-free) is evaluated at the anchor and at structural_q_out random points theta = anchor *
    exp(U(ln output_realization_lo, ln output_realization_hi)), clipped to ``bounds``, at the anchored states; a
    direction is flagged only if its singular value ratio is below structural_sv_tol at ALL points.  As a check that
    never flags, the same points are evaluated with all states perturbed (output_perturb_lo / hi; ``k_generic``):
    nulls that survive there are structural at generic states, the others belong to the fitted operating point.
    Version (ii) (active set: w on a bound at the anchor are pinned) is evaluated at the anchor only.

    Returns a dict: free, ratios_i, Vts_i, N_i / P_i (consensus null basis / projector, relative theta coordinates),
    k_i, ratio_anchor_i, k_generic, ratio_ii, k_ii, P_ii, n_active, classes {param: identifiable | bound_identified |
    non_identifiable} (null if the diagonal of the projector exceeds structural_load_min), null_i, null_ii, groups
    (connected groups of version (i) with their R basis), wall_s.
    """
    started = time.time()
    tol = params["structural_sv_tol"]
    load_min = params["structural_load_min"]
    q = params["structural_q_out"]
    free = [p for p in ALL_PARAMS if p in free]
    layout = output_layout(est)
    full = est.full_theta()
    anchor = {p: float(full[p]) for p in ALL_PARAMS}
    S = np.asarray(est.setup.S, dtype=float)
    thetas = realizations(anchor, bounds, q, seed, free, params["output_realization_lo"],
                          params["output_realization_hi"])
    ratios, Vts = [], []
    for theta in thetas:
        D, _ = output_level_matrix(rate_laws, S, layout, theta, states, free, False, params)
        ratio, Vt = svd_ratio(D)
        ratios.append(ratio)
        Vts.append(Vt)
    N_i, P_i = consensus_null(ratios, Vts, tol)
    generic_ratios, generic_Vts = [], []
    for i in range(1, len(thetas)):
        rng = np.random.default_rng(seed + 7000 + i)
        perturbed = perturb_states(states, rng, params["output_perturb_lo"], params["output_perturb_hi"])
        for condition in perturbed:
            perturbed[condition] = {**perturbed[condition], "b": states[condition]["b"]}
        D, _ = output_level_matrix(rate_laws, S, layout, thetas[i], perturbed, free, False, params)
        ratio, Vt = svd_ratio(D)
        generic_ratios.append(ratio)
        generic_Vts.append(Vt)
    k_generic = int(consensus_null(generic_ratios, generic_Vts, tol)[0].shape[1]) if generic_ratios else 0
    D_active, n_active = output_level_matrix(rate_laws, S, layout, anchor, states, free, True, params)
    ratio_ii, Vt_ii = svd_ratio(D_active)
    P_ii, k_ii = null_projector(Vt_ii, ratio_ii, tol)
    null_i = {p: bool(P_i[j, j] > load_min) for j, p in enumerate(free)}
    null_ii = {p: bool(P_ii[j, j] > load_min) for j, p in enumerate(free)}
    classes = {}
    for p in free:
        if not null_i[p]:
            classes[p] = "identifiable"
        elif null_ii[p]:
            classes[p] = "non_identifiable"
        else:
            classes[p] = "bound_identified"
    groups = []
    if N_i.shape[1] > 0:
        groups = groups_from_projector(P_i, free, load_min)
        for group in groups:
            group["R"], group["pivots"] = group_basis(P_i, group["idx"])
            group["vectors"] = basis_strings(group["R"], group["members"])
    return {"free": free, "ratios_i": ratios, "Vts_i": Vts, "N_i": N_i, "P_i": P_i, "k_i": int(N_i.shape[1]),
            "ratio_anchor_i": ratios[0], "k_generic": k_generic, "ratio_ii": ratio_ii, "k_ii": int(k_ii),
            "P_ii": P_ii, "n_active": int(n_active), "classes": classes, "null_i": null_i, "null_ii": null_ii,
            "groups": groups, "wall_s": time.time() - started}
