"""
structural.py -- rate-level structural identifiability (Stigter & Molenaar 2015).

Local structural identifiability after Stigter & Molenaar (2015, Automatica 58:118-124, "A fast algorithm to assess
local structural identifiability"), applied to the glycolysis model with the Noor et al. (2013) rate laws.  Their
steps 1-5:
  1. q random parameter realizations theta_i = anchor * exp(U(ln lo, ln hi)) (clipped to the bounds) and the exact
     relative output sensitivity matrix (ROSM) S_r(theta_i) of each;
  2. SVD of every S_r; a structural zero is a singular value ratio below structural_sv_tol in ALL realizations;
  3. the right singular vectors of the zero singular values span the null space; a sparse (reduced row echelon)
     basis and the parameter groups it couples are read off;
  4. the exact null vectors of every rate law, computed symbolically (sympy), classify the groups;
  5. structural fixing without reparametrization (identifiability/rules.py): per null group the parameter with the
     highest practical influence stays free, the others are fixed at their anchored values.

The outputs are the 9 reaction rates v_r(x, u, theta) at the fitted states of all conditions; the sensitivities are
exact (CasADi Jacobian of the very same rate-law code the estimator uses) and relative,
S[i, j] = theta_j / v_i * dv_i / dtheta_j.  A direction that is null here is null even if all states and rates were
known without noise.  Rows with |v_i| below structural_guard_rel times the median |v| of the reaction use the absolute
sensitivity scaled by that median instead.  The test is only a NECESSARY condition for the estimation problem: the
free cofactors, drains and unmeasured states can absorb rate changes as well (identifiability/output_level.py).

Position in the pipeline: the structural check at the start of every iteration of the parameter selection (through
identifiability/check.py).
"""
import time
from fractions import Fraction

import numpy as np
import pandas as pd
import casadi as ca

from model.kinetics import GlycolysisRateLaws
from model.parameters import (ALL_PARAMS, PARAM_RXN_MAP, RXN_ORDER, RXN_OF_PARAM, FLUX_KEYS, BALANCED_KEYS,
                              IMBALANCED_KEYS, ENZYME_KEYS)
from model.stoichiometry import load_keq
from data import load_condition

class RateLaws:
    """CasADi functions of the 9 reaction rates v(C, e, k) and their exact Jacobians dv/dk (9 x 37) and dv/dC."""

    def __init__(self):
        rate_laws = GlycolysisRateLaws(load_keq())
        C = ca.SX.sym("C", len(BALANCED_KEYS) + len(IMBALANCED_KEYS))
        e = ca.SX.sym("e", len(ENZYME_KEYS))
        k = ca.SX.sym("k", len(ALL_PARAMS))
        concentrations = {key: C[i] for i, key in enumerate(BALANCED_KEYS + IMBALANCED_KEYS)}
        enzymes = {key: e[i] for i, key in enumerate(ENZYME_KEYS)}
        constants = {key: k[i] for i, key in enumerate(ALL_PARAMS)}
        v = ca.vertcat(*rate_laws.compute_fluxes(concentrations, enzymes, constants))
        self.f_v = ca.Function("rates", [C, e, k], [v])
        self.f_J = ca.Function("drates_dk", [C, e, k], [ca.jacobian(v, k)])
        self.f_JC = ca.Function("drates_dC", [C, e, k], [ca.jacobian(v, C)])

    def rates(self, C, e, theta):
        return self.f_v(C, e, theta).full().ravel()

    def jac(self, C, e, theta):
        """dv/dtheta (9 x 37)."""
        return self.f_J(C, e, theta).full()

    def jac_state(self, C, e, theta):
        """dv/dC (9 x 16, columns = balanced keys then imbalanced keys)."""
        return self.f_JC(C, e, theta).full()

def condition_states(est):
    """
    {condition: {"C": 16-vector (balanced + imbalanced keys), "e": 8 enzyme levels, "b": {key: drain}}} at the fitted
    operating point of the anchored estimator ``est``.
    """
    x_by, u_by, b_by = est.fitted_state()
    states = {}
    for condition in est.conditions:
        C = np.array([x_by[condition][k] for k in BALANCED_KEYS] + [u_by[condition][k] for k in IMBALANCED_KEYS],
                     dtype=float)
        states[condition] = {"C": C, "e": enzyme_vector(condition), "b": b_by[condition]}
    return states

def enzyme_vector(condition):
    enzymes = load_condition(condition)["e"]
    return np.array([float(enzymes[k]) for k in ENZYME_KEYS], dtype=float)

def perturb_states(states, rng, lower, upper):
    """Every state value multiplied by exp(U(ln lower, ln upper)), independently per condition and state."""
    perturbed = {}
    for condition, state in states.items():
        factors = np.exp(rng.uniform(np.log(lower), np.log(upper), size=state["C"].size))
        perturbed[condition] = {"C": state["C"] * factors, "e": state["e"].copy(), "b": state["b"]}
    return perturbed

def realizations(anchor, bounds, q, seed, free, lower, upper):
    """
    q + 1 parameter dicts: the anchor, then q draws theta_i = anchor * exp(U(ln lower, ln upper)) clipped to
    ``bounds``.  The draws for ALL parameters come from one stream (seed); the parameters not in ``free`` stay at
    the anchor.
    """
    rng = np.random.default_rng(seed)
    thetas = [{p: float(anchor[p]) for p in ALL_PARAMS}]
    for _ in range(q):
        factors = np.exp(rng.uniform(np.log(lower), np.log(upper), size=len(ALL_PARAMS)))
        theta = {}
        for p, factor in zip(ALL_PARAMS, factors):
            if p in free:
                theta[p] = float(min(max(anchor[p] * factor, bounds[p][0]), bounds[p][1]))
            else:
                theta[p] = float(anchor[p])
        thetas.append(theta)
    return thetas

def run_names(q):
    return ["anchor"] + ["r%02d" % i for i in range(1, q + 1)]

def rosm_rate_law(rate_laws, theta, states, free, guard_rel):
    """
    Relative output sensitivity matrix of the reaction rates: rows = (condition, reaction) for all conditions,
    columns = ``free``.  Returns (S, number of rows that used the guard).
    """
    theta_vector = np.array([theta[p] for p in ALL_PARAMS], dtype=float)
    columns = [ALL_PARAMS.index(p) for p in free]
    conditions = list(states.keys())
    V = np.zeros((len(conditions), len(FLUX_KEYS)))
    J = np.zeros((len(conditions), len(FLUX_KEYS), len(ALL_PARAMS)))
    for i, condition in enumerate(conditions):
        state = states[condition]
        V[i] = rate_laws.rates(state["C"], state["e"], theta_vector)
        J[i] = rate_laws.jac(state["C"], state["e"], theta_vector)
    reference = np.array([np.median(np.abs(V[:, r])) for r in range(len(FLUX_KEYS))])
    rows = []
    n_guarded = 0
    for i in range(len(conditions)):
        for r in range(len(FLUX_KEYS)):
            denominator = V[i, r]
            if abs(V[i, r]) < guard_rel * max(reference[r], 1e-300):
                denominator = reference[r] if reference[r] > 0 else 1.0
                n_guarded += 1
            rows.append(J[i, r, columns] * theta_vector[columns] / denominator)
    return np.array(rows), n_guarded

def svd_ratio(S):
    """(sigma / sigma_max, Vt) of S (reduced SVD)."""
    _, singular, Vt = np.linalg.svd(S, full_matrices=False)
    return singular / singular[0], Vt

def rref_basis(B, tol=1e-8):
    """Reduced row echelon form of B^T for an (m x d) basis B; returns (R (r x m), pivot columns)."""
    A = np.array(B, dtype=float).T.copy()
    n_rows, n_cols = A.shape
    pivots = []
    r = 0
    for c in range(n_cols):
        if r >= n_rows:
            break
        i = r + int(np.argmax(np.abs(A[r:, c])))
        if abs(A[i, c]) < tol:
            continue
        A[[r, i]] = A[[i, r]]
        A[r] = A[r] / A[r, c]
        for j in range(n_rows):
            if j != r:
                A[j] = A[j] - A[j, c] * A[r]
        pivots.append(c)
        r += 1
    A[np.abs(A) < tol] = 0.0
    return A[:r], pivots

def rationalize(x, max_den=24, atol=1e-6):
    """The nearest simple fraction if it is within atol, else None."""
    fraction = Fraction(float(x)).limit_denominator(max_den)
    if abs(float(fraction) - float(x)) < atol:
        return fraction
    return None

def null_projector(Vt, ratio, tol):
    """(P, n_zero): projector onto the right singular vectors with a singular value ratio below ``tol``."""
    zero = ratio < tol
    N = Vt[zero].T
    return N @ N.T, int(zero.sum())

def consensus_null(ratios, Vts, tol, eig_tol=1e-6):
    """
    Null space common to ALL realizations (a direction counts only if its singular value ratio is below ``tol`` in
    every run): the eigenvectors with eigenvalue 1 of the mean of the per-run null projectors.  Returns
    (N (n x k) orthonormal basis, P = N N^T).
    """
    n = Vts[0].shape[1]
    mean_projector = np.zeros((n, n))
    for ratio, Vt in zip(ratios, Vts):
        projector, _ = null_projector(Vt, ratio, tol)
        mean_projector += projector / len(ratios)
    eigenvalues, vectors = np.linalg.eigh(0.5 * (mean_projector + mean_projector.T))
    N = vectors[:, eigenvalues > 1.0 - eig_tol]
    return N, N @ N.T

def groups_from_projector(P, free, load_min, P_extra=None):
    """
    Connected components of the parameters in the null space: a parameter is a member if P_ii > load_min, two
    members are coupled if |P_ij| > load_min.  ``P_extra`` (the projector of the exact symbolic null vectors) is
    merged by taking the larger magnitude entrywise.  Returns [{idx, members, dim_float, dim, reaction}] ordered by
    the first member.
    """
    n = len(free)
    magnitude = np.abs(P) if P_extra is None else np.maximum(np.abs(P), np.abs(P_extra))
    members = [i for i in range(n) if magnitude[i, i] > load_min]
    seen = set()
    groups = []
    for start in members:
        if start in seen:
            continue
        component = []
        stack = [start]
        while stack:
            a = stack.pop()
            if a in seen:
                continue
            seen.add(a)
            component.append(a)
            for b in members:
                if b not in seen and magnitude[a, b] > load_min:
                    stack.append(b)
        component = sorted(component)
        dim_float = float(np.trace(P[np.ix_(component, component)]))
        reactions = sorted({RXN_OF_PARAM[free[i]] for i in component})
        groups.append({"idx": component, "members": [free[i] for i in component], "dim_float": dim_float,
                       "dim": int(round(dim_float)), "reaction": "+".join(reactions)})
    groups.sort(key=first_member_index)
    return groups

def first_member_index(group):
    return min(group["idx"])

def group_basis(P, idx):
    """Sparse (RREF) basis of the null space restricted to the group ``idx``: (R (d x len(idx)), pivots)."""
    sub = P[np.ix_(idx, idx)]
    eigenvalues, vectors = np.linalg.eigh(0.5 * (sub + sub.T))
    return rref_basis(vectors[:, eigenvalues > 0.5])

def basis_strings(R, members, den=24):
    """Every row of the basis as text, e.g. '+1*Ks_f6p_3 -1*Ks_atp_3' (simple fractions where they fit)."""
    lines = []
    for row in R:
        terms = []
        for name, x in zip(members, row):
            if abs(x) < 1e-8:
                continue
            fraction = rationalize(x, den)
            size = ("%s" % abs(fraction)) if fraction is not None else "%.4g" % abs(x)
            terms.append("%s%s*%s" % ("+" if x > 0 else "-", size, name))
        lines.append(" ".join(terms))
    return lines

def analyse_free_set(rate_laws, anchor, bounds, states, free, params, seed, perturbed=False, state_seed=1000):
    """
    ROSM SVD over q + 1 realizations (index 0 = the anchor) for the free parameters ``free`` (the others stay at the
    anchor); with ``perturbed`` every realization also perturbs the states (seed state_seed + i).  Returns a dict:
    names, thetas, S_list, ratios, Vts, n_guard, counts (ratios below the tolerance per run), N, P (consensus null
    basis and projector), free.
    """
    tol = params["structural_sv_tol"]
    q = params["structural_q"]
    thetas = realizations(anchor, bounds, q, seed, free, params["realization_lo"], params["realization_hi"])
    S_list, ratios, Vts, guards = [], [], [], []
    for i, theta in enumerate(thetas):
        run_states = states
        if perturbed:
            rng = np.random.default_rng(state_seed + i)
            run_states = perturb_states(states, rng, params["perturb_lo"], params["perturb_hi"])
        S, n_guarded = rosm_rate_law(rate_laws, theta, run_states, free, params["structural_guard_rel"])
        ratio, Vt = svd_ratio(S)
        S_list.append(S)
        ratios.append(ratio)
        Vts.append(Vt)
        guards.append(n_guarded)
    N, P = consensus_null(ratios, Vts, tol)
    counts = [int((ratio < tol).sum()) for ratio in ratios]
    return {"names": run_names(q), "thetas": thetas, "S_list": S_list, "ratios": ratios, "Vts": Vts,
            "n_guard": guards, "counts": counts, "N": N, "P": P, "free": list(free)}

def sigma_gap(analysis):
    """
    (largest ratio among the zero singular values over all runs, smallest ratio of the first non-zero singular
    value over all runs): the gap that separates the structural zeros from the rest.
    """
    k = analysis["N"].shape[1]
    n = len(analysis["free"])
    largest_zero = max([float(ratio[n - k:].max()) if k else 0.0 for ratio in analysis["ratios"]])
    smallest_nonzero = float("nan")
    if n - k - 1 >= 0:
        smallest_nonzero = min([float(ratio[n - k - 1]) for ratio in analysis["ratios"]])
    return largest_zero, smallest_nonzero

def spectra_frame(analysis):
    """Singular value ratios of every run (columns) by index k = 1..n (rows)."""
    n = len(analysis["free"])
    frame = pd.DataFrame(dict(zip(analysis["names"], analysis["ratios"])), index=np.arange(1, n + 1))
    frame.index.name = "k"
    return frame

def polynomial_matrix(sp, rows, state_symbols):
    """
    Coefficient matrix M of the polynomial identity sum_j n_j * rows[j](states) = 0: rows[j] are rational functions
    of the states; returns (M, monomials, common denominator).
    """
    denominator = 1
    for row in rows:
        _numerator, row_denominator = sp.fraction(sp.together(row))
        denominator = sp.lcm(denominator, row_denominator)
    polynomials = []
    for row in rows:
        numerator = sp.expand(sp.cancel(sp.together(row * denominator)))
        polynomials.append(sp.Poly(numerator, *state_symbols))
    monomials = set()
    for polynomial in polynomials:
        monomials |= set(polynomial.monoms())
    monomials = sorted(monomials)
    coefficients = []
    for monomial in monomials:
        coefficients.append([polynomial.coeff_monomial(monomial) for polynomial in polynomials])
    M = sp.Matrix(coefficients) if monomials else sp.zeros(0, len(rows))
    return M, monomials, denominator

def reaction_null_vectors(sp, rate, params, theta_symbols, state_symbols, rng, n_points):
    """
    Exact null vectors of one rate law in LOG-parameter space: the vectors n with sum_j n_j theta_j dv/dtheta_j = 0
    identically in the states.  The nullity is evaluated in exact rational arithmetic at ``n_points`` random
    parameter points; the basis is the RREF at the first point.  Every vector is then verified with symbolic
    parameters: J n simplifies to 0 and v is invariant under theta_j -> theta_j * a**n_j.
    Returns (nullity, null space identical at all points, basis (sympy Matrix), verified).
    """
    log_jacobian = [sp.simplify(theta_symbols[p] * sp.diff(rate, theta_symbols[p])) for p in params]
    M, _monomials, _denominator = polynomial_matrix(sp, log_jacobian, state_symbols)
    symbols = set()
    for i in range(M.rows):
        for j in range(M.cols):
            symbols |= sp.sympify(M[i, j]).free_symbols
    symbols = sorted(symbols, key=str) if M.rows else []
    nullities, bases = [], []
    for _ in range(n_points):
        point = {}
        for symbol in symbols:
            point[symbol] = sp.Rational(int(rng.integers(2, 60)), int(rng.integers(2, 60)))
        null_space = M.subs(point).nullspace()
        nullities.append(len(null_space))
        if null_space:
            rows = sp.Matrix.hstack(*null_space).T
            bases.append(rows.rref()[0][:len(null_space), :])
    nullity = int(min(nullities)) if nullities else 0
    identical = all(bases[0] == basis for basis in bases) if bases else True
    basis = bases[0] if bases else sp.zeros(0, len(params))
    if nullity == 0:
        return nullity, identical, basis, True
    a = sp.Symbol("a", positive=True)
    verified = []
    for k in range(basis.rows):
        vector = list(basis.row(k))
        residual = sp.simplify(sum(vector[j] * log_jacobian[j] for j in range(len(params))))
        scaled = {theta_symbols[p]: theta_symbols[p] * a ** vector[j] for j, p in enumerate(params)}
        invariance = sp.simplify(rate.subs(scaled, simultaneous=True) - rate)
        verified.append(residual == 0 and invariance == 0)
    return nullity, identical, basis, all(verified) and identical

def symbolic_null_vectors(log, n_points=3, seed=0):
    """
    {reaction: {"params", "vectors" (float array, rows = exact null vectors in log-parameter space)}} of every rate
    law, written by the very same rate-law methods as the estimator (with sympy symbols for the states, the
    parameters and the equilibrium constants).
    """
    import sympy as sp
    states = {k: sp.Symbol(k, positive=True) for k in BALANCED_KEYS + IMBALANCED_KEYS}
    enzymes = {k: sp.Symbol("E_" + k, positive=True) for k in ENZYME_KEYS}
    theta_symbols = {p: sp.Symbol(p, positive=True) for p in ALL_PARAMS}
    keq = {r: sp.Symbol("Keq_" + r, positive=True) for r in load_keq()}
    rate_laws = GlycolysisRateLaws(keq)
    state_symbols = list(states.values())
    rng = np.random.default_rng(seed)
    vectors = {}
    for reaction in RXN_ORDER:
        started = time.time()
        params = list(PARAM_RXN_MAP[reaction])
        rate = getattr(rate_laws, reaction)(theta_symbols, states, enzymes)
        nullity, _identical, basis, verified = reaction_null_vectors(sp, rate, params, theta_symbols, state_symbols,
                                                                     rng, n_points)
        log("symbolic %s: generic nullity %d (verified %s) [%.1fs]" % (reaction, nullity, verified,
                                                                       time.time() - started))
        rows = [[float(x) for x in basis.row(k)] for k in range(basis.rows)]
        vectors[reaction] = {"params": params, "vectors": np.array(rows).reshape(basis.rows, len(params))}
    return vectors

def combinations_vanishing_on(B, columns):
    """Rows spanning the combinations of the rows of B that vanish on ``columns`` (B itself if there are none)."""
    if not columns:
        return np.eye(B.shape[0])
    _u, singular, vt = np.linalg.svd(B[:, columns].T, full_matrices=True)
    rank = int((singular > 1e-9).sum())
    return vt[rank:]

def exact_vectors_free(symbolic, free):
    """
    Exact (symbolic) null vectors in LOG-parameter space RESTRICTED to the free set: for every reaction the
    combinations of its null vectors that vanish on its parameters that are NOT free (a fixed parameter has no
    coordinate, so a vector with weight on it is no longer a null direction of the reduced model).
    Returns E (d x len(free)); rows of different reactions have disjoint supports.
    """
    rows = []
    for item in symbolic.values():
        B = np.asarray(item["vectors"], dtype=float)
        params = item["params"]
        if B.size == 0:
            continue
        fixed_columns = [j for j, p in enumerate(params) if p not in free]
        if fixed_columns:
            combinations = combinations_vanishing_on(B, fixed_columns)
            B = combinations @ B if combinations.shape[0] else np.zeros((0, len(params)))
        V = np.zeros((B.shape[0], len(free)))
        for i in range(B.shape[0]):
            for p, x in zip(params, B[i]):
                if p in free:
                    V[i, free.index(p)] = x
        V = V[np.linalg.norm(V, axis=1) > 1e-12]
        if V.shape[0]:
            _u, singular, vt = np.linalg.svd(V, full_matrices=False)
            rows.extend(list(vt[singular > 1e-9]))
    return np.array(rows).reshape(len(rows), len(free))

def match_exact(groups, E, free):
    """
    Classify the numerical null groups against the exact null space E of the free set.  Sets per group d_exact (rank
    of the exact vectors supported inside the group), R_exact (sparse basis) and cls: 'exact' (the exact space has
    the full numerical dimension and contains the numerical basis), 'mixed' (0 < d_exact < dim) or
    'point-dependent' (no exact vector: a near-null direction of this anchor or state only).
    """
    n = len(free)
    for group in groups:
        idx = group["idx"]
        outside = [j for j in range(n) if j not in idx]
        live = np.zeros((0, len(idx)))
        if E.shape[0]:
            combinations = combinations_vanishing_on(E, outside)
            if combinations.shape[0]:
                live = combinations @ E[:, idx]
        d_live = int(np.linalg.matrix_rank(live, tol=1e-8)) if live.shape[0] else 0
        R_exact = rref_basis(live.T)[0] if d_live else np.zeros((0, len(idx)))
        R = group["R"]
        contained = bool(d_live) and (np.linalg.matrix_rank(np.vstack([R, R_exact]), tol=1e-8) == d_live)
        group["d_exact"] = d_live
        group["R_exact"] = R_exact
        if d_live >= group["dim"] and contained:
            group["cls"] = "exact"
        elif d_live > 0:
            group["cls"] = "mixed"
        else:
            group["cls"] = "point-dependent"
    return groups
