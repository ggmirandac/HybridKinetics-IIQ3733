import pandas as pd
import numpy as np
import mealpy
from mealpy import FloatVar, PSO, GWO
from mealpy.evolutionary_based import DE
import time


# ── Parameter bounds by type (natural space) ────────────────────────────────
# Organized by prefix so each kinetic constant gets a physiologically
# meaningful search range instead of a flat [0.01, 100] for everything.
PARAM_BOUNDS_BY_PREFIX = {
    "kI"   : (1e-3, 20.0),    # inhibition constants, mM
    "kA"   : (1e-3, 20.0),    # activation constants, mM
    "Km"   : (1e-3, 20.0),    # Michaelis constants, mM
    "kcat" : (0.1,  5000.0),  # catalytic rates, s⁻¹
    "v_max": (1.0,  100.0),   # max velocity, mmol/gDW/h
    "Ka"   : (1e-3, 10.0),    # PTS affinity constants
    "K_"   : (1e-3, 10.0),    # other K-type constants
}

def _bounds_for_param(name: str):
    for prefix, bounds in PARAM_BOUNDS_BY_PREFIX.items():
        if name.startswith(prefix):
            return bounds
    return (1e-3, 1000.0)   # fallback

def build_bounds(param_keys):
    lb = np.array([_bounds_for_param(k)[0] for k in param_keys])
    ub = np.array([_bounds_for_param(k)[1] for k in param_keys])
    return lb, ub


# ── Objective function ───────────────────────────────────────────────────────




# ── PSO ─────────────────────────────────────────────────────────────────────
def param_estimation_mealpy(
    model,
    conditions_enzymes,
    conditions_obs,
    conditions_cell_needs,
    bounds_params,
    max_measure,
    algorithm  = 'PSO',    # 'PSO', 'GWO', 'DE'
    epoch      = 500,
    pop_size   = 50,
    penalty_weight = 1e3,   # penalty for negative fluxes (squared)
    seed       = 42,
    # PSO hyperparameters (ignored for other algorithms)
    c1 = 2.05,
    c2 = 2.05,
    w  = 0.4,
):
    """
    Bi-level parameter estimation:
      - Outer: mealpy population-based global search over kinetic parameters theta.
      - Inner: IPOPT feasibility solve  min 0  s.t. S @ v(C, e, theta) = 0
               to find the steady-state concentrations for each condition.

    Parameters are optimised in log-space so that kcat, Km, kI values
    spanning orders of magnitude are explored proportionally.
    Failed inner solves are penalised with a large objective value.

    Parameters
    ----------
    algorithm : 'PSO' | 'GWO' | 'DE'
        PSO  - Particle Swarm (good default, fast convergence).
        GWO  - Grey Wolf Optimizer (robust on multimodal landscapes).
        DE   - Differential Evolution (strong exploration, slower per epoch).
    epoch     : number of iterations of the outer optimizer.
    pop_size  : population size (more = better exploration, slower per epoch).
    """

    # ── Index alignment check ─────────────────────────────────────────────
    pred_keys = set(model.balanced_keys) | set(model.flux_keys)
    obs_keys  = set(conditions_obs[0].index)
    matched   = pred_keys & obs_keys
    ignored   = obs_keys - pred_keys
    if not matched:
        raise ValueError(
            f"No observations match model outputs.\n"
            f"  Obs index  : {sorted(obs_keys)}\n"
            f"  Model keys : {sorted(pred_keys)}\n"
            f"  Rename your DataFrame indices to match model keys."
        )
    if ignored:
        print(f"[WARNING] Observations silently ignored (name mismatch): {ignored}")
    print(f"[INFO] Fitting on {len(matched)} variables: {matched}")

    n_par    = len(model.params_keys)
    lb_theta = np.array([bounds_params[k][0] for k in model.params_keys])
    ub_theta = np.array([bounds_params[k][1] for k in model.params_keys])

    # Work in log-space: mealpy samples log(theta) uniformly,
    # we exponentiate inside the objective — this is equivalent to log-uniform sampling.
    log_lb = np.log(np.maximum(lb_theta, 1e-12))
    log_ub = np.log(ub_theta)
    
    


    # ── Objective function ────────────────────────────────────────────────
    def objective_fn(log_params):
        theta = {model.params_keys[i]: np.exp(log_params[i]) for i in range(n_par)}
        total_error = 0.0
    
        for k, (enz_k, obs_k, cell_needs_k) in enumerate(zip(conditions_enzymes, conditions_obs, conditions_cell_needs)):

            ss   = model.solve_steady_state(
                enzymes       = enz_k,
                kinetic_params= theta,
                cell_needs    = cell_needs_k,
                condition_key = str(k),   # enables warm-starting across iterations
            )
            pred = ss[0].iloc[0]

            for var in obs_k.index:
                obs_val = obs_k[var]
                if not np.isnan(float(obs_val)) and var in pred.index:
                    
                    
                    rel_err = ((pred[var] - obs_val) / (max_measure[var]))**2
                    total_error += rel_err


        return total_error

    # ── mealpy problem definition ─────────────────────────────────────────
    problem_dict = {
        "obj_func"        : objective_fn,
        "bounds"          : FloatVar(lb=log_lb.tolist(), ub=log_ub.tolist()),
        "minmax"          : "min",
        "save_population" : False,   # saves memory; set True to inspect convergence
    }

    # ── Algorithm selection ───────────────────────────────────────────────
    if algorithm == 'PSO':
        optimizer = PSO.OriginalPSO(epoch=epoch, pop_size=pop_size,
                                     c1=c1, c2=c2, w=w)
    elif algorithm == 'GWO':
        optimizer = GWO.OriginalGWO(epoch=epoch, pop_size=pop_size)
    elif algorithm == 'DE':
        optimizer = DE.OriginalDE(epoch=epoch, pop_size=pop_size)
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose 'PSO', 'GWO', or 'DE'.")

    print(f"Running {algorithm}  (epoch={epoch}, pop_size={pop_size}, "
          f"total evals ≈ {epoch * pop_size:,})...")
    t0     = time.perf_counter()
    g_best = optimizer.solve(problem_dict, mode='single', seed=seed)
    elapsed = time.perf_counter() - t0

    # Exponentiate back from log-space
    p_opt    = {model.params_keys[i]: np.exp(g_best.solution[i]) for i in range(n_par)}
    best_obj = g_best.target.fitness

    print(f"Done in {elapsed:.1f} s  |  Best objective: {best_obj:.4e}")
    return p_opt, best_obj


def param_estimation_ipopt(
    model,
    conditions_enzymes,
    conditions_obs,
    bounds_params,
    n_starts            = 30,
    seed                = 42,
    # ── IPOPT tolerances ──────────────────────────────────────────────────
    tol                 = 1e-4,   # primary convergence tolerance
    acceptable_tol      = 1e-3,   # accept if it stays within this for `acceptable_iter` steps
    acceptable_iter     = 15,     # early-stop after N "acceptable" iterations
    max_iter            = 3000,   # max iterations per start
    # ── Barrier / scaling ─────────────────────────────────────────────────
    mu_strategy         = "adaptive",        # 'adaptive' (robust) or 'monotone' (aggressive)
    nlp_scaling_method  = "gradient-based",  # helps with mixed-scale variables
    bound_push          = 1e-8,              # pushes x0 off bounds to avoid singularities
    bound_frac          = 1e-8,
    # ── Verbosity ─────────────────────────────────────────────────────────
    print_level         = 0,      # 0 = silent, 5 = full IPOPT log per start
):
    """
    Multi-start parameter estimation via a simultaneous IPOPT NLP.
    θ is sampled log-uniformly so parameters spanning orders of magnitude
    (kcat, Km, kI …) are explored proportionally across the full range.
    """

    # ── Index alignment check ─────────────────────────────────────────────
    pred_keys = set(model.balanced_keys) | set(model.flux_keys)
    obs_keys  = set(conditions_obs[0].index)
    matched   = pred_keys & obs_keys
    ignored   = obs_keys - pred_keys
    if not matched:
        raise ValueError(
            f"No observations match model outputs.\n"
            f"  Obs index  : {sorted(obs_keys)}\n"
            f"  Model keys : {sorted(pred_keys)}\n"
            f"  Rename your DataFrame indices to match model keys."
        )
    if ignored:
        print(f"[WARNING] Observations silently ignored (name mismatch): {ignored}")
    print(f"[INFO] Fitting on {len(matched)} variables: {matched}")

    # ── Build IPOPT options ───────────────────────────────────────────────
    ipopt_opts = {
        "ipopt": {
            "print_level"           : print_level,
            "sb"                    : "yes",
            "mu_strategy"           : mu_strategy,
            "nlp_scaling_method"    : nlp_scaling_method,
            "tol"                   : tol,
            "acceptable_tol"        : acceptable_tol,
            "acceptable_iter"       : acceptable_iter,
            "max_iter"              : max_iter,
            "bound_push"            : bound_push,
            "bound_frac"            : bound_frac,
        },
        "print_time": 0,
    }

    # ── Build NLP once, reuse solver for all starts ───────────────────────
    rng = np.random.default_rng(seed)
    solver, lb_x, ub_x, lbg, ubg = model.build_parameter_estimation_nlp(
        conditions_enzymes, conditions_obs, bounds_params, ipopt_opts=ipopt_opts
    )

    n_par    = len(model.params_keys)
    lb_theta = lb_x[:n_par]
    ub_theta = ub_x[:n_par]

    # Log-uniform sampling bounds (clamp away from 0)
    log_lb = np.log(np.maximum(lb_theta, 1e-12))
    log_ub = np.log(ub_theta)

    best_obj, best_x = np.inf, None
    t0 = time.perf_counter()
    print(f"Running {n_starts} multi-start IPOPT solves "
          f"(tol={tol:.0e}, acceptable_tol={acceptable_tol:.0e}, max_iter={max_iter})...")

    for i in range(n_starts):
        # Log-uniform theta: explores proportionally across orders of magnitude
        theta0 = np.exp(log_lb + rng.random(n_par) * (log_ub - log_lb))
        # Metabolite C0: use geometric midpoint of bounds (already in lb_x[n_par:])
        C0 = np.sqrt(lb_x[n_par:] * ub_x[n_par:])   # geometric mean for all conditions
        x0 = np.concatenate([theta0, C0])

        try:
            sol     = solver(x0=x0, lbx=lb_x, ubx=ub_x, lbg=lbg, ubg=ubg)
            obj_val = float(sol["f"])
            if obj_val < best_obj:
                best_obj = obj_val
                best_x   = sol["x"].full().flatten()
                print(f"  Start {i+1:3d}/{n_starts}: new best = {best_obj:.4e}")
        except Exception as exc:
            print(f"  Start {i+1:3d}/{n_starts}: failed ({exc})")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f} s  |  Best objective: {best_obj:.4e}")

    if best_x is None:
        raise RuntimeError("All starts failed — check bounds and model feasibility.")

    p_opt = {model.params_keys[i]: best_x[i] for i in range(n_par)}
    return p_opt, best_obj