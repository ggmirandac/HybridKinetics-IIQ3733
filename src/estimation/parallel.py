"""
parallel.py -- independent estimator fits in a process pool (the engine of the multistart and the bootstrap).

``fit_many`` fits a FRESH ``GlycolysisParameterEstimator`` on every (condition list, starting theta) task,
sequentially or in a ProcessPoolExecutor.  The estimator wraps Pyomo / parmest objects that cannot be pickled, so
every worker builds its own estimator from picklable constructor arguments and only dicts and numbers come back.
The results keep the order of the tasks, so the same seeds give identical results for any n_jobs.

Position in the pipeline: used by estimation/multistart.py and uncertainty/bootstrap.py.
"""
import time
from concurrent.futures import ProcessPoolExecutor

from estimation.estimator import GlycolysisParameterEstimator

# The sets of constructor arguments shared by the tasks, set once in every worker process (see start_worker)
WORKER_ARGS = None

def fit_one(conditions, theta_init, estimator_args):
    """
    Build and fit one estimator.  Returns (theta dict or None, objective or None, converged, error message, wall
    seconds); parmest raises on a non-optimal termination, so converged means no exception.
    """
    started = time.time()
    try:
        est = GlycolysisParameterEstimator(conditions=conditions, theta_init=theta_init,
                                           free_params=estimator_args["free_params"],
                                           solver_options=estimator_args["solver_options"],
                                           x_bounds=estimator_args["x_bounds"], u_bounds=estimator_args["u_bounds"],
                                           drain_bounds=estimator_args["drain_bounds"],
                                           x_init=estimator_args["x_init"], u_init=estimator_args["u_init"],
                                           b_init=estimator_args["b_init"])
        theta = est.estimate()
    except Exception as exc:   # noqa: BLE001  (a fit that does not converge is a recorded outcome)
        return None, None, False, ("%s: %s" % (type(exc).__name__, exc))[:500], time.time() - started
    return dict(theta), float(est.obj_value), True, "", time.time() - started

def start_worker(estimator_args):
    global WORKER_ARGS
    WORKER_ARGS = estimator_args

def fit_task(task):
    conditions, theta_init, args_index = task
    return fit_one(conditions, theta_init, WORKER_ARGS[args_index])

def fit_many(condition_lists, theta_inits, estimator_args, n_jobs, args_indices=None):
    """
    Fit a fresh estimator on every pair (conditions, theta_init) of ``condition_lists`` and ``theta_inits``.
    ``estimator_args`` holds the other constructor arguments (free_params, solver_options, x_bounds, u_bounds,
    drain_bounds, x_init, u_init, b_init); it can also be a list of such dicts, and then ``args_indices`` says which
    one every task uses.  ``n_jobs`` = 1 runs in this process, otherwise in a pool of n_jobs workers.  Returns one
    (theta, objective, converged, error, wall_s) per task, in task order.
    """
    if isinstance(estimator_args, dict):
        estimator_args = [estimator_args]
    if args_indices is None:
        args_indices = [0] * len(theta_inits)
    tasks = list(zip(condition_lists, theta_inits, args_indices))
    if n_jobs == 1:
        results = []
        for conditions, theta_init, args_index in tasks:
            results.append(fit_one(conditions, theta_init, estimator_args[args_index]))
        return results
    with ProcessPoolExecutor(max_workers=int(n_jobs), initializer=start_worker,
                             initargs=(estimator_args,)) as pool:
        return list(pool.map(fit_task, tasks, chunksize=1))
