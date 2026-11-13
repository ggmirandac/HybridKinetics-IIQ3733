"""
solver.py -- solver environment: single-threaded linear algebra, executable search path, solver check.

Importing this module (settings.py imports it first, and every module imports settings) limits BLAS / OpenMP to one
thread.  Multi-threaded MUMPS is not floating-point associative: the same fit converged to points 0.3% apart in the
objective depending on the threads available, and one thread per process also avoids oversubscription when several
solves run in parallel.  The variables reach the IPOPT subprocesses through os.environ.
Set HK_ALLOW_THREADS=1 to opt out.

It also puts the running interpreter's own bin directory first on PATH: a notebook kernel started without
``conda activate`` does not have it, and the IPOPT 3.13 bundled with IDAES (~/.idaes/bin) rejects options such as
acceptable_iter, which makes every solve fail; the conda IPOPT 3.14 next to the interpreter must win.
"""
import os
import sys

THREAD_LIMITS = None
if os.environ.get("HK_ALLOW_THREADS", "0") != "1":
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    # a module that imported numpy before this one has its BLAS pools running already: limit them as well
    from threadpoolctl import threadpool_limits
    THREAD_LIMITS = threadpool_limits(limits=1)

INTERPRETER_BIN = os.path.dirname(os.path.abspath(sys.executable))
if INTERPRETER_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = INTERPRETER_BIN + os.pathsep + os.environ.get("PATH", "")

def check_solver(name="ipopt"):
    """Raise a RuntimeError with the install command if the solver is not available to Pyomo."""
    from pyomo.environ import SolverFactory
    if not SolverFactory(name).available():
        raise RuntimeError("Solver '%s' is not available in the current environment. Install it with:\n"
                           "    conda install -n hybridkin -c conda-forge %s\n" % (name, name))
