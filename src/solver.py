"""
solver.py -- solver environment: single-threaded linear algebra, executable search path, solver check.

Importing this module (settings.py imports it first, and every module imports settings) limits BLAS / OpenMP to one
thread.  Multi-threaded MUMPS is not floating-point associative: the same fit converged to points 0.3% apart in the
objective depending on the threads available, and one thread per process also avoids oversubscription when several
solves run in parallel.  The variables reach the IPOPT subprocesses through os.environ.
Set HK_ALLOW_THREADS=1 to opt out.

It also puts the running interpreter's own bin directory first on PATH: a notebook kernel started without
``conda activate`` does not have it, and the IPOPT 3.13 bundled with IDAES (~/.idaes/bin) rejects options such as
acceptable_iter, which makes every solve fail; the conda IPOPT 3.14 next to the interpreter must win.  ~/.idaes/bin
(k_aug, dot_sens for the covariance) is appended to PATH, and to DYLD_FALLBACK_LIBRARY_PATH because the k_aug binary
does not find the libgfortran bundled next to it otherwise.
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

IDAES_BIN = os.path.join(os.path.expanduser("~"), ".idaes", "bin")
if os.path.isdir(IDAES_BIN):
    if IDAES_BIN not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + IDAES_BIN
    library_path = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    if IDAES_BIN not in library_path.split(os.pathsep):
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = IDAES_BIN + os.pathsep + library_path if library_path else IDAES_BIN

def check_solver(name="ipopt"):
    """Raise a RuntimeError with the install command if the solver is not available to Pyomo."""
    from pyomo.environ import SolverFactory
    if not SolverFactory(name).available():
        raise RuntimeError("Solver '%s' is not available in the current environment. Install it with:\n"
                           "    conda install -n hybridkin -c conda-forge %s\n" % (name, name))
