"""
progress.py -- timestamped progress messages shared by the run modules.

Method: none (infrastructure).  ``log(msg)`` prints "[ elapsed s] msg" with the seconds since this module was
imported (the start of the run).

Position in the pipeline: used by settings.py, the selection, the final analysis, the uncertainty steps and the
estimation modules.
"""
import time

T0 = time.time()

def log(msg):
    print("[%6.0fs] %s" % (time.time() - T0, msg), flush=True)
