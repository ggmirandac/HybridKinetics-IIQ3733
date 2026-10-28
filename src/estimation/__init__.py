"""
estimation -- parameter estimation of the glycolysis model.

experiment.py  one condition as a labeled Pyomo model (GlycolysisExperiment)
estimator.py   GlycolysisParameterEstimator: fit, refine_anchor, fitted state, predictions, objective breakdown
parallel.py    independent fits in a process pool (engine of the multistart and the bootstrap)
multistart.py  recorded multistart and the multistart candidate of the selection loop
refit.py       monotone refit after fixing parameters
anchors.py     make_est, save_anchor, read_anchor, load_point, restore_anchor, check_bounds
"""
