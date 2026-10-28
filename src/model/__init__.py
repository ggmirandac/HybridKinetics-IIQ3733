"""
model -- the glycolysis model, constructed in ONE place.

parameters.py     the 37 parameters, the reaction map, metabolite / flux keys, literature values, reaction groups
kinetics.py       rate laws (EcoliCarbonKinetics) and the CasADi steady-state / closure solvers
stoichiometry.py  stoichiometric matrix S and equilibrium constants Keq

Every other module takes the model from here; to change the model edit these files.
"""
