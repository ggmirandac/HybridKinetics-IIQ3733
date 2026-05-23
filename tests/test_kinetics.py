import io
import sys
import pytest
import numpy as np
import pyomo.environ as pyo
from Model.kinetics import EcoliCarbonKinetics

CONSTANTS = {
    "kI_pyr_1": 0.5, "kA_pep_1": 0.3, "v_max_1": 25.739,
    "Ka1_1": 1.0, "Ka2_1": 0.01, "Ka3_1": 1.0, "K_g6p_1": 0.5,
    "kI_pep_2": 0.12, "Km_g6p_2": 0.48, "Km_f6p_2": 0.19,
    "kcat_f_2": 1475.0, "kcat_r_2": 1000.0,
    "kI_f6p_3": 0.9, "kI_fbp_3": 2.0, "kI_gtp_3": 0.8,
    "kI_pep_3": 0.5, "kI_pi_3": 1.5, "kA_adp_3": 0.12, "kA_gdp_3": 0.15,
    "Km_f6p_3": 0.16, "Km_atp_3": 0.12, "Km_fbp_3": 0.5, "Km_adp_3": 0.2,
    "kcat_f_3": 580.0, "kcat_r_3": 100.0,
    "kI_3pg_4": 2.0, "kI_dhap_4": 0.08, "kI_g3p_4": 0.1, "kA_pep_4": 1.5,
    "kcat_f_4": 95.0, "kcat_r_4": 150.0,
    "Km_fbp_4": 0.3, "Km_g3p_4": 0.4, "Km_dhap_4": 2.0,
    "kcat_f_5": 4300.0, "kcat_r_5": 2400.0,
    "Km_dhap_5": 0.61, "Km_g3p_5": 1.2,
    "kI_adp_6": 0.8, "kI_amp_6": 1.0, "kI_atp_6": 0.2,
    "kcat_f_6": 118.0, "kcat_r_6": 10.0,
    "Km_g3p_6": 0.21, "Km_pi_6": 0.29, "Km_nad_6": 0.09,
    "Km_pgp_6": 0.01, "Km_nadh_6": 0.06,
    "kA_3pg_7": 0.5, "kA_atp_7": 0.3,
    "kcat_f_7": 1150.0, "kcat_r_7": 40.0,
    "Km_pgp_7": 0.05, "Km_adp_7": 0.1, "Km_3pg_7": 0.53, "Km_atp_7": 0.3,
    "kI_pi_8": 10.0, "kcat_f_8": 540.0, "kcat_r_8": 120.0,
    "Km_3pg_8": 0.2, "Km_2pg_8": 1.4,
    "kcat_f_9": 550.0, "kcat_r_9": 210.0,
    "Km_2pg_9": 0.1, "Km_pep_9": 0.5,
}

ENZYMES_A = {
    "Pgi": 0.01, "PfkB": 0.1, "FbaA": 0.1, "TpiA": 0.1,
    "GapA": 0.1, "Pgk": 0.01, "GpmA": 0.1, "Eno": 0.1,
}

ENZYMES_B = {
    "Pgi": 0.02, "PfkB": 0.05, "FbaA": 0.15, "TpiA": 0.08,
    "GapA": 0.12, "Pgk": 0.02, "GpmA": 0.09, "Eno": 0.11,
}

BOUNDS_UNBALANCED = {
    "C_atp":  (0.01, 1.0),
    "C_adp":  (0.001, 0.1),
    "C_amp":  (0.001, 0.1),
    "C_gdp":  (0.001, 0.1),
    "C_glc":  (0.001, 10.0),
    "C_gtp":  (0.001, 0.1),
    "C_nad":  (0.001, 0.1),
    "C_nadh": (0.001, 0.1),
    "C_pi":   (0.001, 10.0),
    "C_pyr":  (0.001, 10.0),
}

@pytest.fixture
def model():
    return EcoliCarbonKinetics(bounds_unbalanced_mets=BOUNDS_UNBALANCED)

@pytest.fixture
def built_model():
    m = EcoliCarbonKinetics(bounds_unbalanced_mets=BOUNDS_UNBALANCED)
    m.build_steady_state_model(ENZYMES_A)
    return m


def test_compute_fluxes_produces_no_stdout(model):
    C_balanced = {
        "C_2pg": 2.84e-5, "C_3pg": 4.52e-4, "C_dhap": 4.48e-6,
        "C_f6p": 3.53e-5, "C_fbp": 3.42e-5, "C_g3p": 4.23e-7,
        "C_g6p": 2.45e-4, "C_pep": 1.02e-4, "C_pgp": 2.56e-6,
    }
    C_unbalanced = {
        "C_atp": 0.1, "C_adp": 0.01, "C_amp": 0.01, "C_gdp": 0.01,
        "C_glc": 0.01, "C_gtp": 0.01, "C_nad": 0.01,
        "C_nadh": 0.01, "C_pi": 0.01, "C_pyr": 0.01,
    }
    captured = io.StringIO()
    sys.stdout = captured
    model.compute_fluxes(C_balanced, C_unbalanced, ENZYMES_A, CONSTANTS)
    sys.stdout = sys.__stdout__
    assert captured.getvalue() == ""
