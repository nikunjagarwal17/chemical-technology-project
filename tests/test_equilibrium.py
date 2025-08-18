"""Simple test to verify equilibrium concentrations for A <=> B"""
from simplecantera import run_simulation


def test_equilibrium():
    spec = {
        'reaction': {'kf': 2.0, 'kr': 1.0},
        'initial': {'temperature': 300.0, 'conc': {'A': 1.0, 'B': 0.0}},
        'sim': {'time_span': 5.0, 'time_step': 0.001}
    }
    times, traj = run_simulation(spec)
    A_final, B_final = traj[-1]
    C = 1.0  # total
    # Analytical equilibrium: A_eq = C * kr/(kf+kr), B_eq = C * kf/(kf+kr)
    kf = 2.0
    kr = 1.0
    A_eq = C * kr / (kf + kr)
    B_eq = C * kf / (kf + kr)
    assert abs(A_final - A_eq) < 0.02
    assert abs(B_final - B_eq) < 0.02
