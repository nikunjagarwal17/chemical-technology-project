"""Test ReactorNetwork series of two well-mixed reactors"""
from pyroxa import run_simulation


def test_series_network_equilibrium():
    # Two well-mixed reactors in series are represented as two specs
    spec = {
        'system': 'series',
        'reactors': [
            {
                'system': 'WellMixed',
                'reaction': {'kf': 2.0, 'kr': 1.0},
                'initial': {'temperature': 300.0, 'conc': {'A': 1.0, 'B': 0.0}}
            },
            {
                'system': 'WellMixed',
                'reaction': {'kf': 2.0, 'kr': 1.0},
                'initial': {'temperature': 300.0, 'conc': {'A': 1.0, 'B': 0.0}}
            }
        ],
        'sim': {'time_span': 5.0, 'time_step': 0.001}
    }
    times, history = run_simulation(spec)
    # history[-1] is list of reactor states; each state is [A,B]
    last = history[-1]
    # analytical equilibrium for each reactor
    kf = 2.0
    kr = 1.0
    C = 1.0
    A_eq = C * kr / (kf + kr)
    B_eq = C * kf / (kf + kr)
    # check outlet reactor (second)
    A_final, B_final = last[1]
    assert abs(A_final - A_eq) < 0.03
    assert abs(B_final - B_eq) < 0.03
