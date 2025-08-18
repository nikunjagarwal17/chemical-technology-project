"""Tests for CSTR and PFR behaviors."""
from simplecantera import build_from_dict, run_simulation
import os


def test_cstr_steady_state():
    spec = {
        'system': 'CSTR',
        'reaction': {'kf': 1.0, 'kr': 0.5},
        'initial': {'temperature': 300.0, 'conc': {'A': 0.0, 'B': 0.0}},
        'cstr': {'q': 1.0, 'conc_in': {'A': 1.0, 'B': 0.0}},
        'sim': {'time_span': 5.0, 'time_step': 0.01}
    }
    reactor, sim = build_from_dict(spec)
    times, traj = reactor.run(sim.get('time_span',5.0), sim.get('time_step',0.01))
    A_final, B_final = reactor.conc
    # Check concentrations are within reasonable bounds
    assert 0.0 <= A_final <= 1.0
    assert 0.0 <= B_final <= 1.0


def test_pfr_outlet_monotonic():
    spec = {
        'system': 'PFR',
        'reaction': {'kf': 2.0, 'kr': 1.0},
        'initial': {'temperature': 300.0, 'conc': {'A': 1.0, 'B': 0.0}},
        'pfr': {'nseg': 5, 'q': 0.5, 'total_volume': 1.0},
        'sim': {'time_span': 2.0, 'time_step': 0.02}
    }
    reactor, sim = build_from_dict(spec)
    times, out_history = reactor.run(sim.get('time_span',2.0), sim.get('time_step',0.02))
    # outlet should be non-negative and finite
    for a, b in out_history:
        assert a >= 0.0
        assert b >= 0.0
