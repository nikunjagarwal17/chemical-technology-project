"""Tests for multi-species multi-reaction reactor"""
from simplecantera import build_from_dict, run_simulation


def test_simple_two_step_series():
    # A -> B -> C (irreversible with given k)
    spec = {
        'species': ['A', 'B', 'C'],
        'reactions': [
            {'kf': 1.0, 'kr': 0.0, 'reactants': {'A': 1}, 'products': {'B': 1}},
            {'kf': 0.5, 'kr': 0.0, 'reactants': {'B': 1}, 'products': {'C': 1}},
        ],
        'initial': {'conc': {'A': 1.0, 'B': 0.0, 'C': 0.0}, 'temperature': 300.0},
        'sim': {'time_span': 5.0, 'time_step': 0.01}
    }
    reactor, sim = build_from_dict(spec)
    times, traj = reactor.run(sim.get('time_span',5.0), sim.get('time_step',0.01))
    # final concentrations should be finite and non-negative
    final = traj[-1]
    assert final[0] >= 0.0 and final[1] >= 0.0 and final[2] >= 0.0
