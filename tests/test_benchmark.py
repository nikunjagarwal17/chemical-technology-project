import pytest
from pyroxa.purepy import Thermodynamics, ReactionMulti, MultiReactor, benchmark_multi_reactor


def test_benchmark_small():
    species = ['A', 'B', 'C']
    # simple two-step chain: A -> B -> C
    rxns = [
        {'kf': 10.0, 'kr': 0.0, 'reactants': {'A': 1}, 'products': {'B': 1}},
        {'kf': 5.0, 'kr': 0.0, 'reactants': {'B': 1}, 'products': {'C': 1}},
    ]
    rxn_objs = []
    for r in rxns:
        react_idx = {0: 1} if 'A' in r.get('reactants', {}) else {1: 1}
        prod_idx = {1: 1} if 'B' in r.get('products', {}) else {2: 1}
        rxn_objs.append(ReactionMulti(r['kf'], r['kr'], react_idx, prod_idx))
    thermo = Thermodynamics()
    reactor = MultiReactor(thermo, rxn_objs, species, conc0=[1.0, 0.0, 0.0])
    results = benchmark_multi_reactor(reactor, time_span=0.1, time_step=0.01)
    assert results['mean_time'] >= 0.0
    assert 'iterations' in results
