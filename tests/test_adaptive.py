from simplecantera.purepy import Thermodynamics, Reaction, WellMixedReactor, MultiReactor, ReactionMulti


def test_adaptive_single_close():
    thermo = Thermodynamics()
    rxn = Reaction(1.0, 0.5)
    reactor = WellMixedReactor(thermo, rxn, conc0=(1.0, 0.0))
    t1, traj1 = reactor.run(1.0, 0.001)
    # adaptive (start from same initial conditions)
    reactor2 = WellMixedReactor(thermo, rxn, conc0=(1.0, 0.0))
    t2, traj2 = reactor2.run_adaptive(1.0, dt_init=0.001, atol=1e-8, rtol=1e-6)
    # compare final concentrations
    assert abs(traj1[-1][0] - traj2[-1][0]) < 1e-3
    assert abs(traj1[-1][1] - traj2[-1][1]) < 1e-3


def test_adaptive_multi_close():
    species = ['A','B','C']
    rxns = [ReactionMulti(10.0, 0.0, {0:1}, {1:1}), ReactionMulti(5.0, 0.0, {1:1}, {2:1})]
    thermo = Thermodynamics()
    reactor = MultiReactor(thermo, rxns, species, conc0=[1.0,0.0,0.0])
    t1, traj1 = reactor.run(0.5, 0.0005)
    # adaptive (fresh reactor)
    reactor2 = MultiReactor(thermo, rxns, species, conc0=[1.0,0.0,0.0])
    t2, traj2 = reactor2.run_adaptive(0.5, dt_init=0.001, atol=1e-8, rtol=1e-6)
    # final concentration of C should be similar
    assert abs(traj1[-1][2] - traj2[-1][2]) < 1e-3
