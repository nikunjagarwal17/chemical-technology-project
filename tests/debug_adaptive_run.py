from pyroxa.purepy import Thermodynamics, Reaction, WellMixedReactor, MultiReactor, ReactionMulti

print('Single reactor debug')
thermo = Thermodynamics()
rxn = Reaction(1.0, 0.5)
reactor = WellMixedReactor(thermo, rxn, conc0=(1.0, 0.0))
t1, traj1 = reactor.run(1.0, 0.001)
print('fixed steps:', len(traj1), 'final:', traj1[-1])

# adaptive: recreate reactor to reset state
reactor2 = WellMixedReactor(thermo, rxn, conc0=(1.0, 0.0))
t2, traj2 = reactor2.run_adaptive(1.0, dt_init=0.001, atol=1e-8, rtol=1e-6)
print('adaptive steps:', len(traj2), 'final:', traj2[-1])
print('last 5 adaptive points:', traj2[-5:])

print('\nMulti reactor debug')
species = ['A','B','C']
rxns = [ReactionMulti(10.0, 0.0, {0:1}, {1:1}), ReactionMulti(5.0, 0.0, {1:1}, {2:1})]
reactor_m = MultiReactor(thermo, rxns, species, conc0=[1.0,0.0,0.0])
tm1, trajm1 = reactor_m.run(0.5, 0.0005)
print('fixed steps:', len(trajm1), 'final:', trajm1[-1])

reactor_m2 = MultiReactor(thermo, rxns, species, conc0=[1.0,0.0,0.0])
tm2, trajm2 = reactor_m2.run_adaptive(0.5, dt_init=0.001, atol=1e-8, rtol=1e-6)
print('adaptive steps:', len(trajm2), 'final:', trajm2[-1])
print('last 5 adaptive points:', trajm2[-5:])
