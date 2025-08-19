import sys
import simplecantera
print('simplecantera import OK')
print('exports:', [a for a in dir(simplecantera) if not a.startswith('_')])
from simplecantera import Reaction, WellMixedReactor
rxn = Reaction(1.0, 0.5)
r = WellMixedReactor(rxn, A0=1.0, B0=0.0, time_span=0.5, dt=0.1)
times, traj = r.run(0.5, 0.1)
print('run result len:', len(times))
print('done')
