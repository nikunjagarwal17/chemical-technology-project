import sys
import pyroxa
print('pyroxa import OK')
print('exports:', [a for a in dir(pyroxa) if not a.startswith('_')])
from pyroxa import Reaction, WellMixedReactor
rxn = Reaction(1.0, 0.5)
r = WellMixedReactor(rxn, A0=1.0, B0=0.0, time_span=0.5, dt=0.1)
times, traj = r.run(0.5, 0.1)
print('run result len:', len(times))
print('done')
