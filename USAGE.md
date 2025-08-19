Usage guide — Pyroxa (local/pure-Python)

Purpose
-------
This page explains how to use the project locally using the pure-Python fallback (no compiled extension required).

Install
-------
```bash
python -m pip install -r requirements.txt
```

Quick examples
--------------
- Import and run a simple well-mixed reactor::

```python
from pyroxa import Reaction, WellMixedReactor
rxn = Reaction(kf=1.0, kr=0.5)
r = WellMixedReactor(rxn, A0=1.0, B0=0.0)
times, traj = r.run(1.0, 0.1)
print('times:', times)
print('trajectory last:', traj[-1])
```

- Use the high-level `build_from_dict` runner for spec-driven runs:

```python
from pyroxa.purepy import build_from_dict, run_simulation_from_dict
spec = {
  'reaction': {'kf': 1.0, 'kr': 0.5},
  'initial': {'conc': {'A': 1.0, 'B': 0.0}},
  'sim': {'time_span': 1.0, 'time_step': 0.1},
  'system': 'WellMixed'
}
reactor, sim = build_from_dict(spec)
results = reactor.run(sim.get('time_span', 1.0), sim.get('time_step', 0.1))
print(results[0])
```

Plotting
--------
If `matplotlib` is installed you can plot A/B vs time after running the simulation.

When you need more speed
-----------------------
- The compiled C++ core and Cython bindings are in the repo. Use CI to produce wheels for distribution, or build locally following instructions in `DEV.md` (use a compatible Python/Cython pairing).

Files of interest
-----------------
- `pyroxa/purepy.py` — pure-Python implementation.
- `pyroxa/pybindings.pyx` — Cython binding source (used for compiled extension).
- `src/core.cpp` / `src/core.h` — C++ numerical core.

Troubleshooting
---------------
- If your build fails with C API or linker errors, check `DEV.md` for the recommended local build steps and reminders about Cython/Python compatibility.
