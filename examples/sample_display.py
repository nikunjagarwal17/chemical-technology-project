"""Sample script that uses pyroxa to run a simulation and print a summary.

Usage:
    python examples/sample_display.py
"""
import yaml
from pyroxa.purepy import build_from_dict

def main():
    with open('examples/sample_spec.yaml') as f:
        spec = yaml.safe_load(f)
    reactor, sim = build_from_dict(spec)
    times, traj = reactor.run(sim.get('time_span', 1.0), sim.get('time_step', 0.1))
    print(f'Steps: {len(times)}')
    print('Final concentrations:', traj[-1])

if __name__ == '__main__':
    main()
