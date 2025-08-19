"""Test 1: Well-mixed A <=> B single reactor example
Run: python -m test1
Outputs: generates examples/test1_plot.png and prints final concentrations
"""
import yaml
import matplotlib.pyplot as plt
from pyroxa import Reaction, WellMixedReactor

def run():
    with open('examples/test1_spec.yaml') as f:
        spec = yaml.safe_load(f)
    # Use builder or direct API
    kf = spec.get('reaction', {}).get('kf', 1.0)
    kr = spec.get('reaction', {}).get('kr', 0.0)
    rxn = Reaction(kf=kf, kr=kr)
    A0 = spec.get('initial', {}).get('conc', {}).get('A', 1.0)
    B0 = spec.get('initial', {}).get('conc', {}).get('B', 0.0)
    r = WellMixedReactor(rxn, A0=A0, B0=B0)
    ts = spec.get('sim', {}).get('time_span', [0.0, 5.0])
    if isinstance(ts, (int, float)):
        t_span = [0.0, float(ts)]
    else:
        t_span = ts
    dt = spec.get('sim', {}).get('time_step', 0.1)
    duration = float(t_span[1] - t_span[0])
    times, traj = r.run(duration, dt)
    A = [p[0] for p in traj]
    B = [p[1] for p in traj]
    plt.figure()
    plt.plot(times, A, label='A')
    plt.plot(times, B, label='B')
    plt.xlabel('time')
    plt.ylabel('concentration')
    plt.legend()
    plt.title('Well-mixed A <=> B')
    out = 'examples/test1_plot.png'
    plt.savefig(out)
    print('Saved plot to', out)
    print('Final concentrations:', traj[-1])

if __name__ == '__main__':
    run()
