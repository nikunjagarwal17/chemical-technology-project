"""Test 3: Multi-step simulation using pure-Python ReactorNetwork
Run: python -m test3
Outputs: examples/test3_plot.png and prints final concentrations
"""
import yaml
import matplotlib.pyplot as plt
from simplecantera import Reaction, WellMixedReactor, ReactorNetwork


def run():
    with open('examples/test3_spec.yaml') as f:
        spec = yaml.safe_load(f)
    kf = spec.get('reaction', {}).get('kf', 1.0)
    kr = spec.get('reaction', {}).get('kr', 0.0)
    rxn = Reaction(kf=kf, kr=kr)
    A0 = spec.get('initial', {}).get('conc', {}).get('A', 1.0)
    B0 = spec.get('initial', {}).get('conc', {}).get('B', 0.0)
    r1 = WellMixedReactor(rxn, A0=A0, B0=B0)
    r2 = WellMixedReactor(rxn, A0=0.0, B0=0.0)
    net = ReactorNetwork([r1, r2])
    ts = spec.get('sim', {}).get('time_span', [0.0, 10.0])
    if isinstance(ts, (int, float)):
        t_span = [0.0, float(ts)]
    else:
        t_span = ts
    dt = spec.get('sim', {}).get('time_step', 0.1)
    duration = float(t_span[1] - t_span[0])
    times, traj = net.run(duration, dt)
    # traj may be network-like (traj[t][reactor_index] == [A,B]) or single-reactor ([A,B] per time)
    plt.figure()
    if len(traj) > 0 and isinstance(traj[0][0], (list, tuple)):
        # network history: plot A and B for each reactor
        nreact = len(traj[0])
        for i in range(nreact):
            Ai = [row[i][0] for row in traj]
            Bi = [row[i][1] for row in traj]
            plt.plot(times, Ai, label=f'A_r{i}')
            plt.plot(times, Bi, label=f'B_r{i}', linestyle='--')
        plt.xlabel('time')
        plt.ylabel('concentration')
        plt.legend()
        out = 'examples/test3_plot.png'
        plt.savefig(out)
        print('Saved plot to', out)
        # Print final concentrations per reactor
        final = traj[-1]
        for i, state in enumerate(final):
            print(f'Final concentrations (reactor {i}):', state)
    else:
        A = [p[0] for p in traj]
        B = [p[1] for p in traj]
        plt.plot(times, A, label='A_total')
        plt.plot(times, B, label='B_total')
        plt.xlabel('time')
        plt.ylabel('concentration')
        plt.legend()
        out = 'examples/test3_plot.png'
        plt.savefig(out)
        print('Saved plot to', out)
        print('Final concentrations (reactor1):', traj[-1])

if __name__ == '__main__':
    run()
