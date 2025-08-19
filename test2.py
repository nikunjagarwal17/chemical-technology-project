"""Test 2: PFR plug-flow example driven by simple spec
Run: python -m test2
Outputs: examples/test2_plot.png and prints final concentrations
"""
import yaml
import numpy as np
import matplotlib.pyplot as plt
from simplecantera import CSTR, PFR


def run():
    with open('examples/test2_spec.yaml') as f:
        spec = yaml.safe_load(f)
    # Build a simple PFR by integrating plug-flow of conversion
    A0 = spec.get('initial', {}).get('conc', {}).get('A', 1.0)
    B0 = spec.get('initial', {}).get('conc', {}).get('B', 0.0)
    # reaction kf may be under reaction or first reactions entry
    if 'reaction' in spec:
        kf = spec.get('reaction', {}).get('kf', 1.0)
    else:
        rlist = spec.get('reactions', [])
        kf = rlist[0].get('kf', 1.0) if rlist else 1.0
    dx = 0.1
    L = spec.get('sim', {}).get('length', 1.0)
    n = int(L/dx)+1
    x = np.linspace(0, L, n)
    A = np.zeros(n)
    B = np.zeros(n)
    A[0] = A0
    B[0] = B0
    for i in range(1, n):
        r = kf * A[i-1]
        A[i] = A[i-1] - r*dx
        B[i] = B[i-1] + r*dx
    plt.figure()
    plt.plot(x, A, label='A')
    plt.plot(x, B, label='B')
    plt.xlabel('position')
    plt.ylabel('concentration')
    plt.legend()
    out = 'examples/test2_plot.png'
    plt.savefig(out)
    print('Saved plot to', out)
    print('Final concentrations at exit:', (A[-1], B[-1]))

if __name__ == '__main__':
    run()
