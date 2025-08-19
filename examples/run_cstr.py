"""Run a small CSTR example and save results to CSV."""
from pyroxa import run_simulation
import os

spec = {
    'system': 'CSTR',
    'reaction': {'kf': 2.0, 'kr': 0.1},
    'initial': {'temperature': 300.0, 'conc': {'A': 1.0, 'B': 0.0}},
    'cstr': {'q': 0.5, 'conc_in': {'A': 1.0, 'B': 0.0}},
    'sim': {'time_span': 5.0, 'time_step': 0.01}
}

out = os.path.join(os.path.dirname(__file__), 'cstr_results.csv')
print('Running CSTR simulation, output:', out)
run_simulation(spec, csv_out=out, plot=False)
print('Done')
