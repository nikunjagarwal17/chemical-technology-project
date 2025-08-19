"""Run a small example simulation and save results to CSV."""
from pyroxa import run_simulation
import os

spec = {
    'reaction': {'kf': 1.0, 'kr': 0.5},
    'initial': {'temperature': 300.0, 'conc': {'A': 1.0, 'B': 0.0}},
    'sim': {'time_span': 10.0, 'time_step': 0.01}
}

out = os.path.join(os.path.dirname(__file__), 'results.csv')
print('Running simulation, output:', out)
run_simulation(spec, csv_out=out, plot=False)
print('Done')
