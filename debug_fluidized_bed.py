#!/usr/bin/env python3
"""Debug fluidized bed reactor conversion issue."""

import sys
import os
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pyroxa.purepy import Reaction, FluidizedBedReactor

def debug_fluidized_bed():
    """Debug the fluidized bed reactor conversion calculation."""
    print("Debugging Fluidized Bed Reactor Conversion...")
    
    # Create reaction A → B (kf=1.0, kr=0.1)
    reaction = Reaction(kf=1.0, kr=0.1)
    
    # Create reactor
    reactor = FluidizedBedReactor(
        bed_height=2.0,
        bed_porosity=0.4,
        bubble_fraction=0.3,
        particle_diameter=0.0005,
        catalyst_density=1500,
        gas_velocity=0.5
    )
    reactor.add_reaction(reaction)
    
    print(f"Initial bubble concentrations: {reactor.conc_bubble}")
    print(f"Initial emulsion concentrations: {reactor.conc_emulsion}")
    
    # Run simulation
    result = reactor.run(time_span=1.0, dt=0.1)
    
    print("\nResults:")
    times = result['times']
    bubble_conc = result['bubble_concentrations']
    emulsion_conc = result['emulsion_concentrations']
    overall_conc = result['overall_concentrations']
    conversion = result['conversion']
    
    print(f"Initial overall A concentration: {overall_conc[0, 0]:.4f}")
    print(f"Final overall A concentration: {overall_conc[-1, 0]:.4f}")
    print(f"Final overall B concentration: {overall_conc[-1, 1]:.4f}")
    print(f"Conversion calculation: 1 - {overall_conc[-1, 0]:.4f} / {reactor.conc_bubble[0]:.4f} = {conversion[-1]:.4f}")
    
    # Check if overall concentration is increasing
    if overall_conc[-1, 0] > reactor.conc_bubble[0]:
        print(f"\nPROBLEM: Final A concentration ({overall_conc[-1, 0]:.4f}) > Initial A concentration ({reactor.conc_bubble[0]:.4f})")
        print("This suggests A is being produced instead of consumed!")
    
    # Check mass balance
    initial_total = overall_conc[0, 0] + overall_conc[0, 1]
    final_total = overall_conc[-1, 0] + overall_conc[-1, 1]
    print(f"\nMass balance check:")
    print(f"Initial total: {initial_total:.4f}")
    print(f"Final total: {final_total:.4f}")
    print(f"Mass balance error: {abs(final_total - initial_total):.4f}")
    
    # Show time evolution
    print(f"\nTime evolution (first 5 points):")
    for i in range(min(5, len(times))):
        print(f"t={times[i]:.1f}: A={overall_conc[i, 0]:.4f}, B={overall_conc[i, 1]:.4f}, Conv={conversion[i]:.4f}")

if __name__ == "__main__":
    debug_fluidized_bed()
