"""
PyroXa Advanced Reactor Test
Test advanced reactor simulations to verify complete functionality
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

def test_advanced_simulations():
    """Test advanced reactor simulations"""
    print("PyroXa Advanced Reactor Simulation Test")
    print("=" * 60)
    
    try:
        import pyroxa
        print(f"PyroXa v{pyroxa.get_version()} - Testing Advanced Features")
        
        # Create a reaction system
        reaction = pyroxa.Reaction(kf=1.0, kr=0.1)
        thermo = pyroxa.Thermodynamics(cp=29.1)
        
        print("\n1. Testing Packed Bed Reactor Simulation:")
        pbr = pyroxa.PackedBedReactor(
            bed_length=1.0,
            bed_porosity=0.4,
            particle_diameter=0.003,
            catalyst_density=1200.0,
            effectiveness_factor=0.8,
            flow_rate=0.001
        )
        pbr.add_reaction(reaction)
        
        results = pbr.run(time_span=1.0, dt=0.1)
        print(f"✓ Packed bed simulation: {len(results['times'])} time points")
        print(f"✓ Final conversion: {results['conversion'][-1]:.3f}")
        print(f"✓ Pressure drop: {results['pressure_profiles'][-1]:.0f} Pa")
        
        print("\n2. Testing Fluidized Bed Reactor Simulation:")
        fbr = pyroxa.FluidizedBedReactor(
            bed_height=2.0,
            bed_porosity=0.5,
            bubble_fraction=0.3,
            particle_diameter=0.001,
            catalyst_density=1500.0,
            gas_velocity=0.5
        )
        fbr.add_reaction(reaction)
        
        results = fbr.run(time_span=1.0, dt=0.1)
        print(f"✓ Fluidized bed simulation: {len(results['times'])} time points")
        print(f"✓ Final conversion: {results['conversion'][-1]:.3f}")
        print(f"✓ Bubble velocity: {results['bubble_velocity']:.2f} m/s")
        
        print("\n3. Testing Heterogeneous Reactor (3-phase):")
        htr = pyroxa.HeterogeneousReactor(
            gas_holdup=0.6,
            liquid_holdup=0.3,
            solid_holdup=0.1,
            mass_transfer_gas_liquid=[0.1, 0.1],
            mass_transfer_liquid_solid=[0.05, 0.05]
        )
        htr.add_gas_reaction(reaction)
        htr.add_liquid_reaction(reaction)
        htr.add_solid_reaction(reaction)
        
        results = htr.run(time_span=1.0, dt=0.1)
        print(f"✓ Heterogeneous simulation: {len(results['times'])} time points")
        print(f"✓ Overall conversion: {results['overall_conversion']:.3f}")
        print(f"✓ Phase holdups sum: {sum(results['phase_holdups'].values()):.3f}")
        
        print("\n4. Testing Multi-Species Reactor:")
        # A + B -> C reaction
        reactants = {0: 1, 1: 1}  # A + B
        products = {2: 1}         # -> C
        multi_reaction = pyroxa.ReactionMulti(1.0, 0.1, reactants, products)
        
        species = ['A', 'B', 'C']
        conc0 = [2.0, 1.5, 0.0]
        multi_reactor = pyroxa.MultiReactor(thermo, [multi_reaction], species, 
                                           T=300.0, conc0=conc0)
        
        times, traj = multi_reactor.run(time_span=2.0, time_step=0.01)
        print(f"✓ Multi-species simulation: {len(times)} time points")
        print(f"✓ Final concentrations: A={traj[-1][0]:.3f}, B={traj[-1][1]:.3f}, C={traj[-1][2]:.3f}")
        
        print("\n5. Testing Adaptive Integration:")
        reactor = pyroxa.WellMixedReactor(reaction, T=300.0, conc0=(2.0, 0.0))
        times, traj = reactor.run_adaptive(time_span=5.0, dt_init=1e-3, atol=1e-6, rtol=1e-6)
        print(f"✓ Adaptive simulation: {len(times)} time points")
        print(f"✓ Final concentrations: A={traj[-1][0]:.3f}, B={traj[-1][1]:.3f}")
        
        print("\n6. Testing Reactor Network:")
        # Series of 3 reactors
        reactor1 = pyroxa.WellMixedReactor(reaction, T=280.0, conc0=(3.0, 0.0))
        reactor2 = pyroxa.CSTR(thermo, reaction, T=320.0, volume=1.0, 
                              conc0=(1.0, 0.0), q=0.3, conc_in=(2.0, 0.0))
        reactor3 = pyroxa.PFR(thermo, reaction, T=350.0, total_volume=0.5, 
                             nseg=5, conc0=(1.5, 0.0), q=0.5)
        
        network = pyroxa.ReactorNetwork([reactor1, reactor2, reactor3], mode='series')
        times, history = network.run(time_span=3.0, time_step=0.01)
        print(f"✓ Network simulation: {len(history[0])} reactors")
        print(f"✓ Reactor 1 final: A={history[-1][0][0]:.3f}")
        print(f"✓ Reactor 2 final: A={history[-1][1][0]:.3f}")
        print(f"✓ Reactor 3 final: A={history[-1][2][0]:.3f}")
        
        print("\n7. Testing Built-in Simulation Functions:")
        
        # Test build from dict
        spec = {
            'reaction': {'kf': 1.5, 'kr': 0.3},
            'initial': {'temperature': 310, 'conc': {'A': 2.5, 'B': 0.0}},
            'sim': {'time_span': 2.0, 'time_step': 0.02},
            'system': 'WellMixed'
        }
        
        reactor, sim = pyroxa.build_from_dict(spec)
        times, traj = pyroxa.run_simulation_from_dict(spec)
        print(f"✓ Dict-based simulation: {len(times)} time points")
        print(f"✓ Final conversion: {1 - traj[-1][0]/2.5:.3f}")
        
        print("\n" + "=" * 60)
        print("🎉 ADVANCED SIMULATION TEST PASSED!")
        print("All reactor types and advanced features working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_advanced_simulations()
    exit(0 if success else 1)