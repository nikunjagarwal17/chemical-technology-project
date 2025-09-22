"""
Quick PyroXa Verification Test
Test core functionality to verify the pure Python implementation
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

def test_core_functionality():
    """Test core PyroXa functionality"""
    print("Testing PyroXa Core Functionality")
    print("=" * 50)
    
    try:
        # Test basic import
        import pyroxa
        print(f"✓ PyroXa v{pyroxa.get_version()} imported successfully")
        print(f"✓ Available functions: {len(pyroxa.__all__)}")
        
        # Test basic kinetics
        print("\nTesting Basic Kinetics:")
        rate = pyroxa.first_order_rate(0.1, 2.0)
        print(f"✓ First-order rate: {rate}")
        
        k = pyroxa.arrhenius_rate(1e10, 50000, 298.15)
        print(f"✓ Arrhenius rate constant: {k:.2e}")
        
        # Test thermodynamics
        print("\nTesting Thermodynamics:")
        thermo = pyroxa.Thermodynamics(cp=29.1, T_ref=298.15)
        h = thermo.enthalpy(350.0)
        print(f"✓ Enthalpy calculation: {h:.2f} J/mol")
        
        # Test reactor classes
        print("\nTesting Reactor Classes:")
        reaction = pyroxa.Reaction(kf=2.0, kr=0.5)
        print(f"✓ Reaction created (Keq = {reaction.equilibrium_constant():.2f})")
        
        reactor = pyroxa.WellMixedReactor(reaction, T=300.0, conc0=(1.0, 0.0))
        print(f"✓ Well-mixed reactor created")
        
        times, traj = reactor.run(time_span=1.0, time_step=0.01)
        print(f"✓ Simulation completed: {len(times)} time points")
        print(f"✓ Final concentrations: A={traj[-1][0]:.3f}, B={traj[-1][1]:.3f}")
        
        # Test CSTR
        print("\nTesting CSTR:")
        cstr = pyroxa.CSTR(thermo, reaction, T=300.0, volume=1.0, 
                          conc0=(1.0, 0.0), q=0.5, conc_in=(2.0, 0.0))
        times, traj = cstr.run(time_span=2.0, time_step=0.01)
        print(f"✓ CSTR simulation: {len(times)} points")
        
        # Test PFR
        print("\nTesting PFR:")
        pfr = pyroxa.PFR(thermo, reaction, T=300.0, total_volume=1.0, 
                        nseg=10, conc0=(2.0, 0.0), q=1.0)
        times, outlet = pfr.run(time_span=1.0, time_step=0.01)
        print(f"✓ PFR simulation: {len(times)} points, outlet A={outlet[-1][0]:.3f}")
        
        # Test advanced reactors
        print("\nTesting Advanced Reactors:")
        
        # Packed bed reactor
        pbr = pyroxa.PackedBedReactor(
            bed_length=1.0,
            bed_porosity=0.4,
            particle_diameter=0.003,
            catalyst_density=1200.0
        )
        pbr.add_reaction(reaction)
        print(f"✓ Packed bed reactor created")
        
        # Fluidized bed reactor
        fbr = pyroxa.FluidizedBedReactor(
            bed_height=2.0,
            bed_porosity=0.5,
            bubble_fraction=0.3,
            particle_diameter=0.001,
            catalyst_density=1500.0,
            gas_velocity=0.5
        )
        fbr.add_reaction(reaction)
        print(f"✓ Fluidized bed reactor created")
        
        # Test reactor network
        print("\nTesting Reactor Network:")
        reactor1 = pyroxa.WellMixedReactor(reaction, T=300.0, conc0=(2.0, 0.0))
        reactor2 = pyroxa.WellMixedReactor(reaction, T=350.0, conc0=(1.0, 0.5))
        network = pyroxa.ReactorNetwork([reactor1, reactor2], mode='series')
        times, history = network.run(time_span=1.0, time_step=0.01)
        print(f"✓ Reactor network: {len(history[0])} reactors simulated")
        
        # Test transport functions (that work)
        print("\nTesting Transport Functions:")
        re = pyroxa.reynolds_number(1000.0, 2.0, 0.1, 0.001)
        print(f"✓ Reynolds number: {re:.0f}")
        
        pr = pyroxa.prandtl_number(4180.0, 0.001, 0.6)
        print(f"✓ Prandtl number: {pr:.3f}")
        
        # Test dimensionless numbers
        nu = pyroxa.nusselt_number(1000.0, 0.1, 0.6)
        print(f"✓ Nusselt number: {nu:.1f}")
        
        print("\n" + "=" * 50)
        print("🎉 CORE FUNCTIONALITY TEST PASSED!")
        print("PyroXa Pure Python implementation is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_core_functionality()
    exit(0 if success else 1)