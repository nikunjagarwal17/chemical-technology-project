#!/usr/bin/env python3
"""Simple test for advanced reactor functionality."""

import sys
import os
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from pyroxa.purepy import (Reaction, PackedBedReactor, FluidizedBedReactor, 
                          HeterogeneousReactor, HomogeneousReactor)

def test_packed_bed():
    """Test packed bed reactor with basic parameters."""
    print("Testing Packed Bed Reactor...")
    
    try:
        # Create reactor with reasonable parameters
        reactor = PackedBedReactor(
            bed_length=0.5,  # 50 cm
            bed_porosity=0.4,
            particle_diameter=0.003,  # 3 mm
            catalyst_density=1000,  # kg/m3
            effectiveness_factor=0.8,
            flow_rate=0.01  # 0.01 m3/s
        )
        
        # Add simple reaction A -> B
        reaction = Reaction(kf=0.1, kr=0.01)
        reactor.add_reaction(reaction)
        
        # Run with small time step
        result = reactor.run(time_span=1.0, dt=0.01)
        
        # Check for reasonable values
        final_conversion = result['conversion'][-1]
        print(f"  Final conversion: {final_conversion:.4f}")
        
        if abs(final_conversion) > 1e10:
            print("  ❌ Conversion values too large - numerical instability")
            return False
        elif final_conversion < 0:
            print("  ❌ Negative conversion detected")
            return False
        else:
            print("  ✓ Packed bed test passed")
            return True
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_homogeneous():
    """Test enhanced homogeneous reactor."""
    print("Testing Enhanced Homogeneous Reactor...")
    
    try:
        # Create reaction and reactor
        reaction = Reaction(kf=1.0, kr=0.1)
        reactor = HomogeneousReactor(reaction, mixing_intensity=1.0, volume=1.0)
        
        # Run simulation
        result = reactor.run(time_span=2.0, dt=0.1)
        
        # Check results
        times = result['times']
        concentrations = result['concentrations']
        
        print(f"  Simulation completed with {len(times)} time points")
        print(f"  Final concentrations: A={concentrations[-1, 0]:.4f}, B={concentrations[-1, 1]:.4f}")
        print(f"  Mixing intensity: {result['mixing_intensity']}")
        
        # Validate
        if len(times) > 0 and not np.any(np.isnan(concentrations)):
            print("  ✓ Homogeneous reactor test passed")
            return True
        else:
            print("  ❌ Invalid results detected")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    print("=" * 50)
    print("Simple Advanced Reactor Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 2
    
    if test_packed_bed():
        tests_passed += 1
        
    if test_homogeneous():
        tests_passed += 1
    
    print("=" * 50)
    print(f"Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")

if __name__ == "__main__":
    main()
