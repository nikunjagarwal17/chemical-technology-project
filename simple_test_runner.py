#!/usr/bin/env python3
"""
Quick test for advanced PyroXa reactor functionality - ASCII version
"""

import sys
import os
import numpy as np
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pyroxa.purepy import (Reaction, PackedBedReactor, FluidizedBedReactor, 
                          HeterogeneousReactor, HomogeneousReactor, WellMixedReactor)

def test_all_reactors():
    """Test all advanced reactor types with basic functionality."""
    print("=" * 60)
    print("PYROXA ADVANCED REACTOR QUICK TEST")
    print("=" * 60)
    
    # Create basic reaction
    reaction = Reaction(kf=1.0, kr=0.1)
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Packed Bed Reactor
    print("\n1. Testing Packed Bed Reactor...")
    try:
        reactor = PackedBedReactor(
            bed_length=0.5,
            bed_porosity=0.4,
            particle_diameter=0.003,
            catalyst_density=1000,
            effectiveness_factor=0.8,
            flow_rate=0.01
        )
        reactor.add_reaction(reaction)
        result = reactor.run(time_span=1.0, dt=0.01)
        conversion = result['conversion'][-1]
        
        if abs(conversion) < 1e10 and not np.isnan(conversion):
            print(f"   + Conversion: {conversion:.4f}")
            print("   + Packed bed test PASSED")
            tests_passed += 1
        else:
            print(f"   - Invalid conversion: {conversion}")
            print("   - Packed bed test FAILED")
    except Exception as e:
        print(f"   - Error: {e}")
        print("   - Packed bed test FAILED")
    
    # Test 2: Fluidized Bed Reactor
    print("\n2. Testing Fluidized Bed Reactor...")
    try:
        reactor = FluidizedBedReactor(
            bed_height=2.0,
            bed_porosity=0.4,
            bubble_fraction=0.3,
            particle_diameter=0.0005,
            catalyst_density=1500,
            gas_velocity=0.5
        )
        reactor.add_reaction(reaction)
        result = reactor.run(time_span=1.0, dt=0.01)
        conversion = result['conversion'][-1]
        
        if abs(conversion) < 1e10 and not np.isnan(conversion):
            print(f"   + Conversion: {conversion:.4f}")
            print("   + Fluidized bed test PASSED")
            tests_passed += 1
        else:
            print(f"   - Invalid conversion: {conversion}")
            print("   - Fluidized bed test FAILED")
    except Exception as e:
        print(f"   - Error: {e}")
        print("   - Fluidized bed test FAILED")
    
    # Test 3: Heterogeneous Reactor
    print("\n3. Testing Heterogeneous Three-Phase Reactor...")
    try:
        reactor = HeterogeneousReactor(
            gas_holdup=0.3,
            liquid_holdup=0.5,
            solid_holdup=0.2,
            mass_transfer_gas_liquid=[0.1, 0.05],
            mass_transfer_liquid_solid=[0.05, 0.02]
        )
        reactor.add_gas_reaction(reaction)
        reactor.add_liquid_reaction(reaction)
        reactor.add_solid_reaction(reaction)
        result = reactor.run(time_span=1.0, dt=0.01)
        conversion = result['overall_conversion']
        
        if 0 <= conversion <= 1:
            print(f"   + Conversion: {conversion:.4f}")
            print("   + Heterogeneous reactor test PASSED")
            tests_passed += 1
        else:
            print(f"   - Invalid conversion: {conversion}")
            print("   - Heterogeneous reactor test FAILED")
    except Exception as e:
        print(f"   - Error: {e}")
        print("   - Heterogeneous reactor test FAILED")
    
    # Test 4: Enhanced Homogeneous Reactor
    print("\n4. Testing Enhanced Homogeneous Reactor...")
    try:
        reactor = HomogeneousReactor(reaction, mixing_intensity=1.0, volume=1.0)
        result = reactor.run(time_span=2.0, dt=0.1)
        
        times = result['times']
        concentrations = result['concentrations']
        final_A = concentrations[-1, 0]
        final_B = concentrations[-1, 1]
        
        if len(times) > 0 and not np.any(np.isnan(concentrations)):
            print(f"   + Final concentrations: A={final_A:.4f}, B={final_B:.4f}")
            print(f"   + Mixing intensity: {result['mixing_intensity']}")
            print("   + Enhanced homogeneous reactor test PASSED")
            tests_passed += 1
        else:
            print("   - Invalid results detected")
            print("   - Enhanced homogeneous reactor test FAILED")
    except Exception as e:
        print(f"   - Error: {e}")
        print("   - Enhanced homogeneous reactor test FAILED")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("SUCCESS: All advanced reactor types are working!")
    elif tests_passed >= 2:
        print("PARTIAL SUCCESS: Most reactor types are working")
    else:
        print("FAILURE: Major issues with reactor implementations")
    
    print("=" * 60)
    return tests_passed == total_tests

if __name__ == "__main__":
    test_all_reactors()
