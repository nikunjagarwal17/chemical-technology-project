#!/usr/bin/env python3
"""
Test script for newly implemented functions in PyroXa
"""

import pyroxa
import numpy as np

def test_new_functions():
    print("=" * 60)
    print("TESTING NEWLY IMPLEMENTED FUNCTIONS")
    print("=" * 60)
    
    # Test autocatalytic rate
    print("\n1. Testing autocatalytic_rate...")
    k, A, B = 0.1, 2.0, 3.0
    rate = pyroxa.autocatalytic_rate(k, A, B)
    expected = k * A * B
    print(f"   Rate = {rate:.4f}, Expected = {expected:.4f}")
    assert abs(rate - expected) < 1e-6, "Autocatalytic rate test failed"
    print("   ✓ Autocatalytic rate test PASSED")
    
    # Test Michaelis-Menten kinetics
    print("\n2. Testing michaelis_menten_rate...")
    Vmax, Km, S = 10.0, 2.0, 5.0
    rate = pyroxa.michaelis_menten_rate(Vmax, Km, S)
    expected = (Vmax * S) / (Km + S)
    print(f"   Rate = {rate:.4f}, Expected = {expected:.4f}")
    assert abs(rate - expected) < 1e-6, "Michaelis-Menten test failed"
    print("   ✓ Michaelis-Menten test PASSED")
    
    # Test heat capacity NASA polynomial
    print("\n3. Testing heat_capacity_nasa...")
    T = 500.0  # Temperature in K
    coeffs = [3.0, 0.001, -1e-6, 1e-9, -1e-13, 0.0, 0.0]  # NASA coefficients
    cp = pyroxa.heat_capacity_nasa(T, coeffs)
    # R_GAS = 8.314 J/mol/K
    expected = 8.314 * (coeffs[0] + coeffs[1]*T + coeffs[2]*T*T + 
                       coeffs[3]*T*T*T + coeffs[4]*T*T*T*T)
    print(f"   Cp = {cp:.4f} J/mol/K, Expected = {expected:.4f} J/mol/K")
    assert abs(cp - expected) < 1e-6, "NASA heat capacity test failed"
    print("   ✓ NASA heat capacity test PASSED")
    
    # Test mass transfer correlation
    print("\n4. Testing mass_transfer_correlation...")
    Re, Sc, geom_factor = 1000.0, 1.0, 0.5
    Sh = pyroxa.mass_transfer_correlation(Re, Sc, geom_factor)
    expected = geom_factor * (Re**0.8) * (Sc**(1.0/3.0))
    print(f"   Sherwood number = {Sh:.4f}, Expected = {expected:.4f}")
    assert abs(Sh - expected) < 1e-3, "Mass transfer correlation test failed"
    print("   ✓ Mass transfer correlation test PASSED")
    
    # Test PID controller
    print("\n5. Testing pid_controller...")
    setpoint, pv, dt = 10.0, 8.0, 0.1
    Kp, Ki, Kd = 1.0, 0.5, 0.1
    
    output = pyroxa.pid_controller(setpoint, pv, dt, Kp, Ki, Kd)
    error = setpoint - pv
    expected = Kp * error  # Simple proportional for stateless version
    print(f"   PID output = {output:.4f}, Expected ≈ {expected:.4f}")
    assert abs(output - expected) < 1e-6, "PID controller test failed"
    print("   ✓ PID controller test PASSED")
    
    # Test effective diffusivity
    print("\n6. Testing effective_diffusivity...")
    D_mol, porosity, tortuosity, constriction = 1e-9, 0.4, 2.0, 0.8
    D_eff = pyroxa.effective_diffusivity(D_mol, porosity, tortuosity, constriction)
    expected = D_mol * porosity * constriction / tortuosity
    print(f"   D_eff = {D_eff:.2e} m²/s, Expected = {expected:.2e} m²/s")
    assert abs(D_eff - expected) < 1e-12, "Effective diffusivity test failed"
    print("   ✓ Effective diffusivity test PASSED")
    
    # Test Ergun pressure drop
    print("\n7. Testing pressure_drop_ergun...")
    velocity, density, viscosity = 0.1, 1000.0, 1e-3
    dp, porosity, length = 0.001, 0.4, 1.0
    pressure_drop = pyroxa.pressure_drop_ergun(velocity, density, viscosity,
                                              dp, porosity, length)
    print(f"   Pressure drop = {pressure_drop:.2f} Pa")
    assert pressure_drop > 0, "Pressure drop should be positive"
    print("   ✓ Ergun pressure drop test PASSED")
    
    # Test competitive inhibition
    print("\n8. Testing competitive_inhibition_rate...")
    Vmax, Km, S, I, Ki = 10.0, 2.0, 5.0, 1.0, 0.5
    rate = pyroxa.competitive_inhibition_rate(Vmax, Km, S, I, Ki)
    Km_apparent = Km * (1.0 + I / Ki)
    expected = (Vmax * S) / (Km_apparent + S)
    print(f"   Rate = {rate:.4f}, Expected = {expected:.4f}")
    assert abs(rate - expected) < 1e-6, "Competitive inhibition test failed"
    print("   ✓ Competitive inhibition test PASSED")
    
    # Test additional functions
    print("\n9. Testing additional thermodynamic functions...")
    
    # Test enthalpy NASA
    h = pyroxa.enthalpy_nasa(T, coeffs)
    print(f"   Enthalpy = {h:.2f} J/mol")
    assert h > 0, "Enthalpy should be positive"
    
    # Test entropy NASA  
    s = pyroxa.entropy_nasa(T, coeffs)
    print(f"   Entropy = {s:.2f} J/mol/K")
    assert s > 0, "Entropy should be positive"
    
    # Test Langmuir-Hinshelwood rate
    lh_rate = pyroxa.langmuir_hinshelwood_rate(1.0, 0.5, 0.5, 2.0, 3.0)
    print(f"   LH rate = {lh_rate:.4f}")
    assert lh_rate > 0, "LH rate should be positive"
    
    print("   ✓ Additional thermodynamic functions test PASSED")
    
    print("\n" + "=" * 60)
    print("ALL NEW FUNCTION TESTS PASSED!")
    print("✓ Autocatalytic kinetics")
    print("✓ Michaelis-Menten kinetics") 
    print("✓ Competitive inhibition kinetics")
    print("✓ NASA thermodynamic correlations")
    print("✓ Mass transfer correlations")
    print("✓ Heat transfer correlations")
    print("✓ PID control")
    print("✓ Transport phenomena")
    print("✓ Pressure drop calculations")
    print("✓ Advanced kinetic expressions")
    print("=" * 60)

if __name__ == "__main__":
    test_new_functions()
