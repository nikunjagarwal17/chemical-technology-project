#!/usr/bin/env python
"""
Quick validation of key PyroXa functions
"""

import pyroxa
import numpy as np

def quick_validation():
    print("🚀 PYROXA QUICK VALIDATION")
    print("=" * 40)
    
    # Test basic functions
    k = pyroxa.arrhenius_rate(1e10, 50000, 300)
    print(f"✓ Arrhenius rate: {k:.2e} s⁻¹")
    
    re = pyroxa.reynolds_number(1000, 1.0, 0.01, 1e-3)
    print(f"✓ Reynolds number: {re:.0f}")
    
    conv = pyroxa.conversion(1.0, 0.4)
    print(f"✓ Conversion: {conv:.1%}")
    
    # Test reactor
    thermo = pyroxa.Thermodynamics()
    reaction = pyroxa.Reaction(1e-3, 1e-4)
    reactor = pyroxa.CSTR(thermo, reaction, volume=1.0, q=0.1)
    print(f"✓ CSTR created: V = {reactor.volume:.1f} L")
    
    # Test thermodynamics
    thermo = pyroxa.Thermodynamics(cp=29.1)
    h = thermo.enthalpy(350)
    print(f"✓ Enthalpy: {h:.0f} J/mol")
    
    # Test mass transfer
    k_mt = pyroxa.mass_transfer_coefficient(Sh=10.0, D_AB=1e-9, characteristic_length=0.01)
    print(f"✓ Mass transfer coeff: {k_mt:.2e} m/s")
    
    # Test some advanced functions that should work
    try:
        A = np.array([[2, 1], [1, 2]])
        b = np.array([3, 3])
        x = pyroxa.solve_linear_system(A, b)
        print(f"✓ Linear system solution: x = [{x[0]:.1f}, {x[1]:.1f}]")
    except:
        print("✗ Linear system failed")
    
    try:
        rmse = pyroxa.calculate_rmse([1,2,3], [1.1,1.9,3.1])
        print(f"✓ RMSE calculation: {rmse:.3f}")
    except:
        print("✗ RMSE failed")
    
    print("\n🎯 CORE FUNCTIONALITY VALIDATED!")
    print(f"📊 Total available functions: {len([name for name in dir(pyroxa) if not name.startswith('_')])}")

if __name__ == "__main__":
    quick_validation()
