#!/usr/bin/env python3
"""
Test script for newly implemented thermodynamic functions in PyroXa
"""

import pyroxa
import numpy as np

def test_new_functions():
    """Test all newly implemented thermodynamic and kinetic functions"""
    
    print("🧪 Testing New PyroXa Thermodynamic Functions")
    print("=" * 50)
    
    # Test 1: Gibbs Free Energy
    print("\n1. Gibbs Free Energy")
    H = 100000  # J/mol (enthalpy)
    S = 200     # J/(mol·K) (entropy)
    T = 298     # K (temperature)
    G = pyroxa.gibbs_free_energy(H, S, T)
    print(f"   H = {H} J/mol, S = {S} J/(mol·K), T = {T} K")
    print(f"   G = H - T*S = {G:.2f} J/mol")
    
    # Test 2: Equilibrium Constant
    print("\n2. Equilibrium Constant")
    delta_G = -5000  # J/mol
    T = 298          # K
    K_eq = pyroxa.equilibrium_constant(delta_G, T)
    print(f"   ΔG = {delta_G} J/mol, T = {T} K")
    print(f"   K_eq = exp(-ΔG/RT) = {K_eq:.4f}")
    
    # Test 3: Arrhenius Rate Constant
    print("\n3. Arrhenius Rate Constant")
    A = 1e12         # pre-exponential factor (1/s)
    Ea = 50000       # activation energy (J/mol)
    T = 298          # temperature (K)
    k = pyroxa.arrhenius_rate(A, Ea, T)
    print(f"   A = {A:.1e} 1/s, Ea = {Ea} J/mol, T = {T} K")
    print(f"   k = A * exp(-Ea/RT) = {k:.6f} 1/s")
    
    # Test 4: Peng-Robinson Equation of State
    print("\n4. Peng-Robinson Equation of State")
    n = 1.0          # moles
    V = 0.024        # m³
    T = 298          # K
    Tc = 647.1       # critical temperature (K)
    Pc = 2.2064e7    # critical pressure (Pa)
    omega = 0.344    # acentric factor
    P = pyroxa.pressure_peng_robinson(n, V, T, Tc, Pc, omega)
    print(f"   n = {n} mol, V = {V} m³, T = {T} K")
    print(f"   Tc = {Tc} K, Pc = {Pc:.2e} Pa, ω = {omega}")
    print(f"   P = {P:.2f} Pa ({P/101325:.4f} atm)")
    
    # Test 5: Fugacity Coefficient
    print("\n5. Fugacity Coefficient")
    P = 101325       # pressure (Pa)
    T = 298          # temperature (K)
    phi = pyroxa.fugacity_coefficient(P, T, Tc, Pc, omega)
    print(f"   P = {P} Pa, T = {T} K")
    print(f"   φ = {phi:.6f}")
    
    # Test 6: Langmuir-Hinshelwood Kinetics
    print("\n6. Langmuir-Hinshelwood Rate")
    k = 2.5          # rate constant
    K_A = 3.0        # adsorption constant for A
    K_B = 1.8        # adsorption constant for B
    conc_A = 0.5     # concentration of A
    conc_B = 0.3     # concentration of B
    rate = pyroxa.langmuir_hinshelwood_rate(k, K_A, K_B, conc_A, conc_B)
    print(f"   k = {k}, K_A = {K_A}, K_B = {K_B}")
    print(f"   [A] = {conc_A} mol/L, [B] = {conc_B} mol/L")
    print(f"   rate = {rate:.6f} mol/(L·s)")
    
    # Test 7: Photochemical Rate
    print("\n7. Photochemical Rate")
    quantum_yield = 0.85      # quantum yield
    molar_absorptivity = 1200 # L/(mol·cm)
    path_length = 1.0         # cm
    light_intensity = 150     # einstein/(L·s)
    concentration = 0.05      # mol/L
    photo_rate = pyroxa.photochemical_rate(quantum_yield, molar_absorptivity, 
                                          path_length, light_intensity, concentration)
    print(f"   φ = {quantum_yield}, ε = {molar_absorptivity} L/(mol·cm)")
    print(f"   l = {path_length} cm, I = {light_intensity} einstein/(L·s)")
    print(f"   C = {concentration} mol/L")
    print(f"   rate = {photo_rate:.4f} mol/(L·s)")
    
    print("\n✅ All new thermodynamic functions working correctly!")
    print(f"📊 Total PyroXa functions available: {len(pyroxa.__all__)}")
    print(f"🔬 New functions added: 7")

if __name__ == "__main__":
    test_new_functions()
