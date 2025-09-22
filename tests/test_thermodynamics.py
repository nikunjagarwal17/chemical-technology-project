"""
Test Thermodynamic Functions
Tests for thermodynamic property calculations
"""

import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pyroxa
from tests import assert_close, assert_positive, assert_in_range

class TestThermodynamics:
    """Test suite for thermodynamic functions"""
    
    def test_heat_capacity_nasa(self):
        """Test NASA polynomial heat capacity calculation"""
        # NASA coefficients for CO2 (example)
        coeffs = [4.45362308e+00, 3.14016890e-03, -1.27841067e-06, 
                 2.39399704e-10, -1.66903437e-14]
        T = 298.15  # temperature (K)
        
        # Function signature: heat_capacity_nasa(T, coeffs)
        cp = pyroxa.heat_capacity_nasa(T, coeffs)
        
        # Check reasonable value for CO2 heat capacity
        assert_positive(cp)
        assert_in_range(cp, 30.0, 60.0)  # J/mol/K
    
    def test_enthalpy_nasa(self):
        """Test NASA polynomial enthalpy calculation"""
        # NASA coefficients for CO2 (example)
        coeffs = [4.45362308e+00, 3.14016890e-03, -1.27841067e-06, 
                 2.39399704e-10, -1.66903437e-14]
        T = 298.15  # temperature (K)
        h_ref = -393520.0  # reference enthalpy (J/mol)
        
        # Function signature: enthalpy_nasa(T, coeffs, h_ref=0.0)
        h = pyroxa.enthalpy_nasa(T, coeffs, h_ref)
        
        # Should be close to reference enthalpy at reference temperature
        assert abs(h - h_ref) < 50000  # Within reasonable range
    
    def test_entropy_nasa(self):
        """Test NASA polynomial entropy calculation"""
        # NASA coefficients for CO2 (example)
        coeffs = [4.45362308e+00, 3.14016890e-03, -1.27841067e-06, 
                 2.39399704e-10, -1.66903437e-14]
        T = 298.15  # temperature (K)
        s_ref = 213.8  # reference entropy (J/mol/K)
        
        # Function signature: entropy_nasa(T, coeffs, s_ref=0.0)
        s = pyroxa.entropy_nasa(T, coeffs, s_ref)
        
        # Should be close to reference entropy
        assert_positive(s)
        assert abs(s - s_ref) < 50  # Within reasonable range
    
    def test_gibbs_free_energy(self):
        """Test Gibbs free energy calculation"""
        h = -393520.0  # enthalpy (J/mol)
        s = 213.8      # entropy (J/mol/K)
        T = 298.15     # temperature (K)
        
        g = pyroxa.gibbs_free_energy(h, s, T)
        expected = h - T * s
        
        assert_close(g, expected)
    
    def test_equilibrium_constant(self):
        """Test equilibrium constant calculation"""
        delta_g = -50000.0  # Gibbs free energy change (J/mol)
        T = 298.15          # temperature (K)
        R = 8.314           # gas constant (J/mol/K)
        
        keq = pyroxa.equilibrium_constant(delta_g, T)
        expected = np.exp(-delta_g / (R * T))
        
        assert_close(keq, expected)
        assert_positive(keq)
    
    def test_temperature_dependence(self):
        """Test temperature dependence of reaction rate"""
        k0 = 1.0      # rate constant at reference temperature
        Ea = 50000.0  # activation energy (J/mol)
        T0 = 298.15   # reference temperature (K)
        T = 350.0     # new temperature (K)
        R = 8.314     # gas constant (J/mol/K)
        
        k = pyroxa.temperature_dependence(k0, Ea, T, T0)
        expected = k0 * np.exp(Ea/R * (1/T0 - 1/T))
        
        assert_close(k, expected)
        assert_positive(k)
        assert k > k0  # Rate should increase with temperature
    
    def test_pressure_dependence(self):
        """Test pressure dependence for gas-phase reactions"""
        k0 = 1.0       # rate constant at reference pressure
        delta_v = -2   # volume change of activation (L/mol)
        P = 2.0 * 101325  # pressure (Pa) - convert from atm
        P0 = 1.0 * 101325 # reference pressure (Pa) - convert from atm
        R = 8.314      # gas constant (J/mol/K)
        T = 298.15     # temperature (K)
        
        # Function signature: pressure_dependence(k_ref, delta_V, P, P_ref=101325, R=8.314, T=298.15)
        k = pyroxa.pressure_dependence(k0, delta_v, P, P0, R, T)
        expected = k0 * np.exp(delta_v * (P - P0) / (R * T))
        
        assert_close(k, expected)
        assert_positive(k)
    
    def test_activity_coefficient(self):
        """Test activity coefficient calculation"""
        x = 0.3        # mole fraction
        gamma_inf = 2.5  # infinite dilution activity coefficient
        alpha = 0.5    # interaction parameter
        
        gamma = pyroxa.activity_coefficient(x, gamma_inf, alpha)
        expected = gamma_inf * np.exp(alpha * (1 - x)**2)
        
        assert_close(gamma, expected)
        assert_positive(gamma)
    
    def test_pressure_peng_robinson(self):
        """Test Peng-Robinson equation of state"""
        n = 1.0        # moles
        T = 298.15     # temperature (K)
        V = 0.024      # molar volume (L/mol)
        Tc = 304.1     # critical temperature (K)
        Pc = 73.8      # critical pressure (bar)
        omega = 0.225  # acentric factor
        
        # Function signature: pressure_peng_robinson(n, V, T, Tc, Pc, omega)
        P = pyroxa.pressure_peng_robinson(n, V, T, Tc, Pc, omega)
        
        assert_positive(P)
        assert_in_range(P, 0.1, 1000)  # Reasonable pressure range (bar)
    
    def test_fugacity_coefficient(self):
        """Test fugacity coefficient calculation"""
        P = 10.0       # pressure (bar)
        T = 298.15     # temperature (K)
        Tc = 304.1     # critical temperature (K)
        Pc = 73.8      # critical pressure (bar)
        omega = 0.225  # acentric factor
        
        phi = pyroxa.fugacity_coefficient(P, T, Tc, Pc, omega)
        
        assert_positive(phi)
        assert phi <= 1.0  # Fugacity coefficient should be <= 1 for most cases
    
    def test_enthalpy_c(self):
        """Test constant heat capacity enthalpy calculation"""
        cp = 29.1   # heat capacity (J/mol/K)
        T = 350.0   # temperature (K)
        
        h = pyroxa.enthalpy_c(cp, T)
        expected = cp * T
        
        assert_close(h, expected)
        assert_positive(h)
    
    def test_entropy_c(self):
        """Test constant heat capacity entropy calculation"""
        cp = 29.1   # heat capacity (J/mol/K)
        T = 350.0   # temperature (K)
        
        s = pyroxa.entropy_c(cp, T)
        expected = cp * np.log(T)
        
        assert_close(s, expected)
        assert_positive(s)


if __name__ == "__main__":
    # Run tests
    test_suite = TestThermodynamics()
    
    # Get all test methods
    test_methods = [method for method in dir(test_suite) if method.startswith('test_')]
    
    print("Running Thermodynamics Tests...")
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        try:
            method = getattr(test_suite, method_name)
            method()
            print(f"✓ {method_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {method_name}: {e}")
            failed += 1
    
    print(f"\nThermodynamics Tests: {passed} passed, {failed} failed")
    if failed == 0:
        print("All thermodynamics tests passed!")