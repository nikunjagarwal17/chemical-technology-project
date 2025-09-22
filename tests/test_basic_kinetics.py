"""
Test Basic Kinetic Functions
Tests for fundamental reaction kinetics functions
"""

import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pyroxa
from tests import assert_close, assert_positive, assert_in_range

class TestBasicKinetics:
    """Test suite for basic kinetic functions"""
    
    def test_first_order_rate(self):
        """Test first-order reaction rate calculation"""
        k = 0.1  # rate constant (1/s)
        concentration = 2.0  # mol/L
        
        rate = pyroxa.first_order_rate(k, concentration)
        expected = k * concentration
        
        assert_close(rate, expected)
        assert_positive(rate)
    
    def test_second_order_rate_single_species(self):
        """Test second-order reaction rate with single species"""
        k = 0.05  # rate constant (L/mol/s)
        conc_A = 1.5  # mol/L
        
        rate = pyroxa.second_order_rate(k, conc_A)
        expected = k * conc_A * conc_A
        
        assert_close(rate, expected)
        assert_positive(rate)
    
    def test_second_order_rate_two_species(self):
        """Test second-order reaction rate with two species"""
        k = 0.05  # rate constant (L/mol/s)
        conc_A = 1.5  # mol/L
        conc_B = 2.0  # mol/L
        
        rate = pyroxa.second_order_rate(k, conc_A, conc_B)
        expected = k * conc_A * conc_B
        
        assert_close(rate, expected)
        assert_positive(rate)
    
    def test_zero_order_rate(self):
        """Test zero-order reaction rate"""
        k = 0.1  # rate constant (mol/L/s)
        
        rate = pyroxa.zero_order_rate(k)
        expected = k
        
        assert_close(rate, expected)
        assert_positive(rate)
    
    def test_arrhenius_rate(self):
        """Test Arrhenius rate equation"""
        A = 1e10  # pre-exponential factor (1/s)
        Ea = 50000  # activation energy (J/mol)
        T = 298.15  # temperature (K)
        R = 8.314  # gas constant (J/mol/K)
        
        k = pyroxa.arrhenius_rate(A, Ea, T)
        expected = A * np.exp(-Ea / (R * T))
        
        assert_close(k, expected)
        assert_positive(k)
    
    def test_michaelis_menten_rate(self):
        """Test Michaelis-Menten kinetics"""
        vmax = 10.0  # maximum rate (mol/L/s)
        km = 2.0     # Michaelis constant (mol/L)
        s = 5.0      # substrate concentration (mol/L)
        
        rate = pyroxa.michaelis_menten_rate(vmax, km, s)
        expected = vmax * s / (km + s)
        
        assert_close(rate, expected)
        assert_positive(rate)
        assert rate < vmax  # Rate should be less than vmax
    
    def test_competitive_inhibition_rate(self):
        """Test competitive inhibition kinetics"""
        vmax = 10.0  # maximum rate (mol/L/s)
        km = 2.0     # Michaelis constant (mol/L)
        s = 5.0      # substrate concentration (mol/L)
        ki = 1.0     # inhibition constant (mol/L)
        i = 0.5      # inhibitor concentration (mol/L)
        
        # Function signature: competitive_inhibition_rate(Vmax, Km, substrate_conc, inhibitor_conc, Ki)
        rate = pyroxa.competitive_inhibition_rate(vmax, km, s, i, ki)
        expected = vmax * s / (km * (1 + i/ki) + s)
        
        assert_close(rate, expected)
        assert_positive(rate)
        assert rate < vmax  # Rate should be less than vmax
    
    def test_autocatalytic_rate(self):
        """Test autocatalytic reaction rate"""
        k = 0.1  # rate constant
        a = 2.0  # reactant concentration
        x = 0.5  # product concentration (catalyst)
        
        rate = pyroxa.autocatalytic_rate(k, a, x)
        expected = k * a * x
        
        assert_close(rate, expected)
        assert_positive(rate)
    
    def test_langmuir_hinshelwood_rate(self):
        """Test Langmuir-Hinshelwood kinetics"""
        k = 1.0   # rate constant
        ka = 0.5  # adsorption constant for A
        kb = 0.3  # adsorption constant for B
        ca = 2.0  # concentration of A
        cb = 1.5  # concentration of B
        
        rate = pyroxa.langmuir_hinshelwood_rate(k, ka, kb, ca, cb)
        expected = k * ka * ca * kb * cb / ((1 + ka * ca + kb * cb)**2)
        
        assert_close(rate, expected)
        assert_positive(rate)
    
    def test_photochemical_rate(self):
        """Test photochemical reaction rate"""
        phi = 0.8    # quantum yield
        i0 = 100.0   # incident light intensity
        alpha = 0.1  # molar absorptivity (absorption coefficient)
        l = 1.0      # path length
        concentration = 1.0  # concentration
        
        # Function signature: photochemical_rate(quantum_yield, molar_absorptivity, path_length, light_intensity, concentration)
        rate = pyroxa.photochemical_rate(phi, alpha, l, i0, concentration)
        expected = phi * i0 * (1 - np.exp(-alpha * concentration * l))
        
        assert_close(rate, expected)
        assert_positive(rate)
    
    def test_reversible_rate(self):
        """Test reversible reaction rate"""
        kf = 2.0  # forward rate constant
        kr = 0.5  # reverse rate constant
        ca = 1.0  # concentration of A
        cb = 0.5  # concentration of B
        
        rate = pyroxa.reversible_rate(kf, kr, ca, cb)
        expected = kf * ca - kr * cb
        
        assert_close(rate, expected)
    
    def test_parallel_reaction_rate(self):
        """Test parallel reaction rates"""
        k1 = 1.0  # rate constant for reaction 1
        k2 = 0.5  # rate constant for reaction 2
        ca = 2.0  # concentration of reactant A
        
        rates = pyroxa.parallel_reaction_rate(k1, k2, ca)
        expected_1 = k1 * ca
        expected_2 = k2 * ca
        
        assert len(rates) == 2
        assert_close(rates[0], expected_1)
        assert_close(rates[1], expected_2)
        assert all(r >= 0 for r in rates)
    
    def test_series_reaction_rate(self):
        """Test series reaction rates A -> B -> C"""
        k1 = 1.0  # rate constant for A -> B
        k2 = 0.5  # rate constant for B -> C
        ca = 2.0  # concentration of A
        cb = 1.0  # concentration of B
        
        rates = pyroxa.series_reaction_rate(k1, k2, ca, cb)
        expected_1 = k1 * ca      # rate of A consumption
        expected_2 = k1 * ca - k2 * cb  # rate of B formation
        expected_3 = k2 * cb      # rate of C formation
        
        assert len(rates) == 3
        assert_close(rates[0], -expected_1)  # A consumption (negative)
        assert_close(rates[1], expected_2)   # B net rate
        assert_close(rates[2], expected_3)   # C formation (positive)
    
    def test_enzyme_inhibition_rate(self):
        """Test enzyme inhibition kinetics"""
        vmax = 10.0  # maximum rate
        km = 2.0     # Michaelis constant
        s = 5.0      # substrate concentration
        ki = 1.0     # inhibition constant
        i = 0.5      # inhibitor concentration
        
        # Function signature: enzyme_inhibition_rate(Vmax, Km, substrate_conc, inhibitor_conc, Ki, inhibition_type='competitive')
        rate = pyroxa.enzyme_inhibition_rate(vmax, km, s, i, ki)
        expected = vmax * s / (km + s * (1 + i/ki))
        
        assert_close(rate, expected)
        assert_positive(rate)
        assert rate < vmax


if __name__ == "__main__":
    # Run tests
    test_suite = TestBasicKinetics()
    
    # Get all test methods
    test_methods = [method for method in dir(test_suite) if method.startswith('test_')]
    
    print("Running Basic Kinetics Tests...")
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
    
    print(f"\nBasic Kinetics Tests: {passed} passed, {failed} failed")
    if failed == 0:
        print("All basic kinetics tests passed!")