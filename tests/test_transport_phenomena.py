"""
Test Transport Phenomena Functions
Tests for heat transfer, mass transfer, and fluid mechanics functions
"""

import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pyroxa
from tests import assert_close, assert_positive, assert_in_range

class TestTransportPhenomena:
    """Test suite for transport phenomena functions"""
    
    def test_reynolds_number(self):
        """Test Reynolds number calculation"""
        density = 1000.0    # kg/m³
        velocity = 2.0      # m/s
        length = 0.1        # m
        viscosity = 0.001   # Pa·s
        
        re = pyroxa.reynolds_number(density, velocity, length, viscosity)
        expected = density * velocity * length / viscosity
        
        assert_close(re, expected)
        assert_positive(re)
    
    def test_prandtl_number(self):
        """Test Prandtl number calculation"""
        cp = 4180.0        # J/kg/K (water)
        viscosity = 0.001  # Pa·s
        k = 0.6            # W/m/K
        
        pr = pyroxa.prandtl_number(cp, viscosity, k)
        expected = cp * viscosity / k
        
        assert_close(pr, expected)
        assert_positive(pr)
    
    def test_schmidt_number(self):
        """Test Schmidt number calculation"""
        viscosity = 0.001    # Pa·s
        density = 1000.0     # kg/m³
        diffusivity = 1e-9   # m²/s
        
        sc = pyroxa.schmidt_number(viscosity, density, diffusivity)
        expected = viscosity / (density * diffusivity)
        
        assert_close(sc, expected)
        assert_positive(sc)
    
    def test_nusselt_number(self):
        """Test Nusselt number calculation"""
        h = 1000.0    # W/m²/K
        L = 0.1       # m
        k = 0.6       # W/m/K
        
        nu = pyroxa.nusselt_number(h, L, k)
        expected = h * L / k
        
        assert_close(nu, expected)
        assert_positive(nu)
    
    def test_sherwood_number(self):
        """Test Sherwood number calculation"""
        kc = 0.001     # m/s
        L = 0.1        # m
        D = 1e-9       # m²/s
        
        sh = pyroxa.sherwood_number(kc, L, D)
        expected = kc * L / D
        
        assert_close(sh, expected)
        assert_positive(sh)
    
    def test_friction_factor(self):
        """Test friction factor calculation"""
        delta_p = 1000.0  # Pa
        L = 1.0           # m
        D = 0.1           # m
        rho = 1000.0      # kg/m³
        v = 2.0           # m/s
        
        f = pyroxa.friction_factor(delta_p, L, D, rho, v)
        expected = delta_p * D / (L * 0.5 * rho * v**2)
        
        assert_close(f, expected)
        assert_positive(f)
    
    def test_heat_transfer_coefficient(self):
        """Test heat transfer coefficient calculation"""
        q = 10000.0   # W/m²
        dt = 50.0     # K
        
        h = pyroxa.heat_transfer_coefficient(q, dt)
        expected = q / dt
        
        assert_close(h, expected)
        assert_positive(h)
    
    def test_mass_transfer_coefficient(self):
        """Test mass transfer coefficient calculation"""
        flux = 0.001   # mol/m²/s
        dc = 100.0     # mol/m³
        
        kc = pyroxa.mass_transfer_coefficient(flux, dc)
        expected = flux / dc
        
        assert_close(kc, expected)
        assert_positive(kc)
    
    def test_diffusion_coefficient(self):
        """Test diffusion coefficient estimation"""
        T = 298.15        # K
        viscosity = 0.001 # Pa·s
        molar_volume = 25.0  # cm³/mol
        
        D = pyroxa.diffusion_coefficient(T, viscosity, molar_volume)
        
        assert_positive(D)
        assert_in_range(D, 1e-12, 1e-8)  # Reasonable range for liquid diffusion
    
    def test_thermal_conductivity(self):
        """Test thermal conductivity calculation"""
        cp = 4180.0      # J/kg/K
        rho = 1000.0     # kg/m³
        alpha = 1.5e-7   # m²/s (thermal diffusivity)
        
        k = pyroxa.thermal_conductivity(cp, rho, alpha)
        expected = cp * rho * alpha
        
        assert_close(k, expected)
        assert_positive(k)
    
    def test_effective_diffusivity(self):
        """Test effective diffusivity in porous media"""
        D_bulk = 1e-9    # m²/s
        porosity = 0.4   # void fraction
        tortuosity = 2.0 # tortuosity factor
        
        D_eff = pyroxa.effective_diffusivity(D_bulk, porosity, tortuosity)
        expected = D_bulk * porosity / tortuosity
        
        assert_close(D_eff, expected)
        assert_positive(D_eff)
        assert D_eff < D_bulk  # Should be less than bulk diffusivity
    
    def test_pressure_drop_ergun(self):
        """Test Ergun equation for pressure drop"""
        velocity = 0.1      # m/s
        length = 1.0        # m
        dp = 0.001          # m (particle diameter)
        porosity = 0.4      # void fraction
        rho = 1000.0        # kg/m³
        mu = 0.001          # Pa·s
        
        # Function signature: pressure_drop_ergun(velocity, density, viscosity, particle_diameter, bed_porosity, bed_length)
        delta_p = pyroxa.pressure_drop_ergun(velocity, rho, mu, dp, porosity, length)
        
        assert_positive(delta_p)
        assert_in_range(delta_p, 100, 100000)  # Reasonable pressure drop range
    
    def test_mass_transfer_correlation(self):
        """Test mass transfer correlation"""
        re = 10000.0    # Reynolds number
        sc = 1.0        # Schmidt number
        
        sh = pyroxa.mass_transfer_correlation(re, sc)
        
        assert_positive(sh)
        assert sh > 1.0  # Should be greater than 1 for forced convection
    
    def test_heat_transfer_correlation(self):
        """Test heat transfer correlation"""
        re = 10000.0    # Reynolds number
        pr = 0.7        # Prandtl number
        
        nu = pyroxa.heat_transfer_correlation(re, pr)
        
        assert_positive(nu)
        assert nu > 1.0  # Should be greater than 1 for forced convection
    
    def test_hydraulic_diameter(self):
        """Test hydraulic diameter calculation"""
        area = 0.01      # m² (cross-sectional area)
        perimeter = 0.4  # m (wetted perimeter)
        
        dh = pyroxa.hydraulic_diameter(area, perimeter)
        expected = 4 * area / perimeter
        
        assert_close(dh, expected)
        assert_positive(dh)
    
    def test_bubble_rise_velocity(self):
        """Test bubble rise velocity calculation"""
        diameter = 0.003   # m
        rho_c = 1000.0     # kg/m³ (continuous phase density)
        rho_d = 1.0        # kg/m³ (dispersed phase density)
        mu = 0.001         # Pa·s
        sigma = 0.072      # N/m (surface tension)
        
        vt = pyroxa.bubble_rise_velocity(diameter, rho_c, rho_d, mu, sigma)
        
        assert_positive(vt)
        assert_in_range(vt, 0.001, 1.0)  # Reasonable range for bubble rise velocity
    
    def test_terminal_velocity(self):
        """Test terminal velocity calculation"""
        diameter = 0.001   # m
        rho_p = 2500.0     # kg/m³ (particle density)
        rho_f = 1000.0     # kg/m³ (fluid density)
        mu = 0.001         # Pa·s
        
        vt = pyroxa.terminal_velocity(diameter, rho_p, rho_f, mu)
        
        assert_positive(vt)
        assert_in_range(vt, 1e-6, 1.0)  # Reasonable range for terminal velocity
    
    def test_drag_coefficient(self):
        """Test drag coefficient calculation"""
        re = 1000.0  # Reynolds number
        
        cd = pyroxa.drag_coefficient(re)
        
        assert_positive(cd)
        assert_in_range(cd, 0.1, 100.0)  # Reasonable range for drag coefficient


if __name__ == "__main__":
    # Run tests
    test_suite = TestTransportPhenomena()
    
    # Get all test methods
    test_methods = [method for method in dir(test_suite) if method.startswith('test_')]
    
    print("Running Transport Phenomena Tests...")
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
    
    print(f"\nTransport Phenomena Tests: {passed} passed, {failed} failed")
    if failed == 0:
        print("All transport phenomena tests passed!")