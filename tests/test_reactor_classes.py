"""
Test Reactor Classes
Tests for all reactor simulation classes
"""

import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pyroxa
from tests import assert_close, assert_positive, assert_in_range

class TestReactorClasses:
    """Test suite for reactor simulation classes"""
    
    def test_thermodynamics_class(self):
        """Test Thermodynamics class"""
        thermo = pyroxa.Thermodynamics(cp=29.1, T_ref=298.15)
        
        # Test enthalpy calculation
        T = 350.0
        h = thermo.enthalpy(T)
        expected = thermo.cp * T
        assert_close(h, expected)
        assert_positive(h)
        
        # Test entropy calculation
        s = thermo.entropy(T)
        expected = thermo.cp * np.log(T / thermo.T_ref)
        assert_close(s, expected)
        
        # Test equilibrium constant
        delta_g = -10000.0  # J/mol
        keq = thermo.equilibrium_constant(T, delta_g)
        expected = np.exp(-delta_g / (8.314 * T))
        assert_close(keq, expected)
        assert_positive(keq)
    
    def test_reaction_class(self):
        """Test Reaction class"""
        kf = 2.0  # forward rate constant
        kr = 0.5  # reverse rate constant
        reaction = pyroxa.Reaction(kf, kr)
        
        # Test rate calculation
        conc = [1.0, 0.5]  # [A, B] concentrations
        rate = reaction.rate(conc)
        expected = kf * conc[0] - kr * conc[1]
        assert_close(rate, expected)
        
        # Test equilibrium constant
        keq = reaction.equilibrium_constant()
        expected = kf / kr
        assert_close(keq, expected)
        
        # Test equilibrium concentrations
        total_conc = 2.0
        a_eq, b_eq = reaction.equilibrium_concentrations(total_conc)
        assert_close(a_eq + b_eq, total_conc)
        assert_positive(a_eq)
        assert_positive(b_eq)
    
    def test_reaction_multi_class(self):
        """Test ReactionMulti class"""
        kf = 1.0
        kr = 0.2
        reactants = {0: 1, 1: 1}  # A + B -> 
        products = {2: 1}         # -> C
        
        reaction = pyroxa.ReactionMulti(kf, kr, reactants, products)
        
        # Test rate calculation
        conc = [1.0, 1.5, 0.2]  # [A, B, C]
        rate = reaction.rate(conc)
        expected = kf * conc[0] * conc[1] - kr * conc[2]
        assert_close(rate, expected)
        
        # Test stoichiometry matrix
        n_species = 3
        matrix = reaction.get_stoichiometry_matrix(n_species)
        assert len(matrix) == 1  # One reaction
        assert len(matrix[0]) == n_species
        assert matrix[0][0] == -1  # A coefficient (reactant)
        assert matrix[0][1] == -1  # B coefficient (reactant)
        assert matrix[0][2] == 1   # C coefficient (product)
    
    def test_well_mixed_reactor(self):
        """Test WellMixedReactor class"""
        # Create reaction and reactor
        reaction = pyroxa.Reaction(kf=1.0, kr=0.1)
        reactor = pyroxa.WellMixedReactor(reaction, T=300.0, conc0=(2.0, 0.0))
        
        # Test initial conditions
        assert_close(reactor.conc[0], 2.0)
        assert_close(reactor.conc[1], 0.0)
        assert reactor.T == 300.0
        
        # Test simulation
        times, traj = reactor.run(time_span=1.0, time_step=0.01)
        
        assert len(times) == len(traj)
        assert len(times) > 50  # Should have many time points
        
        # Check mass conservation (A + B should be constant)
        initial_total = traj[0][0] + traj[0][1]
        final_total = traj[-1][0] + traj[-1][1]
        assert_close(initial_total, final_total, tolerance=1e-3)
        
        # A should decrease, B should increase
        assert traj[-1][0] < traj[0][0]  # A decreases
        assert traj[-1][1] > traj[0][1]  # B increases
    
    def test_cstr_reactor(self):
        """Test CSTR class"""
        thermo = pyroxa.Thermodynamics()
        reaction = pyroxa.Reaction(kf=1.0, kr=0.1)
        
        # Create CSTR with flow
        cstr = pyroxa.CSTR(thermo, reaction, T=300.0, volume=1.0, 
                          conc0=(1.0, 0.0), q=0.5, conc_in=(2.0, 0.0))
        
        # Test initial conditions
        assert_close(cstr.conc[0], 1.0)
        assert_close(cstr.conc[1], 0.0)
        assert cstr.q == 0.5
        
        # Test simulation
        times, traj = cstr.run(time_span=2.0, time_step=0.01)
        
        assert len(times) == len(traj)
        assert len(times) > 100
        
        # With fresh feed, should reach steady state
        # Final concentration should be different from initial
        assert abs(traj[-1][0] - traj[0][0]) > 0.1
    
    def test_pfr_reactor(self):
        """Test PFR class"""
        thermo = pyroxa.Thermodynamics()
        reaction = pyroxa.Reaction(kf=1.0, kr=0.1)
        
        # Create PFR
        pfr = pyroxa.PFR(thermo, reaction, T=300.0, total_volume=1.0, 
                        nseg=10, conc0=(2.0, 0.0), q=1.0)
        
        # Test initial conditions
        assert len(pfr.segs) == 10  # Number of segments
        assert pfr.total_volume == 1.0
        assert pfr.q == 1.0
        
        # Test simulation
        times, outlet_history = pfr.run(time_span=1.0, time_step=0.01)
        
        assert len(times) == len(outlet_history)
        assert len(times) > 50
        
        # Check outlet concentrations are reasonable
        final_outlet = outlet_history[-1]
        assert final_outlet[0] >= 0  # Non-negative concentration
        assert final_outlet[1] >= 0  # Non-negative concentration
        
        # Test spatial profile
        profile = pfr.get_spatial_profile()
        assert 'positions' in profile
        assert 'A_concentrations' in profile
        assert 'B_concentrations' in profile
        assert len(profile['positions']) == pfr.nseg
    
    def test_multi_reactor(self):
        """Test MultiReactor class"""
        thermo = pyroxa.Thermodynamics()
        
        # Create multi-species reaction: A + B -> C
        reactants = {0: 1, 1: 1}  # A + B
        products = {2: 1}         # -> C
        reaction = pyroxa.ReactionMulti(1.0, 0.1, reactants, products)
        
        # Create multi-reactor
        species = ['A', 'B', 'C']
        conc0 = [2.0, 1.5, 0.0]
        reactor = pyroxa.MultiReactor(thermo, [reaction], species, 
                                     T=300.0, conc0=conc0)
        
        # Test initial conditions
        assert len(reactor.species) == 3
        assert len(reactor.conc) == 3
        assert_close(reactor.conc[0], 2.0)
        assert_close(reactor.conc[1], 1.5)
        assert_close(reactor.conc[2], 0.0)
        
        # Test simulation
        times, traj = reactor.run(time_span=1.0, time_step=0.01)
        
        assert len(times) == len(traj)
        assert len(traj[0]) == 3  # Three species
        
        # A and B should decrease, C should increase
        assert traj[-1][0] < traj[0][0]  # A decreases
        assert traj[-1][1] < traj[0][1]  # B decreases
        assert traj[-1][2] > traj[0][2]  # C increases
    
    def test_reactor_network(self):
        """Test ReactorNetwork class"""
        # Create two reactors
        reaction = pyroxa.Reaction(kf=1.0, kr=0.1)
        reactor1 = pyroxa.WellMixedReactor(reaction, T=300.0, conc0=(2.0, 0.0))
        reactor2 = pyroxa.WellMixedReactor(reaction, T=350.0, conc0=(1.0, 0.5))
        
        # Create network
        network = pyroxa.ReactorNetwork([reactor1, reactor2], mode='series')
        
        assert len(network.reactors) == 2
        assert network.mode == 'series'
        
        # Test simulation
        times, history = network.run(time_span=1.0, time_step=0.01)
        
        assert len(times) == len(history)
        assert len(history[0]) == 2  # Two reactors
        assert len(history[0][0]) == 2  # Two species per reactor
    
    def test_packed_bed_reactor(self):
        """Test PackedBedReactor class"""
        reactor = pyroxa.PackedBedReactor(
            bed_length=1.0,
            bed_porosity=0.4,
            particle_diameter=0.003,
            catalyst_density=1200.0,
            effectiveness_factor=0.8,
            flow_rate=0.001
        )
        
        # Add reaction
        reaction = pyroxa.Reaction(kf=1.0, kr=0.1)
        reactor.add_reaction(reaction)
        
        # Test simulation
        results = reactor.run(time_span=1.0, dt=0.01)
        
        assert 'times' in results
        assert 'concentrations' in results
        assert 'conversion' in results
        assert 'bed_length' in results
        assert 'effectiveness_factor' in results
        
        # Check reasonable results
        assert len(results['times']) > 50
        assert len(results['concentrations']) == len(results['times'])
        assert all(c >= 0 for c in results['conversion'])  # Non-negative conversion
    
    def test_fluidized_bed_reactor(self):
        """Test FluidizedBedReactor class"""
        reactor = pyroxa.FluidizedBedReactor(
            bed_height=2.0,
            bed_porosity=0.5,
            bubble_fraction=0.3,
            particle_diameter=0.001,
            catalyst_density=1500.0,
            gas_velocity=0.5
        )
        
        # Add reaction
        reaction = pyroxa.Reaction(kf=1.0, kr=0.1)
        reactor.add_reaction(reaction)
        
        # Test simulation
        results = reactor.run(time_span=1.0, dt=0.01)
        
        assert 'times' in results
        assert 'bubble_concentrations' in results
        assert 'emulsion_concentrations' in results
        assert 'overall_concentrations' in results
        assert 'conversion' in results
        
        # Check reasonable results
        assert len(results['times']) > 50
        assert results['bubble_velocity'] > 0
        assert results['mass_transfer_coefficient'] > 0
    
    def test_heterogeneous_reactor(self):
        """Test HeterogeneousReactor class"""
        reactor = pyroxa.HeterogeneousReactor(
            gas_holdup=0.6,
            liquid_holdup=0.3,
            solid_holdup=0.1,
            mass_transfer_gas_liquid=[0.1, 0.1],
            mass_transfer_liquid_solid=[0.05, 0.05]
        )
        
        # Add reactions in different phases
        reaction = pyroxa.Reaction(kf=1.0, kr=0.1)
        reactor.add_gas_reaction(reaction)
        reactor.add_liquid_reaction(reaction)
        reactor.add_solid_reaction(reaction)
        
        # Test simulation
        results = reactor.run(time_span=1.0, dt=0.01)
        
        assert 'times' in results
        assert 'gas_concentrations' in results
        assert 'liquid_concentrations' in results
        assert 'solid_concentrations' in results
        assert 'overall_conversion' in results
        assert 'phase_holdups' in results
        
        # Check mass conservation
        assert abs(sum(results['phase_holdups'].values()) - 1.0) < 1e-6
    
    def test_homogeneous_reactor(self):
        """Test HomogeneousReactor class"""
        reaction = pyroxa.Reaction(kf=1.0, kr=0.1)
        reactor = pyroxa.HomogeneousReactor(reaction, volume=1.0, mixing_intensity=2.0)
        
        # Test simulation
        results = reactor.run(time_span=1.0, dt=0.01)
        
        assert 'times' in results
        assert 'concentrations' in results
        assert 'mixing_efficiency' in results
        assert 'mixing_intensity' in results
        
        # Check mixing efficiency increases with time
        mixing_eff = results['mixing_efficiency']
        assert mixing_eff[0] < mixing_eff[-1]
        assert all(0 <= eff <= 1 for eff in mixing_eff)


if __name__ == "__main__":
    # Run tests
    test_suite = TestReactorClasses()
    
    # Get all test methods
    test_methods = [method for method in dir(test_suite) if method.startswith('test_')]
    
    print("Running Reactor Classes Tests...")
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
    
    print(f"\nReactor Classes Tests: {passed} passed, {failed} failed")
    if failed == 0:
        print("All reactor classes tests passed!")