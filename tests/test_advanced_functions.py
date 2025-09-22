"""
Test Advanced Functions
Tests for advanced simulation, optimization, and analysis functions
"""

import sys
import os
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pyroxa
from tests import assert_close, assert_positive, assert_in_range

class TestAdvancedFunctions:
    """Test suite for advanced analysis and simulation functions"""
    
    def test_simulate_packed_bed(self):
        """Test packed bed simulation"""
        # Simulation parameters
        params = {
            'bed_length': 1.0,
            'porosity': 0.4,
            'particle_diameter': 0.003,
            'flow_rate': 0.001,
            'inlet_concentration': [2.0, 0.0],
            'reaction_rate_constant': 1.0,
            'time_span': 1.0
        }
        
        results = pyroxa.simulate_packed_bed(params)
        
        assert 'time' in results
        assert 'concentration_profile' in results
        assert 'conversion' in results
        assert 'pressure_drop' in results
        
        # Check reasonable results
        assert len(results['time']) > 10
        assert results['conversion'] >= 0
        assert results['conversion'] <= 1
        assert results['pressure_drop'] > 0
    
    def test_simulate_fluidized_bed(self):
        """Test fluidized bed simulation"""
        params = {
            'bed_height': 2.0,
            'bed_diameter': 1.0,
            'particle_diameter': 0.001,
            'gas_velocity': 0.5,
            'inlet_concentration': [1.0, 0.0],
            'reaction_rate_constant': 0.5,
            'time_span': 2.0
        }
        
        results = pyroxa.simulate_fluidized_bed(params)
        
        assert 'time' in results
        assert 'gas_concentration' in results
        assert 'solid_concentration' in results
        assert 'conversion' in results
        
        # Check reasonable results
        assert len(results['time']) > 10
        assert results['conversion'] >= 0
        assert results['conversion'] <= 1
    
    def test_simulate_homogeneous_batch(self):
        """Test homogeneous batch reactor simulation"""
        params = {
            'initial_concentration': [2.0, 0.0],
            'rate_constant': 1.0,
            'temperature': 298.15,
            'volume': 1.0,
            'time_span': 5.0
        }
        
        results = pyroxa.simulate_homogeneous_batch(params)
        
        assert 'time' in results
        assert 'concentration' in results
        assert 'conversion' in results
        assert 'rate' in results
        
        # Check mass conservation
        initial_total = sum(params['initial_concentration'])
        final_conc = results['concentration'][-1]
        final_total = sum(final_conc)
        assert_close(initial_total, final_total, tolerance=1e-3)
    
    def test_simulate_multi_reactor_adaptive(self):
        """Test adaptive multi-reactor simulation"""
        reactor_specs = [
            {
                'type': 'CSTR',
                'volume': 1.0,
                'flow_rate': 0.5,
                'initial_concentration': [2.0, 0.0]
            },
            {
                'type': 'PFR',
                'volume': 0.5,
                'flow_rate': 0.5,
                'initial_concentration': [1.0, 0.5]
            }
        ]
        
        results = pyroxa.simulate_multi_reactor_adaptive(reactor_specs, time_span=3.0)
        
        assert 'time' in results
        assert 'reactor_concentrations' in results
        assert 'overall_conversion' in results
        
        # Should have results for both reactors
        assert len(results['reactor_concentrations']) == 2
        assert results['overall_conversion'] >= 0
    
    def test_calculate_energy_balance(self):
        """Test energy balance calculation"""
        params = {
            'inlet_temperature': 298.15,
            'reaction_enthalpy': -50000.0,  # J/mol
            'heat_capacity': 75.0,          # J/mol/K
            'flow_rate': 0.001,             # m³/s
            'conversion': 0.8,
            'heat_transfer_coefficient': 100.0,
            'heat_transfer_area': 1.0,
            'ambient_temperature': 298.15
        }
        
        T_outlet = pyroxa.calculate_energy_balance(params)
        
        assert_positive(T_outlet)
        assert T_outlet != params['inlet_temperature']  # Should change due to reaction
    
    def test_calculate_sensitivity(self):
        """Test sensitivity analysis"""
        def model_function(params):
            k, C0 = params
            # Simple first-order decay: C = C0 * exp(-k*t)
            t = 1.0
            return C0 * np.exp(-k * t)
        
        base_params = [1.0, 2.0]  # k=1.0, C0=2.0
        
        sensitivity = pyroxa.calculate_sensitivity(model_function, base_params)
        
        assert len(sensitivity) == len(base_params)
        assert all(isinstance(s, float) for s in sensitivity)
    
    def test_calculate_jacobian(self):
        """Test Jacobian matrix calculation"""
        def system_function(x):
            # Simple 2x2 system
            return [x[0]**2 + x[1], x[0] - x[1]**2]
        
        point = [1.0, 2.0]
        
        jacobian = pyroxa.calculate_jacobian(system_function, point)
        
        assert jacobian.shape == (2, 2)
        assert isinstance(jacobian, np.ndarray)
    
    def test_stability_analysis(self):
        """Test stability analysis"""
        # Create a simple dynamical system matrix
        matrix = np.array([[-1.0, 0.5], [0.2, -2.0]])
        
        eigenvalues, stability = pyroxa.stability_analysis(matrix)
        
        assert len(eigenvalues) == 2
        assert stability in ['stable', 'unstable', 'marginal']
    
    def test_pid_controller(self):
        """Test PID controller"""
        controller = pyroxa.PIDController(kp=1.0, ki=0.1, kd=0.05)
        
        setpoint = 100.0
        current_value = 95.0
        
        output = controller.compute(setpoint, current_value)
        
        assert isinstance(output, float)
        assert output != 0  # Should produce some control action
    
    def test_calculate_objective_function(self):
        """Test objective function calculation"""
        experimental_data = [1.0, 0.8, 0.6, 0.4, 0.2]
        model_predictions = [1.1, 0.7, 0.65, 0.35, 0.25]
        
        objective = pyroxa.calculate_objective_function(experimental_data, model_predictions)
        
        assert_positive(objective)
        assert isinstance(objective, float)
    
    def test_check_mass_conservation(self):
        """Test mass conservation check"""
        initial_mass = [2.0, 0.0, 0.0]  # A, B, C
        final_mass = [1.0, 0.5, 0.5]    # Some A converted to B and C
        stoichiometry = [[-1, 1, 0], [0, -1, 1]]  # A->B, B->C
        
        is_conserved = pyroxa.check_mass_conservation(initial_mass, final_mass, stoichiometry)
        
        assert isinstance(is_conserved, bool)
    
    def test_calculate_rate_constants(self):
        """Test rate constant calculation from data"""
        time_data = [0, 1, 2, 3, 4, 5]
        concentration_data = [2.0, 1.64, 1.35, 1.11, 0.91, 0.75]
        
        k = pyroxa.calculate_rate_constants(time_data, concentration_data, order=1)
        
        assert_positive(k)
        assert isinstance(k, float)
    
    def test_cross_validation_score(self):
        """Test cross-validation scoring"""
        def model_func(params, x):
            a, b = params
            return a * np.exp(-b * x)
        
        x_data = np.array([0, 1, 2, 3, 4])
        y_data = np.array([2.0, 1.64, 1.35, 1.11, 0.91])
        initial_params = [2.0, 0.2]
        
        score = pyroxa.cross_validation_score(model_func, x_data, y_data, initial_params)
        
        assert_positive(score)
        assert isinstance(score, float)
    
    def test_kriging_interpolation(self):
        """Test kriging interpolation"""
        x_known = np.array([0, 1, 2, 3, 4])
        y_known = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        x_unknown = np.array([0.5, 1.5, 2.5, 3.5])
        
        y_predicted = pyroxa.kriging_interpolation(x_known, y_known, x_unknown)
        
        assert len(y_predicted) == len(x_unknown)
        assert all(isinstance(y, float) for y in y_predicted)
    
    def test_bootstrap_uncertainty(self):
        """Test bootstrap uncertainty analysis"""
        data = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        
        def statistic_func(sample):
            return np.mean(sample)
        
        mean_est, ci_lower, ci_upper = pyroxa.bootstrap_uncertainty(data, statistic_func, n_bootstrap=100)
        
        assert isinstance(mean_est, float)
        assert isinstance(ci_lower, float)
        assert isinstance(ci_upper, float)
        assert ci_lower <= mean_est <= ci_upper
    
    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation"""
        def model_func(params):
            k, C0 = params
            t = 1.0
            return C0 * np.exp(-k * t)
        
        param_distributions = {
            'k': {'type': 'normal', 'mean': 1.0, 'std': 0.1},
            'C0': {'type': 'normal', 'mean': 2.0, 'std': 0.2}
        }
        
        results = pyroxa.monte_carlo_simulation(model_func, param_distributions, n_samples=100)
        
        assert 'samples' in results
        assert 'mean' in results
        assert 'std' in results
        assert 'percentiles' in results
        assert len(results['samples']) == 100
    
    def test_parameter_sweep_parallel(self):
        """Test parallel parameter sweep"""
        def model_func(params):
            k, T = params
            # Arrhenius-like model
            return np.exp(-1000.0 / T) * k
        
        param_ranges = {
            'k': np.linspace(0.1, 2.0, 5),
            'T': np.linspace(250, 350, 5)
        }
        
        results = pyroxa.parameter_sweep_parallel(model_func, param_ranges)
        
        assert 'parameter_combinations' in results
        assert 'model_outputs' in results
        assert len(results['model_outputs']) == 25  # 5x5 parameter combinations
    
    def test_residence_time_distribution(self):
        """Test residence time distribution calculation"""
        time_data = np.linspace(0, 10, 100)
        concentration_data = np.exp(-time_data / 2.0) * (time_data > 0)
        
        rtd_results = pyroxa.residence_time_distribution(time_data, concentration_data)
        
        assert 'mean_residence_time' in rtd_results
        assert 'variance' in rtd_results
        assert 'rtd_function' in rtd_results
        assert_positive(rtd_results['mean_residence_time'])
    
    def test_catalyst_deactivation_model(self):
        """Test catalyst deactivation model"""
        time = np.linspace(0, 100, 50)
        kd = 0.01  # deactivation rate constant
        
        activity = pyroxa.catalyst_deactivation_model(time, kd, model_type='exponential')
        
        assert len(activity) == len(time)
        assert all(0 <= a <= 1 for a in activity)
        assert activity[0] == 1.0  # Initial activity should be 1
        assert activity[-1] < activity[0]  # Should decrease with time
    
    def test_process_scale_up(self):
        """Test process scale-up calculations"""
        lab_params = {
            'volume': 0.001,      # m³ (1 L)
            'power': 10.0,        # W
            'heat_transfer_area': 0.1,  # m²
            'flow_rate': 0.0001   # m³/s
        }
        
        scale_factor = 1000  # Scale up 1000x
        
        scaled_params = pyroxa.process_scale_up(lab_params, scale_factor)
        
        assert 'volume' in scaled_params
        assert 'power' in scaled_params
        assert 'heat_transfer_area' in scaled_params
        assert 'flow_rate' in scaled_params
        
        # Check scaling relationships
        assert scaled_params['volume'] == lab_params['volume'] * scale_factor
        assert scaled_params['flow_rate'] == lab_params['flow_rate'] * scale_factor


if __name__ == "__main__":
    # Run tests
    test_suite = TestAdvancedFunctions()
    
    # Get all test methods
    test_methods = [method for method in dir(test_suite) if method.startswith('test_')]
    
    print("Running Advanced Functions Tests...")
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
    
    print(f"\nAdvanced Functions Tests: {passed} passed, {failed} failed")
    if failed == 0:
        print("All advanced functions tests passed!")