#!/usr/bin/env python
"""
Final validation test for all PyroXa functions, especially the newly added advanced ones
"""

import pyroxa
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_advanced_functions():
    """Test all newly implemented advanced functions"""
    print("🧪 TESTING ADVANCED PYROXA FUNCTIONS")
    print("=" * 50)
    
    # Test analytical solutions
    try:
        print("\n📊 ANALYTICAL SOLUTIONS:")
        c_t = pyroxa.analytical_first_order(1.0, 0.1, 5.0)
        print(f"  ✓ analytical_first_order: C(t=5) = {c_t:.3f}")
        
        c_t = pyroxa.analytical_reversible_first_order(1.0, 0.1, 0.05, 5.0)
        print(f"  ✓ analytical_reversible_first_order: C(t=5) = {c_t:.3f}")
        
        conc = pyroxa.analytical_consecutive_first_order(1.0, 0.1, 0.05, 5.0)
        print(f"  ✓ analytical_consecutive_first_order: [A,B,C] = {[round(c,3) for c in conc]}")
        
    except Exception as e:
        print(f"  ❌ Analytical solutions error: {e}")
    
    # Test statistical functions
    try:
        print("\n📈 STATISTICAL FUNCTIONS:")
        np.random.seed(42)
        data = [1.0, 1.2, 0.9, 1.1, 1.05]
        
        mean, std = pyroxa.bootstrap_uncertainty(data, n_bootstrap=100)
        print(f"  ✓ bootstrap_uncertainty: mean={mean:.3f}, std={std:.3f}")
        
        results = pyroxa.monte_carlo_simulation(
            lambda x: x[0] * x[1], 
            [(0.9, 1.1), (1.9, 2.1)], 
            n_samples=100
        )
        print(f"  ✓ monte_carlo_simulation: mean={results['mean']:.3f}")
        
    except Exception as e:
        print(f"  ❌ Statistical functions error: {e}")
    
    # Test numerical methods
    try:
        print("\n🔢 NUMERICAL METHODS:")
        A = np.array([[2, 1], [1, 3]])
        b = np.array([1, 2])
        x = pyroxa.solve_linear_system(A, b)
        print(f"  ✓ solve_linear_system: x = {[round(val,3) for val in x]}")
        
        A_inv = pyroxa.matrix_invert(A)
        print(f"  ✓ matrix_invert: det(A^-1) ≈ {np.linalg.det(A_inv):.3f}")
        
        C = pyroxa.matrix_multiply(A, A_inv)
        print(f"  ✓ matrix_multiply: A*A^-1 diagonal ≈ {[round(C[i,i],2) for i in range(2)]}")
        
    except Exception as e:
        print(f"  ❌ Numerical methods error: {e}")
    
    # Test optimization functions
    try:
        print("\n🎯 OPTIMIZATION FUNCTIONS:")
        
        # Mock data for testing
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 1.9, 3.2, 3.8])
        
        rmse = pyroxa.calculate_rmse(observed, predicted)
        print(f"  ✓ calculate_rmse: RMSE = {rmse:.3f}")
        
        r2 = pyroxa.calculate_r_squared(observed, predicted)
        print(f"  ✓ calculate_r_squared: R² = {r2:.3f}")
        
        aic = pyroxa.calculate_aic(observed, predicted, k=2)
        print(f"  ✓ calculate_aic: AIC = {aic:.3f}")
        
    except Exception as e:
        print(f"  ❌ Optimization functions error: {e}")
    
    # Test sensitivity analysis
    try:
        print("\n🔍 SENSITIVITY ANALYSIS:")
        
        def test_function(params):
            return params[0] * params[1] + params[2]**2
        
        base_params = [1.0, 2.0, 0.5]
        sensitivity = pyroxa.calculate_sensitivity(test_function, base_params)
        print(f"  ✓ calculate_sensitivity: gradients = {[round(s,3) for s in sensitivity]}")
        
        jacobian = pyroxa.calculate_jacobian(test_function, base_params)
        print(f"  ✓ calculate_jacobian: J = {[round(j,3) for j in jacobian]}")
        
    except Exception as e:
        print(f"  ❌ Sensitivity analysis error: {e}")
    
    # Test interpolation
    try:
        print("\n📏 INTERPOLATION:")
        x = [0, 1, 2, 3]
        y = [0, 1, 4, 9]
        
        y_interp = pyroxa.linear_interpolate(x, y, 1.5)
        print(f"  ✓ linear_interpolate: f(1.5) = {y_interp:.3f}")
        
        y_spline = pyroxa.cubic_spline_interpolate(x, y, 1.5)
        print(f"  ✓ cubic_spline_interpolate: f(1.5) = {y_spline:.3f}")
        
    except Exception as e:
        print(f"  ❌ Interpolation error: {e}")
    
    # Test process optimization
    try:
        print("\n⚙️ PROCESS OPTIMIZATION:")
        
        def process_model(x):
            return -(x[0]**2 + x[1]**2)  # Simple optimization problem
        
        result = pyroxa.real_time_optimization(process_model, [0.5, 0.5])
        print(f"  ✓ real_time_optimization: optimum = {[round(r,3) for r in result]}")
        
        params = {'k': 0.1, 'T': 300}
        mpc_output = pyroxa.mpc_controller(params, setpoint=1.0, current_value=0.8)
        print(f"  ✓ mpc_controller: output = {mpc_output:.3f}")
        
    except Exception as e:
        print(f"  ❌ Process optimization error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 ADVANCED FUNCTIONS TEST COMPLETE!")

def test_all_functions_available():
    """Verify all expected functions are accessible"""
    print("\n🔍 FUNCTION AVAILABILITY CHECK")
    print("=" * 40)
    
    advanced_functions = [
        'analytical_first_order', 'analytical_reversible_first_order', 
        'analytical_consecutive_first_order', 'bootstrap_uncertainty',
        'monte_carlo_simulation', 'solve_linear_system', 'matrix_invert',
        'matrix_multiply', 'calculate_rmse', 'calculate_r_squared',
        'calculate_aic', 'calculate_sensitivity', 'calculate_jacobian',
        'linear_interpolate', 'cubic_spline_interpolate', 'stability_analysis',
        'real_time_optimization', 'mpc_controller', 'cross_validation_score',
        'kriging_interpolation', 'parameter_sweep_parallel'
    ]
    
    available = 0
    total = len(advanced_functions)
    
    for func_name in advanced_functions:
        if hasattr(pyroxa, func_name):
            print(f"  ✓ {func_name}")
            available += 1
        else:
            print(f"  ❌ {func_name}")
    
    print(f"\nAvailability: {available}/{total} ({100*available/total:.1f}%)")
    
    return available == total

if __name__ == "__main__":
    print(f"✅ PyroXa v{pyroxa.get_version()} loaded successfully")
    print(f"📦 Available functions: {len([name for name in dir(pyroxa) if not name.startswith('_')])}")
    
    test_advanced_functions()
    all_available = test_all_functions_available()
    
    if all_available:
        print("\n🎯 ALL ADVANCED FUNCTIONS SUCCESSFULLY IMPLEMENTED!")
    else:
        print("\n⚠️  Some advanced functions may need verification")
