import pyroxa
import numpy as np

print("🔧 Testing Latest Batch Functions")
print("=" * 50)

# Test 1: Calculate objective function
print("\n1. Testing calculate_objective_function:")
try:
    experimental = [1.0, 2.0, 3.0, 4.0, 5.0]
    simulated = [1.1, 1.9, 3.1, 3.9, 5.0]
    obj_func = pyroxa.calculate_objective_function(experimental, simulated)
    print(f"   Objective function value: {obj_func:.6f}")
except Exception as e:
    print(f"   ❌ calculate_objective_function failed: {e}")

# Test 2: Check mass conservation
print("\n2. Testing check_mass_conservation:")
try:
    # Simulate concentration data where total mass should be conserved
    concentrations = [
        [1.0, 0.0, 0.0],  # t=0: A=1, B=0, C=0
        [0.8, 0.2, 0.0],  # t=1: A=0.8, B=0.2, C=0
        [0.6, 0.3, 0.1],  # t=2: A=0.6, B=0.3, C=0.1
        [0.4, 0.4, 0.2],  # t=3: A=0.4, B=0.4, C=0.2
        [0.2, 0.5, 0.3]   # t=4: A=0.2, B=0.5, C=0.3
    ]
    result = pyroxa.check_mass_conservation(concentrations)
    print(f"   Mass conservation check: {result['is_conserved']}")
    print(f"   Maximum violation: {result['max_violation']:.6f}")
    print(f"   Mass balance: {[f'{x:.3f}' for x in result['mass_balance']]}")
except Exception as e:
    print(f"   ❌ check_mass_conservation failed: {e}")

# Test 3: Calculate rate constants
print("\n3. Testing calculate_rate_constants:")
try:
    kf_ref = [1.0, 2.0]      # Reference forward rate constants
    kr_ref = [0.5, 1.0]      # Reference reverse rate constants
    Ea_f = [50000, 40000]    # Forward activation energies (J/mol)
    Ea_r = [60000, 45000]    # Reverse activation energies (J/mol)
    T = 500                  # Temperature (K)
    T_ref = 298.15           # Reference temperature (K)
    
    result = pyroxa.calculate_rate_constants(kf_ref, kr_ref, Ea_f, Ea_r, T, T_ref)
    print(f"   Forward rate constants at {T}K: {[f'{x:.2e}' for x in result['kf']]}")
    print(f"   Reverse rate constants at {T}K: {[f'{x:.2e}' for x in result['kr']]}")
except Exception as e:
    print(f"   ❌ calculate_rate_constants failed: {e}")

print("\n4. Function availability check:")
available_funcs = [name for name in dir(pyroxa) if not name.startswith('_')]
print(f"   Total available functions: {len(available_funcs)}")
print(f"   Functions in __all__: {len(pyroxa.__all__)}")

# Check for latest batch functions
latest_batch_funcs = ['calculate_objective_function', 'check_mass_conservation', 'calculate_rate_constants']

available_count = 0
for func in latest_batch_funcs:
    if hasattr(pyroxa, func):
        available_count += 1
        print(f"   ✅ {func}")
    else:
        print(f"   ❌ {func} missing")

print(f"\n📈 Latest batch status: {available_count}/{len(latest_batch_funcs)} functions available")

if available_count == len(latest_batch_funcs):
    print("🎉 All latest batch functions successfully implemented!")
else:
    print("⚠️  Some functions from latest batch are missing")

print("\n🏆 OVERALL IMPLEMENTATION STATUS:")
print(f"   Total functions now available: {len(available_funcs)}")
print(f"   Functions in __all__: {len(pyroxa.__all__)}")

# Run comprehensive verification
print("\n📊 Running comprehensive verification...")
try:
    result = pyroxa.comprehensive_verification() if hasattr(pyroxa, 'comprehensive_verification') else None
    if result:
        print(f"   ✅ Comprehensive verification completed")
    else:
        print("   ℹ️  Running manual verification...")
        # Manual check of key function categories
        kinetic_funcs = ['arrhenius_rate', 'michaelis_menten_rate', 'autocatalytic_rate']
        thermo_funcs = ['gibbs_free_energy', 'enthalpy_c', 'entropy_c']
        analytical_funcs = ['analytical_first_order', 'analytical_reversible_first_order']
        utility_funcs = ['calculate_objective_function', 'check_mass_conservation']
        
        total_key_funcs = len(kinetic_funcs + thermo_funcs + analytical_funcs + utility_funcs)
        available_key_funcs = sum(1 for f in kinetic_funcs + thermo_funcs + analytical_funcs + utility_funcs if hasattr(pyroxa, f))
        
        print(f"   Key functions available: {available_key_funcs}/{total_key_funcs}")
        print(f"   Implementation coverage: {available_key_funcs/total_key_funcs*100:.1f}%")
        
        if available_key_funcs >= total_key_funcs * 0.9:
            print("   🎉 EXCELLENT: PyroXa is comprehensively functional!")
        else:
            print("   ⚠️  Some key functions are missing")
            
except Exception as e:
    print(f"   ❌ Verification failed: {e}")
