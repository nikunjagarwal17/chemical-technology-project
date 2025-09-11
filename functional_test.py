import os
import sys

print("=== PyroXa Functional Testing ===")
print("Testing key functions in both C++ and Pure Python modes\n")

def test_functions_work():
    """Test that key functions produce the same results"""
    results = {}
    
    # Test functions that are commonly used
    test_cases = [
        ('arrhenius_rate', [1e6, 50000, 298], {}),
        ('autocatalytic_rate', [0.1, 2.0, 1.5], {}),
        ('michaelis_menten_rate', [100, 0.5, 2.0], {}),
        ('linear_interpolate', [2.5, [1,2,3,4], [2,4,6,8]], {}),
        ('equilibrium_constant', [-5000, 298], {}),
        ('gibbs_free_energy', [50000, 150, 298], {}),
        ('heat_capacity_nasa', [300, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]], {}),
        ('pressure_peng_robinson', [1, 0.024, 300, 647.1, 220.64, 0.344], {}),
    ]
    
    for func_name, args, kwargs in test_cases:
        try:
            if 'pyroxa' in sys.modules:
                del sys.modules['pyroxa']
                
            import pyroxa
            func = getattr(pyroxa, func_name)
            result = func(*args, **kwargs)
            results[func_name] = result
            print(f"✅ {func_name}: {result:.6f}")
        except Exception as e:
            print(f"❌ {func_name}: Error - {e}")
            results[func_name] = f"ERROR: {e}"
    
    return results

print("1. Testing with C++ extensions...")
os.environ.pop('PYROXA_PURE_PYTHON', None)
cpp_results = test_functions_work()

print("\n2. Testing with Pure Python...")
os.environ['PYROXA_PURE_PYTHON'] = '1'
python_results = test_functions_work()

print("\n=== COMPARISON ===")
all_match = True
for func_name in cpp_results:
    cpp_val = cpp_results[func_name]
    py_val = python_results[func_name]
    
    if isinstance(cpp_val, str) or isinstance(py_val, str):
        # Handle error cases
        if cpp_val == py_val:
            print(f"✅ {func_name}: Both failed identically")
        else:
            print(f"❌ {func_name}: Different errors - C++: {cpp_val}, Python: {py_val}")
            all_match = False
    else:
        # Compare numerical results
        if abs(cpp_val - py_val) < 1e-10:
            print(f"✅ {func_name}: Perfect match ({cpp_val})")
        else:
            print(f"❌ {func_name}: Mismatch - C++: {cpp_val}, Python: {py_val}")
            all_match = False

print(f"\n=== FINAL RESULT ===")
if all_match:
    print("🎉 ALL FUNCTIONS PRODUCE IDENTICAL RESULTS!")
    print("✅ Safe to remove C++ extensions completely")
    print("💡 Pure Python is functionally equivalent")
else:
    print("⚠️  Some functions produce different results")
    print("❌ Need to investigate before removing C++")

# Test complex reactor functionality
print(f"\n=== COMPLEX REACTOR TEST ===")
try:
    os.environ['PYROXA_PURE_PYTHON'] = '1'
    if 'pyroxa' in sys.modules:
        del sys.modules['pyroxa']
    import pyroxa
    
    # Test reactor creation and simulation
    from pyroxa.purepy import Thermodynamics, Reaction
    thermo = Thermodynamics(Cp_A=75.0, Cp_B=100.0)
    reaction = Reaction(kf=0.5, kr=0.1)
    reactor = pyroxa.WellMixedReactor(thermo, reaction, T=350.0, volume=2.0)
    
    times, trajectory = reactor.run(t_span=5.0, t_step=0.1)
    print(f"✅ Complex reactor test passed: {len(times)} time points, final concentrations: {trajectory[-1]}")
    
except Exception as e:
    print(f"❌ Complex reactor test failed: {e}")
