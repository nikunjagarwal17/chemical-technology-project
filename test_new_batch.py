import pyroxa

print("🔧 Testing New Batch Functions")
print("=" * 50)

# Test 1: Check if enthalpy_c and entropy_c are available
print("\n1. Testing basic thermodynamic functions:")
try:
    H = pyroxa.enthalpy_c(29.1, 500)  # cp=29.1 J/mol/K, T=500K
    print(f"   enthalpy_c(29.1, 500): {H:.2f} J/mol")
except Exception as e:
    print(f"   ❌ enthalpy_c failed: {e}")

try:
    S = pyroxa.entropy_c(29.1, 500)  # cp=29.1 J/mol/K, T=500K  
    print(f"   entropy_c(29.1, 500): {S:.2f} J/mol/K")
except Exception as e:
    print(f"   ❌ entropy_c failed: {e}")

# Test 2: Check analytical solutions
print("\n2. Testing analytical solutions:")
try:
    result = pyroxa.analytical_first_order(0.1, 1.0, 10.0, 0.1)
    print(f"   analytical_first_order: {len(result['times'])} time points")
    print(f"   Final A concentration: {result['A'][-1]:.4f}")
    print(f"   Final B concentration: {result['B'][-1]:.4f}")
except Exception as e:
    print(f"   ❌ analytical_first_order failed: {e}")

try:
    result = pyroxa.analytical_reversible_first_order(0.1, 0.05, 1.0, 0.0, 10.0, 0.1)
    print(f"   analytical_reversible_first_order: {len(result['times'])} time points")
    print(f"   Final A concentration: {result['A'][-1]:.4f}")
    print(f"   Final B concentration: {result['B'][-1]:.4f}")
except Exception as e:
    print(f"   ❌ analytical_reversible_first_order failed: {e}")

try:
    result = pyroxa.analytical_consecutive_first_order(0.1, 0.05, 1.0, 10.0, 0.1)
    print(f"   analytical_consecutive_first_order: {len(result['times'])} time points")
    print(f"   Final A concentration: {result['A'][-1]:.4f}")
    print(f"   Final B concentration: {result['B'][-1]:.4f}")
    print(f"   Final C concentration: {result['C'][-1]:.4f}")
except Exception as e:
    print(f"   ❌ analytical_consecutive_first_order failed: {e}")

print("\n3. Function availability check:")
available_funcs = [name for name in dir(pyroxa) if not name.startswith('_')]
print(f"   Total available functions: {len(available_funcs)}")
print(f"   Functions in __all__: {len(pyroxa.__all__)}")

# Check for new functions
new_batch_funcs = ['enthalpy_c', 'entropy_c', 'analytical_first_order', 
                   'analytical_reversible_first_order', 'analytical_consecutive_first_order']

available_count = 0
for func in new_batch_funcs:
    if hasattr(pyroxa, func):
        available_count += 1
        print(f"   ✅ {func}")
    else:
        print(f"   ❌ {func} missing")

print(f"\n📈 New batch status: {available_count}/{len(new_batch_funcs)} functions available")

if available_count == len(new_batch_funcs):
    print("🎉 All new batch functions successfully implemented!")
else:
    print("⚠️  Some functions from new batch are missing")
