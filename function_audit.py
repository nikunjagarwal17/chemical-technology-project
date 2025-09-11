import os
import sys
import inspect

print("=== PyroXa Function Audit ===")
print("Comparing C++ vs Pure Python implementations\n")

# Test with C++ extensions
print("1. Testing with C++ extensions...")
os.environ.pop('PYROXA_PURE_PYTHON', None)  # Remove if set
try:
    # Force reimport to get C++ version
    if 'pyroxa' in sys.modules:
        del sys.modules['pyroxa']
    import pyroxa
    
    cpp_functions = []
    cpp_info = {}
    
    for name in dir(pyroxa):
        if not name.startswith('_'):
            obj = getattr(pyroxa, name)
            if callable(obj):
                cpp_functions.append(name)
                # Get function signature if possible
                try:
                    sig = inspect.signature(obj)
                    cpp_info[name] = {
                        'signature': str(sig),
                        'doc': obj.__doc__[:100] + '...' if obj.__doc__ and len(obj.__doc__) > 100 else obj.__doc__,
                        'type': type(obj).__name__
                    }
                except:
                    cpp_info[name] = {
                        'signature': 'Unable to get signature',
                        'doc': obj.__doc__,
                        'type': type(obj).__name__
                    }
    
    print(f"✅ C++ extensions loaded: {len(cpp_functions)} functions found")
    
except Exception as e:
    print(f"❌ C++ extensions failed: {e}")
    cpp_functions = []
    cpp_info = {}

# Test with pure Python
print("\n2. Testing with pure Python...")
os.environ['PYROXA_PURE_PYTHON'] = '1'

# Force reimport to get pure Python version
if 'pyroxa' in sys.modules:
    del sys.modules['pyroxa']
    
import pyroxa

python_functions = []
python_info = {}

for name in dir(pyroxa):
    if not name.startswith('_'):
        obj = getattr(pyroxa, name)
        if callable(obj):
            python_functions.append(name)
            # Get function signature if possible
            try:
                sig = inspect.signature(obj)
                python_info[name] = {
                    'signature': str(sig),
                    'doc': obj.__doc__[:100] + '...' if obj.__doc__ and len(obj.__doc__) > 100 else obj.__doc__,
                    'type': type(obj).__name__
                }
            except:
                python_info[name] = {
                    'signature': 'Unable to get signature', 
                    'doc': obj.__doc__,
                    'type': type(obj).__name__
                }

print(f"✅ Pure Python loaded: {len(python_functions)} functions found")

# Compare the two
print("\n=== COMPARISON RESULTS ===")

cpp_set = set(cpp_functions)
python_set = set(python_functions)

print(f"\n📊 Function Count Summary:")
print(f"C++ implementation: {len(cpp_functions)} functions")
print(f"Pure Python implementation: {len(python_functions)} functions")

# Functions in C++ but not in Python
missing_in_python = cpp_set - python_set
if missing_in_python:
    print(f"\n❌ MISSING in Pure Python ({len(missing_in_python)} functions):")
    for func in sorted(missing_in_python):
        info = cpp_info.get(func, {})
        print(f"  - {func}{info.get('signature', '')}")
        if info.get('doc'):
            print(f"    Doc: {info['doc']}")
else:
    print("\n✅ All C++ functions are available in Pure Python!")

# Functions in Python but not in C++
extra_in_python = python_set - cpp_set
if extra_in_python:
    print(f"\n➕ EXTRA in Pure Python ({len(extra_in_python)} functions):")
    for func in sorted(extra_in_python):
        info = python_info.get(func, {})
        print(f"  - {func}{info.get('signature', '')}")

# Common functions
common_functions = cpp_set & python_set
if common_functions:
    print(f"\n✅ COMMON functions ({len(common_functions)} functions):")
    for func in sorted(common_functions):
        cpp_sig = cpp_info.get(func, {}).get('signature', '')
        python_sig = python_info.get(func, {}).get('signature', '')
        
        if cpp_sig != python_sig:
            print(f"  ⚠️  {func}: Signature mismatch!")
            print(f"     C++: {cpp_sig}")
            print(f"     Python: {python_sig}")
        else:
            print(f"  ✅ {func}{cpp_sig}")

print(f"\n=== DETAILED FUNCTION ANALYSIS ===")
print("All available functions with their details:")

all_functions = sorted(cpp_set | python_set)
for func in all_functions:
    print(f"\n--- {func} ---")
    
    if func in cpp_info:
        print(f"C++: {cpp_info[func]['signature']}")
        if cpp_info[func]['doc']:
            print(f"C++ Doc: {cpp_info[func]['doc']}")
    else:
        print("C++: NOT AVAILABLE")
        
    if func in python_info:
        print(f"Python: {python_info[func]['signature']}")
        if python_info[func]['doc']:
            print(f"Python Doc: {python_info[func]['doc']}")
    else:
        print("Python: NOT AVAILABLE")

print(f"\n=== SUMMARY ===")
print(f"Total unique functions: {len(all_functions)}")
print(f"Functions needing Python implementation: {len(missing_in_python)}")
print(f"Ready for C++ removal: {'YES' if len(missing_in_python) == 0 else 'NO'}")
