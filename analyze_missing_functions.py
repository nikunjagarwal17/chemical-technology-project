#!/usr/bin/env python3
"""
Analyze missing functions from core.h that need to be implemented in pybindings.pyx
"""

import re

def extract_function_signatures_from_header():
    """Extract all function signatures from core.h"""
    try:
        with open("src/core.h", "r") as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ src/core.h not found")
        return []
    
    # Pattern to match function declarations (return_type function_name(...))
    pattern = r'^(int|double|void)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^;]*\);'
    
    functions = []
    for match in re.finditer(pattern, content, re.MULTILINE):
        return_type = match.group(1)
        func_name = match.group(2)
        full_line = match.group(0)
        functions.append({
            'return_type': return_type,
            'name': func_name,
            'signature': full_line.strip()
        })
    
    return functions

def extract_functions_from_pybindings():
    """Extract functions already declared in pybindings.pyx"""
    try:
        with open("pyroxa/pybindings.pyx", "r") as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ pyroxa/pybindings.pyx not found")
        return []
    
    # Look for extern function declarations
    extern_section = False
    declared_functions = []
    
    for line in content.split('\n'):
        line = line.strip()
        if 'cdef extern from "core.h"' in line:
            extern_section = True
            continue
        if extern_section:
            if line.startswith('cdef') and 'extern' not in line:
                extern_section = False
                continue
            # Extract function names from extern declarations
            if any(keyword in line for keyword in ['int ', 'double ', 'void ']):
                # Extract function name
                match = re.search(r'(int|double|void)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
                if match:
                    declared_functions.append(match.group(2))
    
    return declared_functions

def main():
    print("🔍 Analyzing Missing Functions from Core Library")
    print("=" * 60)
    
    # Get all available functions from core.h
    core_functions = extract_function_signatures_from_header()
    print(f"📊 Total functions in core.h: {len(core_functions)}")
    
    # Get already implemented functions
    implemented_functions = extract_functions_from_pybindings()
    print(f"✅ Already implemented in pybindings.pyx: {len(implemented_functions)}")
    
    # Find missing functions
    implemented_names = set(implemented_functions)
    missing_functions = [f for f in core_functions if f['name'] not in implemented_names]
    
    print(f"⚠️  Missing functions: {len(missing_functions)}")
    print(f"📈 Potential increase: {len(missing_functions)} functions ({len(missing_functions)/len(implemented_functions)*100:.1f}% increase)")
    
    print("\n" + "="*60)
    print("📋 MISSING FUNCTIONS BY CATEGORY")
    print("="*60)
    
    # Categorize missing functions
    categories = {
        'Reactor Simulations': [],
        'Advanced Reactors': [],
        'Thermodynamics': [],
        'Kinetics': [],
        'Analytics': [],
        'Utilities': [],
        'Other': []
    }
    
    for func in missing_functions:
        name = func['name']
        if 'simulate' in name:
            if any(keyword in name for keyword in ['pfr', 'cstr', 'packed', 'fluidized', 'three_phase']):
                categories['Advanced Reactors'].append(func)
            else:
                categories['Reactor Simulations'].append(func)
        elif any(keyword in name for keyword in ['enthalpy', 'entropy', 'gibbs', 'equilibrium', 'temperature']):
            categories['Thermodynamics'].append(func)
        elif any(keyword in name for keyword in ['rate', 'kinetic', 'arrhenius', 'autocatalytic']):
            categories['Kinetics'].append(func)
        elif any(keyword in name for keyword in ['analytical', 'steady', 'optimization', 'parameter']):
            categories['Analytics'].append(func)
        elif any(keyword in name for keyword in ['transfer', 'diffusion', 'pressure', 'correlation']):
            categories['Utilities'].append(func)
        else:
            categories['Other'].append(func)
    
    for category, funcs in categories.items():
        if funcs:
            print(f"\n{category} ({len(funcs)} functions):")
            for i, func in enumerate(funcs, 1):
                print(f"  {i:2}. {func['name']} ({func['return_type']})")
    
    print("\n" + "="*60)
    print("🎯 IMPLEMENTATION STRATEGY")
    print("="*60)
    
    print("\n📈 Priority Levels:")
    print("1. HIGH PRIORITY (Simple utility functions):")
    simple_funcs = [f for f in missing_functions if f['return_type'] == 'double' and 
                   len(f['signature'].split(',')) <= 5]
    for func in simple_funcs[:10]:  # Show first 10
        print(f"   • {func['name']}")
    
    print(f"\n2. MEDIUM PRIORITY (Complex reactor functions):")
    complex_funcs = [f for f in missing_functions if 'simulate' in f['name']]
    for func in complex_funcs[:5]:  # Show first 5
        print(f"   • {func['name']}")
    
    print(f"\n3. LOW PRIORITY (Other functions):")
    other_funcs = [f for f in missing_functions if f not in simple_funcs and f not in complex_funcs]
    for func in other_funcs[:5]:  # Show first 5
        print(f"   • {func['name']}")
    
    print(f"\n🔧 RECOMMENDED IMPLEMENTATION ORDER:")
    print("1. Start with simple double-return functions (5-10 functions)")
    print("2. Test build and functionality")
    print("3. Add medium complexity functions incrementally")
    print("4. Build and test each batch individually")
    print("5. Finally combine all working functions")
    
    return missing_functions

if __name__ == "__main__":
    missing = main()
