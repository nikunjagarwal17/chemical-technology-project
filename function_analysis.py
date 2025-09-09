#!/usr/bin/env python3
"""
Analysis script to find unimplemented functions in PyroXa core.
Compares declarations in core.h with implementations in core.cpp
"""

import re
import os

def extract_function_declarations(header_file):
    """Extract function declarations from header file."""
    with open(header_file, 'r') as f:
        content = f.read()
    
    # Find function declarations (excluding C++ class methods)
    pattern = r'^(int|double|void)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    matches = re.findall(pattern, content, re.MULTILINE)
    
    return [match[1] for match in matches if not match[1].startswith('_')]

def extract_function_implementations(cpp_file):
    """Extract function implementations from cpp file."""
    with open(cpp_file, 'r') as f:
        content = f.read()
    
    # Find function implementations
    pattern = r'^(int|double|void)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    matches = re.findall(pattern, content, re.MULTILINE)
    
    return [match[1] for match in matches if not match[1].startswith('_')]

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    header_file = os.path.join(project_root, 'src', 'core.h')
    cpp_file = os.path.join(project_root, 'src', 'core.cpp')
    
    print("=" * 80)
    print("PYROXA CORE FUNCTION IMPLEMENTATION ANALYSIS")
    print("=" * 80)
    
    # Extract declarations and implementations
    declared_functions = set(extract_function_declarations(header_file))
    implemented_functions = set(extract_function_implementations(cpp_file))
    
    print(f"\nTotal functions declared in core.h: {len(declared_functions)}")
    print(f"Total functions implemented in core.cpp: {len(implemented_functions)}")
    
    # Find unimplemented functions
    unimplemented = declared_functions - implemented_functions
    
    print(f"\nUnimplemented functions: {len(unimplemented)}")
    print("-" * 50)
    
    # Categorize unimplemented functions
    categories = {
        'Control & Optimization': [],
        'Machine Learning': [],
        'Advanced Thermodynamics': [],
        'Numerical Methods': [],
        'Transport Phenomena': [],
        'Data Analysis': [],
        'Reactor Networks': [],
        'Parallel Processing': [],
        'Other': []
    }
    
    for func in sorted(unimplemented):
        if any(keyword in func.lower() for keyword in ['pid', 'mpc', 'optimization', 'control']):
            categories['Control & Optimization'].append(func)
        elif any(keyword in func.lower() for keyword in ['neural', 'gaussian', 'kriging', 'bootstrap']):
            categories['Machine Learning'].append(func)
        elif any(keyword in func.lower() for keyword in ['nasa', 'peng', 'fugacity', 'heat_capacity']):
            categories['Advanced Thermodynamics'].append(func)
        elif any(keyword in func.lower() for keyword in ['bdf', 'implicit', 'gear', 'rk']):
            categories['Numerical Methods'].append(func)
        elif any(keyword in func.lower() for keyword in ['mass_transfer', 'heat_transfer', 'diffusivity', 'pressure_drop']):
            categories['Transport Phenomena'].append(func)
        elif any(keyword in func.lower() for keyword in ['parameter_estimation', 'cross_validation', 'bootstrap']):
            categories['Data Analysis'].append(func)
        elif any(keyword in func.lower() for keyword in ['network', 'rtd', 'connectivity']):
            categories['Reactor Networks'].append(func)
        elif any(keyword in func.lower() for keyword in ['parallel', 'monte_carlo', 'sweep']):
            categories['Parallel Processing'].append(func)
        else:
            categories['Other'].append(func)
    
    # Print categorized results
    for category, functions in categories.items():
        if functions:
            print(f"\n{category}:")
            for func in functions:
                print(f"  - {func}")
    
    print("\n" + "=" * 80)
    print("IMPLEMENTED FUNCTIONS (Working)")
    print("=" * 80)
    
    for func in sorted(implemented_functions):
        print(f"  + {func}")
    
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Declared: {len(declared_functions)}")
    print(f"Implemented: {len(implemented_functions)}")
    print(f"Missing: {len(unimplemented)}")
    print(f"Implementation rate: {len(implemented_functions)/len(declared_functions)*100:.1f}%")
    
    if unimplemented:
        print(f"\nWARNING: {len(unimplemented)} functions are declared but not implemented!")
        print("These functions will cause linking errors if called.")
    else:
        print(f"\nAll declared functions are implemented!")

if __name__ == "__main__":
    main()
