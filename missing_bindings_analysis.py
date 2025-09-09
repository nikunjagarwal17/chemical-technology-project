#!/usr/bin/env python3
"""Analyze missing C++ functions in PyroXa Python bindings"""

import re

# Read pybindings.pyx to see what's currently exposed
with open("pyroxa/pybindings.pyx", "r") as f:
    pyx_content = f.read()

# Extract function declarations from pybindings.pyx
pyx_pattern = r'cdef extern from "core\.h":(.*?)import numpy'
extern_section = re.search(pyx_pattern, pyx_content, re.DOTALL)

if extern_section:
    extern_content = extern_section.group(1)
    # Find function declarations
    func_pattern = r'(?:int|double)\s+(\w+)\s*\('
    exposed_functions = set(re.findall(func_pattern, extern_content))
else:
    exposed_functions = set()

# Read core.h to get all available functions
with open("src/core.h", "r") as f:
    core_content = f.read()

# Extract all function declarations from core.h
all_functions = set(re.findall(r'(?:int|double)\s+(\w+)\s*\(', core_content))

print("=== PYROXA C++ FUNCTION BINDING ANALYSIS ===")
print(f"Functions exposed in Python: {len(exposed_functions)}")
print(f"Functions available in C++: {len(all_functions)}")
print(f"Coverage: {len(exposed_functions)/len(all_functions)*100:.1f}%")

# Find missing functions
missing_functions = all_functions - exposed_functions
print(f"\nMissing from Python bindings: {len(missing_functions)}")

# Categorize missing functions for easier implementation
categories = {
    "Reactor Simulations": [],
    "Thermodynamics": [],
    "Analytics": [],
    "Optimization": [],
    "Control": [],
    "Numerical Methods": [],
    "Utilities": [],
    "Machine Learning": []
}

for func in sorted(missing_functions):
    if any(word in func.lower() for word in ["simulate", "reactor", "pfr", "cstr", "batch", "bed", "phase"]):
        categories["Reactor Simulations"].append(func)
    elif any(word in func.lower() for word in ["enthalpy", "entropy", "gibbs", "equilibrium", "arrhenius", "peng", "fugacity"]):
        categories["Thermodynamics"].append(func)
    elif any(word in func.lower() for word in ["analytical", "first_order", "consecutive"]):
        categories["Analytics"].append(func)
    elif any(word in func.lower() for word in ["optimization", "objective", "sensitivity", "jacobian", "parameter", "nlls"]):
        categories["Optimization"].append(func)
    elif any(word in func.lower() for word in ["controller", "mpc", "rtd"]):
        categories["Control"].append(func)
    elif any(word in func.lower() for word in ["bdf", "gear", "implicit", "adaptive"]):
        categories["Numerical Methods"].append(func)
    elif any(word in func.lower() for word in ["neural", "gaussian", "kriging", "bootstrap"]):
        categories["Machine Learning"].append(func)
    else:
        categories["Utilities"].append(func)

print("\n=== MISSING FUNCTIONS BY CATEGORY ===")
for category, funcs in categories.items():
    if funcs:
        print(f"\n{category} ({len(funcs)}):")
        for func in funcs[:10]:  # Show first 10 to avoid too much output
            print(f"  - {func}")
        if len(funcs) > 10:
            print(f"  ... and {len(funcs) - 10} more")

# Priority functions to implement first
high_priority = [
    "simulate_pfr", "simulate_cstr", "simulate_packed_bed", 
    "simulate_fluidized_bed", "simulate_three_phase_reactor",
    "gibbs_free_energy", "equilibrium_constant", "arrhenius_rate",
    "analytical_first_order", "find_steady_state", 
    "pressure_peng_robinson", "fugacity_coefficient",
    "langmuir_hinshelwood_rate", "photochemical_rate"
]

available_priority = [f for f in high_priority if f in missing_functions]
print(f"\n=== HIGH PRIORITY MISSING FUNCTIONS ===")
print(f"Critical functions to expose: {len(available_priority)}")
for func in available_priority:
    print(f"  ✓ {func}")

print(f"\n=== IMPLEMENTATION POTENTIAL ===")
print(f"Current Python functions: {len(exposed_functions)}")
print(f"Potential total functions: {len(all_functions)}")
print(f"Increase possible: {len(missing_functions)} functions (+{len(missing_functions)/len(exposed_functions)*100:.0f}%)")
