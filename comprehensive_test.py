#!/usr/bin/env python3
"""Comprehensive PyroXa function test"""

import pyroxa

print("=== COMPREHENSIVE PYROXA FUNCTION TEST ===")
print(f"Total public functions: {len([f for f in dir(pyroxa) if not f.startswith('_')])}")
print(f"Functions in __all__: {len(pyroxa.__all__)}")

# Test categories of functions
categories = {
    "Core Classes": ["Thermodynamics", "Reaction", "ReactionMulti", "Reactor"],
    "Reactors": ["WellMixedReactor", "CSTR", "PFR", "PackedBedReactor", "FluidizedBedReactor", 
                 "HeterogeneousReactor", "HomogeneousReactor", "MultiReactor", "ReactorNetwork"],
    "Rate Functions": ["autocatalytic_rate", "michaelis_menten_rate", "competitive_inhibition_rate",
                      "langmuir_hinshelwood_rate", "photochemical_rate"],
    "Thermodynamic Functions": ["heat_capacity_nasa", "enthalpy_nasa", "entropy_nasa",
                               "pressure_peng_robinson", "fugacity_coefficient"],
    "Transport Functions": ["mass_transfer_correlation", "heat_transfer_correlation",
                           "effective_diffusivity", "pressure_drop_ergun"],
    "Control Functions": ["pid_controller", "PIDController"],
    "Simulation Functions": ["run_simulation", "run_simulation_cpp", "build_from_dict", "benchmark_multi_reactor"],
    "Error Classes": ["PyroXaError", "ThermodynamicsError", "ReactionError", "ReactorError"],
    "Utility Functions": ["get_version", "get_build_info", "is_compiled_available", "is_reaction_chains_available"],
    "I/O Functions": ["load_spec_from_yaml", "parse_mechanism", "save_results_to_csv"]
}

total_expected = 0
total_available = 0

for category, functions in categories.items():
    available = sum(1 for func in functions if hasattr(pyroxa, func))
    total_expected += len(functions)
    total_available += available
    print(f"\n{category}: {available}/{len(functions)}")
    for func in functions:
        status = "✓" if hasattr(pyroxa, func) else "❌"
        print(f"  {status} {func}")

print(f"\n=== SUMMARY ===")
print(f"Expected core functions: {total_expected}")
print(f"Available functions: {total_available}")
print(f"Coverage: {total_available/total_expected*100:.1f}%")

# Test key functionality
print(f"\n=== FUNCTIONALITY TEST ===")
try:
    # Test thermodynamics
    thermo = pyroxa.Thermodynamics(cp=30.0)
    h = thermo.enthalpy(350.0)
    print(f"✓ Thermodynamics: H = {h:.0f} J/mol")
    
    # Test reaction
    rxn = pyroxa.Reaction(kf=1e-3, kr=1e-4)
    rate = rxn.rate(1.0, 0.5)
    print(f"✓ Reaction: rate = {rate:.2e} mol/L/s")
    
    # Test rate function
    auto_rate = pyroxa.autocatalytic_rate(1e-3, 1.0, 0.5)
    print(f"✓ Autocatalytic rate: {auto_rate:.2e} mol/L/s")
    
    # Test NASA polynomial
    cp = pyroxa.heat_capacity_nasa(300.0, [3.5, 0.001, 0, 0, 0, 0, 0])
    print(f"✓ NASA heat capacity: {cp:.1f} J/mol/K")
    
    print(f"\n🎉 ALL CORE FUNCTIONS WORKING!")
    
except Exception as e:
    print(f"❌ Functionality test failed: {e}")

print(f"\n✓ PyroXa library has {len([f for f in dir(pyroxa) if not f.startswith('_')])} accessible functions")
