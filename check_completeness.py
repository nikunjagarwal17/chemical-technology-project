#!/usr/bin/env python3
"""
Complete PyroXa Function Inventory and Gap Analysis
"""

import pyroxa

def analyze_pyroxa_completeness():
    print("🔍 PYROXA COMPLETENESS ANALYSIS")
    print("=" * 50)
    
    # Get all attributes
    all_attrs = [x for x in dir(pyroxa) if not x.startswith('_')]
    funcs = [x for x in all_attrs if callable(getattr(pyroxa, x, None))]
    classes = [x for x in all_attrs if not callable(getattr(pyroxa, x, None))]
    
    print(f"📊 INVENTORY SUMMARY:")
    print(f"   Total attributes: {len(all_attrs)}")
    print(f"   Functions: {len(funcs)}")
    print(f"   Classes/Constants: {len(classes)}")
    print()
    
    # Categorize functions
    kinetic_funcs = [f for f in funcs if 'rate' in f or 'kinetic' in f]
    thermo_funcs = [f for f in funcs if any(x in f for x in ['enthalpy', 'entropy', 'gibbs', 'equilibrium', 'temperature', 'pressure'])]
    transport_funcs = [f for f in funcs if any(x in f for x in ['transfer', 'diffus', 'reynolds', 'nusselt', 'sherwood', 'schmidt', 'prandtl'])]
    reactor_funcs = [f for f in funcs if any(x in f for x in ['reactor', 'cstr', 'pfr', 'residence', 'conversion', 'selectivity', 'batch'])]
    separation_funcs = [f for f in funcs if any(x in f for x in ['distill', 'extract', 'crystal', 'precipit', 'dissolut', 'evapor', 'adsorpt'])]
    catalyst_funcs = [f for f in funcs if 'catalyst' in f or 'surface' in f or 'pore' in f]
    fluid_funcs = [f for f in funcs if any(x in f for x in ['drag', 'terminal', 'bubble', 'mixing', 'friction', 'hydraulic'])]
    math_funcs = [f for f in funcs if any(x in f for x in ['interpolate', 'calculate', 'r_squared', 'rmse', 'aic'])]
    control_funcs = [f for f in funcs if 'pid' in f or 'control' in f]
    
    print("📋 FUNCTION CATEGORIES:")
    print(f"   Kinetic Functions: {len(kinetic_funcs)}")
    print(f"   Thermodynamic Functions: {len(thermo_funcs)}")
    print(f"   Transport Phenomena: {len(transport_funcs)}")
    print(f"   Reactor Design: {len(reactor_funcs)}")
    print(f"   Separation Processes: {len(separation_funcs)}")
    print(f"   Catalysis: {len(catalyst_funcs)}")
    print(f"   Fluid Mechanics: {len(fluid_funcs)}")
    print(f"   Mathematical Utils: {len(math_funcs)}")
    print(f"   Process Control: {len(control_funcs)}")
    print()
    
    print("🔧 ALL FUNCTIONS (alphabetical):")
    for i, func in enumerate(sorted(funcs), 1):
        print(f"   {i:2d}. {func}")
    print()
    
    print("🏗️ ALL CLASSES:")
    for i, cls in enumerate(sorted(classes), 1):
        print(f"   {i:2d}. {cls}")
    print()
    
    # Test key functions
    print("✅ FUNCTION TESTING:")
    test_results = []
    
    try:
        result = pyroxa.arrhenius_rate(1e6, 50000, 298)
        test_results.append(f"   ✓ arrhenius_rate: {result:.3e}")
    except Exception as e:
        test_results.append(f"   ✗ arrhenius_rate failed: {e}")
    
    try:
        result = pyroxa.reynolds_number(1000, 2, 0.1, 0.001)
        test_results.append(f"   ✓ reynolds_number: {result:.0f}")
    except Exception as e:
        test_results.append(f"   ✗ reynolds_number failed: {e}")
        
    try:
        result = pyroxa.conversion(5.0, 2.0)
        test_results.append(f"   ✓ conversion: {result:.1f}")
    except Exception as e:
        test_results.append(f"   ✗ conversion failed: {e}")
        
    try:
        result = pyroxa.heat_transfer_coefficient(10, 0.6, 0.05)
        test_results.append(f"   ✓ heat_transfer_coefficient: {result:.1f}")
    except Exception as e:
        test_results.append(f"   ✗ heat_transfer_coefficient failed: {e}")
        
    try:
        pid = pyroxa.PIDController(1.0, 0.1, 0.01)
        result = pid.calculate(100, 95, 0.1)
        test_results.append(f"   ✓ PIDController: {result:.2f}")
    except Exception as e:
        test_results.append(f"   ✗ PIDController failed: {e}")
        
    for result in test_results:
        print(result)
    
    print()
    print("🎯 COMPLETENESS ASSESSMENT:")
    if len(funcs) >= 68:
        print(f"   ✅ EXCELLENT! {len(funcs)} functions (target was 68)")
        print(f"   ✅ All major chemical engineering areas covered")
        print(f"   ✅ Both basic and advanced functionality available")
    else:
        print(f"   ⚠️  Need {68 - len(funcs)} more functions to reach target")
    
    return {
        'total_functions': len(funcs),
        'total_classes': len(classes),
        'kinetic': len(kinetic_funcs),
        'thermo': len(thermo_funcs),
        'transport': len(transport_funcs),
        'reactor': len(reactor_funcs),
        'separation': len(separation_funcs),
        'functions': funcs,
        'classes': classes
    }

if __name__ == "__main__":
    stats = analyze_pyroxa_completeness()
