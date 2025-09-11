#!/usr/bin/env python3
"""
Comprehensive PyroXa Function Audit
Check for any remaining implementations needed
"""

import sys
import os
import importlib.util

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def audit_pyroxa_functions():
    """Comprehensive audit of all PyroXa functions"""
    
    print("=== COMPREHENSIVE PYROXA FUNCTION AUDIT ===\n")
    
    try:
        import pyroxa
        print(f"✅ PyroXa v{pyroxa.get_version()} imported successfully")
    except Exception as e:
        print(f"❌ Failed to import PyroXa: {e}")
        return
    
    # Get all callable functions and classes
    all_attrs = [x for x in dir(pyroxa) if not x.startswith('_')]
    functions = [x for x in all_attrs if callable(getattr(pyroxa, x, None))]
    classes = [x for x in all_attrs if isinstance(getattr(pyroxa, x, None), type)]
    other_attrs = [x for x in all_attrs if x not in functions and x not in classes]
    
    print(f"📊 TOTAL INVENTORY:")
    print(f"   Functions: {len(functions)}")
    print(f"   Classes: {len(classes)}")
    print(f"   Other attributes: {len(other_attrs)}")
    print(f"   GRAND TOTAL: {len(all_attrs)}")
    
    # Categorize functions by type
    kinetic_functions = [f for f in functions if 'rate' in f.lower()]
    thermo_functions = [f for f in functions if any(term in f.lower() for term in ['enthalpy', 'entropy', 'gibbs', 'temperature', 'pressure'])]
    transport_functions = [f for f in functions if any(term in f.lower() for term in ['transfer', 'diffusion', 'reynolds', 'schmidt', 'prandtl', 'nusselt', 'sherwood'])]
    reactor_functions = [f for f in functions if any(term in f.lower() for term in ['reactor', 'cstr', 'pfr', 'conversion', 'residence', 'volume'])]
    separation_functions = [f for f in functions if any(term in f.lower() for term in ['crystallization', 'precipitation', 'dissolution', 'evaporation', 'distillation', 'extraction', 'adsorption'])]
    math_functions = [f for f in functions if any(term in f.lower() for term in ['interpolate', 'calculate', 'solve', 'matrix', 'bootstrap', 'monte_carlo', 'sensitivity'])]
    control_functions = [f for f in functions if any(term in f.lower() for term in ['pid', 'controller', 'mpc', 'optimization'])]
    
    print(f"\n📋 FUNCTION CATEGORIES:")
    print(f"   Kinetic Functions: {len(kinetic_functions)}")
    print(f"   Thermodynamic Functions: {len(thermo_functions)}")
    print(f"   Transport Functions: {len(transport_functions)}")
    print(f"   Reactor Functions: {len(reactor_functions)}")
    print(f"   Separation Functions: {len(separation_functions)}")
    print(f"   Mathematical Functions: {len(math_functions)}")
    print(f"   Control Functions: {len(control_functions)}")
    
    # Check for specific advanced functions that might be missing
    expected_advanced_functions = [
        'analytical_first_order',
        'analytical_reversible_first_order', 
        'analytical_consecutive_first_order',
        'calculate_objective_function',
        'check_mass_conservation',
        'calculate_rate_constants',
        'cross_validation_score',
        'kriging_interpolation',
        'bootstrap_uncertainty',
        'matrix_multiply',
        'matrix_invert',
        'solve_linear_system',
        'calculate_sensitivity',
        'calculate_jacobian',
        'stability_analysis',
        'mpc_controller',
        'real_time_optimization',
        'parameter_sweep_parallel',
        'simulate_packed_bed',
        'simulate_fluidized_bed',
        'simulate_homogeneous_batch',
        'simulate_multi_reactor_adaptive',
        'calculate_energy_balance',
        'monte_carlo_simulation',
        'residence_time_distribution',
        'catalyst_deactivation_model',
        'process_scale_up'
    ]
    
    missing_advanced = [f for f in expected_advanced_functions if f not in functions]
    present_advanced = [f for f in expected_advanced_functions if f in functions]
    
    print(f"\n🔬 ADVANCED FUNCTIONS STATUS:")
    print(f"   Expected Advanced Functions: {len(expected_advanced_functions)}")
    print(f"   Present: {len(present_advanced)} ✅")
    print(f"   Missing: {len(missing_advanced)} ❌")
    
    if missing_advanced:
        print(f"\n❌ MISSING ADVANCED FUNCTIONS:")
        for func in missing_advanced:
            print(f"   - {func}")
    
    # Test a sample of key functions to ensure they work
    print(f"\n🧪 FUNCTION TESTING:")
    test_results = []
    
    test_cases = [
        ('arrhenius_rate', lambda: pyroxa.arrhenius_rate(1e6, 50000, 298)),
        ('reynolds_number', lambda: pyroxa.reynolds_number(1000, 2, 0.1, 0.001)),
        ('first_order_rate', lambda: pyroxa.first_order_rate(0.5, 2.0)),
        ('conversion', lambda: pyroxa.conversion(5.0, 2.0)),
        ('linear_interpolate', lambda: pyroxa.linear_interpolate(2.5, [1,2,3,4], [2,4,6,8])),
    ]
    
    for func_name, test_func in test_cases:
        try:
            result = test_func()
            test_results.append((func_name, "✅", f"Result: {result}"))
        except Exception as e:
            test_results.append((func_name, "❌", f"Error: {e}"))
    
    for func_name, status, message in test_results:
        print(f"   {status} {func_name}: {message}")
    
    # Check class availability
    print(f"\n🏗️ REACTOR CLASSES:")
    reactor_classes = ['WellMixedReactor', 'CSTR', 'PFR', 'ReactorNetwork', 'FluidizedBedReactor']
    for cls_name in reactor_classes:
        if cls_name in classes:
            print(f"   ✅ {cls_name}")
        else:
            print(f"   ❌ {cls_name}")
    
    print(f"\n📈 FINAL ASSESSMENT:")
    total_expected = 68  # Original target
    current_total = len(functions)
    
    if current_total >= total_expected:
        print(f"   🎉 SUCCESS: {current_total} functions (target was {total_expected})")
        print(f"   📊 Achievement: {(current_total/total_expected)*100:.1f}% of target")
        
        if missing_advanced:
            print(f"   ⚠️  Note: {len(missing_advanced)} advanced functions could be added for completeness")
        else:
            print(f"   🏆 COMPLETE: All advanced functions implemented!")
    else:
        print(f"   ⚠️  INCOMPLETE: {current_total} functions (need {total_expected - current_total} more)")
    
    return {
        'total_functions': len(functions),
        'total_classes': len(classes),
        'missing_advanced': missing_advanced,
        'present_advanced': present_advanced,
        'target_met': current_total >= total_expected
    }

if __name__ == "__main__":
    audit_pyroxa_functions()
