#!/usr/bin/env python3
"""
Quick test to check which of the new functions we can actually implement
based on what exists in core.h vs what we tried to implement
"""

import os

def check_core_functions():
    """Check which functions actually exist in core.h"""
    
    core_h_path = r"C:\Users\nikun\OneDrive\Documents\Chemical Technology Project\project\src\core.h"
    
    if not os.path.exists(core_h_path):
        print("❌ core.h not found")
        return
    
    with open(core_h_path, 'r') as f:
        content = f.read()
    
    # Functions we tried to implement in Batches 9-14
    attempted_functions = [
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
    
    print("=== CHECKING FUNCTIONS IN CORE.H ===")
    
    existing_functions = []
    missing_functions = []
    
    for func in attempted_functions:
        if func in content:
            existing_functions.append(func)
            print(f"✓ {func} - EXISTS")
        else:
            missing_functions.append(func)
            print(f"❌ {func} - NOT FOUND")
    
    print(f"\n=== SUMMARY ===")
    print(f"Functions that exist: {len(existing_functions)}/{len(attempted_functions)}")
    print(f"Functions missing: {len(missing_functions)}")
    
    if missing_functions:
        print(f"\nMissing functions to remove/replace:")
        for func in missing_functions:
            print(f"  - {func}")
    
    return existing_functions, missing_functions

if __name__ == "__main__":
    existing, missing = check_core_functions()
