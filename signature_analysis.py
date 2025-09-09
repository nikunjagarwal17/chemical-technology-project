#!/usr/bin/env python3
"""
Comprehensive signature analysis to detect mismatches between:
- core.h (C++ declarations)
- core.cpp (C++ implementations) 
- pybindings.pyx (Cython wrappers)
- purepy.py (Python fallbacks)
"""

import re
import os
from typing import Dict, List, Tuple

def extract_cpp_signatures(file_path: str) -> Dict[str, str]:
    """Extract function signatures from C++ header or source file."""
    signatures = {}
    if not os.path.exists(file_path):
        return signatures
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern for C++ function declarations/definitions
    # Matches: return_type function_name(parameters);
    pattern = r'(?:double|int|void)\s+(\w+)\s*\([^)]*\)(?:\s*{|\s*;)'
    
    for match in re.finditer(pattern, content, re.MULTILINE):
        full_match = match.group(0)
        func_name = match.group(1)
        
        # Extract parameter list
        param_start = full_match.find('(')
        param_end = full_match.find(')')
        if param_start != -1 and param_end != -1:
            params = full_match[param_start+1:param_end].strip()
            # Count parameters (simple approach)
            if params:
                param_count = len([p.strip() for p in params.split(',') if p.strip()])
            else:
                param_count = 0
            signatures[func_name] = f"{param_count} params: {params}"
    
    return signatures

def extract_cython_signatures(file_path: str) -> Dict[str, str]:
    """Extract function signatures from Cython .pyx file."""
    signatures = {}
    if not os.path.exists(file_path):
        return signatures
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern for Cython function definitions: def py_function_name(params):
    pattern = r'def\s+(py_\w+)\s*\([^)]*\):'
    
    for match in re.finditer(pattern, content, re.MULTILINE):
        func_name = match.group(1)
        full_match = match.group(0)
        
        # Extract parameter list
        param_start = full_match.find('(')
        param_end = full_match.find('):')
        if param_start != -1 and param_end != -1:
            params = full_match[param_start+1:param_end].strip()
            # Count parameters (simple approach)
            if params:
                # Remove type annotations for counting
                clean_params = re.sub(r'\b(?:double|int|float)\s+', '', params)
                param_list = [p.strip() for p in clean_params.split(',') if p.strip()]
                param_count = len(param_list)
            else:
                param_count = 0
            signatures[func_name] = f"{param_count} params: {params}"
    
    return signatures

def extract_python_signatures(file_path: str) -> Dict[str, str]:
    """Extract function signatures from Python file."""
    signatures = {}
    if not os.path.exists(file_path):
        return signatures
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern for Python function definitions: def function_name(params):
    pattern = r'def\s+(\w+)\s*\([^)]*\):'
    
    for match in re.finditer(pattern, content, re.MULTILINE):
        func_name = match.group(1)
        full_match = match.group(0)
        
        # Extract parameter list
        param_start = full_match.find('(')
        param_end = full_match.find('):')
        if param_start != -1 and param_end != -1:
            params = full_match[param_start+1:param_end].strip()
            # Count parameters (simple approach)
            if params and params != 'self':
                param_list = [p.strip() for p in params.split(',') if p.strip() and p.strip() != 'self']
                param_count = len(param_list)
            else:
                param_count = 0
            signatures[func_name] = f"{param_count} params: {params}"
    
    return signatures

def main():
    """Main signature analysis."""
    project_root = "C:\\Users\\nikun\\OneDrive\\Documents\\Chemical Technology Project\\project"
    
    # File paths
    files = {
        'core.h': os.path.join(project_root, 'src', 'core.h'),
        'core.cpp': os.path.join(project_root, 'src', 'core.cpp'),
        'pybindings.pyx': os.path.join(project_root, 'pyroxa', 'pybindings.pyx'),
        'purepy.py': os.path.join(project_root, 'pyroxa', 'purepy.py')
    }
    
    # Extract signatures
    signatures = {}
    signatures['core.h'] = extract_cpp_signatures(files['core.h'])
    signatures['core.cpp'] = extract_cpp_signatures(files['core.cpp'])
    signatures['pybindings.pyx'] = extract_cython_signatures(files['pybindings.pyx'])
    signatures['purepy.py'] = extract_python_signatures(files['purepy.py'])
    
    print("=== SIGNATURE ANALYSIS REPORT ===")
    print()
    
    # Find all unique function names
    all_functions = set()
    for file_sigs in signatures.values():
        all_functions.update(file_sigs.keys())
    
    # Focus on key functions that should be consistent
    key_functions = [
        'autocatalytic_rate', 'michaelis_menten_rate', 'competitive_inhibition_rate',
        'simulate_packed_bed', 'simulate_fluidized_bed', 'simulate_homogeneous_batch',
        'calculate_energy_balance', 'monte_carlo_simulation', 'residence_time_distribution',
        'catalyst_deactivation_model', 'process_scale_up'
    ]
    
    print("=== KEY FUNCTION SIGNATURE COMPARISON ===")
    mismatches = []
    
    for func in key_functions:
        print(f"\n--- {func} ---")
        
        # Check if function exists in each file
        found_in = {}
        for file_name, file_sigs in signatures.items():
            # For Cython, look for py_ prefix version
            if file_name == 'pybindings.pyx':
                py_func = f'py_{func}'
                if py_func in file_sigs:
                    found_in[file_name] = file_sigs[py_func]
                elif func in file_sigs:
                    found_in[file_name] = file_sigs[func]
            else:
                if func in file_sigs:
                    found_in[file_name] = file_sigs[func]
        
        if not found_in:
            print(f"  ❌ Not found in any file")
            continue
            
        # Display signatures
        for file_name, sig in found_in.items():
            print(f"  {file_name}: {sig}")
        
        # Check for mismatches
        param_counts = [int(sig.split()[0]) for sig in found_in.values()]
        if len(set(param_counts)) > 1:
            print(f"  ⚠️  MISMATCH: Different parameter counts {set(param_counts)}")
            mismatches.append(func)
        else:
            print(f"  ✅ Consistent")
    
    print(f"\n=== SUMMARY ===")
    print(f"Functions with signature mismatches: {len(mismatches)}")
    if mismatches:
        print("Functions needing fixes:")
        for func in mismatches:
            print(f"  - {func}")
    else:
        print("✅ All key functions have consistent signatures!")

if __name__ == "__main__":
    main()
