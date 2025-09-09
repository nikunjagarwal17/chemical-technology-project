import re
import os

def analyze_core_h():
    """Analyze functions declared in core.h"""
    with open('src/core.h', 'r') as f:
        core_h = f.read()
    
    pattern = r'^(int|double|void)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^;]*\);'
    core_h_funcs = []
    for match in re.finditer(pattern, core_h, re.MULTILINE):
        core_h_funcs.append(match.group(2))
    
    print('📊 CORE.H ANALYSIS:')
    print(f'Total functions declared: {len(core_h_funcs)}')
    return core_h_funcs

def analyze_pybindings():
    """Analyze functions exposed in pybindings.pyx"""
    with open('pyroxa/pybindings.pyx', 'r') as f:
        pyx = f.read()
    
    # Find extern declarations
    extern_pattern = r'^\s+(int|double|void)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    extern_funcs = []
    for match in re.finditer(extern_pattern, pyx, re.MULTILINE):
        extern_funcs.append(match.group(2))
    
    # Find def statements (Python wrappers)
    def_pattern = r'^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    def_funcs = []
    for match in re.finditer(def_pattern, pyx, re.MULTILINE):
        def_funcs.append(match.group(1))
    
    print('\n📊 PYBINDINGS.PYX ANALYSIS:')
    print(f'Extern declarations: {len(extern_funcs)}')
    print(f'Python wrappers (def): {len(def_funcs)}')
    return extern_funcs, def_funcs

def analyze_init():
    """Analyze functions in __init__.py"""
    with open('pyroxa/__init__.py', 'r') as f:
        init = f.read()
    
    # Find __all__ list
    all_pattern = r'__all__\s*=\s*\[(.*?)\]'
    all_match = re.search(all_pattern, init, re.DOTALL)
    if all_match:
        all_content = all_match.group(1)
        # Extract function names from __all__
        func_pattern = r'["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']'
        all_funcs = re.findall(func_pattern, all_content)
    else:
        all_funcs = []
    
    print('\n📊 __INIT__.PY ANALYSIS:')
    print(f'Functions in __all__: {len(all_funcs)}')
    return all_funcs

def check_pipeline():
    """Check the complete pipeline"""
    core_h_funcs = analyze_core_h()
    extern_funcs, def_funcs = analyze_pybindings()
    all_funcs = analyze_init()
    
    print('\n🔍 PIPELINE VERIFICATION:')
    
    # Check missing from pybindings
    missing_from_pyx = set(core_h_funcs) - set(extern_funcs)
    print(f'\n❌ Missing from pybindings extern: {len(missing_from_pyx)}')
    if missing_from_pyx:
        print('Sample missing:', list(missing_from_pyx)[:10])
    
    missing_def = set(extern_funcs) - set(def_funcs)
    print(f'❌ Missing Python wrappers: {len(missing_def)}')
    if missing_def:
        print('Missing wrappers:', list(missing_def))
    
    # Check missing from __init__.py
    missing_from_init = set(def_funcs) - set(all_funcs)
    print(f'❌ Missing from __all__: {len(missing_from_init)}')
    if missing_from_init:
        print('Missing from __all__:', list(missing_from_init))
    
    # Summary
    print(f'\n📈 SUMMARY:')
    print(f'Core.h functions: {len(core_h_funcs)}')
    print(f'Pybindings extern: {len(extern_funcs)}')
    print(f'Python wrappers: {len(def_funcs)}')
    print(f'__all__ exports: {len(all_funcs)}')
    
    return {
        'core_h': core_h_funcs,
        'extern': extern_funcs,
        'def': def_funcs,
        'all': all_funcs
    }

if __name__ == "__main__":
    results = check_pipeline()
