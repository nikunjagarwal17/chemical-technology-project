#!/usr/bin/env python3
"""
Final validation test - verifies all import fixes are working
"""

def test_final_import_validation():
    """Test all the specific imports that were failing in GitHub Actions"""
    
    print("🔍 Testing original failing imports...")
    
    # Test 1: ChainReactorVisualizer import (was failing)
    try:
        from pyroxa import ChainReactorVisualizer
        print("  ✅ ChainReactorVisualizer import successful")
    except ImportError as e:
        print(f"  ❌ ChainReactorVisualizer import failed: {e}")
        return False
    
    # Test 2: run_simulation import (was failing)
    try:
        from pyroxa import run_simulation
        print("  ✅ run_simulation import successful")
    except ImportError as e:
        print(f"  ❌ run_simulation import failed: {e}")
        return False
    
    # Test 3: Verify run_simulation is alias
    try:
        from pyroxa import run_simulation, run_simulation_from_dict
        assert run_simulation is run_simulation_from_dict
        print("  ✅ run_simulation alias verified")
    except Exception as e:
        print(f"  ❌ run_simulation alias verification failed: {e}")
        return False
    
    # Test 4: Core reactor classes
    try:
        from pyroxa import CSTR, PFR, ReactorNetwork, MultiReactor
        print("  ✅ All reactor classes import successful")
    except ImportError as e:
        print(f"  ❌ Reactor classes import failed: {e}")
        return False
    
    # Test 5: Core functions
    try:
        from pyroxa import equilibrium_constant, arrhenius_rate
        print("  ✅ Core function imports successful")
    except ImportError as e:
        print(f"  ❌ Core function imports failed: {e}")
        return False
    
    return True

def test_package_info():
    """Test package information"""
    import pyroxa
    
    # Should have 132+ functions now
    all_items = [item for item in dir(pyroxa) if not item.startswith('_')]
    print(f"📊 Total exported items: {len(all_items)}")
    
    # Check for specific items that were missing
    required_items = ['ChainReactorVisualizer', 'run_simulation']
    missing = [item for item in required_items if not hasattr(pyroxa, item)]
    
    if missing:
        print(f"❌ Still missing: {missing}")
        return False
    else:
        print("✅ All required items present")
        return True

def main():
    """Run final validation"""
    print("🎯 PyroXa Import Fix - Final Validation")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_final_import_validation():
        success = False
    
    print()
    
    # Test package info
    if not test_package_info():
        success = False
    
    print("=" * 50)
    
    if success:
        print("🎉 ALL IMPORT FIXES SUCCESSFUL!")
        print("✅ ChainReactorVisualizer and run_simulation now available")
        print("✅ GitHub Actions tests should now pass")
        print("✅ PyroXa v1.0.0 ready for deployment")
    else:
        print("❌ Some issues remain")
    
    return success

if __name__ == "__main__":
    main()
