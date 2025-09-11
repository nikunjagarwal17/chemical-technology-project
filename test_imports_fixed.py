#!/usr/bin/env python3
"""
Test script to verify all critical imports work after fixing
"""

def test_chainreactor_visualizer_import():
    """Test ChainReactorVisualizer import and basic functionality"""
    from pyroxa import ChainReactorVisualizer, ReactionChain
    
    # Create a simple reaction chain
    species = ['A', 'B', 'C']
    rate_constants = [1.0, 0.5]
    chain = ReactionChain(species, rate_constants)
    
    # Create visualizer
    visualizer = ChainReactorVisualizer(chain)
    
    print("✅ ChainReactorVisualizer imported and instantiated successfully")
    return True

def test_run_simulation_import():
    """Test run_simulation import and availability"""
    from pyroxa import run_simulation, run_simulation_from_dict
    
    # Verify they are the same function (alias)
    assert run_simulation is run_simulation_from_dict
    
    print("✅ run_simulation imported successfully (alias for run_simulation_from_dict)")
    return True

def test_reactor_classes():
    """Test reactor class imports"""
    from pyroxa import CSTR, PFR, ReactorNetwork, MultiReactor
    
    print("✅ All reactor classes imported successfully")
    return True

def test_kinetic_functions():
    """Test kinetic function imports"""
    from pyroxa import arrhenius_rate, equilibrium_constant, first_order_rate
    
    # Test a simple calculation
    k = arrhenius_rate(A=1e10, Ea=50000, T=300, R=8.314)
    assert k > 0
    
    print("✅ All kinetic functions imported and working")
    return True

def main():
    """Run all tests"""
    print("🧪 Testing PyroXa Import Fixes")
    print("=" * 40)
    
    tests = [
        test_chainreactor_visualizer_import,
        test_run_simulation_import,
        test_reactor_classes,
        test_kinetic_functions,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
    
    print("=" * 40)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All import fixes successful!")
        return True
    else:
        print("⚠️ Some tests failed")
        return False

if __name__ == "__main__":
    main()
