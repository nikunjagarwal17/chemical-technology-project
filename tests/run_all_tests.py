"""
PyroXa Comprehensive Test Runner
Run all tests to verify the complete pure Python implementation
"""

import sys
import os
import time
import traceback

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all test modules
try:
    from tests.test_basic_kinetics import TestBasicKinetics
    from tests.test_thermodynamics import TestThermodynamics
    from tests.test_reactor_classes import TestReactorClasses
    from tests.test_transport_phenomena import TestTransportPhenomena
    from tests.test_advanced_functions import TestAdvancedFunctions
except ImportError as e:
    print(f"Error importing test modules: {e}")
    sys.exit(1)

def run_test_suite(test_class, suite_name):
    """Run a complete test suite"""
    print(f"\n{'='*60}")
    print(f"Running {suite_name}")
    print(f"{'='*60}")
    
    test_instance = test_class()
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    failed_tests = []
    
    start_time = time.time()
    
    for method_name in test_methods:
        try:
            method = getattr(test_instance, method_name)
            method()
            print(f"PASS {method_name}")
            passed += 1
        except Exception as e:
            print(f"FAIL {method_name}: {e}")
            failed += 1
            failed_tests.append((method_name, str(e)))
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{suite_name} Results:")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Duration: {duration:.2f}s")
    
    if failed_tests:
        print(f"\nFailed tests in {suite_name}:")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error}")
    
    return passed, failed, failed_tests

def main():
    """Run all PyroXa tests"""
    print("PyroXa Comprehensive Test Suite")
    print("Pure Python Implementation Verification")
    print(f"Python version: {sys.version}")
    
    # Check if PyroXa can be imported
    try:
        import pyroxa
        print(f"PyroXa version: {pyroxa.get_version()}")
        print(f"Available functions: {len([x for x in pyroxa.__all__ if x in dir(pyroxa)])}")
    except ImportError as e:
        print(f"ERROR: Cannot import PyroXa: {e}")
        sys.exit(1)
    
    # Test suites to run
    test_suites = [
        (TestBasicKinetics, "Basic Kinetics Tests"),
        (TestThermodynamics, "Thermodynamics Tests"),
        (TestReactorClasses, "Reactor Classes Tests"),
        (TestTransportPhenomena, "Transport Phenomena Tests"),
        (TestAdvancedFunctions, "Advanced Functions Tests"),
    ]
    
    total_passed = 0
    total_failed = 0
    all_failed_tests = []
    
    overall_start = time.time()
    
    # Run each test suite
    for test_class, suite_name in test_suites:
        try:
            passed, failed, failed_tests = run_test_suite(test_class, suite_name)
            total_passed += passed
            total_failed += failed
            all_failed_tests.extend([(suite_name, test, error) for test, error in failed_tests])
        except Exception as e:
            print(f"ERROR running {suite_name}: {e}")
            traceback.print_exc()
            total_failed += 1
    
    overall_end = time.time()
    overall_duration = overall_end - overall_start
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests Passed: {total_passed}")
    print(f"Total Tests Failed: {total_failed}")
    print(f"Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%" if (total_passed+total_failed) > 0 else "N/A")
    print(f"Total Duration: {overall_duration:.2f}s")
    
    if all_failed_tests:
        print(f"\nAll Failed Tests ({len(all_failed_tests)}):")
        for suite, test, error in all_failed_tests:
            print(f"  {suite} - {test}: {error}")
    else:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("PyroXa Pure Python implementation is working correctly!")
    
    # Return appropriate exit code
    return 0 if total_failed == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
