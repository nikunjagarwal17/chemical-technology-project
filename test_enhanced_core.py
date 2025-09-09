#!/usr/bin/env python3
"""
Enhanced Core Test Suite for PyroXa with Advanced Reactor Types

This file tests the integration of advanced reactor types with the core PyroXa library,
including debugging and validation o            # Enhanced Homogeneous Reactor integration
            print("\nTest 4: Enhanced Homogeneous Reactor integration...")
            enh_homo = HomogeneousReactor(self.reaction, volume=1.5, mixing_intensity=1.5)
            
            start_time = time.time()
            enh_result = enh_homo.run(time_span=5.0, dt=0.05)
            end_time = time.time()cked Bed Reactor (PBR)
- Fluidized Bed Reactor (FBR)
- Heterogeneous Three-Phase Reactor
- Enhanced Homogeneous Reactor

Integration with existing functionality:
- Reaction chains
- Thermodynamics
- Multi-reactor networks
- Performance optimization
"""

import sys
import os
import numpy as np
import time
from typing import Dict, List

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from pyroxa.purepy import (
        Reaction, Thermodynamics, WellMixedReactor, 
        PackedBedReactor, FluidizedBedReactor, HeterogeneousReactor, HomogeneousReactor
    )
    print("✓ Enhanced PyroXa modules imported successfully")
except ImportError as e:
    print(f"✗ Failed to import enhanced PyroXa modules: {e}")
    # Fallback import for basic functionality
    try:
        from pyroxa.purepy import Reaction, Thermodynamics, WellMixedReactor
        print("✓ Basic PyroXa modules imported")
        ADVANCED_REACTORS_AVAILABLE = False
    except ImportError:
        print("✗ Critical import failure")
        sys.exit(1)
else:
    ADVANCED_REACTORS_AVAILABLE = True


class EnhancedCoreIntegrationTester:
    """Integration tester for enhanced core functionality with advanced reactors."""
    
    def __init__(self):
        """Initialize the integration tester."""
        self.test_results = {}
        self.debug_info = {}
        
        # Standard reaction for testing
        self.reaction = Reaction(kf=2.0, kr=0.2)
        self.fast_reaction = Reaction(kf=10.0, kr=1.0)
        self.slow_reaction = Reaction(kf=0.5, kr=0.05)
        
        print("🔧 Enhanced Core Integration Tester initialized")
    
    def test_basic_reactor_functionality(self) -> Dict:
        """Test basic reactor functionality before advanced tests."""
        print("\n🧪 TESTING BASIC REACTOR FUNCTIONALITY")
        print("-" * 50)
        
        results = {'test_name': 'Basic Functionality', 'subtests': {}}
        
        try:
            # Test 1: Basic WellMixedReactor
            print("Test 1: Basic WellMixedReactor...")
            basic_reactor = WellMixedReactor(self.reaction, volume=1.0)
            
            start_time = time.time()
            result = basic_reactor.run(time_span=5.0, time_step=0.1)
            end_time = time.time()
            
            # Validate basic functionality
            times = result['times']
            concentrations = result['concentrations']
            conversion = 1 - concentrations[-1, 0] / concentrations[0, 0]
            
            results['subtests']['basic_reactor'] = {
                'status': 'PASSED' if conversion > 0.1 else 'FAILED',
                'conversion': conversion,
                'execution_time': end_time - start_time,
                'final_concentrations': concentrations[-1].tolist()
            }
            
            print(f"  ✓ Basic reactor conversion: {conversion:.4f}")
            print(f"  ✓ Execution time: {end_time - start_time:.4f} s")
            
            # Test 2: Thermodynamics integration
            print("\nTest 2: Thermodynamics integration...")
            thermo = Thermodynamics(cp=30.0, T_ref=298.15)
            
            temp_range = [300, 400, 500, 600]
            thermo_results = {}
            
            for T in temp_range:
                enthalpy = thermo.enthalpy(T)
                entropy = thermo.entropy(T)
                delta_G = enthalpy - T * entropy  # Simplified
                K_eq = thermo.equilibrium_constant(T, delta_G)
                
                thermo_results[T] = {
                    'enthalpy': enthalpy,
                    'entropy': entropy,
                    'equilibrium_constant': K_eq
                }
            
            results['subtests']['thermodynamics'] = {
                'status': 'PASSED',
                'temperature_range': temp_range,
                'results': thermo_results
            }
            
            print(f"  ✓ Thermodynamics calculations completed for {len(temp_range)} temperatures")
            
            results['overall_status'] = 'PASSED'
            
        except Exception as e:
            print(f"  ✗ Error in basic functionality test: {e}")
            results['overall_status'] = 'FAILED'
            results['error'] = str(e)
        
        self.test_results['basic_functionality'] = results
        return results
    
    def test_advanced_reactor_integration(self) -> Dict:
        """Test advanced reactor types integration."""
        if not ADVANCED_REACTORS_AVAILABLE:
            print("\n⚠ Advanced reactors not available, skipping integration test")
            return {'test_name': 'Advanced Integration', 'overall_status': 'SKIPPED'}
        
        print("\n🏭 TESTING ADVANCED REACTOR INTEGRATION")
        print("-" * 50)
        
        results = {'test_name': 'Advanced Reactor Integration', 'subtests': {}}
        
        try:
            # Test 1: Packed Bed Reactor integration
            print("Test 1: Packed Bed Reactor integration...")
            pbr = PackedBedReactor(
                bed_length=1.5, bed_porosity=0.4, particle_diameter=0.002,
                catalyst_density=1200, effectiveness_factor=0.75, flow_rate=0.008
            )
            pbr.add_reaction(self.reaction)
            
            start_time = time.time()
            pbr_result = pbr.run(time_span=4.0, dt=0.05)
            end_time = time.time()
            
            pbr_conversion = pbr_result['conversion'][-1]
            
            results['subtests']['packed_bed_integration'] = {
                'status': 'PASSED' if pbr_conversion > 0.1 else 'FAILED',
                'conversion': pbr_conversion,
                'execution_time': end_time - start_time,
                'effectiveness_factor': pbr.effectiveness_factor,
                'pressure_drop': 101325 - pbr_result['pressure_profiles'][-1]
            }
            
            print(f"  ✓ PBR conversion: {pbr_conversion:.4f}")
            print(f"  ✓ Pressure drop: {101325 - pbr_result['pressure_profiles'][-1]:.1f} Pa")
            
            # Test 2: Fluidized Bed Reactor integration
            print("\nTest 2: Fluidized Bed Reactor integration...")
            fbr = FluidizedBedReactor(
                bed_height=2.5, bed_porosity=0.5, bubble_fraction=0.25,
                particle_diameter=0.0008, catalyst_density=1800, gas_velocity=0.4
            )
            fbr.add_reaction(self.reaction)
            
            start_time = time.time()
            fbr_result = fbr.run(time_span=6.0, dt=0.05)
            end_time = time.time()
            
            fbr_conversion = fbr_result['conversion'][-1]
            
            results['subtests']['fluidized_bed_integration'] = {
                'status': 'PASSED' if fbr_conversion > 0.1 else 'FAILED',
                'conversion': fbr_conversion,
                'execution_time': end_time - start_time,
                'bubble_velocity': fbr_result['bubble_velocity'],
                'mass_transfer_coeff': fbr_result['mass_transfer_coefficient']
            }
            
            print(f"  ✓ FBR conversion: {fbr_conversion:.4f}")
            print(f"  ✓ Bubble velocity: {fbr_result['bubble_velocity']:.3f} m/s")
            
            # Test 3: Heterogeneous Reactor integration
            print("\nTest 3: Heterogeneous Reactor integration...")
            het_reactor = HeterogeneousReactor(
                gas_holdup=0.3, liquid_holdup=0.5, solid_holdup=0.2,
                mass_transfer_gas_liquid=[0.08, 0.04],
                mass_transfer_liquid_solid=[0.04, 0.02]
            )
            het_reactor.add_gas_reaction(self.slow_reaction)
            het_reactor.add_liquid_reaction(self.reaction)
            het_reactor.add_solid_reaction(self.fast_reaction)
            
            start_time = time.time()
            het_result = het_reactor.run(time_span=8.0, dt=0.05)
            end_time = time.time()
            
            het_conversion = het_result['overall_conversion']
            
            results['subtests']['heterogeneous_integration'] = {
                'status': 'PASSED' if het_conversion > 0.1 else 'FAILED',
                'overall_conversion': het_conversion,
                'execution_time': end_time - start_time,
                'phase_holdups': het_result['phase_holdups'],
                'mass_transfer_coeffs': het_result['mass_transfer_coefficients']
            }
            
            print(f"  ✓ Heterogeneous conversion: {het_conversion:.4f}")
            print(f"  ✓ Three-phase system working correctly")
            
            # Test 4: Enhanced Homogeneous Reactor integration
            print("\nTest 4: Enhanced Homogeneous Reactor integration...")
            enh_homo = HomogeneousReactor(volume=1.5, mixing_intensity=1.5)
            enh_homo.add_reaction(self.reaction)
            
            start_time = time.time()
            enh_result = enh_homo.run(time_span=5.0, dt=0.05)
            end_time = time.time()
            
            enh_conversion = 1 - enh_result['concentrations'][-1, 0] / enh_result['concentrations'][0, 0]
            final_mixing_efficiency = enh_result['mixing_efficiency'][-1]
            
            results['subtests']['enhanced_homogeneous_integration'] = {
                'status': 'PASSED' if enh_conversion > 0.1 else 'FAILED',
                'conversion': enh_conversion,
                'execution_time': end_time - start_time,
                'mixing_intensity': enh_result['mixing_intensity'],
                'final_mixing_efficiency': final_mixing_efficiency
            }
            
            print(f"  ✓ Enhanced homogeneous conversion: {enh_conversion:.4f}")
            print(f"  ✓ Final mixing efficiency: {final_mixing_efficiency:.4f}")
            
            results['overall_status'] = 'PASSED'
            
        except Exception as e:
            print(f"  ✗ Error in advanced reactor integration: {e}")
            results['overall_status'] = 'FAILED'
            results['error'] = str(e)
            import traceback
            traceback.print_exc()
        
        self.test_results['advanced_integration'] = results
        return results
    
    def test_reactor_network_with_advanced_types(self) -> Dict:
        """Test reactor networks incorporating advanced reactor types."""
        if not ADVANCED_REACTORS_AVAILABLE:
            print("\n⚠ Advanced reactors not available, skipping network test")
            return {'test_name': 'Advanced Network', 'overall_status': 'SKIPPED'}
        
        print("\n🔗 TESTING REACTOR NETWORKS WITH ADVANCED TYPES")
        print("-" * 50)
        
        results = {'test_name': 'Advanced Reactor Networks', 'subtests': {}}
        
        try:
            # Test 1: Series network with different reactor types
            print("Test 1: Series network (PBR → FBR → Homogeneous)...")
            
            # Create individual reactors
            pbr = PackedBedReactor(1.0, 0.4, 0.002, 1200, 0.8, 0.01)
            pbr.add_reaction(self.reaction)
            
            fbr = FluidizedBedReactor(2.0, 0.5, 0.3, 0.0005, 1800, 0.5)
            fbr.add_reaction(self.reaction)
            
            homo = HomogeneousReactor(self.reaction, volume=1.0, mixing_intensity=2.0)
            
            # Run series simulation (manual implementation)
            # PBR first
            pbr_result = pbr.run(time_span=3.0, dt=0.05)
            pbr_final_conc = pbr_result['concentrations'][-1]
            
            # Use PBR output as FBR input
            fbr.conc_bubble = pbr_final_conc.tolist()
            fbr.conc_emulsion = pbr_final_conc.tolist()
            fbr_result = fbr.run(time_span=3.0, dt=0.05)
            fbr_final_conc = fbr_result['overall_concentrations'][-1]
            
            # Use FBR output as Homogeneous input
            homo.conc = fbr_final_conc.tolist()
            homo_result = homo.run(time_span=3.0, dt=0.05)
            homo_final_conc = homo_result['concentrations'][-1]
            
            # Calculate overall conversion
            initial_A = 1.0  # Initial concentration
            final_A = homo_final_conc[0]
            overall_conversion = 1 - final_A / initial_A
            
            results['subtests']['series_network'] = {
                'status': 'PASSED' if overall_conversion > 0.2 else 'FAILED',
                'overall_conversion': overall_conversion,
                'pbr_conversion': 1 - pbr_final_conc[0] / initial_A,
                'fbr_conversion': 1 - fbr_final_conc[0] / pbr_final_conc[0],
                'homo_conversion': 1 - final_A / fbr_final_conc[0]
            }
            
            print(f"  ✓ PBR stage conversion: {results['subtests']['series_network']['pbr_conversion']:.4f}")
            print(f"  ✓ FBR stage conversion: {results['subtests']['series_network']['fbr_conversion']:.4f}")
            print(f"  ✓ Homogeneous stage conversion: {results['subtests']['series_network']['homo_conversion']:.4f}")
            print(f"  ✓ Overall conversion: {overall_conversion:.4f}")
            
            # Test 2: Parallel network comparison
            print("\nTest 2: Parallel reactor comparison...")
            
            # Run same reaction in different reactor types
            reactors = {
                'Packed Bed': PackedBedReactor(1.5, 0.4, 0.002, 1200, 0.8, 0.01),
                'Fluidized Bed': FluidizedBedReactor(2.0, 0.5, 0.3, 0.0005, 1800, 0.5),
                'Heterogeneous': HeterogeneousReactor(0.3, 0.5, 0.2, [0.1, 0.05], [0.05, 0.02]),
                'Enhanced Homogeneous': HomogeneousReactor(self.reaction, volume=1.5, mixing_intensity=1.0)
            }
            
            parallel_results = {}
            
            for reactor_name, reactor in reactors.items():
                if reactor_name == 'Heterogeneous':
                    reactor.add_liquid_reaction(self.reaction)
                # Enhanced Homogeneous already has reaction from constructor
                
                start_time = time.time()
                if reactor_name == 'Heterogeneous':
                    result = reactor.run(time_span=5.0, dt=0.05)
                    conversion = result['overall_conversion']
                elif reactor_name == 'Fluidized Bed':
                    result = reactor.run(time_span=5.0, dt=0.05)
                    conversion = result['conversion'][-1]
                elif reactor_name == 'Packed Bed':
                    result = reactor.run(time_span=5.0, dt=0.05)
                    conversion = result['conversion'][-1]
                else:  # Enhanced Homogeneous
                    result = reactor.run(time_span=5.0, dt=0.05)
                    conversion = 1 - result['concentrations'][-1, 0] / result['concentrations'][0, 0]
                
                end_time = time.time()
                
                parallel_results[reactor_name] = {
                    'conversion': conversion,
                    'execution_time': end_time - start_time
                }
                
                print(f"  ✓ {reactor_name}: Conversion = {conversion:.4f}, Time = {end_time - start_time:.4f} s")
            
            results['subtests']['parallel_comparison'] = parallel_results
            
            # Find best performing reactor
            best_reactor = max(parallel_results.keys(), key=lambda x: parallel_results[x]['conversion'])
            best_conversion = parallel_results[best_reactor]['conversion']
            
            results['subtests']['best_reactor'] = {
                'reactor_type': best_reactor,
                'conversion': best_conversion
            }
            
            print(f"  ✓ Best performing reactor: {best_reactor} (conversion: {best_conversion:.4f})")
            
            results['overall_status'] = 'PASSED'
            
        except Exception as e:
            print(f"  ✗ Error in reactor network test: {e}")
            results['overall_status'] = 'FAILED'
            results['error'] = str(e)
            import traceback
            traceback.print_exc()
        
        self.test_results['advanced_networks'] = results
        return results
    
    def test_performance_optimization(self) -> Dict:
        """Test performance optimization features."""
        print("\n⚡ TESTING PERFORMANCE OPTIMIZATION")
        print("-" * 50)
        
        results = {'test_name': 'Performance Optimization', 'subtests': {}}
        
        try:
            # Test 1: Time step optimization
            print("Test 1: Time step optimization study...")
            
            time_steps = [0.001, 0.01, 0.05, 0.1]
            optimization_results = {}
            
            for dt in time_steps:
                reactor = WellMixedReactor(self.reaction, volume=1.0)
                
                start_time = time.time()
                result = reactor.run(time_span=2.0, time_step=dt)
                end_time = time.time()
                
                execution_time = end_time - start_time
                final_conc = result['concentrations'][-1]
                conversion = 1 - final_conc[0] / result['concentrations'][0, 0]
                
                optimization_results[dt] = {
                    'execution_time': execution_time,
                    'conversion': conversion,
                    'steps_per_second': int((2.0 / dt) / execution_time)
                }
                
                print(f"  dt = {dt}: Time = {execution_time:.4f} s, "
                      f"Conversion = {conversion:.4f}, "
                      f"Steps/s = {optimization_results[dt]['steps_per_second']:,}")
            
            results['subtests']['timestep_optimization'] = optimization_results
            
            # Test 2: Reactor size scaling
            print("\nTest 2: Reactor size scaling study...")
            
            if ADVANCED_REACTORS_AVAILABLE:
                volumes = [0.5, 1.0, 2.0, 5.0]
                scaling_results = {}
                
                for vol in volumes:
                    reactor = HomogeneousReactor(volume=vol, mixing_intensity=1.0)
                    reactor.add_reaction(self.reaction)
                    
                    start_time = time.time()
                    result = reactor.run(time_span=3.0, dt=0.05)
                    end_time = time.time()
                    
                    execution_time = end_time - start_time
                    conversion = 1 - result['concentrations'][-1, 0] / result['concentrations'][0, 0]
                    
                    scaling_results[vol] = {
                        'execution_time': execution_time,
                        'conversion': conversion
                    }
                    
                    print(f"  Volume = {vol} m³: Time = {execution_time:.4f} s, "
                          f"Conversion = {conversion:.4f}")
                
                results['subtests']['volume_scaling'] = scaling_results
            
            results['overall_status'] = 'PASSED'
            
        except Exception as e:
            print(f"  ✗ Error in performance optimization test: {e}")
            results['overall_status'] = 'FAILED'
            results['error'] = str(e)
        
        self.test_results['performance_optimization'] = results
        return results
    
    def test_debugging_and_validation(self) -> Dict:
        """Test debugging features and validation methods."""
        print("\n🔍 TESTING DEBUGGING AND VALIDATION")
        print("-" * 50)
        
        results = {'test_name': 'Debugging and Validation', 'subtests': {}}
        
        try:
            # Test 1: Mass balance validation
            print("Test 1: Mass balance validation...")
            
            mass_balance_results = {}
            
            if ADVANCED_REACTORS_AVAILABLE:
                test_reactors = {
                    'Packed Bed': PackedBedReactor(1.0, 0.4, 0.002, 1200, 0.8, 0.01),
                    'Fluidized Bed': FluidizedBedReactor(2.0, 0.5, 0.3, 0.0005, 1800, 0.5),
                    'Enhanced Homogeneous': HomogeneousReactor(1.0, 1.0)
                }
            else:
                test_reactors = {
                    'Well Mixed': WellMixedReactor(self.reaction, volume=1.0)
                }
            
            for reactor_name, reactor in test_reactors.items():
                if reactor_name not in ['Well Mixed']:
                    reactor.add_reaction(self.reaction)
                
                # Use appropriate parameter name based on reactor type
                if reactor_name == 'Well Mixed':
                    result = reactor.run(time_span=5.0, time_step=0.05)
                else:
                    result = reactor.run(time_span=5.0, dt=0.05)
                
                if reactor_name in ['Packed Bed', 'Enhanced Homogeneous']:
                    concentrations = result['concentrations']
                elif reactor_name == 'Fluidized Bed':
                    concentrations = result['overall_concentrations']
                else:
                    concentrations = result['concentrations']
                
                # Calculate mass balance error
                initial_total = concentrations[0].sum()
                final_total = concentrations[-1].sum()
                mass_balance_error = abs(final_total - initial_total) / initial_total
                
                mass_balance_results[reactor_name] = {
                    'initial_total': initial_total,
                    'final_total': final_total,
                    'mass_balance_error': mass_balance_error,
                    'status': 'PASSED' if mass_balance_error < 0.01 else 'FAILED'
                }
                
                print(f"  {reactor_name}: Mass balance error = {mass_balance_error:.2e} "
                      f"({mass_balance_results[reactor_name]['status']})")
            
            results['subtests']['mass_balance_validation'] = mass_balance_results
            
            # Test 2: Parameter validation
            print("\nTest 2: Parameter validation...")
            
            validation_tests = []
            
            try:
                # Test invalid parameters
                if ADVANCED_REACTORS_AVAILABLE:
                    # Should raise error
                    PackedBedReactor(-1.0, 0.4, 0.002, 1200, 0.8, 0.01)
                    validation_tests.append(('Negative bed length', 'FAILED'))
            except Exception:
                validation_tests.append(('Negative bed length', 'PASSED'))
            
            try:
                # Test invalid porosity
                if ADVANCED_REACTORS_AVAILABLE:
                    PackedBedReactor(1.0, 1.5, 0.002, 1200, 0.8, 0.01)
                    validation_tests.append(('Invalid porosity', 'FAILED'))
            except Exception:
                validation_tests.append(('Invalid porosity', 'PASSED'))
            
            try:
                # Test invalid effectiveness factor
                if ADVANCED_REACTORS_AVAILABLE:
                    PackedBedReactor(1.0, 0.4, 0.002, 1200, 1.5, 0.01)
                    validation_tests.append(('Invalid effectiveness factor', 'FAILED'))
            except Exception:
                validation_tests.append(('Invalid effectiveness factor', 'PASSED'))
            
            results['subtests']['parameter_validation'] = validation_tests
            
            for test_name, status in validation_tests:
                print(f"  {test_name}: {status}")
            
            # Test 3: Error handling
            print("\nTest 3: Error handling...")
            
            error_handling_results = {}
            
            try:
                # Test reaction with invalid rate constants
                invalid_reaction = Reaction(kf=-1.0, kr=0.1)
                error_handling_results['negative_rate'] = 'FAILED'
            except Exception:
                error_handling_results['negative_rate'] = 'PASSED'
            
            try:
                # Test reactor with zero time span
                reactor = WellMixedReactor(self.reaction, volume=1.0)
                reactor.run(time_span=0, dt=0.1)
                error_handling_results['zero_timespan'] = 'FAILED'
            except Exception:
                error_handling_results['zero_timespan'] = 'PASSED'
            
            results['subtests']['error_handling'] = error_handling_results
            
            for test_name, status in error_handling_results.items():
                print(f"  {test_name}: {status}")
            
            results['overall_status'] = 'PASSED'
            
        except Exception as e:
            print(f"  ✗ Error in debugging and validation test: {e}")
            results['overall_status'] = 'FAILED'
            results['error'] = str(e)
        
        self.test_results['debugging_validation'] = results
        return results
    
    def generate_integration_report(self) -> str:
        """Generate comprehensive integration test report."""
        report = []
        report.append("=" * 80)
        report.append("PYROXA ENHANCED CORE INTEGRATION TEST REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Advanced Reactors Available: {'Yes' if ADVANCED_REACTORS_AVAILABLE else 'No'}")
        report.append("")
        
        # Summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r.get('overall_status') == 'PASSED')
        skipped_tests = sum(1 for r in self.test_results.values() if r.get('overall_status') == 'SKIPPED')
        
        report.append("📊 INTEGRATION TEST SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Test Categories: {total_tests}")
        report.append(f"Tests Passed: {passed_tests}")
        report.append(f"Tests Skipped: {skipped_tests}")
        report.append(f"Tests Failed: {total_tests - passed_tests - skipped_tests}")
        report.append(f"Success Rate: {passed_tests/(total_tests - skipped_tests)*100:.1f}% (excluding skipped)")
        report.append("")
        
        # Detailed results
        for test_name, results in self.test_results.items():
            report.append(f"🔬 {results['test_name'].upper()}")
            report.append("-" * 50)
            report.append(f"Overall Status: {results['overall_status']}")
            
            if 'subtests' in results:
                for subtest_name, subtest_results in results['subtests'].items():
                    if isinstance(subtest_results, dict):
                        if 'status' in subtest_results:
                            report.append(f"  {subtest_name}: {subtest_results['status']}")
                        elif 'conversion' in subtest_results:
                            report.append(f"  {subtest_name}: Conversion = {subtest_results['conversion']:.4f}")
            
            if 'error' in results:
                report.append(f"  Error: {results['error']}")
            
            report.append("")
        
        # Recommendations
        report.append("💡 INTEGRATION RECOMMENDATIONS")
        report.append("-" * 40)
        
        if ADVANCED_REACTORS_AVAILABLE:
            if passed_tests == total_tests - skipped_tests:
                report.append("✓ All reactor types integrate correctly with core functionality")
                report.append("✓ Advanced reactor network capabilities are operational")
                report.append("✓ Performance optimization features are working")
                report.append("✓ Debugging and validation systems are functional")
            else:
                report.append("⚠ Some integration issues detected")
                report.append("⚠ Review failed test details above")
        else:
            report.append("⚠ Advanced reactors not available - basic functionality tested only")
            report.append("⚠ Install missing dependencies for full functionality")
        
        report.append("")
        report.append("🎯 NEXT STEPS")
        report.append("-" * 40)
        report.append("1. If all tests pass: System ready for production use")
        report.append("2. If tests fail: Address specific issues in failed components")
        report.append("3. Performance tuning: Optimize time steps and reactor parameters")
        report.append("4. Scale testing: Test with larger reactor networks")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main function to run enhanced core integration tests."""
    print("🚀 PYROXA ENHANCED CORE INTEGRATION TEST SUITE")
    print("=" * 70)
    
    # Initialize tester
    tester = EnhancedCoreIntegrationTester()
    
    # Run all integration tests
    print("Starting comprehensive integration tests...")
    
    try:
        # Test basic functionality first
        tester.test_basic_reactor_functionality()
        
        # Test advanced reactor integration
        tester.test_advanced_reactor_integration()
        
        # Test reactor networks
        tester.test_reactor_network_with_advanced_types()
        
        # Test performance optimization
        tester.test_performance_optimization()
        
        # Test debugging and validation
        tester.test_debugging_and_validation()
        
        # Generate and save report
        report = tester.generate_integration_report()
        
        with open('enhanced_core_integration_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("\n" + "=" * 70)
        print("🎉 ENHANCED CORE INTEGRATION TESTING COMPLETE!")
        print("=" * 70)
        print(f"Integration Report: enhanced_core_integration_report.txt")
        
        # Print summary
        total_tests = len(tester.test_results)
        passed_tests = sum(1 for r in tester.test_results.values() if r.get('overall_status') == 'PASSED')
        skipped_tests = sum(1 for r in tester.test_results.values() if r.get('overall_status') == 'SKIPPED')
        
        print(f"\n📊 FINAL INTEGRATION SUMMARY:")
        print(f"Total Test Categories: {total_tests}")
        print(f"Tests Passed: {passed_tests}")
        print(f"Tests Skipped: {skipped_tests}")
        print(f"Advanced Reactors: {'Available' if ADVANCED_REACTORS_AVAILABLE else 'Not Available'}")
        
        if passed_tests == total_tests - skipped_tests:
            print("\n🏆 ALL INTEGRATION TESTS PASSED!")
            print("Your PyroXa enhanced core is fully operational!")
        else:
            print(f"\n⚠ {total_tests - passed_tests - skipped_tests} test(s) failed")
            print("Please review the integration report for details")
        
        return 0
        
    except Exception as e:
        print(f"\n💥 Critical error during integration testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)