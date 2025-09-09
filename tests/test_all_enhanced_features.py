"""
Comprehensive test of all enhanced multi-reaction features.

This demonstrates:
1. Simple A <-> B reactions
2. Multi-step chains A -> B -> C -> D  
3. Branching networks
4. Analytical solutions
5. Enhanced visualization
6. Optimization tools
"""

import sys
import os
# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from pyroxa import (
    create_reaction_chain, ChainReactorVisualizer, 
    Thermodynamics, Reaction, WellMixedReactor,
    ReactionMulti, MultiReactor
)

def test_reaction_chain():
    """Test basic reaction chain A -> B -> C."""
    print("=== Testing Reaction Chain A -> B -> C ===")
    
    # Create chain
    species = ['A', 'B', 'C']
    rate_constants = [2.0, 1.0]
    
    chain = create_reaction_chain(species, rate_constants)
    print(f"? Created chain with {chain.n_species} species and {chain.n_reactions} reactions")
    
    # Create and run reactor
    reactor = chain.create_reactor(conc0=[1.0, 0.0, 0.0])
    times, traj = reactor.run(time_span=3.0, time_step=0.05)
    
    print(f"? Simulation completed: {len(times)} time points")
    print(f"  Initial: {traj[0]}")
    print(f"  Final: {traj[-1]}")
    
    # Test analytical solution
    times_np = np.array(times)
    analytical = chain.get_analytical_solution(times_np, C0=1.0)
    print(f"? Analytical solution computed: shape {analytical.shape}")
    
    # Test kinetic analysis
    concentrations_np = np.array(traj).T
    analysis = chain.analyze_kinetics(times_np, concentrations_np)
    print(f"? Kinetic analysis completed")
    print(f"  Conversion of A: {analysis['conversion']['A']:.1%}")
    print(f"  Max B concentration: {analysis['max_concentrations']['B']:.3f}")
    
    return True

def test_complex_network():
    """Test complex branching network."""
    print("\n=== Testing Complex Branching Network ===")
    
    # Create branching network: A -> B -> C
    #                             ?
    #                             D
    species = ['A', 'B', 'C', 'D']
    reactions = [
        ReactionMulti(kf=2.0, kr=0.0, reactants={0: 1}, products={1: 1}),  # A -> B
        ReactionMulti(kf=1.0, kr=0.0, reactants={1: 1}, products={2: 1}),  # B -> C
        ReactionMulti(kf=0.8, kr=0.0, reactants={1: 1}, products={3: 1})   # B -> D
    ]
    
    thermo = Thermodynamics()
    reactor = MultiReactor(
        thermo=thermo,
        reactions=reactions,
        species=species,
        T=300.0,
        conc0=[1.0, 0.0, 0.0, 0.0]
    )
    
    times, traj = reactor.run(time_span=3.0, time_step=0.02)
    
    print(f"? Complex network simulation completed")
    print(f"  Initial: {traj[0]}")
    print(f"  Final: {traj[-1]}")
    
    # Calculate selectivities
    final_C = traj[-1][2]
    final_D = traj[-1][3]
    total_products = final_C + final_D
    
    if total_products > 0:
        sel_C = final_C / total_products
        sel_D = final_D / total_products
        print(f"  Selectivity to C: {sel_C:.1%}")
        print(f"  Selectivity to D: {sel_D:.1%}")
    
    return True

def test_enhanced_plotting():
    """Test enhanced plotting capabilities."""
    print("\n=== Testing Enhanced Plotting ===")
    
    # Create data for plotting
    species = ['A', 'B', 'C', 'D']
    times = np.linspace(0, 4, 50)
    
    # Simulate concentrations for a 4-step chain
    A = np.exp(-2*times)
    B = 2*(np.exp(-2*times) - np.exp(-1.5*times))
    C = 1.5*(np.exp(-1.5*times) - np.exp(-1*times)) + 0.5*(np.exp(-2*times) - np.exp(-1.5*times))
    D = 1 - A - B - C
    
    concentrations = np.array([A, B, C, D])
    
    try:
        # Test plotting function
        fig = ChainReactorVisualizer.plot_concentration_profiles(
            times, concentrations, species,
            title="Enhanced Multi-Reaction Test",
            save_path='examples/test_enhanced_plotting.png'
        )
        plt.close(fig)  # Close to avoid display issues
        print("? Enhanced plotting completed successfully")
        return True
    except Exception as e:
        print(f"? Plotting failed: {e}")
        return False

def test_optimization_features():
    """Test optimization and analysis features."""
    print("\n=== Testing Optimization Features ===")
    
    try:
        from pyroxa import OptimalReactorDesign
        
        # Create simple A -> B -> C chain
        species = ['A', 'B', 'C']
        rate_constants = [2.0, 1.0]
        chain = create_reaction_chain(species, rate_constants)
        
        # Test temperature optimization
        temp_result = OptimalReactorDesign.find_optimal_temperature(
            chain, temp_range=(250, 350), target_species='B', 
            target_time=2.0, n_temps=5  # Small number for fast testing
        )
        
        optimal_T = temp_result['optimal_temperature']
        optimal_conc = temp_result['optimal_concentration']
        
        print(f"? Temperature optimization completed")
        print(f"  Optimal temperature: {optimal_T:.1f} K")
        print(f"  Maximum B concentration: {optimal_conc:.3f}")
        
        # Test residence time analysis
        flow_rates = [0.5, 1.0, 2.0]
        residence_result = OptimalReactorDesign.residence_time_distribution(
            chain, flow_rates, reactor_volume=1.0
        )
        
        print(f"? Residence time analysis completed")
        print(f"  Tested {len(flow_rates)} different flow rates")
        
        return True
        
    except Exception as e:
        print(f"? Optimization features failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Comprehensive Multi-Reaction Features Test")
    print("=" * 50)
    
    tests = [
        test_reaction_chain,
        test_complex_network,
        test_enhanced_plotting,
        test_optimization_features
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"? Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"? Passed: {sum(results)}/{len(results)} tests")
    
    if all(results):
        print("?? All enhanced multi-reaction features working correctly!")
    else:
        print("??  Some tests failed - check output above")
    
    return all(results)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)