#!/usr/bin/env python3
"""
Quick test of fixed reactor simulations
"""

def test_reactor_simulation():
    """Test that reactor simulations work with fixed rate method"""
    from pyroxa import Reaction, WellMixedReactor, Thermodynamics
    
    # Create simple reaction: A <=> B
    reaction = Reaction(kf=2.0, kr=0.5)
    thermo = Thermodynamics()
    
    # Create reactor
    reactor = WellMixedReactor(thermo, reaction, conc0=[1.0, 0.0])
    
    try:
        # Run simulation
        times, trajectory = reactor.run(time_span=2.0, time_step=0.1)
        print(f"✅ Simulation successful: {len(times)} time points")
        print(f"  Initial concentrations: {trajectory[0]}")
        print(f"  Final concentrations: {trajectory[-1]}")
        return True
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False

def test_reaction_chain():
    """Test reaction chain functionality"""
    from pyroxa import create_reaction_chain
    
    try:
        species = ['A', 'B', 'C']
        rate_constants = [2.0, 1.0]
        
        chain = create_reaction_chain(species, rate_constants)
        print(f"✅ Reaction chain created: {chain.n_species} species, {chain.n_reactions} reactions")
        
        # Test reactor creation
        reactor = chain.create_reactor(conc0=[1.0, 0.0, 0.0])
        times, traj = reactor.run(time_span=1.0, time_step=0.1)
        print(f"✅ Chain reactor simulation: {len(times)} points")
        print(f"  Final concentrations: {traj[-1]}")
        return True
    except Exception as e:
        print(f"❌ Reaction chain test failed: {e}")
        return False

def test_statistical_functions():
    """Test statistical functions"""
    from pyroxa import calculate_r_squared, calculate_rmse, calculate_aic
    
    try:
        experimental = [1.0, 2.0, 3.0, 4.0, 5.0]
        predicted = [1.1, 1.9, 3.1, 3.9, 5.1]
        
        r_squared = calculate_r_squared(experimental, predicted)
        rmse = calculate_rmse(experimental, predicted)
        aic = calculate_aic(experimental, predicted, 2)
        
        print(f"✅ Statistical functions working:")
        print(f"  R²: {r_squared:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  AIC: {aic:.4f}")
        return True
    except Exception as e:
        print(f"❌ Statistical functions failed: {e}")
        return False

def main():
    """Run all quick tests"""
    print("🔧 Testing PyroXa Fixes")
    print("=" * 30)
    
    tests = [
        test_reactor_simulation,
        test_reaction_chain, 
        test_statistical_functions,
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 30)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All critical fixes working!")
    else:
        print("⚠️ Some issues remain")

if __name__ == "__main__":
    main()
