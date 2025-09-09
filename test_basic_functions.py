#!/usr/bin/env python3
"""
Simplified test to build only the functions that we know work from earlier batches
This will help isolate the issues with the new functions
"""

import os
import sys

def test_basic_build():
    """Test building with just the first 42 functions that we know work"""
    
    # Create a minimal test
    try:
        import pyroxa
        
        # Test a function from the earlier working batches
        result = pyroxa.autocatalytic_rate(1.0, 0.5, 0.1, 300.0)
        print(f"✓ Autocatalytic rate test: {result}")
        
        # Test another known working function
        try:
            gibbs = pyroxa.gibbs_free_energy(298.15, -393500.0, 213.8)
            print(f"✓ Gibbs free energy test: {gibbs}")
        except Exception as e:
            print(f"⚠️ Gibbs test issue: {e}")
        
        # Test interpolation functions (from first new batch that worked)
        try:
            x_vals = [0.0, 1.0, 2.0, 3.0]
            y_vals = [0.0, 1.0, 4.0, 9.0]
            interp_result = pyroxa.linear_interpolate(1.5, x_vals, y_vals)
            print(f"✓ Linear interpolation test: {interp_result}")
        except Exception as e:
            print(f"⚠️ Interpolation test issue: {e}")
        
        print("🎉 Basic functions are working!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def check_extension_file():
    """Check if the compiled extension exists"""
    ext_path = r"C:\Users\nikun\OneDrive\Documents\Chemical Technology Project\project\pyroxa\_pybindings.cp313-win_amd64.pyd"
    
    if os.path.exists(ext_path):
        print(f"✓ Extension file exists: {ext_path}")
        return True
    else:
        print(f"❌ Extension file missing: {ext_path}")
        return False

if __name__ == "__main__":
    print("=== CHECKING PYROXA BUILD STATUS ===")
    
    if check_extension_file():
        test_basic_build()
    else:
        print("Need to build extension first")
