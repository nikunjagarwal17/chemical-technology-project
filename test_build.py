#!/usr/bin/env python3
"""
Test script to verify that PyroXa can be built as a wheel
"""
import subprocess
import sys
import os
import tempfile
import shutil

def test_wheel_build():
    """Test building PyroXa as a wheel"""
    print("🔨 Testing PyroXa wheel build...")
    
    try:
        # Clean up any existing build artifacts
        for dir_name in ['build', 'dist', 'pyroxa.egg-info']:
            if os.path.exists(dir_name):
                print(f"Cleaning up {dir_name}")
                shutil.rmtree(dir_name)
        
        # Test setup.py build
        print("\n1. Testing setup.py build...")
        result = subprocess.run([sys.executable, 'setup.py', 'build'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ setup.py build succeeded")
        else:
            print("❌ setup.py build failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        
        # Test wheel building
        print("\n2. Testing wheel build...")
        result = subprocess.run([sys.executable, '-m', 'build', '--wheel'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Wheel build succeeded")
            print("Built wheels:")
            if os.path.exists('dist'):
                for f in os.listdir('dist'):
                    if f.endswith('.whl'):
                        print(f"  - {f}")
        else:
            print("❌ Wheel build failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
            # Try pure Python build
            print("\n3. Testing pure Python wheel build...")
            env = os.environ.copy()
            env['PYROXA_PURE_PYTHON'] = '1'
            result = subprocess.run([sys.executable, '-m', 'build', '--wheel'], 
                                  capture_output=True, text=True, env=env)
            if result.returncode == 0:
                print("✅ Pure Python wheel build succeeded")
                return True
            else:
                print("❌ Even pure Python wheel build failed")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_wheel_build()
    if success:
        print("\n🎉 All build tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Build tests failed!")
        sys.exit(1)
