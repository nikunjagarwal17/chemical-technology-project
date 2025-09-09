from setuptools import setup, Extension
import numpy

# Simple direct C++ extension without Cython
def create_simple_extension():
    return Extension(
        "pyroxa.simple_core",
        sources=["src/simple_binding.cpp", "src/core.cpp"],
        include_dirs=[
            "src",
            numpy.get_include()
        ],
        extra_compile_args=["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"],
        language="c++"
    )

import sys

# Check if we can build the simple extension
if __name__ == "__main__":
    try:
        ext_modules = [create_simple_extension()]
    except:
        print("Failed to create C++ extension - building without it")
        ext_modules = []
    
    setup(
        name="pyroxa",
        version="0.3.0",
        ext_modules=ext_modules,
        packages=["pyroxa"],
        install_requires=["numpy"],
    )
