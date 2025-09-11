"""
Simplified setup script for PyroXa with Python 3.13 compatibility fixes
"""
from setuptools import setup, find_packages, Extension
import os
import sys
import numpy

# Simple configuration for Python 3.13
if sys.platform == 'win32':
    extra_compile_args = ['/std:c++17', '/bigobj', '/EHsc']
    extra_link_args = []
else:
    extra_compile_args = ['-std=c++14', '-O2']
    extra_link_args = []

# Create the extension
ext_modules = []
try:
    from Cython.Build import cythonize
    
    ext = Extension(
        'pyroxa._pybindings',
        sources=['pyroxa/pybindings.pyx', 'src/core.cpp'],
        language='c++',
        include_dirs=[
            numpy.get_include(),
            os.path.abspath('src'),
            os.path.abspath('pyroxa')
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
            ('CYTHON_LIMITED_API', '1'),
            ('PY_SSIZE_T_CLEAN', '1')
        ],
    )
    
    # Simplified cythonize with minimal directives
    ext_modules = cythonize([ext], 
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False
        },
        annotate=False
    )
    
except ImportError:
    print("Cython not available, skipping C++ extensions")
    ext_modules = []

setup(
    name='pyroxa',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=['numpy'],
    description='Chemical kinetics simulation library',
)
