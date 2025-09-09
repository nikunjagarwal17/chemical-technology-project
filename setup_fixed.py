"""
Fixed setup script for Python 3.13 free-threaded C++ extension building
"""

# Import the patch first to fix library configuration
import patch_python_config

from setuptools import setup, find_packages, Extension
import os
import sys

# Minimal runtime requirements
install_requires = ['numpy']

# Base include dirs: our C headers live in src/
include_dirs = [os.path.abspath('src')]

# Try to add numpy include directory
try:
    import numpy
    include_dirs.append(numpy.get_include())
except Exception:
    pass

# Helper to pick C++ standard flags across platforms
if sys.platform == 'win32':
    extra_compile_args = ['/std:c++17']
    extra_link_args = []  # Remove the force unresolved flag for now
    libraries = []
    library_dirs = [os.path.abspath('.')]  # Use local directory for python313t.lib
else:
    extra_compile_args = ['-std=gnu++14']
    extra_link_args = []
    libraries = []
    library_dirs = []

# Build extension with patched configuration
extensions = []
try:
    from Cython.Build import cythonize
    sources = ['pyroxa/pybindings.pyx', 'src/core.cpp']
    ext = Extension(
        'pyroxa._pybindings',
        sources=sources,
        language='c++',
        include_dirs=include_dirs + [os.path.abspath('pyroxa')],
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )
    extensions = cythonize([ext], compiler_directives={
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "embedsignature": True
    })
    print(f"✅ C++ extension configured with {len(extensions)} modules")
except Exception as e:
    print(f"❌ C++ extension setup failed: {e}")
    extensions = []

setup(
    name='pyroxa',
    version='0.1.0',
    packages=find_packages(),
    install_requires=install_requires,
    description='Chemical kinetics library with C++ acceleration',
    ext_modules=extensions,
)
