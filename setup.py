from setuptools import setup, find_packages, Extension
import os
import sys

install_requires = ['numpy']

# Build the compiled extension. Prefer a pre-generated C++ file so builds in
# isolated build environments (cibuildwheel containers) don't require Cython
# to be present at build time. If the generated C++ doesn't exist, fall back
# to using Cython to cythonize the .pyx file.
include_dirs = [os.path.abspath('src')]
try:
    import numpy
    include_dirs.append(numpy.get_include())
except Exception:
    # numpy headers may be provided in the build environment; continue.
    pass

# Try to discover Python's libs dir and release import lib name to help the linker
try:
    import sysconfig, sys
    libdir = sysconfig.get_config_var('LIBDIR') or os.path.join(sys.exec_prefix, 'libs')
    py_lib_name = f'python{sys.version_info.major}{sys.version_info.minor}'
except Exception:
    libdir = os.path.join(sys.exec_prefix, 'libs')
    py_lib_name = None

library_dirs = [os.path.abspath('.')] + ([libdir] if libdir else [])

sources = []
# Prefer to cythonize the .pyx when Cython is available in the build
# environment (PEP 517 isolated builds will install Cython if declared
# in pyproject.toml). If Cython is not available, fall back to using the
# pre-generated .cpp file if present.
try:
    from Cython.Build import cythonize
    sources = ['simplecantera/pybindings.pyx', 'src/core.cpp']
    # Set C++ standard flags so lambdas/auto are supported on all platforms
    if sys.platform == 'win32':
        extra_compile_args = ['/std:c++17']
    else:
        extra_compile_args = ['-std=gnu++14']
    ext = Extension(
        'simplecantera._pybindings',
        sources=sources,
        language='c++',
        include_dirs=include_dirs + [os.path.abspath('simplecantera')],
        extra_compile_args=extra_compile_args,
    )
    extensions = cythonize([ext])
except Exception:
    if os.path.exists(os.path.join('simplecantera', 'pybindings.cpp')):
        # Use the generated C++ file when Cython isn't available
        sources = ['simplecantera/pybindings.cpp', 'src/core.cpp']
        # Set C++ standard flags for fallback path as well
        if sys.platform == 'win32':
            extra_compile_args = ['/std:c++17']
        else:
            extra_compile_args = ['-std=gnu++14']
        ext = Extension(
            'simplecantera._pybindings',
            sources=sources,
            language='c++',
            include_dirs=include_dirs + [os.path.abspath('simplecantera')],
            extra_compile_args=extra_compile_args,
        )
        extensions = [ext]
    else:
        extensions = []

setup(
    name='simplecantera',
    version='0.1.0',
    packages=find_packages(),
    install_requires=install_requires,
    description='Minimal Cantera-inspired MVP: reversible A<=>B reactor',
    ext_modules=extensions,
)
