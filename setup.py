from setuptools import setup, find_packages, Extension
import os

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
if os.path.exists(os.path.join('simplecantera', 'pybindings.cpp')):
    # Use the generated C++ file
    sources = ['simplecantera/pybindings.cpp', 'src/core.cpp']
    ext = Extension(
        'simplecantera._pybindings',
        sources=sources,
        language='c++',
        include_dirs=include_dirs + [os.path.abspath('simplecantera')],
        library_dirs=library_dirs,
        libraries=[py_lib_name] if py_lib_name else [],
    )
    extensions = [ext]
else:
    # Fall back to Cython -> .pyx
    try:
        from Cython.Build import cythonize
        sources = ['simplecantera/pybindings.pyx', 'src/core.cpp']
        ext = Extension(
            'simplecantera._pybindings',
            sources=sources,
            language='c++',
            include_dirs=include_dirs + [os.path.abspath('simplecantera')],
            library_dirs=library_dirs,
            libraries=[py_lib_name] if py_lib_name else [],
        )
        extensions = cythonize([ext])
    except Exception:
        # If Cython isn't available and no generated C++ exists, don't build
        # a compiled extension (fall back to pure-Python package).
        extensions = []

setup(
    name='simplecantera',
    version='0.1.0',
    packages=find_packages(),
    install_requires=install_requires,
    description='Minimal Cantera-inspired MVP: reversible A<=>B reactor',
    ext_modules=extensions,
)
