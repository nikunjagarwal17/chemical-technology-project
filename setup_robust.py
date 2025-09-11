from setuptools import setup, find_packages, Extension
import os
import sys
import platform
import warnings

# Minimal runtime requirements
install_requires = ['numpy>=1.19.0']

# Base include dirs: our C headers live in src/
include_dirs = [os.path.abspath('src')]

# Try to add numpy include directory if available at setup-time
try:
    import numpy
    include_dirs.append(numpy.get_include())
except Exception:
    # If numpy isn't importable at setup-time, the isolated build will provide
    # headers; leave include_dirs as-is.
    pass

# Platform-specific compiler settings
if sys.platform == 'win32':
    extra_compile_args = ['/std:c++17', '/bigobj']
    extra_link_args = []
    libraries = []
    library_dirs = []
elif sys.platform == 'darwin':  # macOS
    extra_compile_args = ['-std=c++14', '-O2', '-fno-strict-aliasing']
    extra_link_args = []
    libraries = []
    library_dirs = []
else:  # Linux and others
    extra_compile_args = ['-std=c++14', '-O2', '-fno-strict-aliasing', '-Wno-unused-function']
    extra_link_args = []
    libraries = []
    library_dirs = []

# Build extension with error handling
extensions = []

def build_extension():
    """Try to build the C++ extension with proper error handling"""
    try:
        from Cython.Build import cythonize
        
        # Check if source files exist
        pyx_file = os.path.join('pyroxa', 'pybindings.pyx')
        cpp_file = os.path.join('src', 'core.cpp')
        
        if not os.path.exists(pyx_file):
            warnings.warn(f"Cython source file {pyx_file} not found, skipping C++ extension")
            return []
            
        if not os.path.exists(cpp_file):
            warnings.warn(f"C++ source file {cpp_file} not found, skipping C++ extension")
            return []
        
        sources = [pyx_file, cpp_file]
        
        ext = Extension(
            'pyroxa._pybindings',
            sources=sources,
            language='c++',
            include_dirs=include_dirs + [os.path.abspath('pyroxa')],
            library_dirs=library_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            libraries=libraries,
            define_macros=[
                ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
                ('PY_SSIZE_T_CLEAN', '1')
            ],
        )
        
        # Use conservative Cython compiler directives for better compatibility
        return cythonize([ext], compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "embedsignature": True
        }, annotate=False)
        
    except ImportError:
        warnings.warn("Cython not available, trying pre-compiled C++ fallback")
        return build_cpp_fallback()
    except Exception as e:
        warnings.warn(f"Failed to build Cython extension: {e}, trying fallback")
        return build_cpp_fallback()

def build_cpp_fallback():
    """Fallback to pre-compiled C++ if available"""
    try:
        cpp_file = os.path.join('pyroxa', 'pybindings.cpp')
        core_file = os.path.join('src', 'core.cpp')
        
        if os.path.exists(cpp_file) and os.path.exists(core_file):
            sources = [cpp_file, core_file]
            ext = Extension(
                'pyroxa._pybindings',
                sources=sources,
                language='c++',
                include_dirs=include_dirs + [os.path.abspath('pyroxa')],
                library_dirs=library_dirs,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                libraries=libraries,
                define_macros=[
                    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
                    ('PY_SSIZE_T_CLEAN', '1')
                ],
            )
            return [ext]
        else:
            warnings.warn("No C++ source files found, building pure Python version")
            return []
    except Exception as e:
        warnings.warn(f"C++ fallback also failed: {e}, building pure Python version")
        return []

# Try to build extensions
try:
    extensions = build_extension()
except Exception as e:
    warnings.warn(f"Extension building failed completely: {e}")
    extensions = []

# If no extensions could be built, that's okay - pure Python will work
if not extensions:
    print("Building PyroXa in pure Python mode (C++ extensions disabled)")
else:
    print(f"Building PyroXa with {len(extensions)} C++ extension(s)")

setup(
    name='pyroxa',
    version='0.3.0',
    packages=find_packages(),
    install_requires=install_requires,
    description='Chemical kinetics and reactor simulation library inspired by Cantera',
    long_description='A chemical kinetics simulation library with optional C++ acceleration',
    author='PyroXa Development Team',
    python_requires='>=3.8',
    ext_modules=extensions,
    zip_safe=False,  # Required for Cython extensions
    include_package_data=True,
    package_data={
        'pyroxa': ['*.pyx', '*.pxd'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
)
