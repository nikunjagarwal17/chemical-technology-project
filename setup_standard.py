from setuptools import setup, find_packages, Extension
import os
import sys

# Get Python installation paths
python_root = os.path.dirname(sys.executable)
python_libs = os.path.join(python_root, 'libs')
python_include = os.path.join(python_root, 'include')

print(f"Python root: {python_root}")
print(f"Python libs: {python_libs}")
print(f"Python include: {python_include}")

# Minimal runtime requirements
install_requires = ['numpy']

# Base include dirs: our C headers live in src/
include_dirs = [
    os.path.abspath('src'),
    python_include  # Use system Python include directory
]

# Try to add numpy include directory
try:
    import numpy
    include_dirs.append(numpy.get_include())
    print(f"Added numpy include: {numpy.get_include()}")
except Exception:
    print("Numpy not available at setup time")

# Force use of standard (non-free-threaded) Python libraries
if sys.platform == 'win32':
    extra_compile_args = ['/std:c++17']
    
    # IMPORTANT: Use standard python313.lib, NOT python313t.lib
    libraries = ['python313']  # This will link to python313.lib
    library_dirs = [python_libs]  # Use system Python libs directory
    
    # Remove any problematic linker flags
    extra_link_args = []
    
    print(f"Using libraries: {libraries}")
    print(f"Using library_dirs: {library_dirs}")
    
else:
    extra_compile_args = ['-std=gnu++14']
    extra_link_args = []
    libraries = []
    library_dirs = []

# Define the Cython extension
try:
    from Cython.Build import cythonize
    
    extensions = [
        Extension(
            "pyroxa._pybindings",
            sources=[
                "pyroxa/pybindings.pyx", 
                "src/core.cpp"
            ],
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            language="c++"
        )
    ]
    
    ext_modules = cythonize(extensions, compiler_directives={'language_level': 3})
    print("Cython extensions configured successfully")
    
except ImportError:
    print("Cython not available - building without C++ extensions")
    ext_modules = []

setup(
    name="pyroxa",
    version="0.3.0",
    author="Pyroxa Development Team",
    description="Chemical kinetics and reactor simulation library",
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=install_requires,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    include_package_data=True,
    package_data={
        'pyroxa': ['*.pyx', '*.pxd'],
    },
)
