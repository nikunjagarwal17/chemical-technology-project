from setuptools import setup, find_packages
import os
import sys

# Minimal runtime requirements
install_requires = ['numpy']

# Pure Python setup - no C++ extensions
setup(
    name='pyroxa',
    version='0.1.0',
    packages=find_packages(),
    install_requires=install_requires,
    description='Chemical kinetics library with advanced reactors - Pure Python version',
    ext_modules=[],  # No C++ extensions
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
)
