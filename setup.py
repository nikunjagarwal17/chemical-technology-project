from setuptools import setup, find_packages

# Pure Python PyroXa - No compilation required!
# Runtime requirements
install_requires = ['numpy>=1.19.0']

print("Building PyroXa in pure Python mode - No compilation required!")

setup(
    name='pyroxa',
    version='1.0.0',  # Major version bump for pure Python transition
    packages=find_packages(),
    install_requires=install_requires,
    description='Pure Python chemical kinetics and reactor simulation library',
    long_description='A chemical kinetics simulation library - now pure Python for easy installation!',
    author='PyroXa Development Team',
    python_requires='>=3.8',
    zip_safe=True,  # Pure Python is zip safe
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
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
