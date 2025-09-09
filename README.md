# PyroXa: Advanced Chemical Kinetics and Reactor Simulation Library

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

PyroXa is a high-performance, comprehensive chemical kinetics and reactor simulation library designed for research and industrial applications. Built with both pure Python and optimized C++ backends, PyroXa provides advanced simulation capabilities for complex chemical systems.

## 🚀 Key Features

- **Advanced Reactor Types**: CSTR, PFR, Packed Bed, Fluidized Bed, and Reactor Networks
- **Sophisticated Reaction Kinetics**: Elementary, enzyme, autocatalytic, surface catalysis
- **Advanced Numerical Methods**: Adaptive integration, stiff system solvers, parallel computing
- **Comprehensive Thermodynamics**: Real gas equations, NASA polynomials, phase equilibrium
- **Analysis Tools**: Sensitivity analysis, optimization, statistical validation

## 📦 Quick Installation

```bash
# Clone the repository
git clone https://github.com/nikunjagarwal17/chemical-technology-project.git
cd chemical-technology-project

# Install dependencies
pip install -r requirements.txt

# Build the library
python setup.py build_ext --inplace

# Verify installation
python -c "import pyroxa; print('PyroXa installed successfully!')"
```

## 🎯 Quick Start

```python
import pyroxa

# Create a simple reaction: A → B
reaction = pyroxa.Reaction(kf=2.0, kr=0.5)

# Set up a well-mixed reactor
reactor = pyroxa.WellMixedReactor(reaction, A0=1.0, B0=0.0)

# Run simulation
times, concentrations = reactor.run(time_span=10.0)

print(f"Final concentrations: A={concentrations[-1][0]:.3f}, B={concentrations[-1][1]:.3f}")
```

## 📚 Documentation

Comprehensive documentation is available in the [`docs/`](./docs/) folder:

- **[Installation Guide](./docs/INSTALLATION_GUIDE.md)** - Detailed installation instructions
- **[API Reference](./docs/API_REFERENCE.md)** - Complete API documentation
- **[Complete Documentation](./docs/PYROXA_COMPLETE_DOCUMENTATION.md)** - All 89 functions documented
- **[Project Guide](./docs/PYROXA_PROJECT_GUIDE.md)** - Architecture and development guide

## 🧪 Examples

Check out the [`examples/`](./examples/) folder for:
- Basic reactor simulations
- Complex reaction networks
- Thermodynamic calculations
- Advanced analysis examples

## 🧪 Testing

Run the test suite to verify functionality:

```bash
python -m pytest tests/
# or
python tests/quick_test.py
```

## 🏗️ Project Structure

```
PyroXa/
├── pyroxa/           # Main library source code
├── src/              # C++ extensions source
├── tests/            # Comprehensive test suite
├── examples/         # Example scripts and tutorials
├── docs/             # Complete documentation
├── requirements.txt  # Python dependencies
└── setup.py         # Package setup
```

## 🤝 Contributing

We welcome contributions! Please see the [Project Guide](./docs/PYROXA_PROJECT_GUIDE.md) for:
- Development setup
- Code structure overview
- Contribution guidelines
- Testing procedures

## 📋 Requirements

- **Python**: 3.8+
- **Core Dependencies**: NumPy, SciPy, PyYAML
- **Optional**: Matplotlib (plotting), C++ compiler (performance)

See [`requirements.txt`](./requirements.txt) for complete dependency list.

## 🏆 Features

### Reactor Types (89+ Functions Available)
- Well-Mixed Batch Reactors
- Continuous Stirred Tank Reactors (CSTR)
- Plug Flow Reactors (PFR)
- Packed Bed Reactors
- Fluidized Bed Reactors
- Multi-reactor Networks

### Analysis Capabilities
- Thermodynamic property calculations
- Reaction kinetics modeling
- Heat and mass transfer
- Process optimization
- Statistical validation

## 📞 Support

- **Documentation**: See [`docs/`](./docs/) folder
- **Examples**: See [`examples/`](./examples/) folder
- **Issues**: Please report on GitHub
- **Testing**: Run tests in [`tests/`](./tests/) folder

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Quick Links:**
- [📖 Full Documentation](./docs/)
- [🚀 Installation Guide](./docs/INSTALLATION_GUIDE.md)
- [🔧 API Reference](./docs/API_REFERENCE.md)
- [💡 Examples](./examples/)
- [🧪 Tests](./tests/)
