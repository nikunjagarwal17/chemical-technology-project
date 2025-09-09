# PyroXa Project Guide - Complete Architecture & Structure Analysis

## Table of Contents

1. [Project Overview](#project-overview)
2. [Directory Structure Analysis](#directory-structure-analysis)
3. [Modular Architecture](#modular-architecture)
4. [Layered System Design](#layered-system-design)
5. [Core Components Deep Dive](#core-components-deep-dive)
6. [File-by-File Analysis](#file-by-file-analysis)
7. [Data Flow & Interaction Patterns](#data-flow--interaction-patterns)
8. [Build System & Compilation](#build-system--compilation)
9. [Testing Framework](#testing-framework)
10. [Configuration & Setup](#configuration--setup)
11. [Development Workflow](#development-workflow)
12. [Key Code Snippets to Remember](#key-code-snippets-to-remember)
13. [Troubleshooting Project Issues](#troubleshooting-project-issues)

---

## Project Overview

PyroXa is a sophisticated chemical kinetics and reactor simulation library with a **dual-architecture design**:

### Project Mission
- **Educational**: Teach chemical kinetics concepts with clear Python code
- **Professional**: Provide industrial-grade performance with C++ extensions
- **Research**: Enable advanced reactor modeling and optimization

### Key Characteristics
- **Language**: Python with C++ extensions via Cython
- **Domain**: Chemical engineering, reaction kinetics, reactor design
- **Architecture**: Layered (Pure Python → Cython Bindings → C++ Core)
- **Deployment**: Pip-installable package with optional C++ compilation

---

## Directory Structure Analysis

```
project/
├── 📁 pyroxa/                          # Main package directory
│   ├── __init__.py                     # Package initialization & exports
│   ├── pybindings.pyx                  # Cython bindings to C++
│   ├── purepy.py                       # Pure Python implementations
│   ├── new_functions.py                # Additional Python functions
│   ├── reaction_chains.py              # Reaction network modeling
│   └── io.py                           # Input/output utilities
│
├── 📁 src/                             # C++ source code
│   ├── core.h                          # C++ function declarations
│   ├── core.cpp                        # C++ function implementations
│   └── thermodynamics.hpp              # Additional C++ headers
│
├── 📁 tests/                           # Test suite
│   ├── test_*.py                       # Various test files
│   └── comprehensive_*.py              # Integration tests
│
├── 📁 examples/                        # Example scripts
│   ├── *.py                           # Usage examples
│   └── *.yaml                         # Configuration examples
│
├── 📁 docs/                           # Documentation
│   ├── conf.py                        # Sphinx configuration
│   ├── index.rst                      # Documentation index
│   └── *.md                          # Markdown documentation
│
├── 📁 build/                          # Build artifacts (generated)
│   ├── lib.win-amd64-cpython-313/     # Compiled extensions
│   └── temp.win-amd64-cpython-313/    # Temporary build files
│
├── 📁 __pycache__/                    # Python bytecode cache
├── 📁 pyroxa.egg-info/                # Package metadata
│
├── setup.py                          # Main build script
├── setup_*.py                        # Alternative build scripts
├── pyproject.toml                     # Modern Python packaging
├── MANIFEST.in                        # Package file inclusion
├── requirements.txt                   # Dependencies
├── README.md                          # Project README
└── *.md                              # Various documentation files
```

### Directory Purposes

| Directory   | Purpose                   | Key Files                                    |
|-------------|---------------------------|----------------------------------------------|
| `pyroxa/`   | Main Python package       | `__init__.py`, `purepy.py`, `pybindings.pyx` |
| `src/`      | C++ high-performance core | `core.h`, `core.cpp`                         |
| `tests/`    | Quality assurance         | `test_*.py`, verification scripts            |
| `examples/` | Usage demonstrations      | Tutorial scripts, YAML configs               |
| `docs/`     | Documentation system      | Sphinx docs, user guides                     |
| `build/`    | Compilation output        | Generated during `setup.py build`            |

---

## Modular Architecture

### 1. Core Module Structure

```python
# The PyroXa package follows a modular design:

pyroxa/
├── Core Classes           # Fundamental chemical objects
├── Reactor Simulations   # Various reactor types
├── Pure Python Layer     # Educational/fallback code
├── C++ Extension Layer   # High-performance functions
├── I/O Module            # File handling utilities
└── Visualization Tools   # Plotting and analysis
```

### 2. Module Responsibilities

#### **Core Classes Module** (`purepy.py`)
```python
# Fundamental chemical engineering objects
class Thermodynamics:     # Calculate thermodynamic properties
class Reaction:           # Define chemical reactions
class Reactor:            # Base reactor class
class WellMixedReactor:   # Batch reactor simulation
class CSTR:               # Continuous stirred tank
class PFR:                # Plug flow reactor
```

#### **Extension Functions Module** (`pybindings.pyx`)
```python
# High-performance C++ wrapped functions
py_simulate_packed_bed()       # Complex packed bed simulation
py_simulate_fluidized_bed()    # Fluidized bed modeling
py_monte_carlo_simulation()    # Statistical analysis
py_calculate_energy_balance()  # Energy calculations
```

#### **Utility Functions Module** (`new_functions.py`)
```python
# Additional Python implementations
autocatalytic_rate()           # Specialized kinetics
michaelis_menten_rate()        # Enzyme kinetics
heat_capacity_nasa()           # NASA polynomial thermodynamics
pressure_drop_ergun()          # Transport phenomena
```

#### **Network Modeling Module** (`reaction_chains.py`)
```python
# Advanced reaction network analysis
class ReactionChain:           # Model reaction sequences
class ChainReactorVisualizer:  # Visualize networks
class OptimalReactorDesign:    # Optimization tools
```

---

## Layered System Design

### Layer 1: User Interface (Top Layer)
```python
# High-level, user-friendly functions
import pyroxa

# Simple interface - abstracts complexity
reactor = pyroxa.WellMixedReactor(reaction, A0=1.0, B0=0.0)
times, trajectory = reactor.run(10.0, 0.1)

# Dictionary-based interface
spec = {'reaction': {'kf': 1.0, 'kr': 0.5}, ...}
reactor, params = pyroxa.build_from_dict(spec)
```

### Layer 2: Python Implementation (Middle Layer)
```python
# Pure Python - educational and portable
class WellMixedReactor:
    def run(self, time_span, dt):
        """Pure Python ODE integration"""
        times = []
        concentrations = []
        
        # Euler integration (simple but educational)
        for t in np.arange(0, time_span, dt):
            rate = self.reaction.rate(current_conc)
            # Update concentrations...
        
        return times, concentrations
```

### Layer 3: C++ Extensions (Bottom Layer)
```cpp
// High-performance C++ core
extern "C" {
    void simulate_packed_bed(
        int n_components, int n_points,
        double reactor_length, double particle_diameter,
        // ... 20+ parameters for industrial-grade simulation
        double* times_out, double* conc_out, double* temp_out
    ) {
        // Advanced numerical methods
        // Heat/mass transfer correlations  
        // Industrial-grade accuracy
    }
}
```

### Layer Architecture Benefits

| Layer | Language | Purpose | Benefits |
|-------|----------|---------|----------|
| **User Interface** | Python | Ease of use | Simple API, quick prototyping |
| **Python Core** | Python | Education & portability | Readable code, platform independent |
| **C++ Extensions** | C++/Cython | Performance | Industrial speed, complex algorithms |

---

## Core Components Deep Dive

### 1. Package Initialization (`__init__.py`)

**Purpose**: Central import hub that manages the dual-architecture system

**Key Mechanisms**:
```python
try:
    # Try to import C++ extensions first (best performance)
    from . import _pybindings as _bind
    print("✓ C++ extension loaded successfully")
    
    # Import high-performance functions
    simulate_packed_bed = _bind.py_simulate_packed_bed
    monte_carlo_simulation = _bind.py_monte_carlo_simulation
    # ... 80+ more functions
    
    _COMPILED_AVAILABLE = True
    
except (ImportError, MemoryError, OSError) as e:
    # Graceful fallback to pure Python
    print(f"⚠ C++ extension failed ({e})")
    print("✓ Falling back to pure Python implementation...")
    
    # Import Python alternatives
    from .purepy import WellMixedReactor, CSTR, PFR
    from .new_functions import autocatalytic_rate, michaelis_menten_rate
    
    _COMPILED_AVAILABLE = False
```

**What This Does**:
- **Attempts C++ first**: Maximum performance when available
- **Graceful degradation**: Falls back to Python if C++ fails
- **Transparent to user**: Same API regardless of backend
- **Error handling**: Provides helpful error messages

### 2. Pure Python Core (`purepy.py`)

**Purpose**: Educational implementation and guaranteed fallback

**Key Classes Architecture**:

```python
class Thermodynamics:
    """Handles thermodynamic property calculations"""
    def __init__(self, cp=29.1, T_ref=298.15):
        self.cp = cp              # Heat capacity
        self.T_ref = T_ref        # Reference temperature
    
    def enthalpy(self, T):
        """H = cp * T (simplified ideal gas)"""
        return self.cp * T
    
    def entropy(self, T):
        """S = cp * ln(T/T_ref) (simplified)"""
        return self.cp * log(T / self.T_ref)

class Reaction:
    """Defines A ⇌ B reversible reaction kinetics"""
    def __init__(self, kf, kr):
        self.kf = kf              # Forward rate constant
        self.kr = kr              # Reverse rate constant
    
    def rate(self, A, B):
        """Rate = kf*[A] - kr*[B] (mass action)"""
        return self.kf * A - self.kr * B

class WellMixedReactor:
    """Perfect mixing batch reactor"""
    def __init__(self, reaction, A0, B0, temperature=298.15):
        self.reaction = reaction
        self.A = A0               # Initial concentration A
        self.B = B0               # Initial concentration B
        self.T = temperature
    
    def run(self, time_span, dt):
        """Integrate ODEs using simple Euler method"""
        times = []
        trajectory = []
        
        A, B = self.A, self.B
        
        for t in np.arange(0, time_span + dt, dt):
            # Store current state
            times.append(t)
            trajectory.append({'A': A, 'B': B, 'T': self.T})
            
            # Calculate reaction rate
            rate = self.reaction.rate(A, B)
            
            # Update concentrations (Euler integration)
            A += -rate * dt       # A decreases
            B += rate * dt        # B increases
        
        return times, trajectory
```

### 3. Cython Bindings (`pybindings.pyx`)

**Purpose**: Bridge between Python and C++ for performance-critical functions

**Architecture Pattern**:
```cython
# Cython declarations for C++ functions
cdef extern from "core.h":
    void simulate_packed_bed_cpp(
        int n_components, int n_points,
        double reactor_length, double particle_diameter,
        # ... many parameters
        double* times_out, double* conc_out_flat, double* temp_out
    )

# Python-callable wrapper
def py_simulate_packed_bed(
    int n_components, int n_points,
    double reactor_length, double particle_diameter,
    # ... expose all C++ parameters to Python
):
    """
    Simulate packed bed reactor with industrial-grade accuracy.
    
    This function exposes the full complexity of the C++ implementation,
    allowing users to specify detailed reactor parameters for professional
    chemical engineering applications.
    """
    
    # Allocate output arrays
    cdef double* times = <double*>malloc(n_points * sizeof(double))
    cdef double* conc_out_flat = <double*>malloc(n_points * n_components * sizeof(double))
    cdef double* temp_out = <double*>malloc(n_points * sizeof(double))
    
    try:
        # Call C++ function with full parameter set
        simulate_packed_bed_cpp(
            n_components, n_points, reactor_length, particle_diameter,
            # ... pass all 20+ parameters to C++
            times, conc_out_flat, temp_out
        )
        
        # Convert C arrays to Python lists
        py_times = [times[i] for i in range(n_points)]
        py_concentrations = []
        py_temperatures = [temp_out[i] for i in range(n_points)]
        
        # Reshape concentration data
        for i in range(n_points):
            conc_point = []
            for j in range(n_components):
                conc_point.append(conc_out_flat[i * n_components + j])
            py_concentrations.append(conc_point)
        
        return py_times, py_concentrations, py_temperatures
        
    finally:
        # Always free allocated memory
        free(times)
        free(conc_out_flat)
        free(temp_out)
```

### 4. C++ Performance Core (`src/core.cpp`)

**Purpose**: Industrial-grade numerical algorithms

**Implementation Approach**:
```cpp
#include "core.h"
#include <vector>
#include <cmath>

void simulate_packed_bed_cpp(
    int n_components, int n_points,
    double reactor_length, double particle_diameter,
    double bed_porosity, double fluid_density,
    // ... 20+ parameters for complete industrial model
    double* times_out, double* conc_out_flat, double* temp_out
) {
    // Professional-grade packed bed simulation
    
    // Initialize spatial discretization
    double dz = reactor_length / (n_points - 1);
    std::vector<double> z_positions(n_points);
    for (int i = 0; i < n_points; i++) {
        z_positions[i] = i * dz;
    }
    
    // Calculate transport properties
    double Reynolds = fluid_density * superficial_velocity * particle_diameter / fluid_viscosity;
    double Schmidt = fluid_viscosity / (fluid_density * molecular_diffusivity);
    double Sherwood = 2.0 + 0.6 * pow(Reynolds, 0.5) * pow(Schmidt, 0.33);
    
    // Mass transfer coefficient
    double kf = Sherwood * molecular_diffusivity / particle_diameter;
    
    // Heat transfer calculations
    double Prandtl = fluid_viscosity * heat_capacity / thermal_conductivity;
    double Nusselt = 2.0 + 0.6 * pow(Reynolds, 0.5) * pow(Prandtl, 0.33);
    double h = Nusselt * thermal_conductivity / particle_diameter;
    
    // Pressure drop (Ergun equation)
    double pressure_drop_per_length = 
        150.0 * pow(1.0 - bed_porosity, 2) * fluid_viscosity * superficial_velocity /
        (pow(bed_porosity, 3) * pow(particle_diameter, 2)) +
        1.75 * (1.0 - bed_porosity) * fluid_density * pow(superficial_velocity, 2) /
        (pow(bed_porosity, 3) * particle_diameter);
    
    // Time integration loop
    double dt = 0.1;  // Time step
    int n_time_steps = 1000;
    
    for (int t_step = 0; t_step < n_time_steps; t_step++) {
        times_out[t_step] = t_step * dt;
        
        // Spatial loop along reactor
        for (int i = 0; i < n_points; i++) {
            // Component loop
            for (int comp = 0; comp < n_components; comp++) {
                // Complex reaction-diffusion-convection equations
                // Industrial-grade numerical methods
                // Heat and mass transfer coupling
                
                int idx = t_step * n_components + comp;
                conc_out_flat[idx] = /* calculated concentration */;
            }
            
            // Temperature calculation with heat effects
            temp_out[t_step] = /* calculated temperature */;
        }
    }
}
```

---

## File-by-File Analysis

### Primary Source Files

#### 1. `pyroxa/__init__.py` (522 lines)
**Role**: Package orchestrator and API gateway

**Key Sections**:
```python
# Lines 1-50: C++ extension loading with error handling
try:
    from . import _pybindings as _bind
    # Import 65+ C++ functions
except ImportError:
    # Fall back to Python

# Lines 51-150: Function imports and aliases
autocatalytic_rate = _bind.py_autocatalytic_rate
simulate_packed_bed = _bind.py_simulate_packed_bed
# ... 80+ more function imports

# Lines 151-250: __all__ list (public API definition)
__all__ = [
    "Thermodynamics", "Reaction", "WellMixedReactor",
    "autocatalytic_rate", "simulate_packed_bed",
    # ... complete public API
]

# Lines 251-522: Pure Python fallbacks and error handling
```

**What to Remember**:
- **Dual import strategy**: C++ first, Python fallback
- **Complete API exposure**: All 88 functions available
- **Error resilience**: Graceful handling of compilation failures

#### 2. `pyroxa/purepy.py` (1555 lines)
**Role**: Pure Python reference implementation

**Key Classes**:
```python
# Lines 1-100: Imports, exceptions, and base classes
class PyroXaError(Exception): pass
class ThermodynamicsError(PyroXaError): pass
class ReactionError(PyroXaError): pass

# Lines 101-300: Core thermodynamics
class Thermodynamics:
    def __init__(self, cp=29.1, T_ref=298.15)
    def enthalpy(self, T)
    def entropy(self, T)

# Lines 301-500: Reaction kinetics
class Reaction:
    def __init__(self, kf, kr, validate=True)
    def rate(self, A, B)
    def equilibrium_constant(self, T)

# Lines 501-800: Well-mixed reactor
class WellMixedReactor:
    def run(self, time_span, dt)
    def steady_state(self)

# Lines 801-1100: Continuous reactors
class CSTR: # Continuous stirred tank
class PFR:  # Plug flow reactor

# Lines 1101-1555: Advanced reactors and utilities
class ReactorNetwork:
class PackedBedReactor:
class FluidizedBedReactor:
```

**What to Remember**:
- **Educational focus**: Clear, readable implementations
- **Complete functionality**: All reactor types available
- **Robust validation**: Input checking and error handling

#### 3. `pyroxa/pybindings.pyx` (Cython interface)
**Role**: High-performance C++ bridge

**Structure Pattern**:
```cython
# C++ function declarations
cdef extern from "core.h":
    void cpp_function(params...)

# Python wrapper functions
def py_function_name(python_params):
    # Parameter validation
    # Memory allocation
    # C++ function call
    # Result conversion
    # Memory cleanup
    return python_results
```

**Critical Functions**:
- `py_simulate_packed_bed()`: 21-parameter industrial reactor
- `py_simulate_fluidized_bed()`: 20-parameter fluidized bed  
- `py_monte_carlo_simulation()`: 17-parameter uncertainty analysis
- `py_calculate_energy_balance()`: 7-parameter energy calculations

#### 4. `src/core.h` & `src/core.cpp`
**Role**: High-performance computational core

**Function Categories**:
```cpp
// Thermodynamic functions
double gibbs_free_energy(double H, double S, double T);
double equilibrium_constant(double delta_G, double T);

// Kinetic functions  
double arrhenius_rate(double A, double Ea, double T);
double autocatalytic_rate(double k, double A, double B);

// Complex reactor simulations
void simulate_packed_bed_cpp(/* 24 parameters */);
void simulate_fluidized_bed_cpp(/* 24 parameters */);

// Statistical and optimization
void monte_carlo_simulation_cpp(/* 18 parameters */);
double calculate_sensitivity_cpp(/* parameters */);
```

### Build and Configuration Files

#### 5. `setup.py` (Main build script)
**Purpose**: Compile C++ extensions and install package

**Key Sections**:
```python
from Cython.Build import cythonize
from pybind11.setup_helpers import Pybind11Extension

# Define C++ extension
extensions = [
    Pybind11Extension(
        "pyroxa._pybindings",
        sources=["pyroxa/pybindings.pyx", "src/core.cpp"],
        include_dirs=["src/"],
        language="c++",
        cxx_std=14,
    )
]

setup(
    name="pyroxa",
    ext_modules=cythonize(extensions),
    # ... package metadata
)
```

#### 6. `pyproject.toml` (Modern Python packaging)
**Purpose**: Modern packaging configuration

```toml
[build-system]
requires = ["setuptools>=45", "wheel", "Cython>=0.29", "pybind11>=2.6"]
build-backend = "setuptools.build_meta"

[project]
name = "pyroxa"
version = "1.0.0"
description = "Chemical kinetics and reactor simulation library"
dependencies = ["numpy>=1.19", "scipy>=1.6", "matplotlib>=3.3"]
```

### Testing Framework

#### 7. Test Files Structure
```
tests/
├── test_basic_functions.py           # Basic function tests
├── test_complex_interfaces.py        # Complex parameter interfaces
├── test_all_68_functions.py         # Comprehensive function tests
├── comprehensive_test.py            # Integration tests
├── final_verification_test.py       # Final validation
└── verify_pipeline.py              # Build pipeline verification
```

**Testing Strategy**:
```python
# Pattern used across all test files
def test_function_name():
    """Test specific functionality"""
    # Arrange: Set up test data
    input_params = {...}
    expected_result = ...
    
    # Act: Call function
    result = pyroxa.function_name(**input_params)
    
    # Assert: Verify results
    assert result is not None
    assert abs(result - expected_result) < tolerance
    print(f"✅ {function_name} test passed")
```

---

## Data Flow & Interaction Patterns

### 1. User Request Flow

```
User Code
    ↓
pyroxa.__init__.py (API Gateway)
    ↓
Decision: C++ Available?
    ↓                    ↓
Yes: C++ Path           No: Python Path
    ↓                    ↓
pybindings.pyx          purepy.py
    ↓                    ↓
core.cpp               Pure Python Logic
    ↓                    ↓
Return Results ←--------Return Results
    ↓
User Gets Results (Same API regardless of path)
```

### 2. Function Call Pattern

```python
# User calls high-level function
result = pyroxa.simulate_packed_bed(
    n_components=2,
    n_points=20,
    reactor_length=1.0,
    # ... 18 more parameters
)

# Internal flow:
# 1. pyroxa.__init__.py routes to appropriate implementation
# 2. If C++ available: pybindings.py_simulate_packed_bed()
# 3. Cython wrapper validates inputs and converts types
# 4. C++ core.cpp:simulate_packed_bed_cpp() does computation
# 5. Results converted back to Python types
# 6. Memory cleaned up
# 7. Results returned to user
```

### 3. Error Handling Flow

```
Function Call
    ↓
Input Validation (Python layer)
    ↓ (if invalid)
Raise PyroXaError subclass
    ↓ (if valid)
Try C++ Implementation
    ↓ (if C++ fails)
Fallback to Python Implementation
    ↓ (if Python fails)
Raise appropriate exception with helpful message
```

### 4. Memory Management Pattern

```cython
# Pattern used in all Cython functions
def py_function_name(parameters):
    # 1. Allocate C arrays
    cdef double* output_array = <double*>malloc(size * sizeof(double))
    
    try:
        # 2. Call C++ function
        cpp_function(parameters, output_array)
        
        # 3. Convert to Python objects
        python_result = [output_array[i] for i in range(size)]
        
        return python_result
        
    finally:
        # 4. Always clean up memory
        free(output_array)
```

---

## Build System & Compilation

### 1. Build Process Overview

```
Source Files                 Build Process                Output
┌─────────────────┐         ┌──────────────────┐         ┌────────────────┐
│ pyroxa/*.py     │────────▶│ Python Packaging │────────▶│ Pure Python    │
│ pyroxa/*.pyx    │         │                  │         │ Package        │
│ src/*.cpp       │         │ Cython           │         │                │
│ src/*.h         │         │ Compilation      │         │ + C++          │
│ setup.py        │         │                  │         │ Extensions     │
└─────────────────┘         └──────────────────┘         └────────────────┘
```

### 2. Build Commands

```bash
# Development build (in-place)
python setup.py build_ext --inplace

# Production build
python setup.py bdist_wheel

# Install for development
pip install -e .

# Clean build (removes build artifacts)
python setup.py clean --all
```

### 3. Build Configuration

**Key Build Settings** (in `setup.py`):
```python
extensions = [
    Pybind11Extension(
        "pyroxa._pybindings",                    # Extension name
        sources=[
            "pyroxa/pybindings.pyx",             # Cython source
            "src/core.cpp",                      # C++ source
        ],
        include_dirs=["src/"],                   # Header locations
        language="c++",                          # C++ compilation
        cxx_std=14,                             # C++14 standard
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]
```

### 4. Compilation Targets

| Platform | Compiler | Output |
|----------|----------|--------|
| Windows | MSVC (Visual Studio) | `.pyd` files |
| Linux | GCC | `.so` files |
| macOS | Clang | `.so` files |

---

## Testing Framework

### 1. Test Categories

```
Testing Hierarchy:
├── Unit Tests              # Individual functions
├── Integration Tests       # Component interactions  
├── System Tests           # Full workflow tests
├── Performance Tests      # Speed/memory benchmarks
└── Validation Tests       # Scientific accuracy
```

### 2. Test Execution Patterns

```python
# Standard test pattern used throughout
def test_function():
    """Test description"""
    print(f"Testing {function_name}...")
    
    try:
        # Test with valid inputs
        result = pyroxa.function_name(valid_params)
        assert result is not None, "Function should return a result"
        
        # Test boundary conditions
        edge_result = pyroxa.function_name(edge_case_params)
        assert edge_result >= 0, "Result should be non-negative"
        
        # Test error conditions
        with pytest.raises(PyroXaError):
            pyroxa.function_name(invalid_params)
            
        print(f"✅ {function_name} passed all tests")
        return True
        
    except Exception as e:
        print(f"❌ {function_name} failed: {e}")
        return False
```

### 3. Test File Purposes

| Test File | Purpose | Functions Tested |
|-----------|---------|------------------|
| `test_basic_functions.py` | Core functionality | Basic thermodynamics, kinetics |
| `test_complex_interfaces.py` | Complex parameters | Packed bed, fluidized bed simulations |
| `test_all_68_functions.py` | Comprehensive coverage | All available functions |
| `comprehensive_test.py` | Integration testing | Cross-module interactions |
| `final_verification_test.py` | Release validation | Critical path verification |

---

## Configuration & Setup

### 1. Package Configuration Files

#### `pyproject.toml` (Modern Python standard)
```toml
[build-system]
requires = [
    "setuptools>=45",
    "wheel", 
    "Cython>=0.29",
    "pybind11>=2.6",
    "numpy>=1.19"
]

[project]
name = "pyroxa"
dynamic = ["version"]
description = "Chemical kinetics and reactor simulation"
authors = [{name = "PyroXa Team"}]
dependencies = [
    "numpy>=1.19",
    "scipy>=1.6", 
    "matplotlib>=3.3"
]
```

#### `MANIFEST.in` (Package file inclusion)
```
include README.md
include LICENSE
include requirements.txt
recursive-include pyroxa *.py *.pyx
recursive-include src *.cpp *.h *.hpp  
recursive-include examples *.py *.yaml
recursive-include docs *.md *.rst
```

#### `requirements.txt` (Dependencies)
```
numpy>=1.19.0
scipy>=1.6.0
matplotlib>=3.3.0
Cython>=0.29.0
pybind11>=2.6.0
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv pyroxa_env
source pyroxa_env/bin/activate  # Linux/Mac
pyroxa_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyroXa in development mode
pip install -e .

# Verify installation
python -c "import pyroxa; print('PyroXa installed successfully')"
```

---

## Development Workflow

### 1. Development Cycle

```
1. Edit Source Code
   ├── Python: Edit .py files directly
   └── C++: Edit .cpp/.h files + rebuild

2. Build/Compile
   ├── Python only: No compilation needed
   └── C++ changes: python setup.py build_ext --inplace

3. Test Changes
   ├── Unit tests: python -m pytest tests/
   └── Manual testing: python examples/test_script.py

4. Validate
   ├── All tests pass
   └── Documentation updated

5. Commit
   └── Git commit with descriptive message
```

### 2. Common Development Tasks

#### Adding a New Function

**Step 1**: Add to C++ core (if performance-critical)
```cpp
// In src/core.h
double new_function(double param1, double param2);

// In src/core.cpp  
double new_function(double param1, double param2) {
    // Implementation
    return result;
}
```

**Step 2**: Add Cython wrapper
```cython
# In pyroxa/pybindings.pyx
cdef extern from "core.h":
    double new_function(double param1, double param2)

def py_new_function(double param1, double param2):
    return new_function(param1, param2)
```

**Step 3**: Add to Python fallback
```python
# In pyroxa/new_functions.py or purepy.py
def new_function(param1, param2):
    """Pure Python implementation"""
    # Python implementation
    return result
```

**Step 4**: Export in package
```python
# In pyroxa/__init__.py
try:
    new_function = _bind.py_new_function
except AttributeError:
    from .new_functions import new_function

__all__.append("new_function")
```

**Step 5**: Add tests
```python
# In tests/test_new_function.py
def test_new_function():
    result = pyroxa.new_function(1.0, 2.0)
    assert result > 0
```

#### Debugging Build Issues

```bash
# Clean all build artifacts
python setup.py clean --all
rm -rf build/ pyroxa.egg-info/ __pycache__/

# Rebuild with verbose output
python setup.py build_ext --inplace --verbose

# Check for missing dependencies
pip list | grep -E "(Cython|pybind11|numpy)"

# Test import without C++ extensions
python -c "
import sys
sys.modules['pyroxa._pybindings'] = None
import pyroxa
print('Pure Python import successful')
"
```

---

## Key Code Snippets to Remember

### 1. Package Import Pattern
```python
# How PyroXa handles dual architecture
try:
    from . import _pybindings as _bind
    function_name = _bind.py_function_name
    _CPP_AVAILABLE = True
except ImportError:
    from .fallback_module import function_name
    _CPP_AVAILABLE = False
```

### 2. Cython Memory Management
```cython
def cython_function(int n):
    cdef double* data = <double*>malloc(n * sizeof(double))
    try:
        # Use data...
        return python_result
    finally:
        free(data)  # Always clean up
```

### 3. Error Handling Pattern
```python
class PyroXaError(Exception):
    """Base exception for PyroXa"""
    pass

def validated_function(param):
    if param <= 0:
        raise PyroXaError(f"Parameter must be positive, got {param}")
    return computation(param)
```

### 4. Reactor Simulation Pattern
```python
def run_simulation(time_span, dt):
    times = []
    states = []
    
    current_state = initial_state
    for t in np.arange(0, time_span + dt, dt):
        times.append(t)
        states.append(current_state.copy())
        
        # Calculate derivatives
        derivatives = calculate_rates(current_state)
        
        # Update state (Euler integration)
        for i, derivative in enumerate(derivatives):
            current_state[i] += derivative * dt
    
    return times, states
```

### 5. High-Level Interface Pattern
```python
def build_from_dict(spec):
    """Build objects from configuration dictionary"""
    reaction_spec = spec['reaction']
    reaction = Reaction(kf=reaction_spec['kf'], kr=reaction_spec['kr'])
    
    system_type = spec.get('system', 'WellMixed')
    if system_type == 'WellMixed':
        reactor = WellMixedReactor(reaction, **spec['initial'])
    elif system_type == 'CSTR':
        reactor = CSTR(reaction, **spec['cstr_params'])
    
    return reactor, spec['sim']
```

---

## Troubleshooting Project Issues

### 1. Common Build Problems

#### C++ Compilation Errors
```bash
# Problem: Missing compiler
# Solution: Install Visual Studio Build Tools (Windows) or GCC (Linux)

# Problem: Missing headers
# Error: "core.h: No such file or directory"
# Solution: Check src/ directory exists and contains headers

# Problem: Cython errors
# Solution: Update Cython
pip install --upgrade Cython
```

#### Import Errors
```python
# Problem: Cannot import _pybindings
# Symptom: Falls back to pure Python unexpectedly

# Debug steps:
import pyroxa
print(f"C++ available: {pyroxa.is_compiled_available()}")

# If False, check build:
python setup.py build_ext --inplace --verbose
```

### 2. Runtime Issues

#### Memory Errors
```python
# Problem: Segmentation fault in C++ functions
# Cause: Usually parameter validation or memory management

# Debug approach:
1. Test with pure Python first
2. Validate all inputs are correct types
3. Check for negative array sizes
4. Verify malloc/free pairing in Cython
```

#### Performance Issues
```python
# Problem: Simulations are slow
# Solutions:

# 1. Verify C++ extensions are loaded
assert pyroxa.is_compiled_available()

# 2. Use appropriate time steps
times, results = reactor.run(time_span=100, dt=0.1)  # Not 0.001

# 3. For parameter studies, minimize function calls
params = [...]
results = [run_simulation(p) for p in params]  # Not individual calls
```

### 3. Development Issues

#### Test Failures
```bash
# Run specific test
python -m pytest tests/test_basic_functions.py::test_specific_function -v

# Run with debugging
python -m pytest tests/ --pdb  # Drops into debugger on failure

# Check test coverage
pip install pytest-cov
python -m pytest tests/ --cov=pyroxa
```

#### Documentation Issues
```bash
# Generate documentation
cd docs/
make html  # Linux/Mac
.\make.bat html  # Windows

# Check for broken links
sphinx-build -b linkcheck . _build/
```

---

## Project Architecture Summary

### Modular Design Principles

1. **Separation of Concerns**
   - **Thermodynamics**: `Thermodynamics` class
   - **Kinetics**: `Reaction` classes  
   - **Reactors**: Reactor hierarchy
   - **I/O**: Separate `io` module
   - **Visualization**: `reaction_chains` module

2. **Layered Architecture**
   - **User Layer**: Simple, high-level functions
   - **Python Layer**: Educational, portable implementations
   - **C++ Layer**: High-performance, industrial-grade

3. **Dual Implementation Strategy**
   - **Pure Python**: Always available, educational
   - **C++ Extensions**: Optional, high-performance
   - **Transparent Fallback**: Same API regardless

4. **Extensible Design**
   - **Plugin Architecture**: Easy to add new reactor types
   - **Function Registration**: Automatic function discovery
   - **Modular Testing**: Independent test modules

### Key Architectural Decisions

| Decision | Rationale | Benefits |
|----------|-----------|----------|
| **Dual Architecture** | Performance + Portability | Best of both worlds |
| **Cython Bridge** | Python-C++ integration | Type safety + performance |
| **Class Hierarchy** | Object-oriented design | Code reuse, extensibility |
| **Configuration-Driven** | YAML/dict specifications | Reproducible experiments |
| **Comprehensive Testing** | Quality assurance | Reliable, maintainable code |

This project represents a sophisticated example of **multi-language integration**, **layered architecture**, and **professional software engineering** practices in the scientific computing domain.

---

*This guide provides complete understanding of the PyroXa project structure, architecture, and development practices. Use it as a reference for understanding, extending, or maintaining the codebase.*
