# PyroXa Test Suite

Comprehensive test suite for PyroXa chemical kinetics library.

## 🧪 Test Organization

### Core Functionality Tests
- **[test_basic_functions.py](test_basic_functions.py)** - Basic function testing
- **[test_enhanced_core.py](test_enhanced_core.py)** - Core functionality tests
- **[test_pyroxa_functionality.py](test_pyroxa_functionality.py)** - Main functionality validation
- **[test_function_availability.py](test_function_availability.py)** - Function availability checks

### Reactor System Tests
- **[test_adaptive.py](test_adaptive.py)** - Adaptive integration methods
- **[test_advanced_reactors.py](test_advanced_reactors.py)** - Advanced reactor types
- **[test_cstr_pfr.py](test_cstr_pfr.py)** - CSTR and PFR reactor tests
- **[test_multi_reactor.py](test_multi_reactor.py)** - Multi-reactor system tests
- **[test_reactor_network.py](test_reactor_network.py)** - Reactor network tests

### Specialized Tests
- **[test_equilibrium.py](test_equilibrium.py)** - Chemical equilibrium calculations
- **[test_matplotlib.py](test_matplotlib.py)** - Plotting and visualization
- **[test_new_functions.py](test_new_functions.py)** - Recently added features
- **[test_new_thermodynamic_functions.py](test_new_thermodynamic_functions.py)** - Thermodynamic properties

### Integration Tests
- **[test_all_enhanced_features.py](test_all_enhanced_features.py)** - Enhanced features validation
- **[test_all_new_functions.py](test_all_new_functions.py)** - New functions comprehensive test
- **[test_complex_interfaces.py](test_complex_interfaces.py)** - Complex interface testing

### Performance Tests
- **[test_benchmark.py](test_benchmark.py)** - Performance benchmarking
- **[test_cpp_success.py](test_cpp_success.py)** - C++ extension validation
- **[test_maximum_capability.py](test_maximum_capability.py)** - Maximum capability assessment

### Quick Tests
- **[quick_test.py](quick_test.py)** - Fast functionality verification
- **[simple_reactor_test.py](simple_reactor_test.py)** - Simple reactor validation
- **[simple_test_runner.py](simple_test_runner.py)** - Basic test runner

### Configuration
- **[capability_test_configs.yaml](capability_test_configs.yaml)** - Test configuration settings
- **[__init__.py](__init__.py)** - Test package initialization

## 🚀 Running Tests

### Quick Verification
```bash
# Fast functionality check
python tests/quick_test.py

# Simple reactor test
python tests/simple_reactor_test.py
```

### Comprehensive Testing
```bash
# Run all tests with pytest
python -m pytest tests/ -v

# Run specific test category
python -m pytest tests/test_basic_functions.py -v
```

### Performance Testing
```bash
# Benchmark performance
python tests/test_benchmark.py

# Test C++ extensions
python tests/test_cpp_success.py

# Maximum capability assessment
python tests/test_maximum_capability.py
```

### Individual Test Modules
```bash
# Test specific functionality
python tests/test_enhanced_core.py
python tests/test_advanced_reactors.py
python tests/test_equilibrium.py
```

## 📊 Test Categories

### 1. Unit Tests
- Individual function validation
- Parameter boundary testing
- Error handling verification

### 2. Integration Tests
- Multi-component system testing
- Reactor network validation
- End-to-end workflow testing

### 3. Performance Tests
- Execution speed benchmarks
- Memory usage validation
- Scalability assessment

### 4. Compatibility Tests
- Python version compatibility
- C++ extension functionality
- Cross-platform validation

## 🔧 Test Configuration

### Environment Setup
```bash
# Install test dependencies
pip install pytest pytest-cov

# Set up test environment
export PYTHONPATH="${PYTHONPATH}:."
```

### Test Data
Test configurations are defined in:
- `capability_test_configs.yaml` - Test parameters and settings
- Individual test files - Specific test cases and data

## 📈 Test Coverage

The test suite covers:
- **89+ Functions** - Complete PyroXa function library
- **All Reactor Types** - CSTR, PFR, Batch, Network reactors
- **Thermodynamics** - Property calculations and correlations
- **Kinetics** - Reaction rate calculations and modeling
- **Numerical Methods** - Integration and optimization algorithms

## 🐛 Debugging Tests

### Running with Debug Output
```bash
# Verbose output
python tests/test_enhanced_core.py --verbose

# Debug mode
python -m pytest tests/ -v -s --tb=long
```

### Test-Specific Debugging
```bash
# Test specific reactor type
python tests/test_cstr_pfr.py

# Test thermodynamic functions
python tests/test_new_thermodynamic_functions.py
```

## 📝 Test Reports

Test execution generates:
- Console output with pass/fail status
- Performance timing information
- Error details and stack traces
- Coverage reports (when using pytest-cov)

## 🔄 Continuous Testing

For development:
1. Run `quick_test.py` for fast validation
2. Use `simple_test_runner.py` for basic checks
3. Execute full test suite before commits
4. Review performance tests for optimization

---

*Start with `quick_test.py` for immediate validation of PyroXa installation and basic functionality.*
