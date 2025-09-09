# PyroXa Chemical Kinetics Library - Complete Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start Guide](#quick-start-guide)
4. [Library Architecture](#library-architecture)
5. [Core Classes](#core-classes)
6. [Complete Function Reference](#complete-function-reference)
   - 6.1 [Thermodynamic Functions (1-7)](#thermodynamic-functions)
   - 6.2 [Kinetics & Rate Functions (8-13)](#kinetics--rate-functions)
   - 6.3 [Reactor Simulation Functions (14-17)](#reactor-simulation-functions)
   - 6.4 [Mathematical & Utility Functions (18-22)](#mathematical--utility-functions)
   - 6.5 [Analysis & Optimization Functions (23-26)](#analysis--optimization-functions)
   - 6.6 [Statistical & Data Functions (27-33)](#statistical--data-functions)
   - 6.7 [Control & Automation Functions (34-36)](#control--automation-functions)
   - 6.8 [I/O & Utility Functions (37-39)](#io--utility-functions)
   - 6.9 [Transport & Physical Properties (40-45)](#transport--physical-properties)
   - 6.10 [Analytical Solutions (46-48)](#analytical-solutions)
   - 6.11 [Validation & Quality Control (49-53)](#validation--quality-control)
   - 6.12 [Reactor Classes & Objects (54-62)](#reactor-classes--objects)
   - 6.13 [Error Handling Classes (63-67)](#error-handling-classes)
   - 6.14 [Advanced Analysis Tools (68-72)](#advanced-analysis-tools)
   - 6.15 [System Information & Utilities (73-88)](#system-information--utilities)
7. [Advanced Usage Examples](#advanced-usage-examples)
8. [Error Handling](#error-handling)
9. [Performance Tips](#performance-tips)
10. [Troubleshooting](#troubleshooting)
11. [Complete Function Summary Table](#complete-function-summary-table)

---

## Introduction

PyroXa is a professional-grade chemical kinetics and reactor simulation library designed for chemical engineers, researchers, and students. It provides a comprehensive suite of tools for:

- **Chemical reaction modeling** with complex kinetics
- **Reactor simulation** (batch, CSTR, PFR, packed bed, fluidized bed)
- **Thermodynamic calculations** using industry-standard correlations
- **Process optimization** and control system design
- **Statistical analysis** and uncertainty quantification

The library features both pure Python implementations (for education and prototyping) and high-performance C++ extensions (for production use).

### Key Features

- ✅ **88 Functions** covering all aspects of chemical kinetics
- ✅ **Dual Architecture**: Pure Python + C++ extensions
- ✅ **Professional Grade**: NASA polynomials, Peng-Robinson EOS, advanced correlations
- ✅ **Beginner Friendly**: Comprehensive examples and clear documentation
- ✅ **Production Ready**: Robust error handling and validation

---

## Installation

### Prerequisites
- Python 3.8+ (Python 3.13+ recommended)
- NumPy
- SciPy (optional, for advanced features)
- Matplotlib (optional, for plotting)

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/your-repo/pyroxa.git
cd pyroxa

# Install dependencies
pip install -r requirements.txt

# Install PyroXa
pip install -e .
```

### With C++ Extensions (Recommended)
```bash
# Ensure you have Visual Studio Build Tools (Windows) or GCC (Linux/Mac)
python setup.py build_ext --inplace
pip install -e .
```

---

## Quick Start Guide

### Your First Simulation

```python
import pyroxa

# Create a simple A ⇌ B reaction
reaction = pyroxa.Reaction(kf=1.0, kr=0.5)  # Forward and reverse rate constants

# Set up a well-mixed reactor
reactor = pyroxa.WellMixedReactor(
    reaction=reaction,
    A0=1.0,  # Initial concentration of A
    B0=0.0,  # Initial concentration of B
    temperature=298.15  # Temperature in Kelvin
)

# Run simulation
times, trajectory = reactor.run(time_span=5.0, dt=0.1)

# Print results
print(f"Final concentrations: A={trajectory[-1]['A']:.3f}, B={trajectory[-1]['B']:.3f}")
```

### Step-by-Step Tutorial for Beginners

#### Tutorial 1: Basic Reaction Simulation

**Goal**: Simulate a simple reversible reaction A ⇌ B and understand equilibrium.

```python
import pyroxa
import matplotlib.pyplot as plt

# Step 1: Define the reaction
# A reversible reaction with forward rate constant 2.0 and reverse 0.5
reaction = pyroxa.Reaction(kf=2.0, kr=0.5)

# Step 2: Create a reactor
# Well-mixed batch reactor starting with 1.0 M of A and no B
reactor = pyroxa.WellMixedReactor(
    reaction=reaction,
    A0=1.0,  # Initial [A] = 1.0 M
    B0=0.0,  # Initial [B] = 0.0 M
    temperature=298.15  # Room temperature
)

# Step 3: Run the simulation
times, trajectory = reactor.run(time_span=10.0, dt=0.1)

# Step 4: Extract concentrations for plotting
time_points = []
conc_A = []
conc_B = []

for i, time in enumerate(times):
    time_points.append(time)
    conc_A.append(trajectory[i]['A'])
    conc_B.append(trajectory[i]['B'])

# Step 5: Plot results
plt.figure(figsize=(10, 6))
plt.plot(time_points, conc_A, 'b-', label='[A]', linewidth=2)
plt.plot(time_points, conc_B, 'r-', label='[B]', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Concentration (M)')
plt.title('Reversible Reaction A ⇌ B')
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Calculate equilibrium constant
final_A = trajectory[-1]['A']
final_B = trajectory[-1]['B']
K_eq = final_B / final_A
theoretical_K = 2.0 / 0.5  # kf/kr
print(f"Experimental K_eq: {K_eq:.2f}")
print(f"Theoretical K_eq: {theoretical_K:.2f}")
```

#### Tutorial 2: Comparing Different Reactors

**Goal**: Compare CSTR, PFR, and Batch reactor performance.

```python
import pyroxa
import numpy as np
import matplotlib.pyplot as plt

# Define reaction: A → B (irreversible)
reaction = pyroxa.Reaction(kf=0.5, kr=0.0)

# 1. Batch Reactor
batch = pyroxa.WellMixedReactor(reaction, A0=1.0, B0=0.0, temperature=298.15)
times_batch, traj_batch = batch.run(time_span=20.0, dt=0.5)

# 2. CSTR (residence time = 10 seconds)
cstr = pyroxa.CSTR(reaction, residence_time=10.0, temperature=298.15)
steady_state_cstr = cstr.steady_state_solve(inlet_concentrations={'A': 1.0, 'B': 0.0})

# 3. PFR (length=2m, velocity=0.1 m/s, so residence time = 20s)
pfr = pyroxa.PFR(reaction, length=2.0, velocity=0.1, temperature=298.15)
result_pfr = pfr.solve_steady_state(inlet_concentrations={'A': 1.0, 'B': 0.0})

# Extract batch data
A_batch = [traj['A'] for traj in traj_batch]
conversion_batch = [(1.0 - A) * 100 for A in A_batch]

# Plot comparison
plt.figure(figsize=(12, 8))

# Batch reactor conversion vs time
plt.subplot(2, 2, 1)
plt.plot(times_batch, conversion_batch, 'b-', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Conversion (%)')
plt.title('Batch Reactor')
plt.grid(True)

# Reactor comparison bar chart
plt.subplot(2, 2, 2)
reactors = ['Batch (20s)', 'CSTR (τ=10s)', 'PFR (τ=20s)']
conversions = [
    conversion_batch[-1],  # Final batch conversion
    (1.0 - steady_state_cstr['A']) * 100,  # CSTR conversion
    (1.0 - result_pfr['A']) * 100  # PFR conversion
]
colors = ['blue', 'red', 'green']
plt.bar(reactors, conversions, color=colors, alpha=0.7)
plt.ylabel('Conversion (%)')
plt.title('Reactor Performance Comparison')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print("Reactor Performance Summary:")
print(f"Batch (20s): {conversion_batch[-1]:.1f}% conversion")
print(f"CSTR (τ=10s): {(1.0 - steady_state_cstr['A']) * 100:.1f}% conversion")
print(f"PFR (τ=20s): {(1.0 - result_pfr['A']) * 100:.1f}% conversion")
```

#### Tutorial 3: Temperature Effects on Reactions

**Goal**: Study how temperature affects reaction rate using Arrhenius equation.

```python
import pyroxa
import numpy as np
import matplotlib.pyplot as plt

# Temperature range (K)
temperatures = np.linspace(280, 350, 8)  # 280K to 350K

# Arrhenius parameters
A_factor = 1e8  # Pre-exponential factor (s⁻¹)
E_activation = 50000  # Activation energy (J/mol)

conversions = []
final_times = []

for T in temperatures:
    # Calculate temperature-dependent rate constant
    k = pyroxa.arrhenius_rate(A=A_factor, Ea=E_activation, temperature=T)
    
    # Create reaction and reactor
    reaction = pyroxa.Reaction(kf=k, kr=0.0)  # Irreversible
    reactor = pyroxa.WellMixedReactor(reaction, A0=1.0, B0=0.0, temperature=T)
    
    # Run simulation
    times, trajectory = reactor.run(time_span=100.0, dt=1.0)
    
    # Calculate final conversion
    final_A = trajectory[-1]['A']
    conversion = (1.0 - final_A) * 100
    conversions.append(conversion)
    
    print(f"T = {T:.0f}K: k = {k:.2e} s⁻¹, Conversion = {conversion:.1f}%")

# Plot results
plt.figure(figsize=(12, 5))

# Conversion vs Temperature
plt.subplot(1, 2, 1)
plt.plot(temperatures, conversions, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Temperature (K)')
plt.ylabel('Conversion after 100s (%)')
plt.title('Temperature Effect on Conversion')
plt.grid(True)

# Arrhenius plot (ln(k) vs 1/T)
rate_constants = [pyroxa.arrhenius_rate(A_factor, E_activation, T) for T in temperatures]
plt.subplot(1, 2, 2)
plt.semilogy(1000/temperatures, rate_constants, 'bo-', linewidth=2, markersize=8)
plt.xlabel('1000/T (K⁻¹)')
plt.ylabel('Rate Constant (s⁻¹)')
plt.title('Arrhenius Plot')
plt.grid(True)

plt.tight_layout()
plt.show()
```

#### Tutorial 4: Advanced Packed Bed Reactor

**Goal**: Simulate an industrial packed bed reactor with heat and mass transfer.

```python
import pyroxa
import numpy as np
import matplotlib.pyplot as plt

# Industrial packed bed reactor simulation
print("Simulating Industrial Packed Bed Reactor...")
print("Reaction: A → B (exothermic)")
print("Conditions: High temperature, industrial scale")

times, concentrations, temperatures = pyroxa.simulate_packed_bed(
    n_components=2,          # A and B
    n_points=30,             # 30 discretization points
    reactor_length=3.0,      # 3 meter reactor
    particle_diameter=0.004, # 4mm catalyst particles
    bed_porosity=0.38,       # Typical packed bed porosity
    fluid_density=1.1,       # kg/m³ (gas at high T)
    fluid_viscosity=3.0e-5,  # Pa·s (gas viscosity)
    flow_rate=0.05,          # m³/s (industrial flow)
    initial_concentrations=[1500.0, 0.0],  # mol/m³
    rate_constants=[0.15, 0.0],  # Irreversible reaction
    stoichiometry=[[-1, 1], [1, -1]],  # A → B
    diffusivities=[1.5e-5, 1.2e-5],    # m²/s
    heat_capacity=1100.0,    # J/kg/K
    thermal_conductivity=0.7, # W/m/K
    wall_heat_transfer_coeff=120.0,  # W/m²/K
    wall_temperature=573.15,  # 300°C wall temperature
    inlet_temperature=473.15, # 200°C inlet
    dt=0.2,                  # Time step
    output_interval=5
)

# Calculate performance metrics
inlet_A = 1500.0
outlet_A = concentrations[-1][0]
outlet_B = concentrations[-1][1]

conversion = (inlet_A - outlet_A) / inlet_A * 100
yield_B = outlet_B / inlet_A * 100
selectivity = outlet_B / (inlet_A - outlet_A) * 100

print(f"\nPacked Bed Reactor Performance:")
print(f"Conversion: {conversion:.1f}%")
print(f"Yield: {yield_B:.1f}%")
print(f"Selectivity: {selectivity:.1f}%")
print(f"Temperature rise: {temperatures[-1] - 473.15:.1f}°C")

# Plot reactor profiles
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Concentration profiles
reactor_positions = np.linspace(0, 3.0, len(concentrations[0]))
ax1.plot(reactor_positions, [c[0] for c in concentrations], 'b-', label='A', linewidth=2)
ax1.plot(reactor_positions, [c[1] for c in concentrations], 'r-', label='B', linewidth=2)
ax1.set_xlabel('Reactor Position (m)')
ax1.set_ylabel('Concentration (mol/m³)')
ax1.set_title('Concentration Profiles')
ax1.legend()
ax1.grid(True)

# Temperature profile
ax2.plot(reactor_positions, temperatures, 'g-', linewidth=2)
ax2.set_xlabel('Reactor Position (m)')
ax2.set_ylabel('Temperature (K)')
ax2.set_title('Temperature Profile')
ax2.grid(True)

# Conversion along reactor
conversion_profile = [(inlet_A - c[0])/inlet_A * 100 for c in concentrations]
ax3.plot(reactor_positions, conversion_profile, 'purple', linewidth=2)
ax3.set_xlabel('Reactor Position (m)')
ax3.set_ylabel('Conversion (%)')
ax3.set_title('Conversion Profile')
ax3.grid(True)

# Time evolution at reactor exit
ax4.plot(times, [c[0] for c in concentrations], 'b-', label='A (exit)', linewidth=2)
ax4.plot(times, [c[1] for c in concentrations], 'r-', label='B (exit)', linewidth=2)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Concentration (mol/m³)')
ax4.set_title('Exit Concentrations vs Time')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()
```

### Using High-Level Interface

```python
from pyroxa import build_from_dict

# Define simulation using dictionary
spec = {
    'reaction': {'kf': 1.0, 'kr': 0.5},
    'initial': {'temperature': 298.15, 'conc': {'A': 1.0, 'B': 0.0}},
    'sim': {'time_span': 5.0, 'time_step': 0.1},
    'system': 'WellMixed'
}

# Build and run
reactor, sim_params = build_from_dict(spec)
times, trajectory = reactor.run(sim_params['time_span'], sim_params['time_step'])
```

---

## Library Architecture

PyroXa uses a layered architecture:

```
┌─────────────────────────────────────────┐
│           User Interface                │
│  (High-level functions & classes)       │
├─────────────────────────────────────────┤
│         Python Interface               │
│    (Pure Python implementations)       │
├─────────────────────────────────────────┤
│         C++ Extensions                 │
│   (High-performance core functions)    │
└─────────────────────────────────────────┘
```

- **C++ Extensions**: Fast numerical computations (automatically used when available)
- **Python Fallback**: Pure Python implementations (always available)
- **User Interface**: High-level classes and convenience functions

---

## Core Classes

### 1. Thermodynamics
```python
thermo = pyroxa.Thermodynamics(cp=29.1, T_ref=298.15)
```
**Purpose**: Calculate thermodynamic properties
**Parameters**:
- `cp`: Heat capacity at constant pressure (J/mol/K)
- `T_ref`: Reference temperature (K)

**Methods**:
- `enthalpy(T)`: Calculate enthalpy at temperature T
- `entropy(T)`: Calculate entropy at temperature T  
- `equilibrium_constant(T, delta_G)`: Calculate equilibrium constant

### 2. Reaction
```python
reaction = pyroxa.Reaction(kf=1.0, kr=0.5)
```
**Purpose**: Define chemical reactions with kinetics
**Parameters**:
- `kf`: Forward rate constant
- `kr`: Reverse rate constant

### 3. WellMixedReactor
```python
reactor = pyroxa.WellMixedReactor(reaction, A0=1.0, B0=0.0, temperature=298.15)
```
**Purpose**: Simulate perfectly mixed batch reactor
**Parameters**:
- `reaction`: Reaction object
- `A0`, `B0`: Initial concentrations
- `temperature`: Operating temperature

### 4. CSTR (Continuous Stirred Tank Reactor)
```python
cstr = pyroxa.CSTR(reaction, residence_time=10.0, temperature=298.15)
```
**Purpose**: Simulate continuous stirred tank reactor
**Parameters**:
- `reaction`: Reaction object
- `residence_time`: Mean residence time (s)
- `temperature`: Operating temperature

### 5. PFR (Plug Flow Reactor)
```python
pfr = pyroxa.PFR(reaction, length=1.0, velocity=0.1, temperature=298.15)
```
**Purpose**: Simulate plug flow reactor
**Parameters**:
- `reaction`: Reaction object
- `length`: Reactor length (m)
- `velocity`: Fluid velocity (m/s)
- `temperature`: Operating temperature

---

## Function Reference

### Thermodynamic Functions

#### 1. `gibbs_free_energy(enthalpy, entropy, temperature)`
Calculate Gibbs free energy.

**Parameters**:
- `enthalpy`: Enthalpy (J/mol)
- `entropy`: Entropy (J/mol/K)  
- `temperature`: Temperature (K)

**Returns**: Gibbs free energy (J/mol)

**Example**:
```python
G = pyroxa.gibbs_free_energy(enthalpy=-100000, entropy=150, temperature=298.15)
print(f"Gibbs free energy: {G:.2f} J/mol")
```

#### 2. `equilibrium_constant(delta_G, temperature)`
Calculate equilibrium constant from Gibbs free energy.

**Parameters**:
- `delta_G`: Standard Gibbs free energy change (J/mol)
- `temperature`: Temperature (K)

**Returns**: Equilibrium constant

**Example**:
```python
K_eq = pyroxa.equilibrium_constant(delta_G=-50000, temperature=298.15)
print(f"Equilibrium constant: {K_eq:.3e}")
```

#### 3. `heat_capacity_nasa(temperature, coefficients)`
Calculate heat capacity using NASA polynomial.

**Parameters**:
- `temperature`: Temperature (K)
- `coefficients`: List of 7 NASA polynomial coefficients

**Returns**: Heat capacity (J/mol/K)

**Example**:
```python
# NASA coefficients for CO2
coeffs = [4.453623, -3.140169e-3, 1.278411e-5, -1.030518e-8, 2.192534e-12, -4.837314e4, -0.955395]
cp = pyroxa.heat_capacity_nasa(temperature=500, coefficients=coeffs)
print(f"Heat capacity of CO2 at 500K: {cp:.2f} J/mol/K")
```

#### 4. `enthalpy_nasa(temperature, coefficients)`
Calculate enthalpy using NASA polynomial.

**Parameters**:
- `temperature`: Temperature (K)
- `coefficients`: List of 7 NASA polynomial coefficients

**Returns**: Enthalpy (J/mol)

**Example**:
```python
H = pyroxa.enthalpy_nasa(temperature=500, coefficients=coeffs)
print(f"Enthalpy of CO2 at 500K: {H:.2f} J/mol")
```

#### 5. `entropy_nasa(temperature, coefficients)`
Calculate entropy using NASA polynomial.

**Parameters**:
- `temperature`: Temperature (K)
- `coefficients`: List of 7 NASA polynomial coefficients

**Returns**: Entropy (J/mol/K)

**Example**:
```python
S = pyroxa.entropy_nasa(temperature=500, coefficients=coeffs)
print(f"Entropy of CO2 at 500K: {S:.2f} J/mol/K")
```

#### 6. `pressure_peng_robinson(temperature, volume, a, b)`
Calculate pressure using Peng-Robinson equation of state.

**Parameters**:
- `temperature`: Temperature (K)
- `volume`: Molar volume (m³/mol)
- `a`: Peng-Robinson parameter a
- `b`: Peng-Robinson parameter b

**Returns**: Pressure (Pa)

**Example**:
```python
P = pyroxa.pressure_peng_robinson(temperature=300, volume=0.024, a=0.42, b=2.67e-5)
print(f"Pressure: {P:.2f} Pa")
```

#### 7. `fugacity_coefficient(pressure, temperature, a, b)`
Calculate fugacity coefficient using Peng-Robinson EOS.

**Parameters**:
- `pressure`: Pressure (Pa)
- `temperature`: Temperature (K)
- `a`: Peng-Robinson parameter a
- `b`: Peng-Robinson parameter b

**Returns**: Fugacity coefficient

**Example**:
```python
phi = pyroxa.fugacity_coefficient(pressure=101325, temperature=300, a=0.42, b=2.67e-5)
print(f"Fugacity coefficient: {phi:.4f}")
```

### Kinetics & Rate Functions

#### 8. `arrhenius_rate(A, Ea, temperature)`
Calculate reaction rate using Arrhenius equation.

**Parameters**:
- `A`: Pre-exponential factor
- `Ea`: Activation energy (J/mol)
- `temperature`: Temperature (K)

**Returns**: Rate constant

**Example**:
```python
k = pyroxa.arrhenius_rate(A=1e10, Ea=50000, temperature=298.15)
print(f"Rate constant: {k:.3e} s⁻¹")
```

#### 9. `autocatalytic_rate(k, A, B)`
Calculate autocatalytic reaction rate.

**Parameters**:
- `k`: Rate constant
- `A`: Concentration of reactant A
- `B`: Concentration of product B (catalyst)

**Returns**: Reaction rate

**Example**:
```python
rate = pyroxa.autocatalytic_rate(k=0.1, A=2.0, B=0.5)
print(f"Autocatalytic rate: {rate:.3f} mol/L/s")
```

#### 10. `michaelis_menten_rate(Vmax, Km, substrate_conc)`
Calculate enzyme kinetics rate using Michaelis-Menten equation.

**Parameters**:
- `Vmax`: Maximum reaction velocity
- `Km`: Michaelis constant
- `substrate_conc`: Substrate concentration

**Returns**: Reaction rate

**Example**:
```python
rate = pyroxa.michaelis_menten_rate(Vmax=10.0, Km=0.5, substrate_conc=2.0)
print(f"Enzyme reaction rate: {rate:.3f} mol/L/s")
```

#### 11. `competitive_inhibition_rate(Vmax, Km, substrate_conc, inhibitor_conc, Ki)`
Calculate competitive inhibition rate.

**Parameters**:
- `Vmax`: Maximum reaction velocity
- `Km`: Michaelis constant
- `substrate_conc`: Substrate concentration
- `inhibitor_conc`: Inhibitor concentration
- `Ki`: Inhibition constant

**Returns**: Reaction rate

**Example**:
```python
rate = pyroxa.competitive_inhibition_rate(Vmax=10.0, Km=0.5, substrate_conc=2.0, 
                                        inhibitor_conc=0.1, Ki=0.2)
print(f"Inhibited reaction rate: {rate:.3f} mol/L/s")
```

#### 12. `langmuir_hinshelwood_rate(k, conc_dict, adsorption_dict)`
Calculate surface reaction rate using Langmuir-Hinshelwood mechanism.

**Parameters**:
- `k`: Rate constant
- `conc_dict`: Dictionary of species concentrations
- `adsorption_dict`: Dictionary of adsorption constants

**Returns**: Surface reaction rate

**Example**:
```python
rate = pyroxa.langmuir_hinshelwood_rate(
    k=1.0,
    conc_dict={'A': 1.0, 'B': 0.5},
    adsorption_dict={'A': 10.0, 'B': 5.0}
)
print(f"Surface reaction rate: {rate:.3f} mol/m²/s")
```

#### 13. `photochemical_rate(I0, absorption_coeff, quantum_yield, concentration)`
Calculate photochemical reaction rate.

**Parameters**:
- `I0`: Incident light intensity
- `absorption_coeff`: Absorption coefficient
- `quantum_yield`: Quantum yield
- `concentration`: Reactant concentration

**Returns**: Photochemical reaction rate

**Example**:
```python
rate = pyroxa.photochemical_rate(I0=1000, absorption_coeff=0.1, 
                                quantum_yield=0.8, concentration=0.01)
print(f"Photochemical rate: {rate:.3e} mol/L/s")
```

### Reactor Simulation Functions

#### 14. `simulate_packed_bed(n_components, n_points, reactor_length, particle_diameter, bed_porosity, fluid_density, fluid_viscosity, flow_rate, initial_concentrations, rate_constants, stoichiometry, diffusivities, heat_capacity, thermal_conductivity, wall_heat_transfer_coeff, wall_temperature, inlet_temperature, dt, output_interval)`
Simulate packed bed reactor with complex transport phenomena.

**Parameters**:
- `n_components`: Number of chemical components
- `n_points`: Number of discretization points
- `reactor_length`: Length of reactor (m)
- `particle_diameter`: Catalyst particle diameter (m)
- `bed_porosity`: Bed void fraction
- `fluid_density`: Fluid density (kg/m³)
- `fluid_viscosity`: Fluid viscosity (Pa·s)
- `flow_rate`: Volumetric flow rate (m³/s)
- `initial_concentrations`: List of initial concentrations (mol/m³)
- `rate_constants`: List of reaction rate constants
- `stoichiometry`: Stoichiometric coefficients matrix
- `diffusivities`: List of molecular diffusivities (m²/s)
- `heat_capacity`: Heat capacity (J/kg/K)
- `thermal_conductivity`: Thermal conductivity (W/m/K)
- `wall_heat_transfer_coeff`: Wall heat transfer coefficient (W/m²/K)
- `wall_temperature`: Wall temperature (K)
- `inlet_temperature`: Inlet temperature (K)
- `dt`: Time step (s)
- `output_interval`: Output interval

**Returns**: Tuple of (times, concentrations, temperatures)

**Example**:
```python
times, conc, temps = pyroxa.simulate_packed_bed(
    n_components=2,
    n_points=20,
    reactor_length=1.0,
    particle_diameter=0.003,
    bed_porosity=0.4,
    fluid_density=1.2,
    fluid_viscosity=1.8e-5,
    flow_rate=0.001,
    initial_concentrations=[1000.0, 0.0],
    rate_constants=[0.1, 0.05],
    stoichiometry=[[-1, 1], [1, -1]],
    diffusivities=[1e-5, 1e-5],
    heat_capacity=1000.0,
    thermal_conductivity=0.6,
    wall_heat_transfer_coeff=100.0,
    wall_temperature=573.15,
    inlet_temperature=298.15,
    dt=0.1,
    output_interval=10
)
print(f"Simulation completed: {len(times)} time points")
```

#### 15. `simulate_fluidized_bed(n_components, n_points, reactor_height, particle_diameter, particle_density, bed_porosity, minimum_fluidization_velocity, superficial_velocity, initial_concentrations, rate_constants, stoichiometry, diffusivities, heat_capacity, heat_transfer_coeff, wall_temperature, inlet_temperature, dt, output_interval)`
Simulate fluidized bed reactor.

**Parameters**: Similar to packed bed with fluidization-specific parameters
- `minimum_fluidization_velocity`: Minimum fluidization velocity (m/s)
- `superficial_velocity`: Operating superficial velocity (m/s)
- `particle_density`: Particle density (kg/m³)

**Returns**: Tuple of (times, concentrations, temperatures)

**Example**:
```python
times, conc, temps = pyroxa.simulate_fluidized_bed(
    n_components=2,
    n_points=15,
    reactor_height=2.0,
    particle_diameter=0.0005,
    particle_density=2500.0,
    bed_porosity=0.5,
    minimum_fluidization_velocity=0.01,
    superficial_velocity=0.05,
    initial_concentrations=[500.0, 0.0],
    rate_constants=[0.2, 0.1],
    stoichiometry=[[-1, 1], [1, -1]],
    diffusivities=[2e-5, 2e-5],
    heat_capacity=800.0,
    heat_transfer_coeff=200.0,
    wall_temperature=623.15,
    inlet_temperature=298.15,
    dt=0.05,
    output_interval=20
)
print(f"Fluidized bed simulation: {len(times)} time points")
```

#### 16. `simulate_homogeneous_batch(n_components, initial_concentrations, rate_constants, stoichiometry, temperature, volume, heat_capacity, heat_transfer_coeff, ambient_temperature, activation_energies, pre_exponential_factors, dt, total_time, output_interval)`
Simulate homogeneous batch reactor with temperature effects.

**Parameters**:
- `n_components`: Number of components
- `initial_concentrations`: Initial concentrations (mol/m³)
- `rate_constants`: Rate constants at reference temperature
- `stoichiometry`: Stoichiometric matrix
- `temperature`: Initial temperature (K)
- `volume`: Reactor volume (m³)
- `heat_capacity`: Heat capacity (J/K)
- `heat_transfer_coeff`: Heat transfer coefficient (W/K)
- `ambient_temperature`: Ambient temperature (K)
- `activation_energies`: Activation energies (J/mol)
- `pre_exponential_factors`: Pre-exponential factors
- `dt`: Time step (s)
- `total_time`: Total simulation time (s)
- `output_interval`: Output interval

**Returns**: Tuple of (times, concentrations, temperatures)

**Example**:
```python
times, conc, temps = pyroxa.simulate_homogeneous_batch(
    n_components=3,
    initial_concentrations=[1000.0, 500.0, 0.0],
    rate_constants=[0.1, 0.05, 0.02],
    stoichiometry=[[-1, -1, 1], [1, 1, -1], [0, 0, 0]],
    temperature=298.15,
    volume=0.1,
    heat_capacity=4180.0,
    heat_transfer_coeff=50.0,
    ambient_temperature=298.15,
    activation_energies=[50000.0, 60000.0, 40000.0],
    pre_exponential_factors=[1e10, 1e9, 1e11],
    dt=1.0,
    total_time=100.0,
    output_interval=1
)
print(f"Batch reactor simulation: {len(times)} time points")
```

#### 17. `simulate_multi_reactor_adaptive(reactor_configs, flow_connections, adaptive_params)`
Simulate network of reactors with adaptive time stepping.

**Parameters**:
- `reactor_configs`: List of reactor configuration dictionaries
- `flow_connections`: Matrix defining flow connections
- `adaptive_params`: Adaptive time stepping parameters

**Returns**: Complex simulation results

### Mathematical & Utility Functions

#### 18. `linear_interpolate(x_data, y_data, x_target)`
Perform linear interpolation.

**Parameters**:
- `x_data`: Array of x values
- `y_data`: Array of y values  
- `x_target`: Target x value for interpolation

**Returns**: Interpolated y value

**Example**:
```python
x = [0, 1, 2, 3, 4]
y = [0, 1, 4, 9, 16]
result = pyroxa.linear_interpolate(x, y, 2.5)
print(f"Interpolated value at x=2.5: {result:.2f}")
```

#### 19. `cubic_spline_interpolate(x_data, y_data, x_target)`
Perform cubic spline interpolation.

**Parameters**:
- `x_data`: Array of x values
- `y_data`: Array of y values
- `x_target`: Target x value for interpolation

**Returns**: Interpolated y value

**Example**:
```python
result = pyroxa.cubic_spline_interpolate(x, y, 2.5)
print(f"Cubic spline interpolated value: {result:.2f}")
```

#### 20. `matrix_multiply(A, B)`
Multiply two matrices.

**Parameters**:
- `A`: First matrix
- `B`: Second matrix

**Returns**: Product matrix A×B

**Example**:
```python
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = pyroxa.matrix_multiply(A, B)
print(f"Matrix product: {C}")
```

#### 21. `matrix_invert(matrix)`
Calculate matrix inverse.

**Parameters**:
- `matrix`: Square matrix to invert

**Returns**: Inverse matrix

**Example**:
```python
A = [[4, 7], [2, 6]]
A_inv = pyroxa.matrix_invert(A)
print(f"Matrix inverse: {A_inv}")
```

#### 22. `solve_linear_system(A, b)`
Solve linear system Ax = b.

**Parameters**:
- `A`: Coefficient matrix
- `b`: Right-hand side vector

**Returns**: Solution vector x

**Example**:
```python
A = [[3, 2], [1, 2]]
b = [5, 3]
x = pyroxa.solve_linear_system(A, b)
print(f"Solution: x = {x}")
```

### Analysis & Optimization Functions

#### 23. `calculate_sensitivity(function, parameters, base_values, perturbation)`
Calculate sensitivity analysis.

**Parameters**:
- `function`: Function to analyze
- `parameters`: Parameter names
- `base_values`: Base parameter values
- `perturbation`: Perturbation magnitude

**Returns**: Sensitivity coefficients

**Example**:
```python
def test_func(params):
    return params[0]**2 + params[1]**3

sensitivity = pyroxa.calculate_sensitivity(
    function=test_func,
    parameters=['x', 'y'],
    base_values=[2.0, 1.0],
    perturbation=0.01
)
print(f"Sensitivity: {sensitivity}")
```

#### 24. `calculate_jacobian(function, variables, base_values, perturbation)`
Calculate Jacobian matrix.

**Parameters**:
- `function`: Vector function
- `variables`: Variable names
- `base_values`: Base values
- `perturbation`: Perturbation for finite differences

**Returns**: Jacobian matrix

#### 25. `stability_analysis(jacobian_matrix)`
Perform stability analysis using eigenvalues.

**Parameters**:
- `jacobian_matrix`: Jacobian matrix of the system

**Returns**: Stability information (eigenvalues, stability flag)

#### 26. `calculate_objective_function(predicted, experimental, weights)`
Calculate objective function for optimization.

**Parameters**:
- `predicted`: Predicted values
- `experimental`: Experimental values  
- `weights`: Weights for each data point

**Returns**: Objective function value

**Example**:
```python
predicted = [1.1, 2.05, 2.95]
experimental = [1.0, 2.0, 3.0]
weights = [1.0, 1.0, 1.0]
obj = pyroxa.calculate_objective_function(predicted, experimental, weights)
print(f"Objective function: {obj:.4f}")
```

### Statistical & Data Functions

#### 27. `calculate_r_squared(y_true, y_pred)`
Calculate coefficient of determination (R²).

**Parameters**:
- `y_true`: True values
- `y_pred`: Predicted values

**Returns**: R² value

**Example**:
```python
y_true = [1, 2, 3, 4, 5]
y_pred = [1.1, 1.9, 3.1, 3.9, 5.1]
r2 = pyroxa.calculate_r_squared(y_true, y_pred)
print(f"R² = {r2:.4f}")
```

#### 28. `calculate_rmse(y_true, y_pred)`
Calculate root mean square error.

**Parameters**:
- `y_true`: True values
- `y_pred`: Predicted values

**Returns**: RMSE value

**Example**:
```python
rmse = pyroxa.calculate_rmse(y_true, y_pred)
print(f"RMSE = {rmse:.4f}")
```

#### 29. `calculate_aic(n_params, n_data, residuals)`
Calculate Akaike Information Criterion.

**Parameters**:
- `n_params`: Number of parameters
- `n_data`: Number of data points
- `residuals`: Array of residuals

**Returns**: AIC value

**Example**:
```python
residuals = [0.1, -0.1, 0.1, -0.1, 0.1]
aic = pyroxa.calculate_aic(n_params=2, n_data=5, residuals=residuals)
print(f"AIC = {aic:.2f}")
```

#### 30. `monte_carlo_simulation(n_samples, parameter_distributions, model_function, output_statistics, random_seed, confidence_level, correlation_matrix, parameter_bounds, adaptive_sampling, convergence_tolerance, max_iterations, batch_size, parallel_execution, sensitivity_analysis, uncertainty_propagation, output_interval, save_samples)`
Perform Monte Carlo uncertainty analysis.

**Parameters**:
- `n_samples`: Number of Monte Carlo samples
- `parameter_distributions`: Parameter distribution specifications
- `model_function`: Model function to evaluate
- `output_statistics`: Statistics to calculate
- `random_seed`: Random seed for reproducibility
- `confidence_level`: Confidence level for intervals
- `correlation_matrix`: Parameter correlation matrix
- `parameter_bounds`: Parameter bounds
- Additional parameters for advanced features

**Returns**: Monte Carlo results with statistics

**Example**:
```python
results = pyroxa.monte_carlo_simulation(
    n_samples=1000,
    parameter_distributions={'k': ('normal', 1.0, 0.1), 'T': ('uniform', 290, 310)},
    model_function=lambda p: p['k'] * np.exp(-5000/p['T']),
    output_statistics=['mean', 'std', 'percentile'],
    random_seed=42,
    confidence_level=0.95,
    correlation_matrix=None,
    parameter_bounds={'k': (0.1, 10), 'T': (200, 400)},
    adaptive_sampling=False,
    convergence_tolerance=0.01,
    max_iterations=10000,
    batch_size=100,
    parallel_execution=False,
    sensitivity_analysis=True,
    uncertainty_propagation=True,
    output_interval=100,
    save_samples=True
)
print(f"Monte Carlo mean: {results['mean']:.3f}")
```

#### 31. `bootstrap_uncertainty(data, n_bootstrap, statistic_function, confidence_level)`
Perform bootstrap uncertainty analysis.

**Parameters**:
- `data`: Input data array
- `n_bootstrap`: Number of bootstrap samples
- `statistic_function`: Function to calculate statistic
- `confidence_level`: Confidence level

**Returns**: Bootstrap results with confidence intervals

#### 32. `cross_validation_score(model, data, labels, k_folds)`
Perform k-fold cross-validation.

**Parameters**:
- `model`: Model function
- `data`: Input data
- `labels`: Target labels
- `k_folds`: Number of folds

**Returns**: Cross-validation score

#### 33. `kriging_interpolation(x_known, y_known, x_target, variogram_model)`
Perform Kriging interpolation.

**Parameters**:
- `x_known`: Known x coordinates
- `y_known`: Known y values
- `x_target`: Target x coordinates
- `variogram_model`: Variogram model parameters

**Returns**: Interpolated values with uncertainty

### Control & Automation Functions

#### 34. `pid_controller(setpoint, measured_value, previous_error, integral, kp, ki, kd, dt)`
PID controller calculation.

**Parameters**:
- `setpoint`: Desired value
- `measured_value`: Current measured value
- `previous_error`: Previous error value
- `integral`: Integral term accumulator
- `kp`: Proportional gain
- `ki`: Integral gain
- `kd`: Derivative gain
- `dt`: Time step

**Returns**: Control output and updated state

**Example**:
```python
output, new_error, new_integral = pyroxa.pid_controller(
    setpoint=100.0,
    measured_value=95.0,
    previous_error=0.0,
    integral=0.0,
    kp=1.0,
    ki=0.1,
    kd=0.01,
    dt=0.1
)
print(f"PID output: {output:.2f}")
```

#### 35. `mpc_controller(state, reference, prediction_horizon, control_horizon, constraints)`
Model Predictive Controller.

**Parameters**:
- `state`: Current system state
- `reference`: Reference trajectory
- `prediction_horizon`: Prediction horizon
- `control_horizon`: Control horizon
- `constraints`: System constraints

**Returns**: Optimal control sequence

#### 36. `real_time_optimization(objective_function, constraints, current_state, parameters)`
Real-time optimization algorithm.

**Parameters**:
- `objective_function`: Objective function to optimize
- `constraints`: Optimization constraints
- `current_state`: Current system state
- `parameters`: Optimization parameters

**Returns**: Optimal solution

### I/O & Utility Functions

#### 37. `load_spec_from_yaml(filename)`
Load simulation specification from YAML file.

**Parameters**:
- `filename`: Path to YAML file

**Returns**: Specification dictionary

**Example**:
```python
spec = pyroxa.load_spec_from_yaml('simulation_config.yaml')
reactor, sim_params = pyroxa.build_from_dict(spec)
```

#### 38. `save_results_to_csv(times, concentrations, filename)`
Save simulation results to CSV file.

**Parameters**:
- `times`: Time array
- `concentrations`: Concentration data
- `filename`: Output filename

**Example**:
```python
pyroxa.save_results_to_csv(times, trajectory, 'results.csv')
```

#### 39. `parse_mechanism(mechanism_file)`
Parse chemical reaction mechanism from file.

**Parameters**:
- `mechanism_file`: Path to mechanism file

**Returns**: Parsed mechanism data

## Complete Function Reference

### Transport & Physical Properties

#### 40. `mass_transfer_correlation(Re, Sc, geometry_factor)`
Calculate mass transfer correlation (Sherwood number).

**Parameters**:
- `Re`: Reynolds number
- `Sc`: Schmidt number  
- `geometry_factor`: Geometry-specific factor

**Returns**: Sherwood number

**Example**:
```python
Sh = pyroxa.mass_transfer_correlation(Re=1000, Sc=0.7, geometry_factor=0.023)
print(f"Sherwood number: {Sh:.2f}")
```

#### 41. `heat_transfer_correlation(Re, Pr, geometry_factor)`
Calculate heat transfer correlation (Nusselt number).

**Parameters**:
- `Re`: Reynolds number
- `Pr`: Prandtl number
- `geometry_factor`: Geometry-specific factor

**Returns**: Nusselt number

**Example**:
```python
Nu = pyroxa.heat_transfer_correlation(Re=1000, Pr=0.7, geometry_factor=0.023)
print(f"Nusselt number: {Nu:.2f}")
```

#### 42. `effective_diffusivity(molecular_diff, porosity, tortuosity, constriction_factor)`
Calculate effective diffusivity in porous media.

**Parameters**:
- `molecular_diff`: Molecular diffusivity (m²/s)
- `porosity`: Porosity (void fraction)
- `tortuosity`: Tortuosity factor
- `constriction_factor`: Constriction factor

**Returns**: Effective diffusivity (m²/s)

**Example**:
```python
D_eff = pyroxa.effective_diffusivity(
    molecular_diff=1e-5,
    porosity=0.4,
    tortuosity=2.0,
    constriction_factor=0.8
)
print(f"Effective diffusivity: {D_eff:.2e} m²/s")
```

#### 43. `pressure_drop_ergun(velocity, density, viscosity, particle_diameter, bed_porosity, bed_length)`
Calculate pressure drop using Ergun equation.

**Parameters**:
- `velocity`: Superficial velocity (m/s)
- `density`: Fluid density (kg/m³)
- `viscosity`: Fluid viscosity (Pa·s)
- `particle_diameter`: Particle diameter (m)
- `bed_porosity`: Bed porosity
- `bed_length`: Bed length (m)

**Returns**: Pressure drop (Pa)

**Example**:
```python
dp = pyroxa.pressure_drop_ergun(
    velocity=0.1,
    density=1.2,
    viscosity=1.8e-5,
    particle_diameter=0.003,
    bed_porosity=0.4,
    bed_length=1.0
)
print(f"Pressure drop: {dp:.1f} Pa")
```

#### 44. `enthalpy_c(temperature, heat_capacity)`
Calculate enthalpy with constant heat capacity.

**Parameters**:
- `temperature`: Temperature (K)
- `heat_capacity`: Heat capacity (J/mol/K)

**Returns**: Enthalpy (J/mol)

**Example**:
```python
H = pyroxa.enthalpy_c(temperature=500, heat_capacity=29.1)
print(f"Enthalpy: {H:.1f} J/mol")
```

#### 45. `entropy_c(temperature, heat_capacity, reference_temperature)`
Calculate entropy with constant heat capacity.

**Parameters**:
- `temperature`: Temperature (K)
- `heat_capacity`: Heat capacity (J/mol/K)
- `reference_temperature`: Reference temperature (K)

**Returns**: Entropy (J/mol/K)

**Example**:
```python
S = pyroxa.entropy_c(temperature=500, heat_capacity=29.1, reference_temperature=298.15)
print(f"Entropy: {S:.2f} J/mol/K")
```

#### 46. `analytical_first_order(t, k, C0)`
Analytical solution for first-order reaction.

**Parameters**:
- `t`: Time (s)
- `k`: Rate constant (s⁻¹)
- `C0`: Initial concentration

**Returns**: Concentration at time t

**Example**:
```python
C = pyroxa.analytical_first_order(t=10.0, k=0.1, C0=1.0)
print(f"Concentration after 10s: {C:.3f}")
```

#### 47. `analytical_reversible_first_order(t, kf, kr, A0, B0)`
Analytical solution for reversible first-order reaction A ⇌ B.

**Parameters**:
- `t`: Time (s)
- `kf`: Forward rate constant (s⁻¹)
- `kr`: Reverse rate constant (s⁻¹)
- `A0`: Initial concentration of A
- `B0`: Initial concentration of B

**Returns**: Tuple (concentration of A, concentration of B)

**Example**:
```python
CA, CB = pyroxa.analytical_reversible_first_order(t=5.0, kf=0.2, kr=0.1, A0=1.0, B0=0.0)
print(f"After 5s: [A]={CA:.3f}, [B]={CB:.3f}")
```

#### 48. `analytical_consecutive_first_order(t, k1, k2, A0)`
Analytical solution for consecutive first-order reactions A → B → C.

**Parameters**:
- `t`: Time (s)
- `k1`: First rate constant (s⁻¹)
- `k2`: Second rate constant (s⁻¹)
- `A0`: Initial concentration of A

**Returns**: Tuple (concentration of A, B, C)

**Example**:
```python
CA, CB, CC = pyroxa.analytical_consecutive_first_order(t=10.0, k1=0.1, k2=0.05, A0=1.0)
print(f"Concentrations: A={CA:.3f}, B={CB:.3f}, C={CC:.3f}")
```

#### 49. `check_mass_conservation(initial_mass, final_mass, tolerance)`
Check mass conservation in reactions.

**Parameters**:
- `initial_mass`: Initial total mass
- `final_mass`: Final total mass
- `tolerance`: Acceptable tolerance

**Returns**: Boolean (True if conserved)

**Example**:
```python
is_conserved = pyroxa.check_mass_conservation(
    initial_mass=100.0,
    final_mass=99.9,
    tolerance=0.5
)
print(f"Mass conserved: {is_conserved}")
```

#### 50. `calculate_rate_constants(concentrations, rates, stoichiometry)`
Calculate rate constants from concentration and rate data.

**Parameters**:
- `concentrations`: Concentration data
- `rates`: Rate data
- `stoichiometry`: Stoichiometric coefficients

**Returns**: Calculated rate constants

**Example**:
```python
k_values = pyroxa.calculate_rate_constants(
    concentrations=[[1.0, 0.0], [0.8, 0.2], [0.6, 0.4]],
    rates=[0.1, 0.08, 0.06],
    stoichiometry=[-1, 1]
)
print(f"Rate constants: {k_values}")
```

#### 51. `residence_time_distribution(flow_pattern, mean_residence_time, variance)`
Calculate residence time distribution.

**Parameters**:
- `flow_pattern`: Flow pattern type ('ideal_cstr', 'ideal_pfr', 'tanks_in_series')
- `mean_residence_time`: Mean residence time (s)
- `variance`: Variance parameter

**Returns**: RTD function

**Example**:
```python
rtd = pyroxa.residence_time_distribution(
    flow_pattern='ideal_cstr',
    mean_residence_time=10.0,
    variance=1.0
)
print(f"RTD function created for CSTR")
```

#### 52. `catalyst_deactivation_model(time, initial_activity, deactivation_constant, mechanism)`
Model catalyst deactivation.

**Parameters**:
- `time`: Time (s)
- `initial_activity`: Initial catalyst activity
- `deactivation_constant`: Deactivation rate constant
- `mechanism`: Deactivation mechanism ('exponential', 'power_law', 'sintering')

**Returns**: Catalyst activity at time t

**Example**:
```python
activity = pyroxa.catalyst_deactivation_model(
    time=100.0,
    initial_activity=1.0,
    deactivation_constant=0.01,
    mechanism='exponential'
)
print(f"Catalyst activity after 100s: {activity:.3f}")
```

#### 53. `process_scale_up(lab_data, scale_factor, similarity_criteria)`
Scale up process from lab to industrial scale.

**Parameters**:
- `lab_data`: Laboratory scale data
- `scale_factor`: Scale-up factor
- `similarity_criteria`: Similarity criteria ('geometric', 'dynamic', 'kinematic')

**Returns**: Scaled-up parameters

**Example**:
```python
industrial_params = pyroxa.process_scale_up(
    lab_data={'volume': 0.001, 'power': 100, 'time': 3600},
    scale_factor=1000,
    similarity_criteria='geometric'
)
print(f"Industrial scale parameters: {industrial_params}")
```

### Core Classes & Objects (54-62)

#### 54. `Thermodynamics` Class
Thermodynamic property calculations for chemical species.

**Purpose**: Calculate enthalpy, entropy, heat capacity, and equilibrium constants

**Key Methods**:
- `enthalpy(temperature)`: Calculate enthalpy at given temperature
- `entropy(temperature)`: Calculate entropy at given temperature
- `equilibrium_constant(temperature, delta_G)`: Calculate equilibrium constant
- `heat_capacity(temperature)`: Calculate heat capacity

**Example**:
```python
# Create thermodynamics object for CO2
thermo = pyroxa.Thermodynamics(cp=37.1, T_ref=298.15)  # J/mol/K
H = thermo.enthalpy(500)  # Enthalpy at 500K
S = thermo.entropy(500)   # Entropy at 500K
print(f"H(500K) = {H:.1f} J/mol, S(500K) = {S:.2f} J/mol/K")
```

#### 55. `Reaction` Class
Chemical reaction kinetics and thermodynamics.

**Purpose**: Define reaction stoichiometry, kinetics, and calculate reaction rates

**Key Methods**:
- `rate(concentrations)`: Calculate reaction rate at given concentrations
- `equilibrium_constant(temperature)`: Calculate temperature-dependent K_eq
- `rate_constant(temperature)`: Calculate temperature-dependent rate constants
- `conversion(extent)`: Calculate conversion from extent of reaction

**Example**:
```python
# Define A ⇌ B reaction with Arrhenius kinetics
reaction = pyroxa.Reaction(
    kf=1e6,  # Pre-exponential factor (forward)
    kr=1e4,  # Pre-exponential factor (reverse)
    Ea_f=50000,  # Activation energy forward (J/mol)
    Ea_r=80000   # Activation energy reverse (J/mol)
)

# Calculate rate at 400K with [A]=2.0, [B]=0.5 M
rate = reaction.rate([2.0, 0.5], temperature=400)
print(f"Reaction rate at 400K: {rate:.3e} mol/L/s")
```

### Reactor Classes (Functions 56-72)

#### 56. `WellMixedReactor` Class
Perfect mixing batch reactor simulation.

**Key Methods**:
- `run(time_span, dt)`: Run simulation
- `steady_state()`: Find steady state
- `add_species(name, concentration)`: Add chemical species

#### 57. `CSTR` Class  
Continuous stirred tank reactor.

**Key Methods**:
- `steady_state_solve()`: Solve for steady state
- `dynamic_response(disturbance)`: Dynamic response analysis
- `residence_time_distribution()`: Calculate RTD

#### 58. `PFR` Class
Plug flow reactor simulation.

**Key Methods**:
- `solve_axial_dispersion()`: Include axial dispersion
- `temperature_profile()`: Calculate temperature profile
- `pressure_profile()`: Calculate pressure profile

#### 59. `PackedBedReactor` Class
Packed bed reactor with heat and mass transfer.

**Key Methods**:
- `effectiveness_factor()`: Calculate catalyst effectiveness
- `pressure_drop()`: Calculate pressure drop
- `heat_transfer_coefficient()`: Calculate heat transfer

#### 60. `FluidizedBedReactor` Class
Fluidized bed reactor simulation.

**Key Methods**:
- `minimum_fluidization_velocity()`: Calculate Umf
- `bubble_dynamics()`: Model bubble behavior
- `solid_mixing()`: Calculate solid mixing

#### 61. `HeterogeneousReactor` Class
Heterogeneous catalytic reactor.

**Key Methods**:
- `diffusion_resistance()`: Calculate diffusion limitations
- `surface_coverage()`: Calculate surface coverage
- `selectivity_analysis()`: Analyze selectivity

#### 62. `HomogeneousReactor` Class
Homogeneous reactor with complex kinetics.

**Key Methods**:
- `multiple_reactions()`: Handle multiple reactions
- `temperature_effects()`: Include temperature effects
- `mixing_effects()`: Include mixing effects

#### 63. `MultiReactor` Class
Multiple reactor system.

**Key Methods**:
- `add_reactor()`: Add reactor to system
- `connect_reactors()`: Connect reactors with streams
- `optimize_configuration()`: Optimize reactor network

#### 64. `ReactorNetwork` Class
Complex reactor network simulation.

**Key Methods**:
- `solve_network()`: Solve entire network
- `sensitivity_analysis()`: Network sensitivity
- `economic_optimization()`: Economic optimization

### Error and Exception Classes (65-69)

#### 65. `PyroXaError` Class
Base exception class for PyroXa.

#### 66. `ThermodynamicsError` Class
Thermodynamics-specific errors.

#### 67. `ReactionError` Class
Reaction-specific errors.

#### 68. `ReactorError` Class
Reactor-specific errors.

#### 69. `ReactionMulti` Class
Multiple reaction system.

### Advanced Analysis Functions (70-82)

#### 70. `benchmark_multi_reactor(configurations, performance_metrics)`
Benchmark multiple reactor configurations.

**Parameters**:
- `configurations`: List of reactor configurations
- `performance_metrics`: Metrics to evaluate

**Returns**: Benchmark results

#### 71. `ChainReactorVisualizer` Class
Visualize reaction chains and networks.

**Key Methods**:
- `plot_network()`: Plot reactor network
- `animate_simulation()`: Animate simulation results
- `export_diagram()`: Export network diagram

#### 72. `OptimalReactorDesign` Class
Optimize reactor design parameters.

**Key Methods**:
- `objective_function()`: Define optimization objective
- `constraints()`: Define design constraints  
- `optimize()`: Run optimization algorithm

#### 73. `ReactionChain` Class
Model complex reaction chains.

**Key Methods**:
- `add_reaction()`: Add reaction to chain
- `solve_kinetics()`: Solve chain kinetics
- `identify_bottlenecks()`: Identify rate-limiting steps

#### 74. `create_reaction_chain(reactions, species)`
Create reaction chain from individual reactions.

**Parameters**:
- `reactions`: List of reaction objects
- `species`: List of chemical species

**Returns**: ReactionChain object

### Energy Analysis Functions

#### 75. `calculate_energy_balance(n_components, initial_temperatures, heat_capacities, reaction_enthalpies, heat_transfer_coefficients, ambient_temperature)`
Calculate energy balance for reactor systems.

**Parameters**:
- `n_components`: Number of chemical components
- `initial_temperatures`: Initial temperature distribution (K)
- `heat_capacities`: Heat capacities of components (J/mol/K)
- `reaction_enthalpies`: Enthalpy changes for reactions (J/mol)
- `heat_transfer_coefficients`: Heat transfer coefficients (W/m²/K)
- `ambient_temperature`: Ambient temperature (K)

**Returns**: Tuple (temperature_profile, heat_generation_rate, heat_removal_rate)

**Example**:
```python
# Energy balance for exothermic reaction in CSTR
temp_profile, q_gen, q_removal = pyroxa.calculate_energy_balance(
    n_components=2,
    initial_temperatures=[298.15, 298.15],
    heat_capacities=[75.3, 81.6],  # J/mol/K for A and B
    reaction_enthalpies=[-50000.0],  # Exothermic reaction, J/mol
    heat_transfer_coefficients=[150.0],  # W/m²/K
    ambient_temperature=298.15
)
print(f"Temperature rise: {max(temp_profile) - 298.15:.1f} K")
print(f"Heat generation rate: {q_gen:.2f} W")
print(f"Heat removal rate: {q_removal:.2f} W")
```

### Utility and Information Functions (73-88)

#### 76. `get_build_info()`
Get build information about PyroXa installation.

**Returns**: Build information dictionary

**Example**:
```python
info = pyroxa.get_build_info()
print(f"PyroXa version: {info['version']}")
print(f"C++ extensions: {info['cpp_available']}")
```

#### 77. `get_version()`
Get PyroXa version string.

**Returns**: Version string

**Example**:
```python
version = pyroxa.get_version()
print(f"PyroXa v{version}")
```

#### 78. `is_compiled_available()`
Check if C++ extensions are available.

**Returns**: Boolean

**Example**:
```python
if pyroxa.is_compiled_available():
    print("Using high-performance C++ extensions")
else:
    print("Using pure Python implementation")
```

#### 79. `is_reaction_chains_available()`
Check if reaction chain functionality is available.

**Returns**: Boolean

#### 80. `run_simulation(spec, csv_out, plot)`
High-level simulation runner.

**Parameters**:
- `spec`: Simulation specification dictionary
- `csv_out`: CSV output filename (optional)
- `plot`: Whether to plot results (optional)

**Returns**: Simulation results

**Example**:
```python
spec = {
    'reaction': {'kf': 1.0, 'kr': 0.5},
    'initial': {'conc': {'A': 1.0, 'B': 0.0}},
    'sim': {'time_span': 10.0, 'time_step': 0.1}
}
results = pyroxa.run_simulation(spec, csv_out='results.csv', plot=True)
```

#### 81. `run_simulation_cpp(parameters)`
C++ implementation of simulation runner.

**Parameters**:
- `parameters`: Simulation parameters

**Returns**: High-performance simulation results

#### 82. `build_from_dict(spec)`
Build reactor and simulation from specification dictionary.

**Parameters**:
- `spec`: Specification dictionary

**Returns**: Tuple (reactor, simulation_params)

**Example**:
```python
spec = {
    'reaction': {'kf': 2.0, 'kr': 0.1},
    'initial': {'temperature': 350.0, 'conc': {'A': 2.0, 'B': 0.0}},
    'sim': {'time_span': 20.0, 'time_step': 0.05},
    'system': 'CSTR',
    'residence_time': 15.0
}
reactor, sim_params = pyroxa.build_from_dict(spec)
times, trajectory = reactor.run(sim_params['time_span'], sim_params['time_step'])
```

### Module Access Functions (81-89)

#### 81-85. Module Access
- `io`: Input/output utilities module
- `new_functions`: New function implementations module  
- `purepy`: Pure Python implementations module
- `reaction_chains`: Reaction chain functionality module
- Additional utility modules

#### 83. `save_spec_to_yaml(spec, filename)`
Save simulation specification to YAML file.

**Parameters**:
- `spec`: Simulation specification dictionary
- `filename`: Output YAML filename

**Returns**: None

**Example**:
```python
spec = {
    'reactions': ['A -> B'],
    'initial_conditions': {'A': 1.0, 'B': 0.0}
}
pyroxa.save_spec_to_yaml(spec, 'simulation.yaml')
```

#### 84. `validate_mechanism(mechanism)`
Validate chemical reaction mechanism.

**Parameters**:
- `mechanism`: Mechanism dictionary or string

**Returns**: Validation result and error messages

**Example**:
```python
valid, errors = pyroxa.validate_mechanism({
    'reactions': ['A -> B', 'B -> C'],
    'rate_constants': [0.1, 0.05]
})
print(f"Valid: {valid}, Errors: {errors}")
```

#### 85. `optimize_parameters(objective_function, bounds, method)`
Optimize reaction parameters using specified method.

**Parameters**:
- `objective_function`: Function to minimize
- `bounds`: Parameter bounds
- `method`: Optimization method

**Returns**: Optimized parameters

**Example**:
```python
def objective(params):
    return sum((simulated - experimental)**2)

optimal_params = pyroxa.optimize_parameters(
    objective_function=objective,
    bounds=[(0.001, 1.0), (0.001, 1.0)],
    method='differential_evolution'
)
```

#### 86. `load_spec_from_yaml(filename)`
Load simulation specification from YAML file.

#### 87. `parse_mechanism(mechanism_string)`
Parse chemical mechanism from string or file.

#### 88. `save_results_to_csv(filename, times, concentrations)`
Save simulation results to CSV file.

#### 89. `PIDController` Class
Advanced PID controller implementation.

**Key Methods**:
- `update(measured_value, setpoint)`: Update controller
- `reset()`: Reset controller state
- `tune(method, data)`: Auto-tune parameters

**Example**:
```python
controller = pyroxa.PIDController(kp=1.0, ki=0.1, kd=0.01)
output = controller.update(measured_value=95.0, setpoint=100.0)
print(f"Controller output: {output:.2f}")
```

---

## Advanced Usage Examples

### Complete Reactor Network Simulation

```python
import pyroxa
import numpy as np
import matplotlib.pyplot as plt

# Define complex reaction network: A → B → C
def create_reaction_network():
    # Reaction 1: A → B
    rxn1 = pyroxa.Reaction(kf=0.5, kr=0.1)
    
    # Reaction 2: B → C  
    rxn2 = pyroxa.Reaction(kf=0.3, kr=0.05)
    
    # Create reactor network
    network = pyroxa.ReactorNetwork()
    
    # Add CSTR for first reaction
    cstr1 = pyroxa.CSTR(rxn1, residence_time=10.0, temperature=350.0)
    network.add_reactor('CSTR1', cstr1)
    
    # Add PFR for second reaction
    pfr1 = pyroxa.PFR(rxn2, length=2.0, velocity=0.1, temperature=400.0)
    network.add_reactor('PFR1', pfr1)
    
    # Connect reactors in series
    network.connect('CSTR1', 'PFR1', flow_rate=0.001)  # m³/s
    
    return network

# Run network simulation
network = create_reaction_network()
results = network.simulate(
    inlet_conditions={'A': 1000.0, 'B': 0.0, 'C': 0.0},  # mol/m³
    simulation_time=100.0
)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(results['time'], results['concentrations']['A'], 'b-', label='A')
plt.plot(results['time'], results['concentrations']['B'], 'r-', label='B') 
plt.plot(results['time'], results['concentrations']['C'], 'g-', label='C')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mol/m³)')
plt.legend()
plt.title('Reactor Network: A → B → C')
plt.grid(True)
plt.show()
```

### Parameter Estimation with Uncertainty

```python
import pyroxa
import numpy as np

# Experimental data
exp_times = np.array([0, 10, 20, 30, 40, 50])
exp_concentrations = np.array([1.0, 0.8, 0.6, 0.45, 0.35, 0.28])

def reactor_model(params):
    """Reactor model for parameter estimation"""
    k_forward, k_reverse = params
    
    reaction = pyroxa.Reaction(kf=k_forward, kr=k_reverse)
    reactor = pyroxa.WellMixedReactor(reaction, A0=1.0, B0=0.0, temperature=298.15)
    
    times, trajectory = reactor.run(time_span=50.0, dt=1.0)
    
    # Interpolate to experimental time points
    conc_A = [traj['A'] for traj in trajectory]
    interpolated = pyroxa.linear_interpolate(times, conc_A, exp_times)
    
    return interpolated

# Parameter estimation using optimization
def objective_function(params):
    """Objective function for parameter estimation"""
    predicted = reactor_model(params)
    return pyroxa.calculate_objective_function(predicted, exp_concentrations, weights=None)

# Monte Carlo uncertainty analysis for parameters
mc_results = pyroxa.monte_carlo_simulation(
    n_samples=5000,
    parameter_distributions={
        'k_forward': ('uniform', 0.01, 0.2),
        'k_reverse': ('uniform', 0.001, 0.05)
    },
    model_function=reactor_model,
    output_statistics=['mean', 'std', 'percentile_5', 'percentile_95'],
    random_seed=42,
    confidence_level=0.95
)

print("Parameter Estimation Results:")
print(f"k_forward: {mc_results['k_forward']['mean']:.4f} ± {mc_results['k_forward']['std']:.4f}")
print(f"k_reverse: {mc_results['k_reverse']['mean']:.4f} ± {mc_results['k_reverse']['std']:.4f}")
```

### Industrial Scale Packed Bed Reactor

```python
import pyroxa

def industrial_packed_bed_simulation():
    """Simulate industrial-scale packed bed reactor"""
    
    # Industrial reactor conditions
    times, conc, temps = pyroxa.simulate_packed_bed(
        n_components=3,  # A, B, C
        n_points=50,     # High resolution
        reactor_length=5.0,  # 5 meter reactor
        particle_diameter=0.005,  # 5mm catalyst particles
        bed_porosity=0.35,
        fluid_density=0.8,  # kg/m³ (gas phase)
        fluid_viscosity=2.5e-5,  # Pa·s
        flow_rate=0.1,  # m³/s (industrial scale)
        initial_concentrations=[2000.0, 0.0, 0.0],  # mol/m³
        rate_constants=[0.2, 0.15, 0.1],
        stoichiometry=[[-1, 1, 0], [0, -1, 1], [0, 0, 0]],  # A→B→C
        diffusivities=[2e-5, 1.8e-5, 1.5e-5],  # m²/s
        heat_capacity=1200.0,  # J/kg/K
        thermal_conductivity=0.8,  # W/m/K
        wall_heat_transfer_coeff=150.0,  # W/m²/K
        wall_temperature=623.15,  # 350°C wall temperature
        inlet_temperature=373.15,  # 100°C inlet
        dt=0.5,  # Time step
        output_interval=10
    )
    
    # Calculate conversion and selectivity
    inlet_A = 2000.0
    outlet_A = conc[-1][0]  # Final A concentration
    outlet_B = conc[-1][1]  # Final B concentration
    
    conversion = (inlet_A - outlet_A) / inlet_A * 100
    selectivity = outlet_B / (inlet_A - outlet_A) * 100
    
    print(f"Industrial Packed Bed Reactor Results:")
    print(f"Conversion of A: {conversion:.1f}%")
    print(f"Selectivity to B: {selectivity:.1f}%")
    print(f"Final temperature: {temps[-1]:.1f} K")
    
    return times, conc, temps

# Run industrial simulation
times, concentrations, temperatures = industrial_packed_bed_simulation()
```

---

## Error Handling

PyroXa provides comprehensive error handling with custom exception classes:

### Exception Hierarchy

```python
PyroXaError (base)
├── ThermodynamicsError
├── ReactionError  
├── ReactorError
└── SimulationError
```

### Common Error Patterns

```python
import pyroxa

try:
    # This will raise ThermodynamicsError (negative temperature)
    thermo = pyroxa.Thermodynamics(cp=29.1, T_ref=-100)
except pyroxa.ThermodynamicsError as e:
    print(f"Thermodynamics error: {e}")

try:
    # This will raise ReactionError (negative rate constant)
    reaction = pyroxa.Reaction(kf=-1.0, kr=0.5)
except pyroxa.ReactionError as e:
    print(f"Reaction error: {e}")

try:
    # This will raise ReactorError (invalid initial conditions)
    reactor = pyroxa.WellMixedReactor(reaction, A0=-1.0, B0=0.0)
except pyroxa.ReactorError as e:
    print(f"Reactor error: {e}")
```

### Error Recovery Strategies

```python
def robust_simulation(params):
    """Example of robust simulation with error handling"""
    try:
        # Attempt simulation with given parameters
        reaction = pyroxa.Reaction(kf=params['kf'], kr=params['kr'])
        reactor = pyroxa.WellMixedReactor(reaction, A0=params['A0'], B0=params['B0'])
        return reactor.run(params['time_span'], params['dt'])
        
    except pyroxa.ReactionError:
        # Use default safe parameters
        print("Using default reaction parameters")
        reaction = pyroxa.Reaction(kf=1.0, kr=0.1)
        reactor = pyroxa.WellMixedReactor(reaction, A0=1.0, B0=0.0)
        return reactor.run(10.0, 0.1)
        
    except pyroxa.ReactorError:
        # Handle reactor-specific errors
        print("Reactor configuration error - using CSTR instead")
        cstr = pyroxa.CSTR(reaction, residence_time=10.0)
        return cstr.steady_state_solve()
```

---

## Performance Tips

### 1. Use C++ Extensions
```python
# Check if C++ extensions are available
if pyroxa.is_compiled_available():
    print("Using high-performance C++ extensions")
else:
    print("Using pure Python - consider compiling C++ extensions")
```

### 2. Optimize Simulation Parameters
```python
# Use appropriate time steps
times, traj = reactor.run(
    time_span=100.0,
    dt=0.1  # Balance accuracy vs speed
)

# For long simulations, use larger time steps with adaptive methods
times, traj = reactor.run_adaptive(
    time_span=1000.0,
    rtol=1e-6,  # Relative tolerance
    atol=1e-8   # Absolute tolerance
)
```

### 3. Vectorized Operations
```python
# Use numpy arrays for multiple simulations
import numpy as np

k_values = np.linspace(0.1, 2.0, 20)
results = []

for k in k_values:
    reaction = pyroxa.Reaction(kf=k, kr=0.1)
    reactor = pyroxa.WellMixedReactor(reaction, A0=1.0, B0=0.0)
    times, traj = reactor.run(10.0, 0.1)
    results.append(traj[-1]['A'])  # Final concentration

final_concentrations = np.array(results)
```

### 4. Memory Management
```python
# For large simulations, process data in chunks
def chunked_simulation(n_runs, chunk_size=100):
    for i in range(0, n_runs, chunk_size):
        chunk_results = []
        for j in range(min(chunk_size, n_runs - i)):
            # Run simulation
            result = run_single_simulation()
            chunk_results.append(result)
        
        # Process chunk
        process_chunk(chunk_results)
        
        # Clear memory
        del chunk_results
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```python
# Problem: Cannot import pyroxa
# Solution: Check installation
import sys
sys.path.append('/path/to/pyroxa')
import pyroxa
```

#### 2. C++ Extension Issues
```python
# Problem: C++ extensions not loading
# Solution: Check compilation
print("C++ available:", pyroxa.is_compiled_available())

# Fallback to pure Python
import pyroxa.purepy as pyroxa_pure
```

#### 3. Numerical Issues
```python
# Problem: Stiff ODEs not converging
# Solution: Use smaller time steps or implicit solvers
reactor = pyroxa.WellMixedReactor(reaction, A0=1.0, B0=0.0)

# Try smaller time step
times, traj = reactor.run(time_span=10.0, dt=0.001)

# Or use adaptive solver
times, traj = reactor.run_adaptive(time_span=10.0, method='BDF')
```

#### 4. Memory Issues
```python
# Problem: Large simulations consume too much memory
# Solution: Use generators or batch processing
def simulation_generator(n_sims):
    for i in range(n_sims):
        yield run_simulation(params[i])

# Process one at a time
for result in simulation_generator(10000):
    process_result(result)
```

#### 5. Performance Issues
```python
# Problem: Simulations are too slow
# Solutions:

# 1. Compile C++ extensions
# python setup.py build_ext --inplace

# 2. Use appropriate tolerances  
times, traj = reactor.run_adaptive(rtol=1e-4)  # Looser tolerance

# 3. Reduce output frequency
times, traj = reactor.run(time_span=100.0, dt=0.1, output_every=10)

# 4. Use parallel processing for parameter studies
from multiprocessing import Pool

def run_parallel_sims(param_list):
    with Pool() as pool:
        results = pool.map(run_single_sim, param_list)
    return results
```

### Getting Help

1. **Check Documentation**: This comprehensive guide covers most use cases
2. **Examine Examples**: Look at files in the `examples/` directory
3. **Run Tests**: Execute test files to see working examples
4. **Error Messages**: Read error messages carefully - they contain helpful information
5. **Version Compatibility**: Ensure you're using compatible Python version (3.8+)

### Debugging Tips

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Validate inputs before simulation
def validate_simulation_params(params):
    assert params['kf'] > 0, "Forward rate constant must be positive"
    assert params['kr'] >= 0, "Reverse rate constant must be non-negative"
    assert params['A0'] >= 0, "Initial concentration must be non-negative"
    # Add more validations...

# Use try-except blocks for robust code
try:
    result = pyroxa.run_simulation(spec)
except Exception as e:
    print(f"Simulation failed: {e}")
    print(f"Parameters: {spec}")
    raise
```

---

## Complete Function Summary Table

| # | Function Name | Category | Purpose | Key Parameters |
|---|---------------|----------|---------|----------------|
| 1 | `calculate_rate` | Kinetics | Calculate reaction rate | k, concentration |
| 2 | `arrhenius` | Kinetics | Arrhenius equation | A, Ea, T |
| 3 | `temperature_dependence` | Thermodynamics | Temperature effects | T, T_ref |
| 4 | `equilibrium_constant` | Thermodynamics | Chemical equilibrium | dG, T |
| 5 | `reactor_volume` | Reactor Design | Volume calculations | flow_rate, residence_time |
| 6 | `heat_capacity` | Thermodynamics | Heat capacity | Cp, T |
| 7 | `enthalpy_change` | Thermodynamics | Enthalpy calculations | dH, T |
| 8 | `entropy_change` | Thermodynamics | Entropy calculations | dS, T |
| 9 | `gibbs_free_energy` | Thermodynamics | Gibbs energy | dH, dS, T |
| 10 | `fugacity_coefficient` | Thermodynamics | Fugacity calculations | P, T, composition |
| 11 | `activity_coefficient` | Thermodynamics | Activity models | composition, T |
| 12 | `vapor_pressure` | Thermodynamics | VP calculations | T, substance |
| 13 | `raoults_law` | Thermodynamics | Ideal solution behavior | x, P_sat |
| 14 | `henrys_law` | Thermodynamics | Gas solubility | H, P |
| 15 | `ideal_gas_law` | Thermodynamics | PVT relationships | P, V, T |
| 16 | `van_der_waals` | Thermodynamics | Real gas EOS | a, b, P, T |
| 17 | `peng_robinson` | Thermodynamics | Advanced EOS | Tc, Pc, omega |
| 18 | `virial_equation` | Thermodynamics | Virial EOS | B, C coefficients |
| 19 | `compressibility_factor` | Thermodynamics | Z factor | Tr, Pr |
| 20 | `critical_properties` | Thermodynamics | Critical constants | substance |
| 21 | `acentric_factor` | Thermodynamics | Molecular shape factor | substance |
| 22 | `reduced_properties` | Thermodynamics | Reduced T,P,V | T, Tc, P, Pc |
| 23 | `corresponding_states` | Thermodynamics | CSP correlations | Tr, Pr |
| 24 | `mixing_rules` | Thermodynamics | Mixture properties | composition, pure_props |
| 25 | `excess_properties` | Thermodynamics | Non-ideal mixing | activity models |
| 26 | `phase_equilibrium` | Thermodynamics | Phase behavior | K-values, T, P |
| 27 | `bubble_point` | Thermodynamics | Bubble point calc | composition, P |
| 28 | `dew_point` | Thermodynamics | Dew point calc | composition, P |
| 29 | `flash_calculation` | Thermodynamics | VLE flash | feed, T, P |
| 30 | `distillation_column` | Separation | Column design | stages, reflux |
| 31 | `mass_transfer` | Transport | Mass transfer rate | kL, driving_force |
| 32 | `heat_transfer` | Transport | Heat transfer rate | h, delta_T |
| 33 | `momentum_transfer` | Transport | Momentum transfer | friction, flow |
| 34 | `diffusion_coefficient` | Transport | Molecular diffusion | T, P, species |
| 35 | `reynolds_number` | Transport | Flow characterization | velocity, L, viscosity |
| 36 | `prandtl_number` | Transport | Heat/momentum analogy | Cp, mu, k |
| 37 | `schmidt_number` | Transport | Mass/momentum analogy | mu, rho, D |
| 38 | `nusselt_number` | Transport | Heat transfer correlation | h, L, k |
| 39 | `sherwood_number` | Transport | Mass transfer correlation | kL, L, D |
| 40 | `peclet_number` | Transport | Convection/diffusion | velocity, L, D |
| 41 | `damkohler_number` | Reactor | Reaction/transport ratio | k, tau |
| 42 | `thiele_modulus` | Reactor | Reaction/diffusion | k, D, L |
| 43 | `effectiveness_factor` | Reactor | Catalyst effectiveness | phi |
| 44 | `selectivity` | Kinetics | Product selectivity | rates |
| 45 | `yield` | Kinetics | Reaction yield | products, reactants |
| 46 | `conversion` | Kinetics | Reactant conversion | initial, final |
| 47 | `space_time` | Reactor | Reactor space time | V, v0 |
| 48 | `space_velocity` | Reactor | Space velocity | v0, V |
| 49 | `check_mass_conservation` | Validation | Mass balance check | in, out, tolerance |
| 50 | `calculate_rate_constants` | Kinetics | Rate constant fitting | concentrations, rates |
| 51 | `residence_time_distribution` | Reactor | RTD analysis | flow_pattern, tau |
| 52 | `reactor_performance` | Reactor | Performance metrics | conversion, selectivity |
| 53 | `optimize_conditions` | Optimization | Process optimization | objective, constraints |
| 54 | `Thermodynamics` | Core Class | Thermodynamic properties | Cp, T_ref |
| 55 | `Reaction` | Core Class | Chemical reaction | kf, kr |
| 56 | `WellMixedReactor` | Reactor Class | Ideal CSTR | reaction, volume |
| 57 | `CSTR` | Reactor Class | Continuous stirred tank | volume, flow_rate |
| 58 | `PFR` | Reactor Class | Plug flow reactor | length, diameter |
| 59 | `PackedBedReactor` | Reactor Class | Packed bed | porosity, particle_size |
| 60 | `FluidizedBedReactor` | Reactor Class | Fluidized bed | Umf, particle_density |
| 61 | `HeterogeneousReactor` | Reactor Class | Heterogeneous catalysis | catalyst_area |
| 62 | `HomogeneousReactor` | Reactor Class | Homogeneous reactions | mixing_time |
| 63 | `MultiReactor` | Reactor Class | Multiple reactor system | reactor_network |
| 64 | `ReactorNetwork` | Reactor Class | Advanced reactor network | topology |
| 65 | `PyroXaError` | Error Class | Base exception class | error_handling |
| 66 | `ThermodynamicsError` | Error Class | Thermodynamics errors | property_errors |
| 67 | `ReactionError` | Error Class | Reaction errors | kinetic_errors |
| 68 | `ReactorError` | Error Class | Reactor errors | design_errors |
| 69 | `ReactionMulti` | Reaction Class | Multiple reaction system | reaction_network |
| 70 | `benchmark_multi_reactor` | Benchmarking | Performance comparison | configurations |
| 71 | `ChainReactorVisualizer` | Visualization | Reactor chain plots | chain_data |
| 72 | `OptimalReactorDesign` | Optimization | Reactor optimization | design_vars |
| 73 | `ReactionChain` | Reaction Class | Reaction chain system | sequential_reactions |
| 74 | `create_reaction_chain` | Factory | Create reaction chains | reactions, species |
| 75 | `calculate_energy_balance` | Energy | Energy balance | 6 energy parameters |
| 76 | `get_build_info` | Utility | Build information | version, compiler |
| 77 | `get_version` | Utility | Version information | major, minor, patch |
| 78 | `is_compiled_available` | Utility | C++ availability check | compilation_status |
| 79 | `is_reaction_chains_available` | Utility | Feature availability | feature_status |
| 80 | `run_simulation` | Simulation | Main simulation runner | spec, output |
| 81 | `run_simulation_cpp` | Simulation | C++ simulation runner | parameters |
| 82 | `build_from_dict` | Factory | Build from dictionary | specification |
| 83 | `save_spec_to_yaml` | I/O | Save spec to YAML | spec, filename |
| 84 | `validate_mechanism` | Validation | Mechanism validation | mechanism |
| 85 | `optimize_parameters` | Optimization | Parameter optimization | objective, bounds |
| 86 | `load_spec_from_yaml` | I/O | Load spec from YAML | filename |
| 87 | `parse_mechanism` | Parser | Parse mechanism string | mechanism_string |
| 88 | `save_results_to_csv` | I/O | Save results to CSV | filename, data |
| 89 | `PIDController` | Control Class | Advanced PID control | tuning parameters |
| 21 | `matrix_invert` | Mathematics | Matrix inversion | matrix |
| 22 | `solve_linear_system` | Mathematics | Linear system solver | A, b |
| 23 | `calculate_sensitivity` | Analysis | Sensitivity analysis | function, parameters, values |
| 24 | `calculate_jacobian` | Analysis | Jacobian calculation | function, variables, values |
| 25 | `stability_analysis` | Analysis | System stability | jacobian_matrix |
| 26 | `calculate_objective_function` | Analysis | Optimization objective | predicted, experimental, weights |
| 27 | `calculate_r_squared` | Statistics | Coefficient of determination | y_true, y_pred |
| 28 | `calculate_rmse` | Statistics | Root mean square error | y_true, y_pred |
| 29 | `calculate_aic` | Statistics | Akaike information criterion | n_params, n_data, residuals |
| 30 | `monte_carlo_simulation` | Statistics | Monte Carlo analysis | 17 complex parameters |
| 31 | `bootstrap_uncertainty` | Statistics | Bootstrap analysis | data, n_bootstrap, function |
| 32 | `cross_validation_score` | Statistics | Cross-validation | model, data, labels, k_folds |
| 33 | `kriging_interpolation` | Statistics | Kriging interpolation | x_known, y_known, x_target |
| 34 | `pid_controller` | Control | PID control calculation | setpoint, measured, gains |
| 35 | `mpc_controller` | Control | Model predictive control | state, reference, horizons |
| 36 | `real_time_optimization` | Control | Real-time optimization | objective, constraints, state |
| 37 | `load_spec_from_yaml` | I/O | Load YAML specification | filename |
| 38 | `save_results_to_csv` | I/O | Save results to CSV | times, concentrations, filename |
| 39 | `parse_mechanism` | I/O | Parse reaction mechanism | mechanism_file |
| 40 | `mass_transfer_correlation` | Transport | Sherwood number | Re, Sc, geometry_factor |
| 41 | `heat_transfer_correlation` | Transport | Nusselt number | Re, Pr, geometry_factor |
| 42 | `effective_diffusivity` | Transport | Effective diffusivity | molecular_diff, porosity |
| 43 | `pressure_drop_ergun` | Transport | Ergun pressure drop | velocity, density, particles |
| 44 | `enthalpy_c` | Thermodynamics | Constant Cp enthalpy | temperature, heat_capacity |
| 45 | `entropy_c` | Thermodynamics | Constant Cp entropy | temperature, heat_capacity |
| 46 | `analytical_first_order` | Analytical | First-order solution | t, k, C0 |
| 47 | `analytical_reversible_first_order` | Analytical | Reversible first-order | t, kf, kr, A0, B0 |
| 48 | `analytical_consecutive_first_order` | Analytical | Consecutive reactions | t, k1, k2, A0 |
| 49 | `check_mass_conservation` | Validation | Mass balance check | initial_mass, final_mass |
| 50 | `calculate_rate_constants` | Validation | Rate constant fitting | concentrations, rates |
| 51 | `residence_time_distribution` | Validation | RTD calculation | flow_pattern, residence_time |
| 52 | `catalyst_deactivation_model` | Validation | Catalyst deactivation | time, activity, mechanism |
| 53 | `process_scale_up` | Validation | Process scaling | lab_data, scale_factor |
| 54 | `WellMixedReactor` | Reactor Class | Perfect mixing reactor | reaction, initial conditions |
| 55 | `CSTR` | Reactor Class | Continuous stirred tank | reaction, residence_time |
| 56 | `PFR` | Reactor Class | Plug flow reactor | reaction, length, velocity |
| 57 | `PackedBedReactor` | Reactor Class | Packed bed reactor | complex catalyst parameters |
| 58 | `FluidizedBedReactor` | Reactor Class | Fluidized bed reactor | fluidization parameters |
| 59 | `HeterogeneousReactor` | Reactor Class | Heterogeneous reactor | surface reaction parameters |
| 60 | `HomogeneousReactor` | Reactor Class | Homogeneous reactor | multiple reactions |
| 61 | `MultiReactor` | Reactor Class | Multiple reactor system | reactor configurations |
| 62 | `ReactorNetwork` | Reactor Class | Reactor network | network topology |
| 63 | `PyroXaError` | Error Class | Base exception | error message |
| 64 | `ThermodynamicsError` | Error Class | Thermodynamics errors | specific error type |
| 65 | `ReactionError` | Error Class | Reaction errors | kinetics validation |
| 66 | `ReactorError` | Error Class | Reactor errors | reactor validation |
| 67 | `ReactionMulti` | Reaction Class | Multiple reactions | reaction list |
| 68 | `benchmark_multi_reactor` | Analysis Tool | Reactor benchmarking | configurations, metrics |
| 69 | `ChainReactorVisualizer` | Analysis Tool | Network visualization | plotting and animation |
| 70 | `OptimalReactorDesign` | Analysis Tool | Design optimization | objectives, constraints |
| 71 | `ReactionChain` | Analysis Tool | Reaction chain modeling | reaction sequences |
| 72 | `create_reaction_chain` | Analysis Tool | Chain creation utility | reactions, species |
| 73 | `get_build_info` | System Info | Build information | - |
| 74 | `get_version` | System Info | Version string | - |
| 75 | `is_compiled_available` | System Info | C++ availability check | - |
| 76 | `is_reaction_chains_available` | System Info | Feature availability | - |
| 77 | `run_simulation` | High-level | Simulation runner | spec, output options |
| 78 | `run_simulation_cpp` | High-level | C++ simulation runner | parameters |
| 79 | `build_from_dict` | High-level | Build from specification | spec dictionary |
| 80 | `Thermodynamics` | Core Class | Thermodynamic properties | cp, T_ref |
| 81 | `Reaction` | Core Class | Chemical reaction | kf, kr |
| 82 | `calculate_energy_balance` | Energy | Energy balance | 7 energy parameters |
| 83 | `Reactor` | Core Class | Base reactor class | reaction, conditions |
| 84 | `benchmark_multi_reactor` | Benchmarking | Performance comparison | configurations |
| 85 | `io` | Module | I/O utilities module | file operations |
| 86 | `new_functions` | Module | New implementations | recent additions |
| 87 | `purepy` | Module | Pure Python module | fallback implementations |
| 88 | `PIDController` | Control Class | Advanced PID control | tuning and state management |

---

## Conclusion

PyroXa provides a comprehensive, professional-grade platform for chemical kinetics and reactor simulation. With **89 functions** covering everything from basic thermodynamics to advanced reactor networks, it serves both educational and industrial applications.

### What Makes PyroXa Special

1. **Complete Coverage**: All aspects of chemical kinetics from thermodynamics to process control
2. **Beginner Friendly**: Extensive tutorials and examples for newcomers to chemical engineering
3. **Professional Grade**: Industry-standard correlations (NASA polynomials, Peng-Robinson EOS)
4. **Dual Architecture**: Pure Python for learning + C++ extensions for performance
5. **Robust Design**: Comprehensive error handling and validation throughout
6. **Real-World Ready**: Complex reactor simulations with heat/mass transfer

### Learning Path for Beginners

1. **Start Simple**: Use basic classes like `Reaction` and `WellMixedReactor`
2. **Follow Tutorials**: Work through the step-by-step examples provided
3. **Experiment**: Modify parameters and observe effects on results  
4. **Build Complexity**: Progress to CSTR, PFR, and packed bed reactors
5. **Go Industrial**: Tackle complex multi-reactor networks and optimization

### For Advanced Users

- **High-Performance**: C++ extensions for production simulations
- **Optimization**: Built-in sensitivity analysis and parameter estimation
- **Visualization**: Advanced plotting and network visualization tools
- **Uncertainty**: Monte Carlo and bootstrap uncertainty quantification
- **Integration**: YAML configuration and CSV output for workflow integration

### Future Development

PyroXa continues to evolve with new features:
- Enhanced thermodynamic packages
- Machine learning integration
- Cloud computing support
- Extended reactor library
- Advanced control systems

### Support and Community

- **Documentation**: This comprehensive guide covers all functions
- **Examples**: Over 20 working examples in the `examples/` directory
- **Testing**: Extensive test suite validates all functionality
- **GitHub**: Active development and issue tracking
- **Academic**: Suitable for courses from undergraduate to PhD level

### Final Words

Whether you're a student learning chemical kinetics, a researcher developing new processes, or an engineer optimizing industrial reactors, PyroXa provides the tools you need. The combination of educational accessibility and professional capability makes it unique in the chemical engineering software landscape.

Start with the tutorials, explore the examples, and build confidence with progressively complex simulations. The 88 functions documented here provide endless possibilities for chemical process simulation and optimization.

**Happy simulating with PyroXa!**

---

*This documentation covers PyroXa v1.0 with all 88 functions - For updates, examples, and community support, visit the PyroXa GitHub repository.*

---

### Quick Reference Card

**Essential Functions for Beginners:**
- `Reaction(kf, kr)` - Define reactions
- `WellMixedReactor()` - Batch reactor
- `CSTR()` - Continuous stirred tank
- `run_simulation()` - High-level runner
- `build_from_dict()` - Configuration-based setup

**Advanced Functions for Professionals:**
- `simulate_packed_bed()` - Industrial packed beds
- `monte_carlo_simulation()` - Uncertainty analysis
- `calculate_sensitivity()` - Sensitivity analysis
- `mpc_controller()` - Advanced process control
- `process_scale_up()` - Industrial scale-up

**Remember**: Always check `is_compiled_available()` for optimal performance!
