# PyroXa API Reference

## Core C++ Functions

### Basic Reactor Simulation

#### `simulate_reactor`
```c
int simulate_reactor(double kf, double kr, double A0, double B0,
                     double time_span, double dt,
                     double* times, double* Aout, double* Bout, int max_len);
```
Simulates a simple reversible reaction A ⇌ B in a well-mixed reactor using RK4 integration.

**Parameters:**
- `kf`: Forward rate constant (1/s)
- `kr`: Reverse rate constant (1/s)  
- `A0`: Initial concentration of species A (mol/L)
- `B0`: Initial concentration of species B (mol/L)
- `time_span`: Total simulation time (s)
- `dt`: Time step (s)
- `times`: Output array for time points
- `Aout`: Output array for species A concentrations
- `Bout`: Output array for species B concentrations
- `max_len`: Maximum number of output points

**Returns:** Number of time points simulated, or -1 on error

#### `simulate_multi_reactor`
```c
int simulate_multi_reactor(int N, int M,
                           double* kf, double* kr,
                           int* reac_idx, double* reac_nu, int* reac_off,
                           int* prod_idx, double* prod_nu, int* prod_off,
                           double* conc0,
                           double time_span, double dt,
                           double* times, double* conc_out_flat, int max_len);
```
Simulates multi-species, multi-reaction systems using RK4 integration.

**Parameters:**
- `N`: Number of species
- `M`: Number of reactions
- `kf`: Array of forward rate constants
- `kr`: Array of reverse rate constants
- `reac_idx`: Reactant species indices
- `reac_nu`: Reactant stoichiometric coefficients
- `reac_off`: Reactant offset array for indexing
- `prod_idx`: Product species indices
- `prod_nu`: Product stoichiometric coefficients
- `prod_off`: Product offset array for indexing
- `conc0`: Initial concentrations
- Other parameters as above

### Adaptive Integration

#### `simulate_reactor_adaptive`
```c
int simulate_reactor_adaptive(double kf, double kr, double A0, double B0,
                              double time_span, double dt_init, double atol, double rtol,
                              double* times, double* Aout, double* Bout, int max_len);
```
Adaptive Cash-Karp RK45 integration for single A ⇌ B reaction.

**Parameters:**
- Standard reactor parameters plus:
- `dt_init`: Initial time step
- `atol`: Absolute tolerance for error control
- `rtol`: Relative tolerance for error control

#### `simulate_multi_reactor_adaptive`
Similar to above but for multi-species systems.

### Advanced Reactor Types

#### `simulate_pfr`
```c
int simulate_pfr(int N, int M, int nseg,
                 double* kf, double* kr,
                 int* reac_idx, double* reac_nu, int* reac_off,
                 int* prod_idx, double* prod_nu, int* prod_off,
                 double* conc0, double flow_rate, double total_volume,
                 double time_span, double dt,
                 double* times, double* conc_out_flat, int max_len);
```
Plug Flow Reactor simulation with spatial discretization.

**Additional Parameters:**
- `nseg`: Number of spatial segments
- `flow_rate`: Volumetric flow rate (L/s)
- `total_volume`: Total reactor volume (L)

#### `simulate_cstr`
```c
int simulate_cstr(int N, int M,
                  double* kf, double* kr,
                  int* reac_idx, double* reac_nu, int* reac_off,
                  int* prod_idx, double* prod_nu, int* prod_off,
                  double* conc0, double* conc_in, double flow_rate, double volume,
                  double time_span, double dt,
                  double* times, double* conc_out_flat, int max_len);
```
Continuous Stirred Tank Reactor with flow.

**Additional Parameters:**
- `conc_in`: Inlet concentrations
- `volume`: Reactor volume (L)

#### `simulate_batch_variable_temp`
```c
int simulate_batch_variable_temp(int N, int M,
                                 double* kf_ref, double* kr_ref, double* Ea_f, double* Ea_r,
                                 int* reac_idx, double* reac_nu, int* reac_off,
                                 int* prod_idx, double* prod_nu, int* prod_off,
                                 double* conc0, double* temp_profile, double T_ref,
                                 double time_span, double dt,
                                 double* times, double* conc_out_flat, 
                                 double* temp_out, int max_len);
```
Batch reactor with variable temperature profile and Arrhenius kinetics.

**Additional Parameters:**
- `kf_ref`, `kr_ref`: Reference rate constants at T_ref
- `Ea_f`, `Ea_r`: Activation energies (J/mol)
- `temp_profile`: Temperature vs time array
- `T_ref`: Reference temperature (K)

## Thermodynamics Functions

#### `enthalpy_c`
```c
double enthalpy_c(double cp, double T);
```
Calculate enthalpy using constant heat capacity model.

#### `entropy_c`
```c
double entropy_c(double cp, double T);
```
Calculate entropy using constant heat capacity model.

#### `gibbs_free_energy`
```c
double gibbs_free_energy(double enthalpy, double entropy, double T);
```
Calculate Gibbs free energy: G = H - TS

#### `equilibrium_constant`
```c
double equilibrium_constant(double delta_G, double T);
```
Calculate equilibrium constant from Gibbs free energy: K = exp(-ΔG/RT)

#### `arrhenius_rate`
```c
double arrhenius_rate(double A, double Ea, double T, double R);
```
Calculate rate constant using Arrhenius equation: k = A × exp(-Ea/RT)

#### `calculate_rate_constants`
```c
void calculate_rate_constants(int M, double* kf_ref, double* kr_ref,
                             double* Ea_f, double* Ea_r, double T, double T_ref,
                             double* kf_out, double* kr_out);
```
Calculate temperature-dependent rate constants for multiple reactions.

### Advanced Thermodynamic Properties

#### `heat_capacity_nasa`
```c
double heat_capacity_nasa(double T, double* coeffs);
```
Calculate heat capacity using NASA polynomial coefficients.

#### `enthalpy_nasa`
```c
double enthalpy_nasa(double T, double* coeffs);
```
Calculate enthalpy from NASA polynomials.

#### `entropy_nasa`
```c
double entropy_nasa(double T, double* coeffs);
```
Calculate entropy from NASA polynomials.

#### `pressure_peng_robinson`
```c
double pressure_peng_robinson(double n, double V, double T, 
                             double Tc, double Pc, double omega);
```
Peng-Robinson equation of state for real gas behavior.

#### `fugacity_coefficient`
```c
double fugacity_coefficient(double P, double T, double Tc, double Pc, double omega);
```
Calculate fugacity coefficient for non-ideal gas behavior.

## Reaction Kinetics Functions

#### `reaction_rate`
```c
double reaction_rate(double kf, double kr, double A, double B);
```
Basic reaction rate: r = kf×A - kr×B

#### `reaction_rate_temperature`
```c
double reaction_rate_temperature(double kf_ref, double kr_ref, 
                               double Ea_f, double Ea_r,
                               double T, double T_ref,
                               double A, double B);
```
Temperature-dependent reaction rate with Arrhenius kinetics.

#### `michaelis_menten_rate`
```c
double michaelis_menten_rate(double Vmax, double Km, double substrate_conc);
```
Michaelis-Menten enzyme kinetics: v = Vmax×S/(Km + S)

#### `competitive_inhibition_rate`
```c
double competitive_inhibition_rate(double Vmax, double Km, 
                                 double substrate_conc, double inhibitor_conc,
                                 double Ki);
```
Competitive enzyme inhibition kinetics.

#### `autocatalytic_rate`
```c
double autocatalytic_rate(double k, double A, double B);
```
Autocatalytic reaction rate: r = k×A×B

#### `langmuir_hinshelwood_rate`
```c
double langmuir_hinshelwood_rate(double k, double K_A, double K_B,
                               double conc_A, double conc_B);
```
Langmuir-Hinshelwood surface reaction kinetics.

#### `photochemical_rate`
```c
double photochemical_rate(double quantum_yield, double molar_absorptivity,
                        double path_length, double light_intensity,
                        double concentration);
```
Photochemical reaction rate with Beer-Lambert light absorption.

## Analytical Solutions

#### `analytical_first_order`
```c
int analytical_first_order(double k, double A0, double time_span, double dt,
                          double* times, double* A_out, double* B_out, int max_len);
```
Analytical solution for first-order reaction A → B.

#### `analytical_consecutive_first_order`
```c
int analytical_consecutive_first_order(double k1, double k2, double A0,
                                      double time_span, double dt,
                                      double* times, double* A_out, 
                                      double* B_out, double* C_out, int max_len);
```
Analytical solution for consecutive reactions A → B → C.

#### `analytical_reversible_first_order`
```c
int analytical_reversible_first_order(double kf, double kr, double A0, double B0,
                                     double time_span, double dt,
                                     double* times, double* A_out, double* B_out, int max_len);
```
Analytical solution for reversible first-order reaction A ⇌ B.

## Analysis and Optimization

#### `calculate_sensitivity`
```c
int calculate_sensitivity(int N, int M, int nparam,
                         double* kf, double* kr, double* param_perturbations,
                         int* reac_idx, double* reac_nu, int* reac_off,
                         int* prod_idx, double* prod_nu, int* prod_off,
                         double* conc0, double time_span, double dt,
                         double* sensitivity_matrix, int max_len);
```
Calculate parameter sensitivity using finite differences.

#### `calculate_objective_function`
```c
double calculate_objective_function(int ndata, double* experimental_data,
                                   double* simulated_data, double* weights);
```
Calculate weighted sum of squared residuals for parameter estimation.

#### `find_steady_state`
```c
int find_steady_state(int N, int M,
                     double* kf, double* kr,
                     int* reac_idx, double* reac_nu, int* reac_off,
                     int* prod_idx, double* prod_nu, int* prod_off,
                     double* conc_guess, double* conc_steady,
                     double tolerance, int max_iterations);
```
Find steady-state concentrations using Newton-Raphson iteration.

## Utility Functions

#### `matrix_multiply`
```c
int matrix_multiply(double* A, double* B, double* C, int m, int n, int p);
```
Matrix multiplication: C = A × B

#### `linear_interpolate`
```c
double linear_interpolate(double x, double* x_data, double* y_data, int n);
```
Linear interpolation between data points.

#### `calculate_r_squared`
```c
double calculate_r_squared(double* experimental, double* predicted, int n);
```
Calculate coefficient of determination (R²) for model validation.

#### `calculate_rmse`
```c
double calculate_rmse(double* experimental, double* predicted, int n);
```
Calculate root mean square error.

#### `calculate_aic`
```c
double calculate_aic(double* experimental, double* predicted, int ndata, int nparams);
```
Calculate Akaike Information Criterion for model selection.

## C++ Classes (Advanced Interface)

### `pyroxa::ReactionNetwork`
```cpp
class ReactionNetwork {
public:
    ReactionNetwork(int n_species, int n_reactions);
    
    void add_reaction(const std::vector<int>& reactants, 
                     const std::vector<double>& reactant_stoich,
                     const std::vector<int>& products,
                     const std::vector<double>& product_stoich,
                     double kf, double kr);
    
    std::vector<double> simulate(const std::vector<double>& initial_conc,
                               double time_span, double dt);
    
    std::vector<double> find_steady_state(const std::vector<double>& guess);
};
```

### `pyroxa::ReactorSimulator`
```cpp
class ReactorSimulator {
public:
    ReactorSimulator();
    
    void set_parallel_threads(int nthreads);
    void enable_adaptive_stepping(double atol, double rtol);
    void enable_conservation_checking(double tolerance);
    
    std::vector<std::vector<double>> simulate_batch(
        const ReactionNetwork& network,
        const std::vector<double>& initial_conc,
        double time_span, double dt);
};
```

### `pyroxa::AdvancedReaction`
```cpp
class AdvancedReaction {
public:
    AdvancedReaction(const std::vector<int>& reactants,
                    const std::vector<double>& reactant_stoich,
                    const std::vector<int>& products,
                    const std::vector<double>& product_stoich,
                    double kf, double kr = 0.0);
    
    void set_arrhenius_parameters(double Ea_forward, double Ea_reverse, double T_ref);
    void set_reaction_type(int type);
    
    double calculate_rate(const std::vector<double>& concentrations, 
                         double temperature = 298.15) const;
};
```

### `pyroxa::ThermodynamicProperties`
```cpp
class ThermodynamicProperties {
public:
    ThermodynamicProperties(const std::string& name, double mw,
                          const std::vector<double>& coeffs_low,
                          const std::vector<double>& coeffs_high,
                          double T_transition = 1000.0);
    
    double heat_capacity(double T) const;
    double enthalpy(double T) const;
    double entropy(double T) const;
    double gibbs_free_energy(double T) const;
};
```

### `pyroxa::MixtureThermodynamics`
```cpp
class MixtureThermodynamics {
public:
    void add_species(const ThermodynamicProperties& species, double mole_fraction);
    
    double mixture_heat_capacity(double T) const;
    double mixture_enthalpy(double T) const;
    double mixture_entropy(double T) const;
    double mixture_molecular_weight() const;
};
```

## Error Codes

- `0`: Success
- `-1`: Invalid input parameters
- `-2`: Memory allocation failure
- `-3`: Convergence failure
- `-4`: Integration failure
- `-5`: Matrix inversion failure

## Constants

- `PI = 3.14159265358979323846`
- `R_GAS = 8.31446261815324` (J/mol/K)
- `BOLTZMANN = 1.380649e-23` (J/K)
- `AVOGADRO = 6.02214076e23` (mol⁻¹)

## Advanced Reactor Types (Python Interface)

### `PackedBedReactor`
```python
class PackedBedReactor:
    def __init__(self, bed_length: float, bed_porosity: float, 
                 particle_diameter: float, catalyst_density: float,
                 effectiveness_factor: float = 1.0, flow_rate: float = 1.0)
    
    def add_reaction(self, reaction)
    def run(self, time_span: float, dt: float = 0.01) -> Dict
```
Packed bed reactor with spatial discretization and catalyst effectiveness factor.

**Key Features:**
- Spatial concentration profiles along bed length
- Pressure drop calculation (Ergun equation)
- Catalyst effectiveness factor for mass transfer limitations
- Enhanced numerical stability with bounds checking

**Parameters:**
- `bed_length`: Length of catalyst bed (m)
- `bed_porosity`: Void fraction of bed (0-1)
- `particle_diameter`: Catalyst particle diameter (m)
- `catalyst_density`: Catalyst bulk density (kg/m³)
- `effectiveness_factor`: Catalyst effectiveness factor (0-1)
- `flow_rate`: Volumetric flow rate (m³/s)

**Returns:** Dictionary with concentration profiles, conversion, and pressure drop

### `FluidizedBedReactor`
```python
class FluidizedBedReactor:
    def __init__(self, bed_height: float, bed_porosity: float, 
                 bubble_fraction: float, particle_diameter: float,
                 catalyst_density: float, gas_velocity: float)
    
    def add_reaction(self, reaction)
    def run(self, time_span: float, dt: float = 0.01) -> Dict
```
Fluidized bed reactor with two-phase model (bubble and emulsion phases).

**Key Features:**
- Two-phase hydrodynamics modeling
- Inter-phase mass transfer
- Bubble velocity calculation
- Phase-specific concentration tracking

**Parameters:**
- `bed_height`: Height of fluidized bed (m)
- `bed_porosity`: Overall bed porosity (0-1)
- `bubble_fraction`: Fraction of bed occupied by bubbles (0-1)
- `particle_diameter`: Catalyst particle diameter (m)
- `catalyst_density`: Catalyst density (kg/m³)
- `gas_velocity`: Superficial gas velocity (m/s)

**Returns:** Dictionary with bubble/emulsion concentrations, overall conversion, and bubble velocity

### `HeterogeneousReactor`
```python
class HeterogeneousReactor:
    def __init__(self, gas_holdup: float, liquid_holdup: float, solid_holdup: float,
                 mass_transfer_gas_liquid: List[float], 
                 mass_transfer_liquid_solid: List[float])
    
    def add_gas_reaction(self, reaction)
    def add_liquid_reaction(self, reaction)
    def add_solid_reaction(self, reaction)
    def run(self, time_span: float, dt: float = 0.01) -> Dict
```
Three-phase heterogeneous reactor with gas-liquid-solid interactions.

**Key Features:**
- Independent reactions in each phase
- Inter-phase mass transfer modeling
- Phase holdup considerations
- Comprehensive mass balance tracking

**Parameters:**
- `gas_holdup`: Gas phase volume fraction (0-1)
- `liquid_holdup`: Liquid phase volume fraction (0-1)
- `solid_holdup`: Solid phase volume fraction (0-1)
- `mass_transfer_gas_liquid`: Mass transfer coefficients [A, B] between gas-liquid
- `mass_transfer_liquid_solid`: Mass transfer coefficients [A, B] between liquid-solid

**Returns:** Dictionary with phase-specific concentrations and overall conversion

### `HomogeneousReactor`
```python
class HomogeneousReactor(WellMixedReactor):
    def __init__(self, reaction, mixing_intensity: float = 1.0, 
                 volume: float = 1.0, **kwargs)
    
    def run(self, time_span: float, dt: float = 0.01) -> Dict
```
Enhanced homogeneous reactor with mixing intensity effects (inherits from WellMixedReactor).

**Key Features:**
- Mixing intensity effects on reaction rates
- Enhanced reaction rates due to improved mixing
- Time-dependent mixing efficiency calculation
- Comprehensive results dictionary

**Parameters:**
- `reaction`: Reaction object to simulate
- `mixing_intensity`: Mixing intensity parameter (higher = better mixing)
- `volume`: Reactor volume (m³)
- `**kwargs`: Additional parameters passed to WellMixedReactor

**Returns:** Dictionary with concentrations, mixing efficiency, and mixing intensity

## C++ Advanced Functions

### `simulate_packed_bed`
```c
int simulate_packed_bed(double bed_length, double bed_porosity, 
                       double particle_diameter, double catalyst_density,
                       double effectiveness_factor, double flow_rate,
                       double* conc0, double time_span, double dt,
                       double* conc_profiles, double* pressure_profiles,
                       int nseg, int max_len);
```

### `simulate_fluidized_bed`
```c
int simulate_fluidized_bed(double bed_height, double bed_porosity,
                          double bubble_fraction, double particle_diameter,
                          double catalyst_density, double gas_velocity,
                          double* conc_bubble, double* conc_emulsion,
                          double time_span, double dt,
                          double* bubble_out, double* emulsion_out,
                          double* overall_out, int max_len);
```

### `simulate_three_phase_reactor`
```c
int simulate_three_phase_reactor(double gas_holdup, double liquid_holdup,
                                double solid_holdup, double* kgl, double* kls,
                                double* conc_gas, double* conc_liquid, double* conc_solid,
                                double time_span, double dt,
                                double* gas_out, double* liquid_out, double* solid_out,
                                int max_len);
```

## Performance Notes

- Use adaptive integration for stiff systems
- Enable OpenMP for parallel computations
- Prefer analytical solutions when available
- Use appropriate tolerances to balance accuracy and speed
- Consider C++ interface for high-performance applications
- Advanced reactors include numerical stability controls and bounds checking

## Thread Safety

- All C functions are thread-safe when called with separate data
- C++ classes are not thread-safe - use separate instances per thread
- Parallel functions manage their own thread safety internally
