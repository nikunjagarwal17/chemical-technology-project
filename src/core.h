#ifndef PYROXA_CORE_H
#define PYROXA_CORE_H

#include <vector>
#include <memory>

extern "C" {

// ============================================================================
// CORE SIMULATION FUNCTIONS
// ============================================================================

// Single well-mixed reactor (A <=> B) with enhanced RK4
int simulate_reactor(double kf, double kr, double A0, double B0,
                     double time_span, double dt,
                     double* times, double* Aout, double* Bout, int max_len);

// Multi-species multi-reaction RK4 simulator with optimizations
int simulate_multi_reactor(int N, int M,
                           double* kf, double* kr,
                           int* reac_idx, double* reac_nu, int* reac_off,
                           int* prod_idx, double* prod_nu, int* prod_off,
                           double* conc0,
                           double time_span, double dt,
                           double* times, double* conc_out_flat, int max_len);

// ============================================================================
// ADAPTIVE INTEGRATION (RK45 Cash-Karp)
// ============================================================================

// Adaptive single reactor with advanced error control
int simulate_reactor_adaptive(double kf, double kr, double A0, double B0,
                              double time_span, double dt_init, double atol, double rtol,
                              double* times, double* Aout, double* Bout, int max_len);

// Adaptive multi-species with enhanced stability
int simulate_multi_reactor_adaptive(int N, int M,
                                    double* kf, double* kr,
                                    int* reac_idx, double* reac_nu, int* reac_off,
                                    int* prod_idx, double* prod_nu, int* prod_off,
                                    double* conc0,
                                    double time_span, double dt_init, double atol, double rtol,
                                    double* times, double* conc_out_flat, int max_len);

// ============================================================================
// ADVANCED REACTOR TYPES
// ============================================================================

// Plug Flow Reactor (PFR) with spatial discretization
int simulate_pfr(int N, int M, int nseg,
                 double* kf, double* kr,
                 int* reac_idx, double* reac_nu, int* reac_off,
                 int* prod_idx, double* prod_nu, int* prod_off,
                 double* conc0, double flow_rate, double total_volume,
                 double time_span, double dt,
                 double* times, double* conc_out_flat, int max_len);

// Continuous Stirred Tank Reactor (CSTR) with flow
int simulate_cstr(int N, int M,
                  double* kf, double* kr,
                  int* reac_idx, double* reac_nu, int* reac_off,
                  int* prod_idx, double* prod_nu, int* prod_off,
                  double* conc0, double* conc_in, double flow_rate, double volume,
                  double time_span, double dt,
                  double* times, double* conc_out_flat, int max_len);

// ============================================================================
// PACKED BED REACTOR TYPES
// ============================================================================

// Packed Bed Reactor (PBR) with catalyst particles
int simulate_packed_bed(int N, int M, int nseg,
                       double* kf, double* kr,
                       int* reac_idx, double* reac_nu, int* reac_off,
                       int* prod_idx, double* prod_nu, int* prod_off,
                       double* conc0, double flow_rate, double bed_length,
                       double bed_porosity, double particle_diameter,
                       double catalyst_density, double effectiveness_factor,
                       double time_span, double dt,
                       double* times, double* conc_out_flat, 
                       double* pressure_out, int max_len);

// Heterogeneous Packed Bed with mass transfer limitations
int simulate_heterogeneous_packed_bed(int N, int M, int nseg,
                                     double* kf_intrinsic, double* kr_intrinsic,
                                     int* reac_idx, double* reac_nu, int* reac_off,
                                     int* prod_idx, double* prod_nu, int* prod_off,
                                     double* conc0, double flow_rate, double bed_length,
                                     double bed_porosity, double particle_diameter,
                                     double catalyst_density, double* mass_transfer_coeff,
                                     double time_span, double dt,
                                     double* times, double* conc_out_flat,
                                     double* surface_conc_out, int max_len);

// ============================================================================
// FLUIDIZED BED REACTOR TYPES
// ============================================================================

// Bubbling Fluidized Bed Reactor
int simulate_fluidized_bed(int N, int M,
                          double* kf, double* kr,
                          int* reac_idx, double* reac_nu, int* reac_off,
                          int* prod_idx, double* prod_nu, int* prod_off,
                          double* conc0, double gas_velocity, double bed_height,
                          double bed_porosity, double bubble_fraction,
                          double particle_diameter, double catalyst_density,
                          double time_span, double dt,
                          double* times, double* conc_out_flat,
                          double* bubble_conc_out, double* emulsion_conc_out, int max_len);

// Circulating Fluidized Bed Reactor with riser-regenerator
int simulate_circulating_fluidized_bed(int N, int M,
                                      double* kf_riser, double* kr_riser,
                                      double* kf_regen, double* kr_regen,
                                      int* reac_idx, double* reac_nu, int* reac_off,
                                      int* prod_idx, double* prod_nu, int* prod_off,
                                      double* conc0, double circulation_rate,
                                      double riser_height, double regen_height,
                                      double riser_diameter, double regen_diameter,
                                      double catalyst_activity,
                                      double time_span, double dt,
                                      double* times, double* riser_conc_out,
                                      double* regen_conc_out, int max_len);

// ============================================================================
// HOMOGENEOUS AND HETEROGENEOUS REACTOR TYPES
// ============================================================================

// Homogeneous Batch Reactor with enhanced mixing
int simulate_homogeneous_batch(int N, int M,
                              double* kf, double* kr,
                              int* reac_idx, double* reac_nu, int* reac_off,
                              int* prod_idx, double* prod_nu, int* prod_off,
                              double* conc0, double volume, double mixing_intensity,
                              double time_span, double dt,
                              double* times, double* conc_out_flat,
                              double* mixing_efficiency_out, int max_len);

// Heterogeneous Three-Phase Reactor (Gas-Liquid-Solid)
int simulate_three_phase_reactor(int N, int M,
                                double* kf_gas, double* kr_gas,
                                double* kf_liquid, double* kr_liquid,
                                double* kf_solid, double* kr_solid,
                                int* reac_idx, double* reac_nu, int* reac_off,
                                int* prod_idx, double* prod_nu, int* prod_off,
                                double* conc0_gas, double* conc0_liquid, double* conc0_solid,
                                double* mass_transfer_gas_liquid,
                                double* mass_transfer_liquid_solid,
                                double gas_holdup, double liquid_holdup, double solid_holdup,
                                double time_span, double dt,
                                double* times, double* gas_conc_out,
                                double* liquid_conc_out, double* solid_conc_out, int max_len);

// Batch reactor with variable temperature profile
int simulate_batch_variable_temp(int N, int M,
                                 double* kf_ref, double* kr_ref, double* Ea_f, double* Ea_r,
                                 int* reac_idx, double* reac_nu, int* reac_off,
                                 int* prod_idx, double* prod_nu, int* prod_off,
                                 double* conc0, double* temp_profile, double T_ref,
                                 double time_span, double dt,
                                 double* times, double* conc_out_flat, 
                                 double* temp_out, int max_len);

// ============================================================================
// THERMODYNAMICS AND KINETICS
// ============================================================================

// Enhanced thermodynamics with real gas effects
double enthalpy_c(double cp, double T);
double entropy_c(double cp, double T);
double gibbs_free_energy(double enthalpy, double entropy, double T);
double equilibrium_constant(double delta_G, double T);

// Arrhenius rate calculation
double arrhenius_rate(double A, double Ea, double T, double R);

// Temperature-dependent rate constants
void calculate_rate_constants(int M, double* kf_ref, double* kr_ref,
                             double* Ea_f, double* Ea_r, double T, double T_ref,
                             double* kf_out, double* kr_out);

// ============================================================================
// ANALYTICAL SOLUTIONS
// ============================================================================

// Analytical solution for A -> B (first order)
int analytical_first_order(double k, double A0, double time_span, double dt,
                          double* times, double* A_out, double* B_out, int max_len);

// Analytical solution for A -> B -> C (consecutive first order)
int analytical_consecutive_first_order(double k1, double k2, double A0,
                                      double time_span, double dt,
                                      double* times, double* A_out, 
                                      double* B_out, double* C_out, int max_len);

// Analytical solution for A <=> B (reversible first order)
int analytical_reversible_first_order(double kf, double kr, double A0, double B0,
                                     double time_span, double dt,
                                     double* times, double* A_out, double* B_out, int max_len);

// ============================================================================
// OPTIMIZATION AND SENSITIVITY ANALYSIS
// ============================================================================

// Parameter sensitivity calculation (finite differences)
int calculate_sensitivity(int N, int M, int nparam,
                         double* kf, double* kr, double* param_perturbations,
                         int* reac_idx, double* reac_nu, int* reac_off,
                         int* prod_idx, double* prod_nu, int* prod_off,
                         double* conc0, double time_span, double dt,
                         double* sensitivity_matrix, int max_len);

// Objective function for optimization (sum of squared residuals)
double calculate_objective_function(int ndata, double* experimental_data,
                                   double* simulated_data, double* weights);

// Jacobian matrix calculation for parameter estimation
int calculate_jacobian(int N, int M, int nparam, int ndata,
                      double* parameters, double* experimental_data,
                      double* jacobian_matrix);

// ============================================================================
// STEADY STATE AND STABILITY ANALYSIS
// ============================================================================

// Find steady state using Newton-Raphson
int find_steady_state(int N, int M,
                     double* kf, double* kr,
                     int* reac_idx, double* reac_nu, int* reac_off,
                     int* prod_idx, double* prod_nu, int* prod_off,
                     double* conc_guess, double* conc_steady,
                     double tolerance, int max_iterations);

// Eigenvalue analysis for stability
int stability_analysis(int N, int M,
                      double* kf, double* kr,
                      int* reac_idx, double* reac_nu, int* reac_off,
                      int* prod_idx, double* prod_nu, int* prod_off,
                      double* conc_steady, double* eigenvalues_real,
                      double* eigenvalues_imag);

// ============================================================================
// MASS AND ENERGY CONSERVATION
// ============================================================================

// Check mass conservation during simulation
int check_mass_conservation(int N, int npoints, double* conc_trajectory,
                           double* mass_balance, double tolerance);

// Energy balance calculation for non-isothermal systems
int calculate_energy_balance(int N, int M, double* conc, double* reaction_rates,
                            double* enthalpies_formation, double* heat_capacities,
                            double T, double* heat_generation);

// ============================================================================
// ADVANCED NUMERICAL METHODS
// ============================================================================

// Backward Differentiation Formula (BDF) for stiff systems
int simulate_reactor_bdf(int N, int M,
                        double* kf, double* kr,
                        int* reac_idx, double* reac_nu, int* reac_off,
                        int* prod_idx, double* prod_nu, int* prod_off,
                        double* conc0, double time_span, double dt,
                        double* times, double* conc_out_flat, int max_len);

// Implicit Runge-Kutta methods for very stiff systems
int simulate_reactor_implicit_rk(int N, int M,
                               double* kf, double* kr,
                               int* reac_idx, double* reac_nu, int* reac_off,
                               int* prod_idx, double* prod_nu, int* prod_off,
                               double* conc0, double time_span, double dt,
                               double* times, double* conc_out_flat, int max_len);

// Gear's method for stiff ODEs
int simulate_reactor_gear(int N, int M, int order,
                         double* kf, double* kr,
                         int* reac_idx, double* reac_nu, int* reac_off,
                         int* prod_idx, double* prod_nu, int* prod_off,
                         double* conc0, double time_span, double dt,
                         double* times, double* conc_out_flat, int max_len);

// ============================================================================
// PARALLEL PROCESSING SUPPORT
// ============================================================================

// Multi-threaded parameter sweep
int parameter_sweep_parallel(int N, int M, int nsweep,
                            double* kf_base, double* kr_base,
                            double* param_ranges, int* param_indices,
                            int* reac_idx, double* reac_nu, int* reac_off,
                            int* prod_idx, double* prod_nu, int* prod_off,
                            double* conc0, double time_span, double dt,
                            double* results_matrix, int nthreads);

// Parallel Monte Carlo simulation
int monte_carlo_simulation(int N, int M, int nsamples,
                          double* kf_mean, double* kr_mean,
                          double* kf_std, double* kr_std,
                          int* reac_idx, double* reac_nu, int* reac_off,
                          int* prod_idx, double* prod_nu, int* prod_off,
                          double* conc0, double time_span, double dt,
                          double* statistics_output, int nthreads);

// ============================================================================
// ADVANCED REACTOR NETWORK ANALYSIS
// ============================================================================

// Multi-reactor network simulation
int simulate_reactor_network(int n_reactors, int N, int M,
                            double* reactor_volumes, double* flow_rates,
                            int* connectivity_matrix,
                            double* kf, double* kr,
                            int* reac_idx, double* reac_nu, int* reac_off,
                            int* prod_idx, double* prod_nu, int* prod_off,
                            double* conc0, double time_span, double dt,
                            double* times, double* conc_out_flat, int max_len);

// Residence time distribution analysis
int calculate_rtd(int n_reactors, double* volumes, double* flow_rates,
                 int* connectivity, double time_span, double dt,
                 double* rtd_output);

// ============================================================================
// CONTROL AND OPTIMIZATION
// ============================================================================

// PID controller for reactor temperature
double pid_controller(double setpoint, double process_variable, double dt,
                     double Kp, double Ki, double Kd,
                     double* integral_term, double* previous_error);

// Model predictive control
int mpc_controller(int N, int M, int horizon,
                  double* current_state, double* setpoints,
                  double* control_bounds,
                  double* kf, double* kr,
                  int* reac_idx, double* reac_nu, int* reac_off,
                  int* prod_idx, double* prod_nu, int* prod_off,
                  double* optimal_controls);

// Real-time optimization
int real_time_optimization(int N, int M, int n_controls,
                          double* current_concentrations,
                          double* economic_objective_coeffs,
                          double* control_bounds,
                          double* kf, double* kr,
                          int* reac_idx, double* reac_nu, int* reac_off,
                          int* prod_idx, double* prod_nu, int* prod_off,
                          double* optimal_controls, double* predicted_profit);

// ============================================================================
// ADVANCED THERMODYNAMICS
// ============================================================================

// NASA polynomial for heat capacity
double heat_capacity_nasa(double T, double* coeffs);

// Enthalpy from NASA polynomials
double enthalpy_nasa(double T, double* coeffs);

// Entropy from NASA polynomials  
double entropy_nasa(double T, double* coeffs);

// Real gas equation of state (Peng-Robinson)
double pressure_peng_robinson(double n, double V, double T, 
                             double Tc, double Pc, double omega);

// Fugacity coefficient calculation
double fugacity_coefficient(double P, double T, double Tc, double Pc, double omega);

// ============================================================================
// REACTION KINETICS EXTENSIONS
// ============================================================================

// Michaelis-Menten enzyme kinetics
double michaelis_menten_rate(double Vmax, double Km, double substrate_conc);

// Competitive inhibition
double competitive_inhibition_rate(double Vmax, double Km, 
                                 double substrate_conc, double inhibitor_conc,
                                 double Ki);

// Autocatalytic reaction rate with temperature dependency
double autocatalytic_rate(double k, double A, double B, double temperature);

// Langmuir-Hinshelwood surface reaction
double langmuir_hinshelwood_rate(double k, double K_A, double K_B,
                               double conc_A, double conc_B);

// Photochemical reaction rate
double photochemical_rate(double quantum_yield, double molar_absorptivity,
                        double path_length, double light_intensity,
                        double concentration);

// ============================================================================
// TRANSPORT PHENOMENA
// ============================================================================

// Mass transfer coefficient correlation
double mass_transfer_correlation(double Re, double Sc, double geometry_factor);

// Heat transfer coefficient correlation  
double heat_transfer_correlation(double Re, double Pr, double geometry_factor);

// Effective diffusivity in porous media
double effective_diffusivity(double molecular_diff, double porosity, 
                           double tortuosity, double constriction_factor);

// Pressure drop in packed bed
double pressure_drop_ergun(double velocity, double density, double viscosity,
                          double particle_diameter, double bed_porosity, double bed_length);

// ============================================================================
// DATA ANALYSIS AND FITTING
// ============================================================================

// Nonlinear least squares parameter estimation
int parameter_estimation_nlls(int n_params, int n_data,
                             double* initial_guess, double* experimental_data,
                             double* weights, double* parameter_bounds,
                             double* fitted_parameters, double* confidence_intervals);

// Cross-validation for model selection
double cross_validation_score(int n_folds, int n_data, double* data,
                             int n_params, double* parameters);

// Bootstrap parameter uncertainty
int bootstrap_uncertainty(int n_bootstrap, int n_data, int n_params,
                         double* data, double* parameters,
                         double* parameter_distribution);

// ============================================================================
// MACHINE LEARNING INTEGRATION
// ============================================================================

// Neural network surrogate model
int train_neural_network(int n_inputs, int n_outputs, int n_hidden,
                        int n_training_data, double* inputs, double* outputs,
                        double* network_weights);

// Gaussian process regression
int gaussian_process_prediction(int n_training, int n_test,
                               double* training_inputs, double* training_outputs,
                               double* test_inputs, double* predictions,
                               double* uncertainties);

// Kriging interpolation
double kriging_interpolation(double* x_new, int n_known, double* x_known,
                           double* y_known, double* variogram_params);

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Matrix operations for linear algebra
int matrix_multiply(double* A, double* B, double* C, int m, int n, int p);
int matrix_invert(double* A, double* A_inv, int n);
int solve_linear_system(double* A, double* b, double* x, int n);

// Interpolation and data processing
double linear_interpolate(double x, double* x_data, double* y_data, int n);
double cubic_spline_interpolate(double x, double* x_data, double* y_data, int n);

// Statistical analysis
double calculate_r_squared(double* experimental, double* predicted, int n);
double calculate_rmse(double* experimental, double* predicted, int n);
double calculate_aic(double* experimental, double* predicted, int ndata, int nparams);

// Memory management helpers
void* allocate_aligned_memory(size_t size, size_t alignment);
void free_aligned_memory(void* ptr);

// ============================================================================
// SIMPLIFIED WRAPPER FUNCTIONS (matching Python interface)
// ============================================================================

// Simplified packed bed reactor (matches Python interface)
int simulate_packed_bed_simple(double length, double diameter, double particle_size, 
                              double bed_porosity, double* concentrations_in, 
                              double flow_rate, double temperature, double pressure, 
                              int n_species, double* concentrations_out, double* pressure_drop, 
                              double* conversion);

// Simplified fluidized bed reactor (matches Python interface)
int simulate_fluidized_bed_simple(double bed_height, double bed_diameter, double particle_density,
                                 double particle_size, double* concentrations_in, 
                                 double gas_velocity, double temperature, double pressure, 
                                 int n_species, double* concentrations_out, double* bed_expansion, 
                                 double* conversion);

// Simplified homogeneous batch reactor (matches Python interface)
int simulate_homogeneous_batch_simple(double* concentrations_initial, double volume, 
                                     double temperature, double pressure, double reaction_time,
                                     int n_species, int n_reactions, double* concentrations_final,
                                     double* conversion);

// Simplified energy balance (matches Python interface)
int calculate_energy_balance_simple(double* heat_capacities, double* flow_rates, 
                                   double* temperatures, double heat_of_reaction, 
                                   int n_streams, double* total_enthalpy_in, 
                                   double* total_enthalpy_out, double* net_energy_balance);

// Simplified Monte Carlo simulation (matches Python interface)
int monte_carlo_simulation_simple(double* parameter_distributions, int n_samples, 
                                 double* statistics_mean, double* statistics_std,
                                 double* statistics_min, double* statistics_max);

}

// ============================================================================
// C++ ONLY INTERFACE (for internal use and future extensions)
// ============================================================================

#ifdef __cplusplus

namespace pyroxa {

// Modern C++ interface for reaction networks
class ReactionNetwork {
public:
    ReactionNetwork(int n_species, int n_reactions);
    ~ReactionNetwork();
    
    void add_reaction(const std::vector<int>& reactants, 
                     const std::vector<double>& reactant_stoich,
                     const std::vector<int>& products,
                     const std::vector<double>& product_stoich,
                     double kf, double kr);
    
    std::vector<double> simulate(const std::vector<double>& initial_conc,
                               double time_span, double dt);
    
    std::vector<double> find_steady_state(const std::vector<double>& guess);
    
private:
    int n_species_;
    int n_reactions_;
    std::vector<double> kf_, kr_;
    std::vector<std::vector<int>> reactants_, products_;
    std::vector<std::vector<double>> reactant_stoich_, product_stoich_;
};

// High-performance reactor simulator with modern C++ features
class ReactorSimulator {
public:
    ReactorSimulator();
    ~ReactorSimulator();
    
    void set_parallel_threads(int nthreads);
    void enable_adaptive_stepping(double atol, double rtol);
    void enable_conservation_checking(double tolerance);
    
    std::vector<std::vector<double>> simulate_batch(
        const ReactionNetwork& network,
        const std::vector<double>& initial_conc,
        double time_span, double dt);
    
    std::vector<std::vector<double>> simulate_pfr(
        const ReactionNetwork& network,
        const std::vector<double>& initial_conc,
        double flow_rate, double volume, int nseg,
        double time_span, double dt);
    
private:
    bool adaptive_stepping_;
    bool conservation_checking_;
    double atol_, rtol_, conservation_tol_;
    int nthreads_;
};

} // namespace pyroxa

#endif // __cplusplus

#endif // PYROXA_CORE_H
