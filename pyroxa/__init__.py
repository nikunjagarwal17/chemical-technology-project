"""
PyroXa - Chemical Kinetics and Reactor Simulation Library
Pure Python Implementation (v1.0.0)
"""

import math
import numpy as np

# Version information
__version__ = "1.0.0"
__author__ = "PyroXa Development Team"

def get_version():
    """Return the PyroXa version string"""
    return __version__

# Import core functions from new_functions module
from .new_functions import (
    # Basic kinetic functions
    autocatalytic_rate,
    michaelis_menten_rate,
    competitive_inhibition_rate,
    arrhenius_rate,
    langmuir_hinshelwood_rate,
    photochemical_rate,
    
    # Reaction kinetics
    first_order_rate,
    second_order_rate,
    zero_order_rate,
    reversible_rate,
    parallel_reaction_rate,
    series_reaction_rate,
    enzyme_inhibition_rate,
    
    # Thermodynamic functions
    heat_capacity_nasa,
    enthalpy_nasa,
    entropy_nasa,
    gibbs_free_energy,
    equilibrium_constant,
    temperature_dependence,
    pressure_dependence,
    activity_coefficient,
    
    # Transport phenomena
    mass_transfer_correlation,
    heat_transfer_correlation,
    effective_diffusivity,
    pressure_drop_ergun,
    diffusion_coefficient,
    thermal_conductivity,
    heat_transfer_coefficient,
    mass_transfer_coefficient,
    
    # Dimensionless numbers
    reynolds_number,
    prandtl_number,
    schmidt_number,
    nusselt_number,
    sherwood_number,
    friction_factor,
    
    # Equation of state
    pressure_peng_robinson,
    fugacity_coefficient,
    
    # Reactor design
    hydraulic_diameter,
    residence_time,
    conversion,
    selectivity,
    yield_coefficient,
    space_time,
    space_velocity,
    reaction_quotient,
    extent_of_reaction,
    batch_reactor_time,
    cstr_volume,
    pfr_volume,
    
    # Advanced reactor operations
    fluidized_bed_hydrodynamics,
    packed_bed_pressure_drop,
    bubble_column_dynamics,
    
    # Separation processes
    crystallization_rate,
    precipitation_rate,
    dissolution_rate,
    evaporation_rate,
    distillation_efficiency,
    extraction_efficiency,
    adsorption_isotherm,
    desorption_rate,
    
    # Catalysis
    catalyst_activity,
    catalyst_deactivation,
    surface_reaction_rate,
    pore_diffusion_rate,
    film_mass_transfer,
    
    # Fluid mechanics
    bubble_rise_velocity,
    terminal_velocity,
    drag_coefficient,
    
    # Process engineering
    mixing_time,
    power_consumption,
    pumping_power,
    compression_work,
    heat_exchanger_effectiveness,
    overall_heat_transfer_coefficient,
    fouling_resistance,
    
    # Advanced simulation functions
    simulate_packed_bed,
    simulate_fluidized_bed,
    simulate_homogeneous_batch,
    simulate_multi_reactor_adaptive,
    calculate_energy_balance,
    
    # Mathematical utilities
    linear_interpolate,
    cubic_spline_interpolate,
    calculate_r_squared,
    calculate_rmse,
    calculate_aic,
    
    # Process control
    PIDController,
    pid_controller,
    
    # Advanced analytical methods
    analytical_first_order,
    analytical_reversible_first_order,
    analytical_consecutive_first_order,
    
    # Statistical and optimization methods
    calculate_objective_function,
    check_mass_conservation,
    calculate_rate_constants,
    cross_validation_score,
    kriging_interpolation,
    bootstrap_uncertainty,
    
    # Matrix operations
    matrix_multiply,
    matrix_invert,
    solve_linear_system,
    
    # Sensitivity and stability analysis
    calculate_sensitivity,
    calculate_jacobian,
    stability_analysis,
    
    # Advanced process control
    mpc_controller,
    real_time_optimization,
    parameter_sweep_parallel,
    monte_carlo_simulation,
    
    # Process engineering analysis
    residence_time_distribution,
    catalyst_deactivation_model,
    process_scale_up,
)

# Import additional utility functions from new_functions if they exist
try:
    from .new_functions import (
        # Additional functions that might be defined
        first_order_rate,
        second_order_rate,
        zero_order_rate,
        reversible_rate,
        parallel_reaction_rate,
        series_reaction_rate,
        enzyme_inhibition_rate,
        temperature_dependence,
        pressure_dependence,
        activity_coefficient,
        diffusion_coefficient,
        thermal_conductivity,
        heat_transfer_coefficient,
        mass_transfer_coefficient,
        reynolds_number,
        prandtl_number,
        schmidt_number,
        nusselt_number,
        sherwood_number,
        friction_factor,
        hydraulic_diameter,
        residence_time,
        conversion,
        selectivity,
        yield_coefficient,
        space_time,
        space_velocity,
        reaction_quotient,
        extent_of_reaction,
        batch_reactor_time,
        cstr_volume,
        pfr_volume,
        fluidized_bed_hydrodynamics,
        packed_bed_pressure_drop,
        bubble_column_dynamics,
        crystallization_rate,
        precipitation_rate,
        dissolution_rate,
        evaporation_rate,
        distillation_efficiency,
        extraction_efficiency,
        adsorption_isotherm,
        desorption_rate,
        catalyst_activity,
        catalyst_deactivation,
        surface_reaction_rate,
        pore_diffusion_rate,
        film_mass_transfer,
        bubble_rise_velocity,
        terminal_velocity,
        drag_coefficient,
        mixing_time,
        power_consumption,
        pumping_power,
        compression_work,
        heat_exchanger_effectiveness,
        overall_heat_transfer_coefficient,
        fouling_resistance,
    )
except ImportError:
    # These functions might not be implemented yet
    pass

# Import classes and advanced functions from purepy module
try:
    from .purepy import (
        # Core classes
        Thermodynamics,
        Reaction,
        ReactionMulti,
        MultiReactor,
        
        # Reactor classes
        WellMixedReactor,
        CSTR,
        PFR,
        ReactorNetwork,
        PackedBedReactor,
        FluidizedBedReactor,
        HeterogeneousReactor,
        HomogeneousReactor,
        
        # Utility functions
        build_from_dict,
        run_simulation_from_dict,
        benchmark_multi_reactor,
        enthalpy_c,
        entropy_c,
        
        # Exception classes
        PyroXaError,
        ThermodynamicsError,
        ReactionError,
        ReactorError,
    )
    
    # Create alias for run_simulation
    run_simulation = run_simulation_from_dict
    
except ImportError as e:
    # These might not be available in simplified version
    print(f"Warning: Some advanced classes not available: {e}")
    pass

# Import reaction chains if available
try:
    from .reaction_chains import (
        # Reaction chain classes and functions that actually exist
        ReactionChain,
        create_reaction_chain,
        ChainReactorVisualizer,
        OptimalReactorDesign,
    )
except ImportError as e:
    # Some reaction chain functions might not be implemented
    pass

# Define essential functions that might be missing
if 'first_order_rate' not in globals():
    def first_order_rate(k, concentration):
        """First-order reaction rate: r = k * [A]"""
        return k * concentration

if 'second_order_rate' not in globals():
    def second_order_rate(k, conc_A, conc_B=None):
        """Second-order reaction rate: r = k * [A] * [B] or r = k * [A]^2"""
        if conc_B is None:
            return k * conc_A * conc_A
        return k * conc_A * conc_B

if 'zero_order_rate' not in globals():
    def zero_order_rate(k):
        """Zero-order reaction rate: r = k"""
        return k

if 'reynolds_number' not in globals():
    def reynolds_number(density, velocity, length, viscosity):
        """Calculate Reynolds number"""
        return density * velocity * length / viscosity

if 'conversion' not in globals():
    def conversion(initial_conc, final_conc):
        """Calculate conversion: X = (C0 - C) / C0"""
        if initial_conc == 0:
            return 0.0
        return (initial_conc - final_conc) / initial_conc

# List all available functions for easy discovery
__all__ = [
    'get_version',
    # Basic kinetics
    'autocatalytic_rate',
    'michaelis_menten_rate', 
    'competitive_inhibition_rate',
    'arrhenius_rate',
    'langmuir_hinshelwood_rate',
    'photochemical_rate',
    'first_order_rate',
    'second_order_rate',
    'zero_order_rate',
    'reversible_rate',
    'parallel_reaction_rate',
    'series_reaction_rate',
    'enzyme_inhibition_rate',
    # Thermodynamics
    'heat_capacity_nasa',
    'enthalpy_nasa',
    'entropy_nasa',
    'gibbs_free_energy',
    'equilibrium_constant',
    'temperature_dependence',
    'pressure_dependence',
    'activity_coefficient',
    # Transport
    'mass_transfer_correlation',
    'heat_transfer_correlation',
    'effective_diffusivity',
    'pressure_drop_ergun',
    'diffusion_coefficient',
    'thermal_conductivity',
    'heat_transfer_coefficient',
    'mass_transfer_coefficient',
    # Dimensionless numbers
    'reynolds_number',
    'prandtl_number',
    'schmidt_number',
    'nusselt_number',
    'sherwood_number',
    'friction_factor',
    # Equation of state
    'pressure_peng_robinson',
    'fugacity_coefficient',
    # Reactor design
    'hydraulic_diameter',
    'residence_time',
    'conversion',
    'selectivity',
    'yield_coefficient',
    'space_time',
    'space_velocity',
    'reaction_quotient',
    'extent_of_reaction',
    'batch_reactor_time',
    'cstr_volume',
    'pfr_volume',
    # Advanced reactors
    'fluidized_bed_hydrodynamics',
    'packed_bed_pressure_drop',
    'bubble_column_dynamics',
    # Separation processes
    'crystallization_rate',
    'precipitation_rate',
    'dissolution_rate',
    'evaporation_rate',
    'distillation_efficiency',
    'extraction_efficiency',
    'adsorption_isotherm',
    'desorption_rate',
    # Catalysis
    'catalyst_activity',
    'catalyst_deactivation',
    'surface_reaction_rate',
    'pore_diffusion_rate',
    'film_mass_transfer',
    # Fluid mechanics
    'bubble_rise_velocity',
    'terminal_velocity',
    'drag_coefficient',
    # Process engineering
    'mixing_time',
    'power_consumption',
    'pumping_power',
    'compression_work',
    'heat_exchanger_effectiveness',
    'overall_heat_transfer_coefficient',
    'fouling_resistance',
    # Advanced simulations
    'simulate_packed_bed',
    'simulate_fluidized_bed', 
    'simulate_homogeneous_batch',
    'simulate_multi_reactor_adaptive',
    'calculate_energy_balance',
    # Mathematical utilities
    'linear_interpolate',
    'cubic_spline_interpolate',
    'calculate_r_squared',
    'calculate_rmse',
    'calculate_aic',
    # Process control
    'PIDController',
    'pid_controller',
    # Advanced analytical methods
    'analytical_first_order',
    'analytical_reversible_first_order',
    'analytical_consecutive_first_order',
    # Statistical and optimization
    'calculate_objective_function',
    'check_mass_conservation',
    'calculate_rate_constants',
    'cross_validation_score',
    'kriging_interpolation',
    'bootstrap_uncertainty',
    # Matrix operations
    'matrix_multiply',
    'matrix_invert',
    'solve_linear_system',
    # Sensitivity analysis
    'calculate_sensitivity',
    'calculate_jacobian',
    'stability_analysis',
    # Advanced control
    'mpc_controller',
    'real_time_optimization',
    'parameter_sweep_parallel',
    'monte_carlo_simulation',
    # Process analysis
    'residence_time_distribution',
    'catalyst_deactivation_model',
    'process_scale_up',
    # Core classes
    'Thermodynamics',
    'Reaction',
    'ReactionMulti', 
    'MultiReactor',
    # Reactor classes
    'WellMixedReactor',
    'CSTR',
    'PFR',
    'ReactorNetwork',
    'PackedBedReactor',
    'FluidizedBedReactor',
    'HeterogeneousReactor',
    'HomogeneousReactor',
    # Exception classes
    'PyroXaError',
    'ThermodynamicsError',
    'ReactionError',
    'ReactorError',
    # Reaction chains
    'ReactionChain',
    'create_reaction_chain',
    'ChainReactorVisualizer',
    'OptimalReactorDesign',
    # Simulation utilities
    'run_simulation_from_dict',
    'run_simulation',
]

print("✅ PyroXa v1.0.0 loaded successfully (Pure Python)")
print(f"📦 Available functions: {len([x for x in __all__ if x in globals()])}")
