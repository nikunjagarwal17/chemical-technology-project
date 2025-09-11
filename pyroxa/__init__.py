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
    
    # Thermodynamic functions
    heat_capacity_nasa,
    enthalpy_nasa,
    entropy_nasa,
    gibbs_free_energy,
    equilibrium_constant,
    
    # Transport phenomena
    mass_transfer_correlation,
    heat_transfer_correlation,
    effective_diffusivity,
    pressure_drop_ergun,
    
    # Equation of state
    pressure_peng_robinson,
    fugacity_coefficient,
    
    # Mathematical utilities
    linear_interpolate,
    cubic_spline_interpolate,
    calculate_r_squared,
    calculate_rmse,
    calculate_aic,
    
    # Process control
    PIDController,
    pid_controller,
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
        Thermodynamics,
        Reaction,
        ReactionMulti,
        Reactor,
        MultiReactor,
        FluidizedBedReactor,
        build_from_dict,
        run_simulation_from_dict,
        benchmark_multi_reactor,
        enthalpy_c,
        entropy_c,
    )
except ImportError:
    # These might not be available in simplified version
    pass

# Import reaction chains if available
try:
    from .reaction_chains import (
        # Reaction chain functions if they exist
        chain_reaction_rate,
        branching_factor,
        chain_length,
        initiation_rate,
        propagation_rate,
        termination_rate,
    )
except ImportError:
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
    # Thermodynamics
    'heat_capacity_nasa',
    'enthalpy_nasa',
    'entropy_nasa',
    'gibbs_free_energy',
    'equilibrium_constant',
    # Transport
    'mass_transfer_correlation',
    'heat_transfer_correlation',
    'effective_diffusivity',
    'pressure_drop_ergun',
    'reynolds_number',
    # Equation of state
    'pressure_peng_robinson',
    'fugacity_coefficient',
    # Mathematical utilities
    'linear_interpolate',
    'cubic_spline_interpolate',
    'calculate_r_squared',
    'calculate_rmse',
    'calculate_aic',
    # Process control
    'PIDController',
    'pid_controller',
    # Reactor design
    'conversion',
    # Classes (if available)
    'Thermodynamics',
    'Reaction',
    'ReactionMulti', 
    'Reactor',
    'MultiReactor',
    'FluidizedBedReactor',
]

print("✅ PyroXa v1.0.0 loaded successfully (Pure Python)")
print(f"📦 Available functions: {len([x for x in __all__ if x in globals()])}")
