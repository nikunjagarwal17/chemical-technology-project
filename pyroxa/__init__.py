try:
	# Try to import simplified C++ extension first
	from . import simple_bindings as _bind
	print("✓ C++ extension loaded successfully")
	
	# Create function aliases for compatibility
	simulate_reactor_cpp = _bind.py_simulate_reactor
	enthalpy_c_cpp = _bind.py_enthalpy_c
	entropy_c_cpp = _bind.py_entropy_c
	autocatalytic_rate_cpp = _bind.py_autocatalytic_rate
	michaelis_menten_rate_cpp = _bind.py_michaelis_menten_rate
	arrhenius_rate_cpp = _bind.py_arrhenius_rate
	
	_cpp_available = True
	
	# Since simplified bindings don't have classes, we'll use pure Python classes
	# but with C++ acceleration for computationally intensive functions
	print("✓ Using C++ accelerated functions for core operations")
	
	# Set flags for available functions
	_NEW_FUNCTIONS_FROM_CPP = False  # Most functions will use Python fallback
	_EXTENDED_FUNCTIONS_FROM_CPP = False
	_ALL_65_FUNCTIONS_FROM_CPP = False
	
	# Always import all functions from Python implementations (C++ may not have all)
	if not _EXTENDED_FUNCTIONS_FROM_CPP:
		from .new_functions import (
			langmuir_hinshelwood_rate, photochemical_rate, 
			pressure_peng_robinson, fugacity_coefficient, PIDController
		)
	else:
		# Import only missing functions when extended C++ functions available
		from .new_functions import PIDController
	
	# Import missing functions from Python if not in C++
	if not _NEW_FUNCTIONS_FROM_CPP:
		from .new_functions import (
			autocatalytic_rate, michaelis_menten_rate, competitive_inhibition_rate,
			heat_capacity_nasa, enthalpy_nasa, entropy_nasa,
			mass_transfer_correlation, heat_transfer_correlation,
			effective_diffusivity, pressure_drop_ergun, pid_controller
		)
	
	# Fallback pure-Python helpers still available
	from .purepy import (
		WellMixedReactor, CSTR, PFR, ReactorNetwork, run_simulation, build_from_dict,
		ReactionMulti, MultiReactor, benchmark_multi_reactor,
		PackedBedReactor, FluidizedBedReactor, HeterogeneousReactor, HomogeneousReactor,
		PyroXaError, ThermodynamicsError, ReactionError, ReactorError
	)
	# Import enhanced multi-reaction features
	try:
		from .reaction_chains import ReactionChain, ChainReactorVisualizer, OptimalReactorDesign
		_REACTION_CHAINS_AVAILABLE = True
	except ImportError:
		_REACTION_CHAINS_AVAILABLE = False
	
	__all__ = [
		"Thermodynamics",
		"Reaction", 
		"ReactionMulti",
		"Reactor",
		"WellMixedReactor",
		"CSTR",
		"PFR", 
		"MultiReactor",
		"ReactorNetwork",
		"run_simulation_cpp",
		"run_simulation",
		"build_from_dict",
		"benchmark_multi_reactor",
		"PyroXaError",
		"ThermodynamicsError", 
		"ReactionError",
		"ReactorError",
		# Newly implemented functions
		"autocatalytic_rate",
		"michaelis_menten_rate", 
		"competitive_inhibition_rate",
		"heat_capacity_nasa",
		"enthalpy_nasa",
		"entropy_nasa",
		"mass_transfer_correlation",
		"heat_transfer_correlation",
		"effective_diffusivity",
		"pressure_drop_ergun",
		"pid_controller",
		"langmuir_hinshelwood_rate",
		"photochemical_rate",
		"pressure_peng_robinson",
		"fugacity_coefficient",
		"gibbs_free_energy",
		"equilibrium_constant",
		"arrhenius_rate",
		"PIDController",
		# Batch 1: Statistical and interpolation functions
		"linear_interpolate",
		"cubic_spline_interpolate", 
		"calculate_r_squared",
		"calculate_rmse",
		"calculate_aic",
		# Batch 5: Core thermodynamic functions
		"enthalpy_c",
		"entropy_c",
		# Batch 6: Analytical solutions
		"analytical_first_order",
		"analytical_reversible_first_order",
		"analytical_consecutive_first_order",
		# Batch 8: Utility and optimization functions
		"calculate_objective_function",
		"check_mass_conservation",
		"calculate_rate_constants",
		# Batch 9: Simple utility and validation functions
		"cross_validation_score",
		"kriging_interpolation", 
		"bootstrap_uncertainty",
		# Batch 10: Matrix operations
		"matrix_multiply",
		"matrix_invert",
		"solve_linear_system",
		# Batch 11: Advanced optimization and sensitivity analysis
		"calculate_sensitivity",
		"calculate_jacobian",
		"stability_analysis",
		"mpc_controller",
		"real_time_optimization",
		# "parameter_sweep_parallel",  # TEMPORARILY DISABLED
		# Batch 12: Advanced reactor simulations
		"simulate_packed_bed",
		"simulate_fluidized_bed",
		"simulate_homogeneous_batch",
		"simulate_multi_reactor_adaptive",
		# Batch 13: Energy analysis and statistical methods
		"calculate_energy_balance",
		"monte_carlo_simulation",
		"residence_time_distribution",
		# Batch 14: Final functions
		"catalyst_deactivation_model",
		"process_scale_up"
	]
	
	if _REACTION_CHAINS_AVAILABLE:
		__all__.extend(["ReactionChain", "ChainReactorVisualizer", "OptimalReactorDesign"])
	
	# Import I/O utilities for convenience
	try:
		from .io import load_spec_from_yaml, parse_mechanism, save_results_to_csv
	except ImportError:
		# I/O utilities are optional
		load_spec_from_yaml = None
		parse_mechanism = None
		save_results_to_csv = None
	
	_COMPILED_AVAILABLE = True
except (ImportError, MemoryError, OSError) as e:
	print(f"⚠ C++ extension failed to load ({type(e).__name__}: {e})")
	if isinstance(e, MemoryError):
		print("  This is expected with free-threaded Python 3.13 due to missing symbols:")
		print("  - __imp__Py_MergeZeroLocalRefcount")
		print("  - __imp_PyUnstable_Module_SetGIL") 
		print("  - __imp__Py_DecRefShared")
		print("  Solution: Install standard Python 3.13 (not free-threaded) for C++ extensions")
	print("✓ Falling back to pure Python implementation...")
	
	# Pure-Python fallback
	from .purepy import (
		Thermodynamics,
		Reaction,
		ReactionMulti,
		WellMixedReactor,
		CSTR,
		PFR,
		MultiReactor,
		ReactorNetwork,
		PackedBedReactor,
		FluidizedBedReactor,
		HeterogeneousReactor,
		HomogeneousReactor,
		run_simulation,
		build_from_dict,
		benchmark_multi_reactor,
		PyroXaError,
		ThermodynamicsError,
		ReactionError,
		ReactorError
	)
	
	# Import all new functions from Python implementations
	from .new_functions import (
		autocatalytic_rate, michaelis_menten_rate, competitive_inhibition_rate,
		heat_capacity_nasa, enthalpy_nasa, entropy_nasa,
		mass_transfer_correlation, heat_transfer_correlation,
		effective_diffusivity, pressure_drop_ergun, pid_controller,
		langmuir_hinshelwood_rate, photochemical_rate, 
		pressure_peng_robinson, fugacity_coefficient, PIDController,
		# Statistical and interpolation functions
		linear_interpolate, cubic_spline_interpolate,
		calculate_r_squared, calculate_rmse, calculate_aic,
		gibbs_free_energy, equilibrium_constant, arrhenius_rate
	)
	
	# Import basic thermodynamic functions from purepy
	from .purepy import enthalpy_c, entropy_c
	
	# Analytical solutions (if available in purepy, otherwise create simple fallbacks)
	try:
		from .purepy import analytical_first_order, analytical_reversible_first_order, analytical_consecutive_first_order
	except ImportError:
		# Simple fallback implementations
		def analytical_first_order(k, A0, time_span, dt, max_len=1000):
			"""Simple analytical solution for A -> B (first order)"""
			import numpy as np
			times = np.arange(0, time_span + dt, dt)
			A = A0 * np.exp(-k * times)
			B = A0 * (1 - np.exp(-k * times))
			return {'times': times.tolist(), 'A': A.tolist(), 'B': B.tolist()}
		
		def analytical_reversible_first_order(kf, kr, A0, B0, time_span, dt, max_len=1000):
			"""Simple analytical solution for A <=> B (reversible first order)"""
			import numpy as np
			times = np.arange(0, time_span + dt, dt)
			k_total = kf + kr
			K_eq = kf / kr if kr > 0 else 1e6
			A_eq = (A0 + B0) / (1 + K_eq)
			B_eq = (A0 + B0) * K_eq / (1 + K_eq)
			A = A_eq + (A0 - A_eq) * np.exp(-k_total * times)
			B = B_eq + (B0 - B_eq) * np.exp(-k_total * times)
			return {'times': times.tolist(), 'A': A.tolist(), 'B': B.tolist()}
		
		def analytical_consecutive_first_order(k1, k2, A0, time_span, dt, max_len=1000):
			"""Simple analytical solution for A -> B -> C (consecutive first order)"""
			import numpy as np
			times = np.arange(0, time_span + dt, dt)
			A = A0 * np.exp(-k1 * times)
			if abs(k1 - k2) > 1e-10:
				B = A0 * k1 / (k2 - k1) * (np.exp(-k1 * times) - np.exp(-k2 * times))
			else:
				B = A0 * k1 * times * np.exp(-k1 * times)
			C = A0 * (1 - np.exp(-k1 * times) - k1/(k2-k1) * (np.exp(-k1 * times) - np.exp(-k2 * times))) if abs(k1-k2) > 1e-10 else A0 * (1 - (1 + k1*times) * np.exp(-k1 * times))
			return {'times': times.tolist(), 'A': A.tolist(), 'B': B.tolist(), 'C': C.tolist()}
	
	# Simple fallback utility functions
	def calculate_objective_function(experimental_data, simulated_data, weights=None):
		"""Calculate objective function (sum of squared residuals)"""
		import numpy as np
		exp = np.array(experimental_data)
		sim = np.array(simulated_data)
		w = np.array(weights) if weights else np.ones_like(exp)
		return np.sum(w * (exp - sim)**2)
	
	def check_mass_conservation(concentrations, tolerance=1e-6):
		"""Check mass conservation during simulation"""
		import numpy as np
		conc_array = np.array(concentrations)
		total_mass = np.sum(conc_array, axis=1)
		mass_balance = total_mass - total_mass[0]
		max_violation = np.max(np.abs(mass_balance))
		return {
			'is_conserved': max_violation < tolerance,
			'mass_balance': mass_balance.tolist(),
			'max_violation': max_violation
		}
	
	def calculate_rate_constants(kf_ref, kr_ref, Ea_f, Ea_r, T, T_ref=298.15, R=8.314):
		"""Calculate temperature-dependent rate constants using Arrhenius equation"""
		import numpy as np
		kf_ref = np.array(kf_ref)
		kr_ref = np.array(kr_ref)
		Ea_f = np.array(Ea_f)
		Ea_r = np.array(Ea_r)
		
		kf_out = kf_ref * np.exp(-(Ea_f/R) * (1/T - 1/T_ref))
		kr_out = kr_ref * np.exp(-(Ea_r/R) * (1/T - 1/T_ref))
		
		return {
			'kf': kf_out.tolist(),
			'kr': kr_out.tolist()
		}
	
	_NEW_FUNCTIONS_FROM_CPP = False
	
	# Import enhanced multi-reaction features
	try:
		from .reaction_chains import ReactionChain, ChainReactorVisualizer, OptimalReactorDesign
		_REACTION_CHAINS_AVAILABLE = True
	except ImportError:
		_REACTION_CHAINS_AVAILABLE = False
	
	# Alias for compatibility
	Reactor = WellMixedReactor
	
	def run_simulation_cpp(*args, **kwargs):
		"""Fallback function when compiled extension is not available."""
		import warnings
		warnings.warn("Compiled extension not available, falling back to pure Python implementation")
		return run_simulation(*args, **kwargs)
	
	__all__ = [
		# Core classes
		"Thermodynamics",
		"Reaction",
		"ReactionMulti", 
		"WellMixedReactor",
		"CSTR",
		"PFR",
		"MultiReactor",
		"ReactorNetwork",
		"PackedBedReactor",
		"FluidizedBedReactor",
		"HeterogeneousReactor",
		"HomogeneousReactor",
		"Reactor",
		
		# Simulation functions
		"run_simulation_cpp",
		"run_simulation",
		"build_from_dict",
		"benchmark_multi_reactor",
		
		# Error classes
		"PyroXaError",
		"ThermodynamicsError",
		"ReactionError", 
		"ReactorError",
		
		# Rate calculation functions
		"autocatalytic_rate",
		"michaelis_menten_rate", 
		"competitive_inhibition_rate",
		"langmuir_hinshelwood_rate",
		"photochemical_rate",
		
		# Thermodynamic functions
		"heat_capacity_nasa",
		"enthalpy_nasa",
		"entropy_nasa",
		"pressure_peng_robinson",
		"fugacity_coefficient",
		"gibbs_free_energy",
		"equilibrium_constant",
		"arrhenius_rate",
		
		# Transport phenomena functions
		"mass_transfer_correlation",
		"heat_transfer_correlation",
		"effective_diffusivity",
		"pressure_drop_ergun",
		
		# Control functions
		"pid_controller",
		"PIDController",
		
		# Statistical and interpolation functions
		"linear_interpolate",
		"cubic_spline_interpolate", 
		"calculate_r_squared",
		"calculate_rmse",
		"calculate_aic",
		
		# Core thermodynamic functions
		"enthalpy_c",
		"entropy_c",
		
		# Analytical solutions
		"analytical_first_order",
		"analytical_reversible_first_order",
		"analytical_consecutive_first_order",
		
		# Utility and optimization functions
		"calculate_objective_function",
		"check_mass_conservation",
		"calculate_rate_constants",
		
		# Utility functions
		"get_version",
		"get_build_info",
		"is_compiled_available",
		"is_reaction_chains_available",
		"create_reaction_chain",
		
		# I/O functions
		"load_spec_from_yaml",
		"parse_mechanism",
		"save_results_to_csv"
	]
	
	if _REACTION_CHAINS_AVAILABLE:
		__all__.extend(["ReactionChain", "ChainReactorVisualizer", "OptimalReactorDesign"])
	
	_COMPILED_AVAILABLE = False

# Version information
__version__ = "0.3.0"
__author__ = "Pyroxa Development Team"
__description__ = "Chemical kinetics and reactor simulation library inspired by Cantera"

# Convenience functions for users
def get_version():
	"""Get the current version of Pyroxa."""
	return __version__

def is_compiled_available():
	"""Check if the compiled C++ extension is available."""
	return _COMPILED_AVAILABLE

def is_reaction_chains_available():
	"""Check if the enhanced reaction chain features are available."""
	return _REACTION_CHAINS_AVAILABLE

def get_build_info():
	"""Get information about the current build."""
	info = {
		'version': __version__,
		'compiled_extension': _COMPILED_AVAILABLE,
		'python_fallback': True,
		'reaction_chains': _REACTION_CHAINS_AVAILABLE,
	}
	
	if _COMPILED_AVAILABLE:
		try:
			# Try to get additional info from compiled extension
			info['cpp_core'] = True
		except:
			info['cpp_core'] = False
	else:
		info['cpp_core'] = False
		
	return info

def create_reaction_chain(species, rate_constants, **kwargs):
	"""Convenience function to create a reaction chain.
	
	Args:
		species: List of species names
		rate_constants: List of forward rate constants
		**kwargs: Additional arguments for ReactionChain
		
	Returns:
		ReactionChain object
	"""
	if not _REACTION_CHAINS_AVAILABLE:
		raise ImportError("ReactionChain features not available. Check dependencies.")
	
	return ReactionChain(species, rate_constants, **kwargs)

# Import I/O utilities for convenience
try:
	from .io import load_spec_from_yaml, parse_mechanism, save_results_to_csv
	__all__.extend(['load_spec_from_yaml', 'parse_mechanism', 'save_results_to_csv'])
except ImportError:
	pass  # I/O utilities are optional
