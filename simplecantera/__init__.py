try:
	# Prefer compiled extension
	from . import _pybindings as _bind
	Thermodynamics = _bind.Thermodynamics
	Reaction = _bind.Reaction
	Reactor = _bind.Reactor
	run_simulation_cpp = _bind.run_simulation_cpp
	# Fallback pure-Python helpers still available
	from .purepy import WellMixedReactor, CSTR, PFR, ReactorNetwork, run_simulation, build_from_dict
	__all__ = [
		"Thermodynamics",
		"Reaction",
		"Reactor",
		"run_simulation_cpp",
		"WellMixedReactor",
		"CSTR",
		"PFR",
		"ReactorNetwork",
		"run_simulation",
		"build_from_dict",
	]
except Exception:
	# Pure-Python fallback
	from .purepy import (
		Thermodynamics,
		Reaction,
		WellMixedReactor,
		CSTR,
		PFR,
		ReactorNetwork,
		run_simulation,
		build_from_dict,
	)
	Reactor = WellMixedReactor
	__all__ = [
		"Thermodynamics",
		"Reaction",
		"WellMixedReactor",
		"CSTR",
		"PFR",
		"ReactorNetwork",
		"Reactor",
		"run_simulation",
		"build_from_dict",
	]
