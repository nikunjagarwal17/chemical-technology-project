# distutils: language = c++
from libc.stdlib cimport malloc, free
from cpython.ref cimport PyObject
import cython

cdef extern from "../src/core.h":
	int simulate_reactor(double kf, double kr, double A0, double B0,
						 double time_span, double dt,
						 double* times, double* Aout, double* Bout, int max_len)
	double enthalpy_c(double cp, double T)
	double entropy_c(double cp, double T)
	int simulate_multi_reactor(int N, int M,
							   double* kf, double* kr,
							   int* reac_idx, double* reac_nu, int* reac_off,
							   int* prod_idx, double* prod_nu, int* prod_off,
							   double* conc0,
							   double time_span, double dt,
							   double* times, double* conc_out_flat, int max_len)

try:
	import numpy as np
	cimport numpy as np
	_HAS_NUMPY = True
except Exception:
	_HAS_NUMPY = False


cdef class Thermodynamics:
	cdef double cp

	def __cinit__(self, double cp=29.1):
		self.cp = cp

	def enthalpy(self, double T):
		return enthalpy_c(self.cp, T)

	def entropy(self, double T):
		return entropy_c(self.cp, T)


cdef class Reaction:
	cdef double kf
	cdef double kr

	def __cinit__(self, double kf=1.0, double kr=0.5):
		self.kf = kf
		self.kr = kr

	def rate(self, double A, double B):
		return self.kf * A - self.kr * B


cdef class ReactionMulti:
	cdef double kf
	cdef double kr
	cdef dict reactants
	cdef dict products

	def __cinit__(self, double kf=1.0, double kr=0.0, reactants=None, products=None):
		self.kf = kf
		self.kr = kr
		self.reactants = reactants if reactants is not None else {}
		self.products = products if products is not None else {}

	cpdef double rate(self, list conc):
		cdef double f = 1.0
		cdef double r = 1.0
		cdef int idx
		for key, nu in self.reactants.items():
			idx = int(key)
			val = conc[idx]
			if val <= 0:
				f = 0.0
				break
			f *= val ** nu
		for key, nu in self.products.items():
			idx = int(key)
			val = conc[idx]
			if val <= 0:
				r = 0.0
				break
			r *= val ** nu
		return self.kf * f - self.kr * r


cdef class MultiReactor:
	cdef list species
	cdef list conc
	cdef list reactions
	cdef double T

	def __cinit__(self, list species, list conc0, list reactions, double T=300.0):
		self.species = species
		self.conc = [float(x) for x in conc0]
		self.reactions = reactions
		self.T = T

	cpdef list _dcdt(self, list conc):
		cdef int N = len(conc)
		cdef list d = [0.0] * N
		cdef double rate
		cdef object rxn
		for rxn in self.reactions:
			rate = rxn.rate(conc)
			for key, nu in rxn.reactants.items():
				d[int(key)] -= nu * rate
			for key, nu in rxn.products.items():
				d[int(key)] += nu * rate
		return d

	def step(self, double dt):
		y0 = self.conc
		k1 = self._dcdt(y0)
		y1 = [y0[i] + 0.5 * dt * k1[i] for i in range(len(y0))]
		k2 = self._dcdt(y1)
		y2 = [y0[i] + 0.5 * dt * k2[i] for i in range(len(y0))]
		k3 = self._dcdt(y2)
		y3 = [y0[i] + dt * k3[i] for i in range(len(y0))]
		k4 = self._dcdt(y3)
		for i in range(len(y0)):
			self.conc[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
			if self.conc[i] < 0:
				self.conc[i] = 0.0

	def run(self, double time_span, double dt):
		cdef int nsteps = <int>round(time_span / dt)
		cdef int i
		times = [0.0]
		traj = [list(self.conc)]
		for i in range(nsteps):
			self.step(dt)
			times.append((i + 1) * dt)
			traj.append(list(self.conc))
		return times, traj


cdef class Reactor:
	cdef double kf
	cdef double kr
	cdef double A0
	cdef double B0
	cdef double time_span
	cdef double dt

	def __cinit__(self, Reaction rxn, double A0=1.0, double B0=0.0, double time_span=10.0, double dt=0.01):
		self.kf = rxn.kf
		self.kr = rxn.kr
		self.A0 = A0
		self.B0 = B0
		self.time_span = time_span
		self.dt = dt

	def run(self):
		cdef int nsteps = <int>round(self.time_span / self.dt)
		cdef int npts = nsteps + 1
		cdef int max_len = npts
		cdef double* times = <double*>malloc(max_len * sizeof(double))
		cdef double* Aout = <double*>malloc(max_len * sizeof(double))
		cdef double* Bout = <double*>malloc(max_len * sizeof(double))
		if not times or not Aout or not Bout:
			if times: free(times)
			if Aout: free(Aout)
			if Bout: free(Bout)
			raise MemoryError("allocation failed")
		try:
			cdef int written = simulate_reactor(self.kf, self.kr, self.A0, self.B0, self.time_span, self.dt, times, Aout, Bout, max_len)
			if written <= 0:
				raise RuntimeError("simulation failed or insufficient buffer size")
			if _HAS_NUMPY:
				tarr = np.empty(written, dtype=np.float64)
				aarr = np.empty((written, 2), dtype=np.float64)
				for i in range(written):
					tarr[i] = times[i]
					aarr[i, 0] = Aout[i]
					aarr[i, 1] = Bout[i]
				return tarr, aarr
			else:
				py_times = [times[i] for i in range(written)]
				traj = [[Aout[i], Bout[i]] for i in range(written)]
				return py_times, traj
		finally:
			free(times)
			free(Aout)
			free(Bout)


def run_simulation_cpp(spec):
	"""High-level helper that takes a Python dict spec and runs the C++ reactor.

	Expected spec keys similar to pure-Python runner:
	  reaction: {'kf', 'kr'}
	  initial: {'conc': {'A','B'}}
	  sim: {'time_span', 'time_step'}
	"""
	if not isinstance(spec, dict):
		raise TypeError('spec must be a dict')
	reaction = spec.get('reaction', {})
	initial = spec.get('initial', {})
	sim = spec.get('sim', {})
	# multi-species branch: try calling optimized C++ multi-reactor
	if 'species' in spec and 'reactions' in spec:
		species = spec.get('species', [])
		N = len(species)
		rxns = spec.get('reactions', [])
		M = len(rxns)
		# prepare arrays
		# kf/kr
		cdef double[:] kf_arr = None
		cdef double[:] kr_arr = None
		import array
		kf_py = array.array('d', [float(r.get('kf', 1.0)) for r in rxns])
		kr_py = array.array('d', [float(r.get('kr', 0.0)) for r in rxns])
		cdef double* kf_ptr = <double*>kf_py.buffer_info()[0]
		cdef double* kr_ptr = <double*>kr_py.buffer_info()[0]
		# reactant/product flattened lists and offsets
		reac_idx_list = []
		reac_nu_list = []
		reac_off = [0]
		prod_idx_list = []
		prod_nu_list = []
		prod_off = [0]
		for r in rxns:
			reactants = r.get('reactants', {})
			products = r.get('products', {})
			for s, nu in reactants.items():
				reac_idx_list.append(int(species.index(s)))
				reac_nu_list.append(float(nu))
			reac_off.append(len(reac_idx_list))
			for s, nu in products.items():
				prod_idx_list.append(int(species.index(s)))
				prod_nu_list.append(float(nu))
			prod_off.append(len(prod_idx_list))
		# create arrays
		kf_c = kf_ptr
		kr_c = kr_ptr
		import ctypes
		c_reac_idx = (ctypes.c_int * len(reac_idx_list))(*reac_idx_list) if len(reac_idx_list) > 0 else <int*>0
		c_reac_nu = (ctypes.c_double * len(reac_nu_list))(*reac_nu_list) if len(reac_nu_list) > 0 else <double*>0
		c_reac_off = (ctypes.c_int * len(reac_off))(*reac_off)
		c_prod_idx = (ctypes.c_int * len(prod_idx_list))(*prod_idx_list) if len(prod_idx_list) > 0 else <int*>0
		c_prod_nu = (ctypes.c_double * len(prod_nu_list))(*prod_nu_list) if len(prod_nu_list) > 0 else <double*>0
		c_prod_off = (ctypes.c_int * len(prod_off))(*prod_off)
		# conc0
		conc0_py = [float(spec.get('initial', {}).get('conc', {}).get(s, 0.0)) for s in species]
		c_conc0 = (ctypes.c_double * N)(*conc0_py)
		# output buffers
		nsteps = int(round(float(sim.get('time_span', 10.0)) / float(sim.get('time_step', 0.01))))
		npts = nsteps + 1
		max_len = npts
		times_buf = (ctypes.c_double * max_len)()
		conc_out_flat = (ctypes.c_double * (max_len * N))()
		# call
		written = simulate_multi_reactor(N, M, kf_c, kr_c,
										 <int*>c_reac_idx, <double*>c_reac_nu, <int*>c_reac_off,
										 <int*>c_prod_idx, <double*>c_prod_nu, <int*>c_prod_off,
										 <double*>c_conc0,
										 float(sim.get('time_span', 10.0)), float(sim.get('time_step', 0.01)),
										 <double*>times_buf, <double*>conc_out_flat, max_len)
		if written <= 0:
			raise RuntimeError('multi-reactor C++ simulation failed')
		# build numpy arrays if available
		if _HAS_NUMPY:
			tarr = np.empty(written, dtype=np.float64)
			carr = np.empty((written, N), dtype=np.float64)
			for i in range(written):
				tarr[i] = times_buf[i]
				for j in range(N):
					carr[i, j] = conc_out_flat[i*N + j]
			return tarr, carr
		else:
			times_py = [times_buf[i] for i in range(written)]
			traj = [[conc_out_flat[i*N + j] for j in range(N)] for i in range(written)]
			return times_py, traj
	else:
		kf = float(reaction.get('kf', 1.0))
		kr = float(reaction.get('kr', 0.5))
		conc = initial.get('conc', {})
		A0 = float(conc.get('A', 1.0))
		B0 = float(conc.get('B', 0.0))
		time_span = float(sim.get('time_span', 10.0))
		dt = float(sim.get('time_step', 0.01))
		rxn = Reaction(kf, kr)
		reactor = Reactor(rxn, A0=A0, B0=B0, time_span=time_span, dt=dt)
		return reactor.run()
