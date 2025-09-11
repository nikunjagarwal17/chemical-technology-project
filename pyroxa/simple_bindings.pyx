# Simplified PyroXa bindings for Python 3.13 compatibility  
# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free

# Import essential C functions from core.h
cdef extern from "core.h":
    int simulate_reactor(double kf, double kr, double A0, double B0,
                         double time_span, double dt,
                         double* times, double* Aout, double* Bout, int max_len)
    double enthalpy_c(double cp, double T)
    double entropy_c(double cp, double T)
    double autocatalytic_rate(double k, double A, double B, double temperature)
    double michaelis_menten_rate(double Vmax, double Km, double substrate_conc)
    double arrhenius_rate(double A, double Ea, double T, double R)

def py_simulate_reactor(double kf, double kr, double A0, double B0, 
                       double time_span, double dt=0.01):
    """Simulate simple A <=> B reactor"""
    cdef int max_len = int(time_span / dt) + 10
    cdef double* times = <double*>malloc(max_len * sizeof(double))
    cdef double* Aout = <double*>malloc(max_len * sizeof(double))  
    cdef double* Bout = <double*>malloc(max_len * sizeof(double))
    
    cdef int actual_len = simulate_reactor(kf, kr, A0, B0, time_span, dt, 
                                          times, Aout, Bout, max_len)
    
    # Convert to Python lists
    times_py = [times[i] for i in range(actual_len)]
    A_py = [Aout[i] for i in range(actual_len)]
    B_py = [Bout[i] for i in range(actual_len)]
    
    free(times)
    free(Aout)
    free(Bout)
    
    return {'times': times_py, 'A': A_py, 'B': B_py}

def py_enthalpy_c(double cp, double T):
    """Calculate enthalpy"""
    return enthalpy_c(cp, T)

def py_entropy_c(double cp, double T):
    """Calculate entropy"""  
    return entropy_c(cp, T)

def py_autocatalytic_rate(double k, double A, double B, double temperature):
    """Calculate autocatalytic rate"""
    return autocatalytic_rate(k, A, B, temperature)

def py_michaelis_menten_rate(double Vmax, double Km, double substrate_conc):
    """Calculate Michaelis-Menten rate"""
    return michaelis_menten_rate(Vmax, Km, substrate_conc)

def py_arrhenius_rate(double A, double Ea, double T, double R=8.314):
    """Calculate Arrhenius rate"""
    return arrhenius_rate(A, Ea, T, R)
