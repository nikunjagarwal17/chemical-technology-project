#include <Python.h>
#include <numpy/arrayobject.h>
#include "core.h"

// Simple wrapper for autocatalytic_rate
static PyObject* py_autocatalytic_rate_simple(PyObject* self, PyObject* args) {
    double k, A, B;
    if (!PyArg_ParseTuple(args, "ddd", &k, &A, &B)) {
        return NULL;
    }
    
    double result = autocatalytic_rate(k, A, B);
    return PyFloat_FromDouble(result);
}

// Simple wrapper for pressure_drop_ergun  
static PyObject* py_pressure_drop_ergun_simple(PyObject* self, PyObject* args) {
    double velocity, density, viscosity, particle_diameter, bed_porosity, bed_length;
    if (!PyArg_ParseTuple(args, "dddddd", &velocity, &density, &viscosity, 
                          &particle_diameter, &bed_porosity, &bed_length)) {
        return NULL;
    }
    
    double result = pressure_drop_ergun(velocity, density, viscosity, 
                                       particle_diameter, bed_porosity, bed_length);
    return PyFloat_FromDouble(result);
}

// Method definitions
static PyMethodDef SimpleMethods[] = {
    {"autocatalytic_rate", py_autocatalytic_rate_simple, METH_VARARGS, "Calculate autocatalytic rate"},
    {"pressure_drop_ergun", py_pressure_drop_ergun_simple, METH_VARARGS, "Calculate Ergun pressure drop"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef simplemodule = {
    PyModuleDef_HEAD_INIT,
    "simple_core",
    NULL,
    -1,
    SimpleMethods
};

// Module initialization
PyMODINIT_FUNC PyInit_simple_core(void) {
    PyObject* module = PyModule_Create(&simplemodule);
    if (module == NULL) {
        return NULL;
    }
    
    // Initialize numpy
    import_array();
    
    return module;
}
