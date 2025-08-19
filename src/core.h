#ifndef SIMPLECANTERA_CORE_H
#define SIMPLECANTERA_CORE_H

extern "C" {

// Simulate a single well-mixed reactor (A <=> B) using forward Euler.
// Returns number of points written (nsteps+1) on success, or -1 on error.
int simulate_reactor(double kf, double kr, double A0, double B0,
                     double time_span, double dt,
                     double* times, double* Aout, double* Bout, int max_len);

// Simple thermodynamics helpers
double enthalpy_c(double cp, double T);
double entropy_c(double cp, double T);

// Multi-species multi-reaction RK4 simulator
// N: number of species, M: number of reactions
// kf, kr: length M
// reac_idx, reac_nu, reac_off: flattened reactant indices, stoich, offsets (size M+1)
// prod_idx, prod_nu, prod_off: same for products
// conc0: length N initial concentrations
// times: output times array length max_len
// conc_out_flat: output flattened concentrations length max_len * N (row-major times)
int simulate_multi_reactor(int N, int M,
                           double* kf, double* kr,
                           int* reac_idx, double* reac_nu, int* reac_off,
                           int* prod_idx, double* prod_nu, int* prod_off,
                           double* conc0,
                           double time_span, double dt,
                           double* times, double* conc_out_flat, int max_len);

// Adaptive RK45 versions (embedded) with relative & absolute tolerances.
// dt_init: initial time step to try
// atol, rtol: absolute and relative tolerances
int simulate_reactor_adaptive(double kf, double kr, double A0, double B0,
                              double time_span, double dt_init, double atol, double rtol,
                              double* times, double* Aout, double* Bout, int max_len);

int simulate_multi_reactor_adaptive(int N, int M,
                                    double* kf, double* kr,
                                    int* reac_idx, double* reac_nu, int* reac_off,
                                    int* prod_idx, double* prod_nu, int* prod_off,
                                    double* conc0,
                                    double time_span, double dt_init, double atol, double rtol,
                                    double* times, double* conc_out_flat, int max_len);

}

#endif // SIMPLECANTERA_CORE_H
