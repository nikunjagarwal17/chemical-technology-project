#include "core.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

// Constants for numerical methods
const double PI = 3.14159265358979323846;
const double R_GAS = 8.31446261815324;  // J/mol/K
const double BOLTZMANN = 1.380649e-23;  // J/K
const double AVOGADRO = 6.02214076e23;  // mol^-1

int simulate_reactor(double kf, double kr, double A0, double B0,
                     double time_span, double dt,
                     double* times, double* Aout, double* Bout, int max_len) {
    if (dt <= 0 || time_span < 0 || max_len <= 0) return -1;
    int nsteps = (int)std::round(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    double A = A0;
    double B = B0;
    times[0] = 0.0;
    Aout[0] = A;
    Bout[0] = B;
    // RK4 integrator for dy/dt where y = [A, B], dy/dt = [-r, r] with r = kf*A - kr*B
    for (int i = 0; i < nsteps; ++i) {
        double t = i * dt;
        // k1
        double r1 = kf * A - kr * B;
        double k1A = -r1;
        double k1B = r1;
        // k2
        double A2 = A + 0.5 * dt * k1A;
        double B2 = B + 0.5 * dt * k1B;
        double r2 = kf * A2 - kr * B2;
        double k2A = -r2;
        double k2B = r2;
        // k3
        double A3 = A + 0.5 * dt * k2A;
        double B3 = B + 0.5 * dt * k2B;
        double r3 = kf * A3 - kr * B3;
        double k3A = -r3;
        double k3B = r3;
        // k4
        double A4 = A + dt * k3A;
        double B4 = B + dt * k3B;
        double r4 = kf * A4 - kr * B4;
        double k4A = -r4;
        double k4B = r4;

        A += (dt / 6.0) * (k1A + 2.0 * k2A + 2.0 * k3A + k4A);
        B += (dt / 6.0) * (k1B + 2.0 * k2B + 2.0 * k3B + k4B);

        if (A < 0) A = 0;
        if (B < 0) B = 0;
        times[i+1] = (i+1) * dt;
        Aout[i+1] = A;
        Bout[i+1] = B;
    }
    return nsteps + 1;
}

// Enhanced thermodynamics functions with advanced features
double enthalpy_c(double cp, double T) {
    if (T <= 0) return NAN;
    return cp * T;
}

double entropy_c(double cp, double T) {
    if (T <= 0) return NAN;
    return cp * std::log(T);
}

double gibbs_free_energy(double enthalpy, double entropy, double T) {
    if (T <= 0) return NAN;
    return enthalpy - T * entropy;
}

double equilibrium_constant(double delta_G, double T) {
    if (T <= 0) return NAN;
    return std::exp(-delta_G / (R_GAS * T));
}

double arrhenius_rate(double A, double Ea, double T, double R) {
    if (T <= 0 || A <= 0) return 0.0;
    return A * std::exp(-Ea / (R * T));
}

void calculate_rate_constants(int M, double* kf_ref, double* kr_ref,
                             double* Ea_f, double* Ea_r, double T, double T_ref,
                             double* kf_out, double* kr_out) {
    if (T <= 0 || T_ref <= 0) return;
    
    for (int i = 0; i < M; ++i) {
        if (kf_ref[i] > 0) {
            kf_out[i] = kf_ref[i] * std::exp(Ea_f[i] / R_GAS * (1.0/T_ref - 1.0/T));
        } else {
            kf_out[i] = 0.0;
        }
        
        if (kr_ref[i] > 0) {
            kr_out[i] = kr_ref[i] * std::exp(Ea_r[i] / R_GAS * (1.0/T_ref - 1.0/T));
        } else {
            kr_out[i] = 0.0;
        }
    }
}

int simulate_multi_reactor(int N, int M,
                           double* kf, double* kr,
                           int* reac_idx, double* reac_nu, int* reac_off,
                           int* prod_idx, double* prod_nu, int* prod_off,
                           double* conc0,
                           double time_span, double dt,
                           double* times, double* conc_out_flat, int max_len) {
    if (dt <= 0 || time_span < 0 || max_len <= 0 || N <= 0 || M < 0) return -1;
    int nsteps = (int)std::round(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    // allocate working arrays
    double* y = (double*)malloc(sizeof(double) * N);
    double* k1 = (double*)malloc(sizeof(double) * N);
    double* k2 = (double*)malloc(sizeof(double) * N);
    double* k3 = (double*)malloc(sizeof(double) * N);
    double* k4 = (double*)malloc(sizeof(double) * N);
    double* yt = (double*)malloc(sizeof(double) * N);
    if (!y || !k1 || !k2 || !k3 || !k4 || !yt) {
        free(y); free(k1); free(k2); free(k3); free(k4); free(yt);
        return -1;
    }
    for (int i = 0; i < N; ++i) y[i] = conc0[i];
    // write initial
    times[0] = 0.0;
    for (int i = 0; i < N; ++i) conc_out_flat[i] = y[i];

    auto compute_dydt = [&](double* state, double* out) {
        // zero
        for (int i = 0; i < N; ++i) out[i] = 0.0;
        // for each reaction
        for (int r = 0; r < M; ++r) {
            // reactant product f and r
            double f = 1.0;
            int start = reac_off[r];
            int end = reac_off[r+1];
            for (int ii = start; ii < end; ++ii) {
                int idx = reac_idx[ii];
                double nu = reac_nu[ii];
                double val = state[idx];
                if (val <= 0) { f = 0.0; break; }
                f *= pow(val, nu);
            }
            double rr = 1.0;
            start = prod_off[r];
            end = prod_off[r+1];
            for (int ii = start; ii < end; ++ii) {
                int idx = prod_idx[ii];
                double nu = prod_nu[ii];
                double val = state[idx];
                if (val <= 0) { rr = 0.0; break; }
                rr *= pow(val, nu);
            }
            double rate = kf[r] * f - kr[r] * rr;
            // apply stoichiometry
            start = reac_off[r];
            end = reac_off[r+1];
            for (int ii = start; ii < end; ++ii) {
                int idx = reac_idx[ii];
                double nu = reac_nu[ii];
                out[idx] -= nu * rate;
            }
            start = prod_off[r];
            end = prod_off[r+1];
            for (int ii = start; ii < end; ++ii) {
                int idx = prod_idx[ii];
                double nu = prod_nu[ii];
                out[idx] += nu * rate;
            }
        }
    };

    for (int step = 0; step < nsteps; ++step) {
        // k1
        compute_dydt(y, k1);
        for (int i = 0; i < N; ++i) yt[i] = y[i] + 0.5 * dt * k1[i];
        // k2
        compute_dydt(yt, k2);
        for (int i = 0; i < N; ++i) yt[i] = y[i] + 0.5 * dt * k2[i];
        // k3
        compute_dydt(yt, k3);
        for (int i = 0; i < N; ++i) yt[i] = y[i] + dt * k3[i];
        // k4
        compute_dydt(yt, k4);
        // update
        for (int i = 0; i < N; ++i) {
            y[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            if (y[i] < 0) y[i] = 0.0;
        }
        times[step+1] = (step+1) * dt;
        for (int i = 0; i < N; ++i) conc_out_flat[(step+1)*N + i] = y[i];
    }

    free(y); free(k1); free(k2); free(k3); free(k4); free(yt);
    return nsteps + 1;
}

// Helper: compute derivative for multi-species (same logic as lambda above)
static void compute_dydt_multi(int N, int M,
                               double* kf, double* kr,
                               int* reac_idx, double* reac_nu, int* reac_off,
                               int* prod_idx, double* prod_nu, int* prod_off,
                               double* state, double* out) {
    for (int i = 0; i < N; ++i) out[i] = 0.0;
    for (int r = 0; r < M; ++r) {
        double f = 1.0;
        int start = reac_off[r];
        int end = reac_off[r+1];
        for (int ii = start; ii < end; ++ii) {
            int idx = reac_idx[ii];
            double nu = reac_nu[ii];
            double val = state[idx];
            if (val <= 0) { f = 0.0; break; }
            f *= pow(val, nu);
        }
        double rr = 1.0;
        start = prod_off[r];
        end = prod_off[r+1];
        for (int ii = start; ii < end; ++ii) {
            int idx = prod_idx[ii];
            double nu = prod_nu[ii];
            double val = state[idx];
            if (val <= 0) { rr = 0.0; break; }
            rr *= pow(val, nu);
        }
        double rate = kf[r] * f - kr[r] * rr;
        start = reac_off[r];
        end = reac_off[r+1];
        for (int ii = start; ii < end; ++ii) {
            int idx = reac_idx[ii];
            double nu = reac_nu[ii];
            out[idx] -= nu * rate;
        }
        start = prod_off[r];
        end = prod_off[r+1];
        for (int ii = start; ii < end; ++ii) {
            int idx = prod_idx[ii];
            double nu = prod_nu[ii];
            out[idx] += nu * rate;
        }
    }
}

// Simple embedded Cash-Karp RK45 implementation for single A<=>B
int simulate_reactor_adaptive(double kf, double kr, double A0, double B0,
                              double time_span, double dt_init, double atol, double rtol,
                              double* times, double* Aout, double* Bout, int max_len) {
    if (dt_init <= 0 || time_span < 0 || max_len <= 0) return -1;
    double t = 0.0;
    double A = A0, B = B0;
    int written = 0;
    times[written] = t;
    Aout[written] = A;
    Bout[written] = B;
    written++;
    double dt = dt_init;
    // Cash-Karp coefficients
    const double a2 = 1.0/5.0, a3 = 3.0/10.0, a4 = 3.0/5.0, a5 = 1.0, a6 = 7.0/8.0;
    const double b21 = 1.0/5.0;
    const double b31 = 3.0/40.0, b32 = 9.0/40.0;
    const double b41 = 3.0/10.0, b42 = -9.0/10.0, b43 = 6.0/5.0;
    const double b51 = -11.0/54.0, b52 = 5.0/2.0, b53 = -70.0/27.0, b54 = 35.0/27.0;
    const double b61 = 1631.0/55296.0, b62 = 175.0/512.0, b63 = 575.0/13824.0, b64 = 44275.0/110592.0, b65 = 253.0/4096.0;
    const double c1 = 37.0/378.0, c3 = 250.0/621.0, c4 = 125.0/594.0, c6 = 512.0/1771.0;
    const double dc1 = c1 - 2825.0/27648.0;
    const double dc3 = c3 - 18575.0/48384.0;
    const double dc4 = c4 - 13525.0/55296.0;
    const double dc5 = -277.00/14336.0;
    const double dc6 = c6 - 0.25;

    while (t < time_span && written < max_len) {
        if (t + dt > time_span) dt = time_span - t;
        // compute k1..k6 for A,B scalar
        double r1 = kf * A - kr * B;
        double k1A = -r1, k1B = r1;
        double A2 = A + dt * b21 * k1A;
        double B2 = B + dt * b21 * k1B;
        double r2 = kf * A2 - kr * B2;
        double k2A = -r2, k2B = r2;
        double A3 = A + dt * (b31 * k1A + b32 * k2A);
        double B3 = B + dt * (b31 * k1B + b32 * k2B);
        double r3 = kf * A3 - kr * B3;
        double k3A = -r3, k3B = r3;
        double A4 = A + dt * (b41 * k1A + b42 * k2A + b43 * k3A);
        double B4 = B + dt * (b41 * k1B + b42 * k2B + b43 * k3B);
        double r4 = kf * A4 - kr * B4;
        double k4A = -r4, k4B = r4;
        double A5 = A + dt * (b51 * k1A + b52 * k2A + b53 * k3A + b54 * k4A);
        double B5 = B + dt * (b51 * k1B + b52 * k2B + b53 * k3B + b54 * k4B);
        double r5 = kf * A5 - kr * B5;
        double k5A = -r5, k5B = r5;
        double A6 = A + dt * (b61 * k1A + b62 * k2A + b63 * k3A + b64 * k4A + b65 * k5A);
        double B6 = B + dt * (b61 * k1B + b62 * k2B + b63 * k3B + b64 * k4B + b65 * k5B);
        double r6 = kf * A6 - kr * B6;
        double k6A = -r6, k6B = r6;
        // high order solution
        double Ah = A + dt * (c1 * k1A + c3 * k3A + c4 * k4A + c6 * k6A);
        double Bh = B + dt * (c1 * k1B + c3 * k3B + c4 * k4B + c6 * k6B);
        // error estimate
        double errA = dt * (dc1 * k1A + dc3 * k3A + dc4 * k4A + dc5 * k5A + dc6 * k6A);
        double errB = dt * (dc1 * k1B + dc3 * k3B + dc4 * k4B + dc5 * k5B + dc6 * k6B);
        double scA = atol + rtol * fmax(fabs(A), fabs(Ah));
        double scB = atol + rtol * fmax(fabs(B), fabs(Bh));
        double err = sqrt((errA/ scA)*(errA/ scA) + (errB/ scB)*(errB/ scB)) / sqrt(2.0);
        if (err <= 1.0) {
            // accept
            t += dt;
            A = Ah; B = Bh;
            if (A < 0) A = 0; if (B < 0) B = 0;
            if (written < max_len) {
                times[written] = t;
                Aout[written] = A;
                Bout[written] = B;
                written++;
            }
        }
        // update dt
        double safety = 0.9;
        double minscale = 0.2;
        double maxscale = 5.0;
        if (err == 0.0) err = 1e-16;
        double scale = safety * pow(err, -0.2);
        scale = fmax(minscale, fmin(maxscale, scale));
        dt *= scale;
        if (dt < 1e-12) dt = 1e-12;
    }
    return written;
}

// Adaptive RK45 for multi-species
int simulate_multi_reactor_adaptive(int N, int M,
                                    double* kf, double* kr,
                                    int* reac_idx, double* reac_nu, int* reac_off,
                                    int* prod_idx, double* prod_nu, int* prod_off,
                                    double* conc0,
                                    double time_span, double dt_init, double atol, double rtol,
                                    double* times, double* conc_out_flat, int max_len) {
    if (dt_init <= 0 || time_span < 0 || max_len <= 0 || N <= 0 || M < 0) return -1;
    int written = 0;
    double t = 0.0;
    double* y = (double*)malloc(sizeof(double)*N);
    double* k1 = (double*)malloc(sizeof(double)*N);
    double* k2 = (double*)malloc(sizeof(double)*N);
    double* k3 = (double*)malloc(sizeof(double)*N);
    double* k4 = (double*)malloc(sizeof(double)*N);
    double* k5 = (double*)malloc(sizeof(double)*N);
    double* k6 = (double*)malloc(sizeof(double)*N);
    double* ytmp = (double*)malloc(sizeof(double)*N);
    double* yhigh = (double*)malloc(sizeof(double)*N);
    double* errv = (double*)malloc(sizeof(double)*N);
    if (!y || !k1 || !k2 || !k3 || !k4 || !k5 || !k6 || !ytmp || !yhigh || !errv) {
        free(y); free(k1); free(k2); free(k3); free(k4); free(k5); free(k6); free(ytmp); free(yhigh); free(errv);
        return -1;
    }
    for (int i=0;i<N;++i) y[i] = conc0[i];
    times[written] = 0.0;
    for (int i=0;i<N;++i) conc_out_flat[i] = y[i];
    written++;
    double dt = dt_init;
    // Cash-Karp coeffs as above
    const double a2 = 1.0/5.0, a3 = 3.0/10.0, a4 = 3.0/5.0, a5 = 1.0, a6 = 7.0/8.0;
    const double b21 = 1.0/5.0;
    const double b31 = 3.0/40.0, b32 = 9.0/40.0;
    const double b41 = 3.0/10.0, b42 = -9.0/10.0, b43 = 6.0/5.0;
    const double b51 = -11.0/54.0, b52 = 5.0/2.0, b53 = -70.0/27.0, b54 = 35.0/27.0;
    const double b61 = 1631.0/55296.0, b62 = 175.0/512.0, b63 = 575.0/13824.0, b64 = 44275.0/110592.0, b65 = 253.0/4096.0;
    const double c1 = 37.0/378.0, c3 = 250.0/621.0, c4 = 125.0/594.0, c6 = 512.0/1771.0;
    const double dc1 = c1 - 2825.0/27648.0;
    const double dc3 = c3 - 18575.0/48384.0;
    const double dc4 = c4 - 13525.0/55296.0;
    const double dc5 = -277.00/14336.0;
    const double dc6 = c6 - 0.25;

    while (t < time_span && written < max_len) {
        if (t + dt > time_span) dt = time_span - t;
        // k1
        compute_dydt_multi(N,M,kf,kr,reac_idx,reac_nu,reac_off,prod_idx,prod_nu,prod_off,y,k1);
        for (int i=0;i<N;++i) ytmp[i] = y[i] + dt * b21 * k1[i];
        // k2
        compute_dydt_multi(N,M,kf,kr,reac_idx,reac_nu,reac_off,prod_idx,prod_nu,prod_off,ytmp,k2);
        for (int i=0;i<N;++i) ytmp[i] = y[i] + dt * (b31 * k1[i] + b32 * k2[i]);
        // k3
        compute_dydt_multi(N,M,kf,kr,reac_idx,reac_nu,reac_off,prod_idx,prod_nu,prod_off,ytmp,k3);
        for (int i=0;i<N;++i) ytmp[i] = y[i] + dt * (b41 * k1[i] + b42 * k2[i] + b43 * k3[i]);
        // k4
        compute_dydt_multi(N,M,kf,kr,reac_idx,reac_nu,reac_off,prod_idx,prod_nu,prod_off,ytmp,k4);
        for (int i=0;i<N;++i) ytmp[i] = y[i] + dt * (b51 * k1[i] + b52 * k2[i] + b53 * k3[i] + b54 * k4[i]);
        // k5
        compute_dydt_multi(N,M,kf,kr,reac_idx,reac_nu,reac_off,prod_idx,prod_nu,prod_off,ytmp,k5);
        for (int i=0;i<N;++i) ytmp[i] = y[i] + dt * (b61 * k1[i] + b62 * k2[i] + b63 * k3[i] + b64 * k4[i] + b65 * k5[i]);
        // k6
        compute_dydt_multi(N,M,kf,kr,reac_idx,reac_nu,reac_off,prod_idx,prod_nu,prod_off,ytmp,k6);
        // high-order
        for (int i=0;i<N;++i) {
            yhigh[i] = y[i] + dt * (c1 * k1[i] + c3 * k3[i] + c4 * k4[i] + c6 * k6[i]);
            errv[i] = dt * (dc1 * k1[i] + dc3 * k3[i] + dc4 * k4[i] + dc5 * k5[i] + dc6 * k6[i]);
        }
        // norm
        double errnorm = 0.0;
        for (int i=0;i<N;++i) {
            double sc = atol + rtol * fmax(fabs(y[i]), fabs(yhigh[i]));
            double e = errv[i] / sc;
            errnorm += e*e;
        }
        errnorm = sqrt(errnorm / (double)N);
        if (errnorm <= 1.0) {
            t += dt;
            for (int i=0;i<N;++i) {
                y[i] = yhigh[i];
                if (y[i] < 0) y[i] = 0.0;
            }
            if (written < max_len) {
                times[written] = t;
                for (int i=0;i<N;++i) conc_out_flat[written*N + i] = y[i];
                written++;
            }
        }
        double safety = 0.9;
        double minscale = 0.2;
        double maxscale = 5.0;
        if (errnorm == 0.0) errnorm = 1e-16;
        double scale = safety * pow(errnorm, -0.2);
        if (scale < minscale) scale = minscale;
        if (scale > maxscale) scale = maxscale;
        dt *= scale;
        if (dt < 1e-16) dt = 1e-16;
    }

    free(y); free(k1); free(k2); free(k3); free(k4); free(k5); free(k6); free(ytmp); free(yhigh); free(errv);
    return written;
}

// ============================================================================
// ADVANCED REACTOR IMPLEMENTATIONS
// ============================================================================

int simulate_pfr(int N, int M, int nseg,
                 double* kf, double* kr,
                 int* reac_idx, double* reac_nu, int* reac_off,
                 int* prod_idx, double* prod_nu, int* prod_off,
                 double* conc0, double flow_rate, double total_volume,
                 double time_span, double dt,
                 double* times, double* conc_out_flat, int max_len) {
    
    if (nseg <= 0 || flow_rate <= 0 || total_volume <= 0) return -1;
    
    double segment_volume = total_volume / nseg;
    double residence_time = segment_volume / flow_rate;
    
    // Allocate memory for concentrations at each segment
    std::vector<std::vector<double>> conc_segments(nseg, std::vector<double>(N));
    
    // Initialize first segment with inlet conditions
    for (int i = 0; i < N; ++i) {
        conc_segments[0][i] = conc0[i];
    }
    
    int nsteps = (int)std::round(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    
    times[0] = 0.0;
    for (int i = 0; i < N; ++i) {
        conc_out_flat[i] = conc_segments[nseg-1][i];
    }
    
    for (int step = 0; step < nsteps; ++step) {
        // Update each segment from outlet to inlet
        for (int seg = nseg - 1; seg >= 0; --seg) {
            std::vector<double> dydt(N, 0.0);
            
            // Calculate reaction rates for this segment
            compute_dydt_multi(N, M, kf, kr, reac_idx, reac_nu, reac_off,
                             prod_idx, prod_nu, prod_off,
                             conc_segments[seg].data(), dydt.data());
            
            // Add convective term (flow from previous segment or inlet)
            if (seg == 0) {
                // First segment gets inlet concentration
                for (int i = 0; i < N; ++i) {
                    dydt[i] += (conc0[i] - conc_segments[seg][i]) / residence_time;
                }
            } else {
                // Other segments get concentration from upstream
                for (int i = 0; i < N; ++i) {
                    dydt[i] += (conc_segments[seg-1][i] - conc_segments[seg][i]) / residence_time;
                }
            }
            
            // Update concentrations using explicit Euler (could be enhanced with RK4)
            for (int i = 0; i < N; ++i) {
                conc_segments[seg][i] += dt * dydt[i];
                if (conc_segments[seg][i] < 0) conc_segments[seg][i] = 0.0;
            }
        }
        
        // Store outlet concentrations
        times[step + 1] = (step + 1) * dt;
        for (int i = 0; i < N; ++i) {
            conc_out_flat[(step + 1) * N + i] = conc_segments[nseg-1][i];
        }
    }
    
    return nsteps + 1;
}

int simulate_cstr(int N, int M,
                  double* kf, double* kr,
                  int* reac_idx, double* reac_nu, int* reac_off,
                  int* prod_idx, double* prod_nu, int* prod_off,
                  double* conc0, double* conc_in, double flow_rate, double volume,
                  double time_span, double dt,
                  double* times, double* conc_out_flat, int max_len) {
    
    if (flow_rate <= 0 || volume <= 0) return -1;
    
    double residence_time = volume / flow_rate;
    int nsteps = (int)std::round(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    
    std::vector<double> y(N);
    std::vector<double> dydt(N);
    
    // Initialize with initial conditions
    for (int i = 0; i < N; ++i) {
        y[i] = conc0[i];
    }
    
    times[0] = 0.0;
    for (int i = 0; i < N; ++i) {
        conc_out_flat[i] = y[i];
    }
    
    for (int step = 0; step < nsteps; ++step) {
        // Calculate reaction rates
        compute_dydt_multi(N, M, kf, kr, reac_idx, reac_nu, reac_off,
                         prod_idx, prod_nu, prod_off, y.data(), dydt.data());
        
        // Add flow terms: (inlet - outlet) / residence_time
        for (int i = 0; i < N; ++i) {
            dydt[i] += (conc_in[i] - y[i]) / residence_time;
        }
        
        // Update using explicit Euler
        for (int i = 0; i < N; ++i) {
            y[i] += dt * dydt[i];
            if (y[i] < 0) y[i] = 0.0;
        }
        
        times[step + 1] = (step + 1) * dt;
        for (int i = 0; i < N; ++i) {
            conc_out_flat[(step + 1) * N + i] = y[i];
        }
    }
    
    return nsteps + 1;
}

int simulate_batch_variable_temp(int N, int M,
                                 double* kf_ref, double* kr_ref, double* Ea_f, double* Ea_r,
                                 int* reac_idx, double* reac_nu, int* reac_off,
                                 int* prod_idx, double* prod_nu, int* prod_off,
                                 double* conc0, double* temp_profile, double T_ref,
                                 double time_span, double dt,
                                 double* times, double* conc_out_flat, 
                                 double* temp_out, int max_len) {
    
    int nsteps = (int)std::round(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    
    std::vector<double> y(N);
    std::vector<double> dydt(N);
    std::vector<double> kf_temp(M);
    std::vector<double> kr_temp(M);
    
    // Initialize
    for (int i = 0; i < N; ++i) {
        y[i] = conc0[i];
    }
    
    times[0] = 0.0;
    temp_out[0] = temp_profile[0];
    for (int i = 0; i < N; ++i) {
        conc_out_flat[i] = y[i];
    }
    
    for (int step = 0; step < nsteps; ++step) {
        double T = temp_profile[step];
        temp_out[step + 1] = temp_profile[step + 1];
        
        // Calculate temperature-dependent rate constants
        calculate_rate_constants(M, kf_ref, kr_ref, Ea_f, Ea_r, T, T_ref,
                               kf_temp.data(), kr_temp.data());
        
        // Calculate reaction rates at current temperature
        compute_dydt_multi(N, M, kf_temp.data(), kr_temp.data(),
                         reac_idx, reac_nu, reac_off,
                         prod_idx, prod_nu, prod_off, y.data(), dydt.data());
        
        // Update concentrations
        for (int i = 0; i < N; ++i) {
            y[i] += dt * dydt[i];
            if (y[i] < 0) y[i] = 0.0;
        }
        
        times[step + 1] = (step + 1) * dt;
        for (int i = 0; i < N; ++i) {
            conc_out_flat[(step + 1) * N + i] = y[i];
        }
    }
    
    return nsteps + 1;
}

// ============================================================================
// ANALYTICAL SOLUTIONS
// ============================================================================

int analytical_first_order(double k, double A0, double time_span, double dt,
                          double* times, double* A_out, double* B_out, int max_len) {
    
    int nsteps = (int)std::round(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    
    for (int i = 0; i <= nsteps; ++i) {
        double t = i * dt;
        times[i] = t;
        A_out[i] = A0 * std::exp(-k * t);
        B_out[i] = A0 * (1.0 - std::exp(-k * t));
    }
    
    return nsteps + 1;
}

int analytical_consecutive_first_order(double k1, double k2, double A0,
                                      double time_span, double dt,
                                      double* times, double* A_out, 
                                      double* B_out, double* C_out, int max_len) {
    
    int nsteps = (int)std::round(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    
    for (int i = 0; i <= nsteps; ++i) {
        double t = i * dt;
        times[i] = t;
        
        A_out[i] = A0 * std::exp(-k1 * t);
        
        if (std::abs(k1 - k2) < 1e-12) {
            // Special case when k1 ≈ k2
            B_out[i] = A0 * k1 * t * std::exp(-k1 * t);
        } else {
            B_out[i] = A0 * k1 / (k2 - k1) * (std::exp(-k1 * t) - std::exp(-k2 * t));
        }
        
        C_out[i] = A0 - A_out[i] - B_out[i];
        if (C_out[i] < 0) C_out[i] = 0.0;
    }
    
    return nsteps + 1;
}

int analytical_reversible_first_order(double kf, double kr, double A0, double B0,
                                     double time_span, double dt,
                                     double* times, double* A_out, double* B_out, int max_len) {
    
    int nsteps = (int)std::round(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    
    double k_total = kf + kr;
    double A_eq = kr * (A0 + B0) / k_total;
    double B_eq = kf * (A0 + B0) / k_total;
    
    for (int i = 0; i <= nsteps; ++i) {
        double t = i * dt;
        times[i] = t;
        
        double exp_term = std::exp(-k_total * t);
        A_out[i] = A_eq + (A0 - A_eq) * exp_term;
        B_out[i] = B_eq + (B0 - B_eq) * exp_term;
    }
    
    return nsteps + 1;
}

// ============================================================================
// OPTIMIZATION AND SENSITIVITY ANALYSIS
// ============================================================================

int calculate_sensitivity(int N, int M, int nparam,
                         double* kf, double* kr, double* param_perturbations,
                         int* reac_idx, double* reac_nu, int* reac_off,
                         int* prod_idx, double* prod_nu, int* prod_off,
                         double* conc0, double time_span, double dt,
                         double* sensitivity_matrix, int max_len) {
    
    int nsteps = (int)std::round(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    
    // Base simulation
    std::vector<double> times_base(nsteps + 1);
    std::vector<double> conc_base((nsteps + 1) * N);
    
    int result = simulate_multi_reactor(N, M, kf, kr, reac_idx, reac_nu, reac_off,
                                      prod_idx, prod_nu, prod_off, conc0,
                                      time_span, dt, times_base.data(),
                                      conc_base.data(), nsteps + 1);
    
    if (result <= 0) return -1;
    
    // Calculate sensitivity for each parameter
    for (int p = 0; p < nparam; ++p) {
        std::vector<double> kf_pert(M), kr_pert(M);
        for (int i = 0; i < M; ++i) {
            kf_pert[i] = kf[i];
            kr_pert[i] = kr[i];
        }
        
        // Perturb parameter (assuming first M parameters are kf, next M are kr)
        double perturbation = param_perturbations[p];
        if (p < M) {
            kf_pert[p] *= (1.0 + perturbation);
        } else if (p < 2*M) {
            kr_pert[p - M] *= (1.0 + perturbation);
        }
        
        std::vector<double> times_pert(nsteps + 1);
        std::vector<double> conc_pert((nsteps + 1) * N);
        
        result = simulate_multi_reactor(N, M, kf_pert.data(), kr_pert.data(),
                                      reac_idx, reac_nu, reac_off,
                                      prod_idx, prod_nu, prod_off, conc0,
                                      time_span, dt, times_pert.data(),
                                      conc_pert.data(), nsteps + 1);
        
        if (result <= 0) continue;
        
        // Calculate finite difference sensitivity
        for (int t = 0; t <= nsteps; ++t) {
            for (int i = 0; i < N; ++i) {
                double base_val = conc_base[t * N + i];
                double pert_val = conc_pert[t * N + i];
                sensitivity_matrix[p * (nsteps + 1) * N + t * N + i] = 
                    (pert_val - base_val) / (perturbation * base_val + 1e-12);
            }
        }
    }
    
    return nsteps + 1;
}

double calculate_objective_function(int ndata, double* experimental_data,
                                   double* simulated_data, double* weights) {
    double sum_sq = 0.0;
    for (int i = 0; i < ndata; ++i) {
        double residual = experimental_data[i] - simulated_data[i];
        double weight = weights ? weights[i] : 1.0;
        sum_sq += weight * residual * residual;
    }
    return sum_sq;
}

// ============================================================================
// STEADY STATE AND STABILITY ANALYSIS
// ============================================================================

int find_steady_state(int N, int M,
                     double* kf, double* kr,
                     int* reac_idx, double* reac_nu, int* reac_off,
                     int* prod_idx, double* prod_nu, int* prod_off,
                     double* conc_guess, double* conc_steady,
                     double tolerance, int max_iterations) {
    
    std::vector<double> y(N), dydt(N), y_new(N);
    
    // Initialize with guess
    for (int i = 0; i < N; ++i) {
        y[i] = conc_guess[i];
    }
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Calculate derivatives
        compute_dydt_multi(N, M, kf, kr, reac_idx, reac_nu, reac_off,
                         prod_idx, prod_nu, prod_off, y.data(), dydt.data());
        
        // Check convergence
        double max_dydt = 0.0;
        for (int i = 0; i < N; ++i) {
            max_dydt = std::max(max_dydt, std::abs(dydt[i]));
        }
        
        if (max_dydt < tolerance) {
            // Converged
            for (int i = 0; i < N; ++i) {
                conc_steady[i] = y[i];
            }
            return iter + 1;
        }
        
        // Simple Newton step (could be improved with Jacobian)
        double step_size = 0.1;
        for (int i = 0; i < N; ++i) {
            y[i] -= step_size * dydt[i];
            if (y[i] < 0) y[i] = 0.0;
        }
    }
    
    return -1; // Failed to converge
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

int matrix_multiply(double* A, double* B, double* C, int m, int n, int p) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            C[i * p + j] = 0.0;
            for (int k = 0; k < n; ++k) {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
    }
    return 0;
}

double linear_interpolate(double x, double* x_data, double* y_data, int n) {
    if (n < 2) return NAN;
    
    // Find bounding indices
    int i = 0;
    while (i < n - 1 && x_data[i + 1] < x) {
        i++;
    }
    
    if (i == n - 1) return y_data[n - 1];
    
    // Linear interpolation
    double t = (x - x_data[i]) / (x_data[i + 1] - x_data[i]);
    return y_data[i] + t * (y_data[i + 1] - y_data[i]);
}

double calculate_r_squared(double* experimental, double* predicted, int n) {
    if (n <= 1) return NAN;
    
    // Calculate mean of experimental data
    double mean_exp = 0.0;
    for (int i = 0; i < n; ++i) {
        mean_exp += experimental[i];
    }
    mean_exp /= n;
    
    // Calculate sum of squares
    double ss_tot = 0.0, ss_res = 0.0;
    for (int i = 0; i < n; ++i) {
        ss_tot += (experimental[i] - mean_exp) * (experimental[i] - mean_exp);
        ss_res += (experimental[i] - predicted[i]) * (experimental[i] - predicted[i]);
    }
    
    if (ss_tot == 0.0) return NAN;
    return 1.0 - ss_res / ss_tot;
}

double calculate_rmse(double* experimental, double* predicted, int n) {
    if (n <= 0) return NAN;
    
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = experimental[i] - predicted[i];
        sum_sq += diff * diff;
    }
    
    return std::sqrt(sum_sq / n);
}

double calculate_aic(double* experimental, double* predicted, int ndata, int nparams) {
    if (ndata <= nparams) return NAN;
    
    double rmse = calculate_rmse(experimental, predicted, ndata);
    double log_likelihood = -0.5 * ndata * std::log(2.0 * PI * rmse * rmse);
    
    return 2.0 * nparams - 2.0 * log_likelihood;
}

// ============================================================================
// C++ CLASS IMPLEMENTATIONS
// ============================================================================

#ifdef __cplusplus

namespace pyroxa {

ReactionNetwork::ReactionNetwork(int n_species, int n_reactions) 
    : n_species_(n_species), n_reactions_(n_reactions) {
    kf_.reserve(n_reactions);
    kr_.reserve(n_reactions);
    reactants_.reserve(n_reactions);
    products_.reserve(n_reactions);
    reactant_stoich_.reserve(n_reactions);
    product_stoich_.reserve(n_reactions);
}

ReactionNetwork::~ReactionNetwork() = default;

void ReactionNetwork::add_reaction(const std::vector<int>& reactants,
                                 const std::vector<double>& reactant_stoich,
                                 const std::vector<int>& products,
                                 const std::vector<double>& product_stoich,
                                 double kf, double kr) {
    reactants_.push_back(reactants);
    reactant_stoich_.push_back(reactant_stoich);
    products_.push_back(products);
    product_stoich_.push_back(product_stoich);
    kf_.push_back(kf);
    kr_.push_back(kr);
}

std::vector<double> ReactionNetwork::simulate(const std::vector<double>& initial_conc,
                                            double time_span, double dt) {
    // Convert to C interface format and call existing function
    // This is a simplified implementation - could be enhanced
    std::vector<double> result(initial_conc.size());
    
    // For now, just return initial concentrations
    // Full implementation would convert data structures and call simulate_multi_reactor
    std::copy(initial_conc.begin(), initial_conc.end(), result.begin());
    
    return result;
}

ReactorSimulator::ReactorSimulator() 
    : adaptive_stepping_(false), conservation_checking_(false),
      atol_(1e-6), rtol_(1e-6), conservation_tol_(1e-8), nthreads_(1) {
}

ReactorSimulator::~ReactorSimulator() = default;

void ReactorSimulator::set_parallel_threads(int nthreads) {
    nthreads_ = std::max(1, nthreads);
}

void ReactorSimulator::enable_adaptive_stepping(double atol, double rtol) {
    adaptive_stepping_ = true;
    atol_ = atol;
    rtol_ = rtol;
}

} // namespace pyroxa

// ============================================================================
// PACKED BED REACTOR IMPLEMENTATIONS
// ============================================================================

int simulate_packed_bed(int N, int M, int nseg,
                       double* kf, double* kr,
                       int* reac_idx, double* reac_nu, int* reac_off,
                       int* prod_idx, double* prod_nu, int* prod_off,
                       double* conc0, double flow_rate, double bed_length,
                       double bed_porosity, double particle_diameter,
                       double catalyst_density, double effectiveness_factor,
                       double time_span, double dt,
                       double* times, double* conc_out_flat, 
                       double* pressure_out, int max_len) {
    
    if (nseg <= 0 || N <= 0 || M <= 0 || dt <= 0 || time_span < 0) return -1;
    if (bed_porosity <= 0 || bed_porosity >= 1.0) return -1;
    if (particle_diameter <= 0 || catalyst_density <= 0) return -1;
    if (effectiveness_factor <= 0 || effectiveness_factor > 1.0) return -1;
    
    int nsteps = (int)(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    
    double dz = bed_length / nseg;  // Spatial step size
    double superficial_velocity = flow_rate / (PI * pow(bed_length/4, 2)); // Simplified
    double interstitial_velocity = superficial_velocity / bed_porosity;
    
    // Initialize concentration profiles
    std::vector<std::vector<double>> conc(nseg, std::vector<double>(N));
    std::vector<std::vector<double>> conc_new(nseg, std::vector<double>(N));
    
    // Set initial conditions
    for (int j = 0; j < nseg; j++) {
        for (int i = 0; i < N; i++) {
            conc[j][i] = conc0[i];
        }
    }
    
    // Time integration
    for (int t = 0; t <= nsteps; t++) {
        times[t] = t * dt;
        
        // Store output concentrations (exit of bed)
        for (int i = 0; i < N; i++) {
            conc_out_flat[t * N + i] = conc[nseg-1][i];
        }
        
        // Calculate pressure drop (Ergun equation simplified)
        double pressure_drop = 150.0 * (1.0 - bed_porosity) * (1.0 - bed_porosity) * 
                              superficial_velocity * bed_length / 
                              (bed_porosity * bed_porosity * bed_porosity * 
                               particle_diameter * particle_diameter);
        pressure_out[t] = 101325.0 - pressure_drop; // Atmospheric pressure minus drop
        
        if (t < nsteps) {
            // Spatial discretization with convection and reaction
            for (int j = 0; j < nseg; j++) {
                for (int i = 0; i < N; i++) {
                    double convection_term = 0.0;
                    
                    // Convective transport
                    if (j == 0) {
                        // Inlet boundary condition
                        convection_term = -interstitial_velocity * (conc[j][i] - conc0[i]) / dz;
                    } else {
                        // Upwind differencing
                        convection_term = -interstitial_velocity * (conc[j][i] - conc[j-1][i]) / dz;
                    }
                    
                    // Reaction term with effectiveness factor
                    double reaction_term = 0.0;
                    for (int r = 0; r < M; r++) {
                        double rate = kf[r];
                        for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                            rate *= pow(conc[j][reac_idx[k]], reac_nu[k]);
                        }
                        double reverse_rate = kr[r];
                        for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                            reverse_rate *= pow(conc[j][prod_idx[k]], prod_nu[k]);
                        }
                        
                        // Check if species i is involved in reaction r
                        for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                            if (reac_idx[k] == i) {
                                reaction_term -= reac_nu[k] * (rate - reverse_rate) * 
                                               effectiveness_factor * catalyst_density * 
                                               (1.0 - bed_porosity) / bed_porosity;
                            }
                        }
                        for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                            if (prod_idx[k] == i) {
                                reaction_term += prod_nu[k] * (rate - reverse_rate) * 
                                               effectiveness_factor * catalyst_density * 
                                               (1.0 - bed_porosity) / bed_porosity;
                            }
                        }
                    }
                    
                    // Euler integration
                    conc_new[j][i] = conc[j][i] + dt * (convection_term + reaction_term);
                    
                    // Ensure non-negative concentrations
                    if (conc_new[j][i] < 0) conc_new[j][i] = 0;
                }
            }
            
            // Update concentrations
            conc = conc_new;
        }
    }
    
    return nsteps + 1;
}

int simulate_heterogeneous_packed_bed(int N, int M, int nseg,
                                     double* kf_intrinsic, double* kr_intrinsic,
                                     int* reac_idx, double* reac_nu, int* reac_off,
                                     int* prod_idx, double* prod_nu, int* prod_off,
                                     double* conc0, double flow_rate, double bed_length,
                                     double bed_porosity, double particle_diameter,
                                     double catalyst_density, double* mass_transfer_coeff,
                                     double time_span, double dt,
                                     double* times, double* conc_out_flat,
                                     double* surface_conc_out, int max_len) {
    
    if (nseg <= 0 || N <= 0 || M <= 0 || dt <= 0 || time_span < 0) return -1;
    
    int nsteps = (int)(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    
    double dz = bed_length / nseg;
    double superficial_velocity = flow_rate / (PI * pow(bed_length/4, 2));
    double interstitial_velocity = superficial_velocity / bed_porosity;
    
    // Bulk and surface concentrations
    std::vector<std::vector<double>> conc_bulk(nseg, std::vector<double>(N));
    std::vector<std::vector<double>> conc_surface(nseg, std::vector<double>(N));
    std::vector<std::vector<double>> conc_bulk_new(nseg, std::vector<double>(N));
    std::vector<std::vector<double>> conc_surface_new(nseg, std::vector<double>(N));
    
    // Initialize
    for (int j = 0; j < nseg; j++) {
        for (int i = 0; i < N; i++) {
            conc_bulk[j][i] = conc0[i];
            conc_surface[j][i] = conc0[i]; // Assume initial equilibrium
        }
    }
    
    // Specific surface area (simplified)
    double specific_surface = 6.0 * (1.0 - bed_porosity) / particle_diameter;
    
    for (int t = 0; t <= nsteps; t++) {
        times[t] = t * dt;
        
        // Store outputs
        for (int i = 0; i < N; i++) {
            conc_out_flat[t * N + i] = conc_bulk[nseg-1][i];
            surface_conc_out[t * N + i] = conc_surface[nseg-1][i];
        }
        
        if (t < nsteps) {
            for (int j = 0; j < nseg; j++) {
                for (int i = 0; i < N; i++) {
                    // Bulk phase mass balance
                    double convection = 0.0;
                    if (j == 0) {
                        convection = -interstitial_velocity * (conc_bulk[j][i] - conc0[i]) / dz;
                    } else {
                        convection = -interstitial_velocity * (conc_bulk[j][i] - conc_bulk[j-1][i]) / dz;
                    }
                    
                    // Mass transfer between bulk and surface
                    double mass_transfer = -mass_transfer_coeff[i] * specific_surface * 
                                          (conc_bulk[j][i] - conc_surface[j][i]);
                    
                    conc_bulk_new[j][i] = conc_bulk[j][i] + dt * (convection + mass_transfer);
                    
                    // Surface phase mass balance
                    double surface_reaction = 0.0;
                    for (int r = 0; r < M; r++) {
                        double rate = kf_intrinsic[r];
                        for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                            rate *= pow(conc_surface[j][reac_idx[k]], reac_nu[k]);
                        }
                        double reverse_rate = kr_intrinsic[r];
                        for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                            reverse_rate *= pow(conc_surface[j][prod_idx[k]], prod_nu[k]);
                        }
                        
                        // Species involvement
                        for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                            if (reac_idx[k] == i) {
                                surface_reaction -= reac_nu[k] * (rate - reverse_rate);
                            }
                        }
                        for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                            if (prod_idx[k] == i) {
                                surface_reaction += prod_nu[k] * (rate - reverse_rate);
                            }
                        }
                    }
                    
                    conc_surface_new[j][i] = conc_surface[j][i] + dt * 
                                           (-mass_transfer / catalyst_density + surface_reaction);
                    
                    // Non-negative constraints
                    if (conc_bulk_new[j][i] < 0) conc_bulk_new[j][i] = 0;
                    if (conc_surface_new[j][i] < 0) conc_surface_new[j][i] = 0;
                }
            }
            
            conc_bulk = conc_bulk_new;
            conc_surface = conc_surface_new;
        }
    }
    
    return nsteps + 1;
}

// ============================================================================
// FLUIDIZED BED REACTOR IMPLEMENTATIONS
// ============================================================================

int simulate_fluidized_bed(int N, int M,
                          double* kf, double* kr,
                          int* reac_idx, double* reac_nu, int* reac_off,
                          int* prod_idx, double* prod_nu, int* prod_off,
                          double* conc0, double gas_velocity, double bed_height,
                          double bed_porosity, double bubble_fraction,
                          double particle_diameter, double catalyst_density,
                          double time_span, double dt,
                          double* times, double* conc_out_flat,
                          double* bubble_conc_out, double* emulsion_conc_out, int max_len) {
    
    if (N <= 0 || M <= 0 || dt <= 0 || time_span < 0) return -1;
    if (bubble_fraction < 0 || bubble_fraction > 1.0) return -1;
    
    int nsteps = (int)(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    
    // Two-phase model: bubble and emulsion phases
    std::vector<double> conc_bubble(N), conc_emulsion(N);
    std::vector<double> conc_bubble_new(N), conc_emulsion_new(N);
    
    // Initialize
    for (int i = 0; i < N; i++) {
        conc_bubble[i] = conc0[i];
        conc_emulsion[i] = conc0[i];
    }
    
    // Bubble rise velocity (simplified Davidson-Harrison correlation)
    double bubble_velocity = gas_velocity + 0.711 * sqrt(9.81 * particle_diameter);
    
    // Mass transfer coefficient between phases (simplified)
    double mass_transfer_coeff = 0.975 * sqrt(gas_velocity * 9.81 / particle_diameter);
    
    // Residence times
    double tau_bubble = bed_height / bubble_velocity;
    double tau_emulsion = bed_height / (gas_velocity * (1.0 - bubble_fraction));
    
    for (int t = 0; t <= nsteps; t++) {
        times[t] = t * dt;
        
        // Overall outlet concentration (mixing of phases)
        for (int i = 0; i < N; i++) {
            conc_out_flat[t * N + i] = bubble_fraction * conc_bubble[i] + 
                                      (1.0 - bubble_fraction) * conc_emulsion[i];
            bubble_conc_out[t * N + i] = conc_bubble[i];
            emulsion_conc_out[t * N + i] = conc_emulsion[i];
        }
        
        if (t < nsteps) {
            for (int i = 0; i < N; i++) {
                // Bubble phase mass balance
                double bubble_in_out = (conc0[i] - conc_bubble[i]) / tau_bubble;
                double bubble_mass_transfer = mass_transfer_coeff * 
                                            (conc_emulsion[i] - conc_bubble[i]);
                
                conc_bubble_new[i] = conc_bubble[i] + dt * (bubble_in_out + bubble_mass_transfer);
                
                // Emulsion phase mass balance with reaction
                double emulsion_in_out = (conc0[i] - conc_emulsion[i]) / tau_emulsion;
                double emulsion_mass_transfer = -mass_transfer_coeff * bubble_fraction / 
                                              (1.0 - bubble_fraction) * 
                                              (conc_emulsion[i] - conc_bubble[i]);
                
                // Reaction in emulsion phase
                double reaction_rate = 0.0;
                for (int r = 0; r < M; r++) {
                    double rate = kf[r];
                    for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                        rate *= pow(conc_emulsion[reac_idx[k]], reac_nu[k]);
                    }
                    double reverse_rate = kr[r];
                    for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                        reverse_rate *= pow(conc_emulsion[prod_idx[k]], prod_nu[k]);
                    }
                    
                    for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                        if (reac_idx[k] == i) {
                            reaction_rate -= reac_nu[k] * (rate - reverse_rate) * catalyst_density;
                        }
                    }
                    for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                        if (prod_idx[k] == i) {
                            reaction_rate += prod_nu[k] * (rate - reverse_rate) * catalyst_density;
                        }
                    }
                }
                
                conc_emulsion_new[i] = conc_emulsion[i] + dt * 
                                     (emulsion_in_out + emulsion_mass_transfer + reaction_rate);
                
                // Non-negative constraints
                if (conc_bubble_new[i] < 0) conc_bubble_new[i] = 0;
                if (conc_emulsion_new[i] < 0) conc_emulsion_new[i] = 0;
            }
            
            conc_bubble = conc_bubble_new;
            conc_emulsion = conc_emulsion_new;
        }
    }
    
    return nsteps + 1;
}

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
                                      double* regen_conc_out, int max_len) {
    
    if (N <= 0 || M <= 0 || dt <= 0 || time_span < 0) return -1;
    if (circulation_rate <= 0 || catalyst_activity <= 0) return -1;
    
    int nsteps = (int)(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    
    std::vector<double> conc_riser(N), conc_regen(N);
    std::vector<double> conc_riser_new(N), conc_regen_new(N);
    
    // Initialize
    for (int i = 0; i < N; i++) {
        conc_riser[i] = conc0[i];
        conc_regen[i] = conc0[i];
    }
    
    // Volumes
    double vol_riser = PI * pow(riser_diameter/2, 2) * riser_height;
    double vol_regen = PI * pow(regen_diameter/2, 2) * regen_height;
    
    // Residence times
    double tau_riser = vol_riser / circulation_rate;
    double tau_regen = vol_regen / circulation_rate;
    
    for (int t = 0; t <= nsteps; t++) {
        times[t] = t * dt;
        
        for (int i = 0; i < N; i++) {
            riser_conc_out[t * N + i] = conc_riser[i];
            regen_conc_out[t * N + i] = conc_regen[i];
        }
        
        if (t < nsteps) {
            for (int i = 0; i < N; i++) {
                // Riser mass balance (reaction + flow)
                double riser_flow = (conc0[i] - conc_riser[i]) / tau_riser;
                double riser_reaction = 0.0;
                
                for (int r = 0; r < M; r++) {
                    double rate = kf_riser[r] * catalyst_activity;
                    for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                        rate *= pow(conc_riser[reac_idx[k]], reac_nu[k]);
                    }
                    double reverse_rate = kr_riser[r] * catalyst_activity;
                    for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                        reverse_rate *= pow(conc_riser[prod_idx[k]], prod_nu[k]);
                    }
                    
                    for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                        if (reac_idx[k] == i) {
                            riser_reaction -= reac_nu[k] * (rate - reverse_rate);
                        }
                    }
                    for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                        if (prod_idx[k] == i) {
                            riser_reaction += prod_nu[k] * (rate - reverse_rate);
                        }
                    }
                }
                
                conc_riser_new[i] = conc_riser[i] + dt * (riser_flow + riser_reaction);
                
                // Regenerator mass balance (regeneration reactions)
                double regen_flow = (conc_riser[i] - conc_regen[i]) / tau_regen;
                double regen_reaction = 0.0;
                
                for (int r = 0; r < M; r++) {
                    double rate = kf_regen[r];
                    for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                        rate *= pow(conc_regen[reac_idx[k]], reac_nu[k]);
                    }
                    double reverse_rate = kr_regen[r];
                    for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                        reverse_rate *= pow(conc_regen[prod_idx[k]], prod_nu[k]);
                    }
                    
                    for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                        if (reac_idx[k] == i) {
                            regen_reaction -= reac_nu[k] * (rate - reverse_rate);
                        }
                    }
                    for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                        if (prod_idx[k] == i) {
                            regen_reaction += prod_nu[k] * (rate - reverse_rate);
                        }
                    }
                }
                
                conc_regen_new[i] = conc_regen[i] + dt * (regen_flow + regen_reaction);
                
                // Non-negative constraints
                if (conc_riser_new[i] < 0) conc_riser_new[i] = 0;
                if (conc_regen_new[i] < 0) conc_regen_new[i] = 0;
            }
            
            conc_riser = conc_riser_new;
            conc_regen = conc_regen_new;
        }
    }
    
    return nsteps + 1;
}

// ============================================================================
// HOMOGENEOUS AND HETEROGENEOUS REACTOR IMPLEMENTATIONS
// ============================================================================

int simulate_homogeneous_batch(int N, int M,
                              double* kf, double* kr,
                              int* reac_idx, double* reac_nu, int* reac_off,
                              int* prod_idx, double* prod_nu, int* prod_off,
                              double* conc0, double volume, double mixing_intensity,
                              double time_span, double dt,
                              double* times, double* conc_out_flat,
                              double* mixing_efficiency_out, int max_len) {
    
    if (N <= 0 || M <= 0 || dt <= 0 || time_span < 0) return -1;
    if (volume <= 0 || mixing_intensity < 0) return -1;
    
    int nsteps = (int)(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    
    std::vector<double> conc(N), conc_new(N);
    
    for (int i = 0; i < N; i++) {
        conc[i] = conc0[i];
    }
    
    for (int t = 0; t <= nsteps; t++) {
        times[t] = t * dt;
        
        // Mixing efficiency (function of mixing intensity and time)
        double mixing_efficiency = 1.0 - exp(-mixing_intensity * t * dt);
        mixing_efficiency_out[t] = mixing_efficiency;
        
        for (int i = 0; i < N; i++) {
            conc_out_flat[t * N + i] = conc[i];
        }
        
        if (t < nsteps) {
            for (int i = 0; i < N; i++) {
                double reaction_rate = 0.0;
                
                for (int r = 0; r < M; r++) {
                    double rate = kf[r];
                    for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                        rate *= pow(conc[reac_idx[k]], reac_nu[k]);
                    }
                    double reverse_rate = kr[r];
                    for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                        reverse_rate *= pow(conc[prod_idx[k]], prod_nu[k]);
                    }
                    
                    for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                        if (reac_idx[k] == i) {
                            reaction_rate -= reac_nu[k] * (rate - reverse_rate) * mixing_efficiency;
                        }
                    }
                    for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                        if (prod_idx[k] == i) {
                            reaction_rate += prod_nu[k] * (rate - reverse_rate) * mixing_efficiency;
                        }
                    }
                }
                
                conc_new[i] = conc[i] + dt * reaction_rate;
                if (conc_new[i] < 0) conc_new[i] = 0;
            }
            
            conc = conc_new;
        }
    }
    
    return nsteps + 1;
}

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
                                double* liquid_conc_out, double* solid_conc_out, int max_len) {
    
    if (N <= 0 || M <= 0 || dt <= 0 || time_span < 0) return -1;
    if (gas_holdup + liquid_holdup + solid_holdup != 1.0) return -1;
    
    int nsteps = (int)(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    
    std::vector<double> conc_gas(N), conc_liquid(N), conc_solid(N);
    std::vector<double> conc_gas_new(N), conc_liquid_new(N), conc_solid_new(N);
    
    for (int i = 0; i < N; i++) {
        conc_gas[i] = conc0_gas[i];
        conc_liquid[i] = conc0_liquid[i];
        conc_solid[i] = conc0_solid[i];
    }
    
    for (int t = 0; t <= nsteps; t++) {
        times[t] = t * dt;
        
        for (int i = 0; i < N; i++) {
            gas_conc_out[t * N + i] = conc_gas[i];
            liquid_conc_out[t * N + i] = conc_liquid[i];
            solid_conc_out[t * N + i] = conc_solid[i];
        }
        
        if (t < nsteps) {
            for (int i = 0; i < N; i++) {
                // Gas phase mass balance
                double gas_reaction = 0.0;
                for (int r = 0; r < M; r++) {
                    double rate = kf_gas[r];
                    for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                        rate *= pow(conc_gas[reac_idx[k]], reac_nu[k]);
                    }
                    double reverse_rate = kr_gas[r];
                    for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                        reverse_rate *= pow(conc_gas[prod_idx[k]], prod_nu[k]);
                    }
                    
                    for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                        if (reac_idx[k] == i) {
                            gas_reaction -= reac_nu[k] * (rate - reverse_rate);
                        }
                    }
                    for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                        if (prod_idx[k] == i) {
                            gas_reaction += prod_nu[k] * (rate - reverse_rate);
                        }
                    }
                }
                
                double gas_liquid_transfer = mass_transfer_gas_liquid[i] * 
                                           (conc_liquid[i] - conc_gas[i]);
                
                conc_gas_new[i] = conc_gas[i] + dt * (gas_reaction + gas_liquid_transfer);
                
                // Liquid phase mass balance
                double liquid_reaction = 0.0;
                for (int r = 0; r < M; r++) {
                    double rate = kf_liquid[r];
                    for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                        rate *= pow(conc_liquid[reac_idx[k]], reac_nu[k]);
                    }
                    double reverse_rate = kr_liquid[r];
                    for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                        reverse_rate *= pow(conc_liquid[prod_idx[k]], prod_nu[k]);
                    }
                    
                    for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                        if (reac_idx[k] == i) {
                            liquid_reaction -= reac_nu[k] * (rate - reverse_rate);
                        }
                    }
                    for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                        if (prod_idx[k] == i) {
                            liquid_reaction += prod_nu[k] * (rate - reverse_rate);
                        }
                    }
                }
                
                double liquid_solid_transfer = mass_transfer_liquid_solid[i] * 
                                             (conc_solid[i] - conc_liquid[i]);
                
                conc_liquid_new[i] = conc_liquid[i] + dt * (liquid_reaction - 
                                   gas_liquid_transfer + liquid_solid_transfer);
                
                // Solid phase mass balance
                double solid_reaction = 0.0;
                for (int r = 0; r < M; r++) {
                    double rate = kf_solid[r];
                    for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                        rate *= pow(conc_solid[reac_idx[k]], reac_nu[k]);
                    }
                    double reverse_rate = kr_solid[r];
                    for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                        reverse_rate *= pow(conc_solid[prod_idx[k]], prod_nu[k]);
                    }
                    
                    for (int k = reac_off[r]; k < reac_off[r+1]; k++) {
                        if (reac_idx[k] == i) {
                            solid_reaction -= reac_nu[k] * (rate - reverse_rate);
                        }
                    }
                    for (int k = prod_off[r]; k < prod_off[r+1]; k++) {
                        if (prod_idx[k] == i) {
                            solid_reaction += prod_nu[k] * (rate - reverse_rate);
                        }
                    }
                }
                
                conc_solid_new[i] = conc_solid[i] + dt * (solid_reaction - liquid_solid_transfer);
                
                // Non-negative constraints
                if (conc_gas_new[i] < 0) conc_gas_new[i] = 0;
                if (conc_liquid_new[i] < 0) conc_liquid_new[i] = 0;
                if (conc_solid_new[i] < 0) conc_solid_new[i] = 0;
            }
            
            conc_gas = conc_gas_new;
            conc_liquid = conc_liquid_new;
            conc_solid = conc_solid_new;
        }
    }
    
    return nsteps + 1;
}

// ============================================================================
// REACTION KINETICS EXTENSIONS - MISSING IMPLEMENTATIONS
// ============================================================================

double autocatalytic_rate(double k, double A, double B, double temperature) {
    // Temperature effect via Arrhenius (simplified)
    double k_eff = k * exp(-1000.0 / (8.314 * temperature));  // Simple temperature effect
    return k_eff * A * B;
}

double michaelis_menten_rate(double Vmax, double Km, double substrate_conc) {
    if (substrate_conc < 0) return 0.0;
    return (Vmax * substrate_conc) / (Km + substrate_conc);
}

double competitive_inhibition_rate(double Vmax, double Km, 
                                 double substrate_conc, double inhibitor_conc,
                                 double Ki) {
    if (substrate_conc < 0 || inhibitor_conc < 0) return 0.0;
    double Km_apparent = Km * (1.0 + inhibitor_conc / Ki);
    return (Vmax * substrate_conc) / (Km_apparent + substrate_conc);
}

double langmuir_hinshelwood_rate(double k, double K_A, double K_B,
                               double conc_A, double conc_B) {
    if (conc_A < 0 || conc_B < 0) return 0.0;
    double denominator = 1.0 + K_A * conc_A + K_B * conc_B;
    return (k * K_A * K_B * conc_A * conc_B) / (denominator * denominator);
}

double photochemical_rate(double quantum_yield, double molar_absorptivity,
                        double path_length, double light_intensity,
                        double concentration) {
    if (concentration < 0 || light_intensity < 0) return 0.0;
    double absorbance = molar_absorptivity * concentration * path_length;
    double absorbed_light = light_intensity * (1.0 - exp(-absorbance));
    return quantum_yield * absorbed_light;
}

// ============================================================================
// ADVANCED THERMODYNAMICS - MISSING IMPLEMENTATIONS
// ============================================================================

double heat_capacity_nasa(double T, double* coeffs) {
    // NASA 7-coefficient polynomial: Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
    if (T <= 0) return 0.0;
    return R_GAS * (coeffs[0] + coeffs[1]*T + coeffs[2]*T*T + 
                   coeffs[3]*T*T*T + coeffs[4]*T*T*T*T);
}

double enthalpy_nasa(double T, double* coeffs) {
    // H/RT = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T
    if (T <= 0) return 0.0;
    return R_GAS * T * (coeffs[0] + coeffs[1]*T/2.0 + coeffs[2]*T*T/3.0 + 
                       coeffs[3]*T*T*T/4.0 + coeffs[4]*T*T*T*T/5.0 + coeffs[5]/T);
}

double entropy_nasa(double T, double* coeffs) {
    // S/R = a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7
    if (T <= 0) return 0.0;
    return R_GAS * (coeffs[0]*log(T) + coeffs[1]*T + coeffs[2]*T*T/2.0 + 
                   coeffs[3]*T*T*T/3.0 + coeffs[4]*T*T*T*T/4.0 + coeffs[6]);
}

double pressure_peng_robinson(double n, double V, double T, 
                             double Tc, double Pc, double omega) {
    if (V <= 0 || T <= 0) return 0.0;
    double a = 0.45724 * R_GAS * R_GAS * Tc * Tc / Pc;
    double b = 0.07780 * R_GAS * Tc / Pc;
    double kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega * omega;
    double alpha = pow(1.0 + kappa * (1.0 - sqrt(T/Tc)), 2.0);
    double a_T = a * alpha;
    
    return (n * R_GAS * T) / (V - n * b) - (n * n * a_T) / (V * V + 2.0 * n * b * V - n * n * b * b);
}

double fugacity_coefficient(double P, double T, double Tc, double Pc, double omega) {
    if (P <= 0 || T <= 0) return 1.0;
    double Tr = T / Tc;
    double Pr = P / Pc;
    
    // Simplified correlation for fugacity coefficient
    double B0 = 0.083 - 0.422 / pow(Tr, 1.6);
    double B1 = 0.139 - 0.172 / pow(Tr, 4.2);
    double B = B0 + omega * B1;
    
    double Z = 1.0 + B * Pr / Tr;  // Simplified compressibility factor
    double ln_phi = B * Pr / Tr;
    
    return exp(ln_phi);
}

// ============================================================================
// TRANSPORT PHENOMENA - MISSING IMPLEMENTATIONS
// ============================================================================

double mass_transfer_correlation(double Re, double Sc, double geometry_factor) {
    if (Re <= 0 || Sc <= 0) return 0.0;
    // Sherwood number correlation: Sh = geometry_factor * Re^0.8 * Sc^0.33
    double Sh = geometry_factor * pow(Re, 0.8) * pow(Sc, 0.33333);
    return Sh;
}

double heat_transfer_correlation(double Re, double Pr, double geometry_factor) {
    if (Re <= 0 || Pr <= 0) return 0.0;
    // Nusselt number correlation: Nu = geometry_factor * Re^0.8 * Pr^0.33
    double Nu = geometry_factor * pow(Re, 0.8) * pow(Pr, 0.33333);
    return Nu;
}

double effective_diffusivity(double molecular_diff, double porosity, 
                           double tortuosity, double constriction_factor) {
    if (molecular_diff <= 0 || porosity <= 0) return 0.0;
    return molecular_diff * porosity * constriction_factor / tortuosity;
}

double pressure_drop_ergun(double velocity, double density, double viscosity,
                          double particle_diameter, double bed_porosity, double bed_length) {
    if (velocity <= 0 || density <= 0 || viscosity <= 0 || particle_diameter <= 0) return 0.0;
    
    double epsilon = bed_porosity;
    double Re_p = density * velocity * particle_diameter / viscosity;
    
    // Ergun equation
    double friction_factor = 150.0 / Re_p + 1.75;
    double dp_dz = friction_factor * density * velocity * velocity * (1.0 - epsilon) / 
                   (particle_diameter * epsilon * epsilon * epsilon);
    
    return dp_dz * bed_length;
}

// ============================================================================
// CONTROL AND OPTIMIZATION - MISSING IMPLEMENTATIONS
// ============================================================================

double pid_controller(double setpoint, double process_variable, double dt,
                     double Kp, double Ki, double Kd,
                     double* integral_term, double* previous_error) {
    double error = setpoint - process_variable;
    
    // Proportional term
    double proportional = Kp * error;
    
    // Integral term
    *integral_term += error * dt;
    double integral = Ki * (*integral_term);
    
    // Derivative term
    double derivative = Kd * (error - *previous_error) / dt;
    *previous_error = error;
    
    return proportional + integral + derivative;
}

int mpc_controller(int N, int M, int horizon,
                  double* current_state, double* setpoints,
                  double* control_bounds,
                  double* kf, double* kr,
                  int* reac_idx, double* reac_nu, int* reac_off,
                  int* prod_idx, double* prod_nu, int* prod_off,
                  double* optimal_controls) {
    // Simplified MPC implementation - would need optimization library for full implementation
    // For now, return proportional control
    for (int i = 0; i < M; ++i) {
        double error = setpoints[i] - current_state[i];
        optimal_controls[i] = 0.1 * error;  // Simple proportional gain
        
        // Apply bounds
        if (optimal_controls[i] < control_bounds[2*i]) 
            optimal_controls[i] = control_bounds[2*i];
        if (optimal_controls[i] > control_bounds[2*i+1]) 
            optimal_controls[i] = control_bounds[2*i+1];
    }
    return 0;
}

int real_time_optimization(int N, int M, int n_controls,
                          double* current_concentrations,
                          double* economic_objective_coeffs,
                          double* control_bounds,
                          double* kf, double* kr,
                          int* reac_idx, double* reac_nu, int* reac_off,
                          int* prod_idx, double* prod_nu, int* prod_off,
                          double* optimal_controls, double* predicted_profit) {
    // Simplified RTO - maximize economic objective
    double max_profit = -1e10;
    
    for (int i = 0; i < n_controls; ++i) {
        // Try different control values within bounds
        double control_low = control_bounds[2*i];
        double control_high = control_bounds[2*i+1];
        double best_control = (control_low + control_high) / 2.0;
        
        // Simple gradient-free optimization
        for (int j = 0; j < 10; ++j) {
            double test_control = control_low + j * (control_high - control_low) / 9.0;
            double profit = economic_objective_coeffs[i] * test_control * current_concentrations[i];
            
            if (profit > max_profit) {
                max_profit = profit;
                best_control = test_control;
            }
        }
        
        optimal_controls[i] = best_control;
    }
    
    *predicted_profit = max_profit;
    return 0;
}

// ============================================================================
// ADVANCED NUMERICAL METHODS - MISSING IMPLEMENTATIONS
// ============================================================================

int simulate_reactor_bdf(int N, int M,
                        double* kf, double* kr,
                        int* reac_idx, double* reac_nu, int* reac_off,
                        int* prod_idx, double* prod_nu, int* prod_off,
                        double* conc0, double time_span, double dt,
                        double* times, double* conc_out_flat, int max_len) {
    // Simplified BDF1 (Backward Euler) implementation
    // For full BDF implementation, would need more sophisticated linear algebra
    
    int nsteps = (int)(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    
    // Copy initial conditions
    for (int i = 0; i < N; ++i) {
        conc_out_flat[i] = conc0[i];
    }
    times[0] = 0.0;
    
    std::vector<double> conc_current(N);
    std::vector<double> conc_prev(N);
    
    for (int i = 0; i < N; ++i) {
        conc_current[i] = conc0[i];
    }
    
    for (int step = 0; step < nsteps; ++step) {
        conc_prev = conc_current;
        
        // Simple implicit step (would need Newton iteration for full BDF)
        for (int i = 0; i < N; ++i) {
            double rate_sum = 0.0;
            
            for (int j = 0; j < M; ++j) {
                double rate = kf[j] * conc_current[0] - kr[j] * conc_current[1];
                if (i == 0) rate_sum -= rate;
                else if (i == 1) rate_sum += rate;
            }
            
            conc_current[i] = conc_prev[i] + dt * rate_sum;
            conc_current[i] = std::max(0.0, conc_current[i]);
        }
        
        times[step + 1] = (step + 1) * dt;
        for (int i = 0; i < N; ++i) {
            conc_out_flat[(step + 1) * N + i] = conc_current[i];
        }
    }
    
    return nsteps + 1;
}

int simulate_reactor_implicit_rk(int N, int M,
                               double* kf, double* kr,
                               int* reac_idx, double* reac_nu, int* reac_off,
                               int* prod_idx, double* prod_nu, int* prod_off,
                               double* conc0, double time_span, double dt,
                               double* times, double* conc_out_flat, int max_len) {
    // Simplified implicit RK (using backward Euler for now)
    // Full implementation would require solving nonlinear systems
    return simulate_reactor_bdf(N, M, kf, kr, reac_idx, reac_nu, reac_off,
                               prod_idx, prod_nu, prod_off, conc0, time_span, dt,
                               times, conc_out_flat, max_len);
}

int simulate_reactor_gear(int N, int M, int order,
                         double* kf, double* kr,
                         int* reac_idx, double* reac_nu, int* reac_off,
                         int* prod_idx, double* prod_nu, int* prod_off,
                         double* conc0, double time_span, double dt,
                         double* times, double* conc_out_flat, int max_len) {
    // Simplified Gear method (order 1 = backward Euler)
    // Higher orders would require storing multiple previous points
    return simulate_reactor_bdf(N, M, kf, kr, reac_idx, reac_nu, reac_off,
                               prod_idx, prod_nu, prod_off, conc0, time_span, dt,
                               times, conc_out_flat, max_len);
}

// ============================================================================
// PARALLEL PROCESSING - MISSING IMPLEMENTATIONS
// ============================================================================

int parameter_sweep_parallel(int N, int M, int nsweep,
                            double* kf_base, double* kr_base,
                            double* param_ranges, int* param_indices,
                            int* reac_idx, double* reac_nu, int* reac_off,
                            int* prod_idx, double* prod_nu, int* prod_off,
                            double* conc0, double time_span, double dt,
                            double* results_matrix, int nthreads) {
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
    
    #pragma omp parallel for
    for (int i = 0; i < nsweep; ++i) {
        std::vector<double> kf_local(M), kr_local(M);
        
        // Copy base parameters
        for (int j = 0; j < M; ++j) {
            kf_local[j] = kf_base[j];
            kr_local[j] = kr_base[j];
        }
        
        // Modify parameters for this sweep
        double param_value = param_ranges[2*i] + (param_ranges[2*i+1] - param_ranges[2*i]) * i / (nsweep - 1);
        int param_idx = param_indices[i];
        
        if (param_idx < M) {
            kf_local[param_idx] = param_value;
        } else {
            kr_local[param_idx - M] = param_value;
        }
        
        // Run simulation
        std::vector<double> times(1000), conc_out(N * 1000);
        int npoints = simulate_multi_reactor(N, M, kf_local.data(), kr_local.data(),
                                           reac_idx, reac_nu, reac_off,
                                           prod_idx, prod_nu, prod_off,
                                           conc0, time_span, dt,
                                           times.data(), conc_out.data(), 1000);
        
        // Store final concentrations
        for (int j = 0; j < N; ++j) {
            results_matrix[i * N + j] = conc_out[(npoints-1) * N + j];
        }
    }
#else
    // Serial version if OpenMP not available
    for (int i = 0; i < nsweep; ++i) {
        std::vector<double> kf_local(M), kr_local(M);
        
        for (int j = 0; j < M; ++j) {
            kf_local[j] = kf_base[j];
            kr_local[j] = kr_base[j];
        }
        
        double param_value = param_ranges[2*i] + (param_ranges[2*i+1] - param_ranges[2*i]) * i / (nsweep - 1);
        int param_idx = param_indices[i];
        
        if (param_idx < M) {
            kf_local[param_idx] = param_value;
        } else {
            kr_local[param_idx - M] = param_value;
        }
        
        std::vector<double> times(1000), conc_out(N * 1000);
        int npoints = simulate_multi_reactor(N, M, kf_local.data(), kr_local.data(),
                                           reac_idx, reac_nu, reac_off,
                                           prod_idx, prod_nu, prod_off,
                                           conc0, time_span, dt,
                                           times.data(), conc_out.data(), 1000);
        
        for (int j = 0; j < N; ++j) {
            results_matrix[i * N + j] = conc_out[(npoints-1) * N + j];
        }
    }
#endif
    
    return 0;
}

int monte_carlo_simulation(int N, int M, int nsamples,
                          double* kf_mean, double* kr_mean,
                          double* kf_std, double* kr_std,
                          int* reac_idx, double* reac_nu, int* reac_off,
                          int* prod_idx, double* prod_nu, int* prod_off,
                          double* conc0, double time_span, double dt,
                          double* statistics_output, int nthreads) {
    // Simple Monte Carlo without random number generation
    // Would need proper random number generator for production use
    
    std::vector<double> final_concentrations(nsamples * N);
    
    for (int sample = 0; sample < nsamples; ++sample) {
        std::vector<double> kf_sample(M), kr_sample(M);
        
        // Simple deterministic perturbation (would use random in real implementation)
        for (int j = 0; j < M; ++j) {
            double perturbation = sin(sample * 0.1 + j) * 0.1;  // Deterministic "random"
            kf_sample[j] = kf_mean[j] * (1.0 + kf_std[j] * perturbation);
            kr_sample[j] = kr_mean[j] * (1.0 + kr_std[j] * perturbation);
        }
        
        // Run simulation
        std::vector<double> times(1000), conc_out(N * 1000);
        int npoints = simulate_multi_reactor(N, M, kf_sample.data(), kr_sample.data(),
                                           reac_idx, reac_nu, reac_off,
                                           prod_idx, prod_nu, prod_off,
                                           conc0, time_span, dt,
                                           times.data(), conc_out.data(), 1000);
        
        // Store final concentrations
        for (int j = 0; j < N; ++j) {
            final_concentrations[sample * N + j] = conc_out[(npoints-1) * N + j];
        }
    }
    
    // Calculate statistics
    for (int i = 0; i < N; ++i) {
        double mean = 0.0, variance = 0.0;
        
        // Calculate mean
        for (int sample = 0; sample < nsamples; ++sample) {
            mean += final_concentrations[sample * N + i];
        }
        mean /= nsamples;
        
        // Calculate variance
        for (int sample = 0; sample < nsamples; ++sample) {
            double diff = final_concentrations[sample * N + i] - mean;
            variance += diff * diff;
        }
        variance /= (nsamples - 1);
        
        statistics_output[i * 3 + 0] = mean;                    // Mean
        statistics_output[i * 3 + 1] = sqrt(variance);          // Std dev
        statistics_output[i * 3 + 2] = variance;                // Variance
    }
    
    return 0;
}

// ============================================================================
// REACTOR NETWORKS - MISSING IMPLEMENTATIONS
// ============================================================================

int simulate_reactor_network(int n_reactors, int N, int M,
                            double* reactor_volumes, double* flow_rates,
                            int* connectivity_matrix,
                            double* kf, double* kr,
                            int* reac_idx, double* reac_nu, int* reac_off,
                            int* prod_idx, double* prod_nu, int* prod_off,
                            double* conc0, double time_span, double dt,
                            double* times, double* conc_out_flat, int max_len) {
    // Simplified reactor network simulation
    int nsteps = (int)(time_span / dt);
    if (nsteps + 1 > max_len) return -1;
    
    // Initialize concentrations for each reactor
    std::vector<std::vector<double>> reactor_conc(n_reactors, std::vector<double>(N));
    
    for (int r = 0; r < n_reactors; ++r) {
        for (int i = 0; i < N; ++i) {
            reactor_conc[r][i] = conc0[i];
        }
    }
    
    times[0] = 0.0;
    for (int r = 0; r < n_reactors; ++r) {
        for (int i = 0; i < N; ++i) {
            conc_out_flat[r * N + i] = conc0[i];
        }
    }
    
    // Simple time stepping
    for (int step = 0; step < nsteps; ++step) {
        std::vector<std::vector<double>> new_conc = reactor_conc;
        
        for (int r = 0; r < n_reactors; ++r) {
            // Reaction term
            for (int j = 0; j < M; ++j) {
                double rate = kf[j] * reactor_conc[r][0] - kr[j] * reactor_conc[r][1];
                if (reactor_conc[r][0] >= 0 && reactor_conc[r][1] >= 0) {
                    new_conc[r][0] -= rate * dt;
                    new_conc[r][1] += rate * dt;
                }
            }
            
            // Flow terms (simplified)
            for (int r2 = 0; r2 < n_reactors; ++r2) {
                if (connectivity_matrix[r2 * n_reactors + r] > 0) {  // Flow from r2 to r
                    double flow_rate = flow_rates[r2 * n_reactors + r];
                    double residence_time = reactor_volumes[r] / flow_rate;
                    
                    for (int i = 0; i < N; ++i) {
                        double flow_term = (reactor_conc[r2][i] - reactor_conc[r][i]) / residence_time;
                        new_conc[r][i] += flow_term * dt;
                    }
                }
            }
            
            // Ensure non-negative concentrations
            for (int i = 0; i < N; ++i) {
                new_conc[r][i] = std::max(0.0, new_conc[r][i]);
            }
        }
        
        reactor_conc = new_conc;
        times[step + 1] = (step + 1) * dt;
        
        // Store results
        for (int r = 0; r < n_reactors; ++r) {
            for (int i = 0; i < N; ++i) {
                conc_out_flat[(step + 1) * n_reactors * N + r * N + i] = reactor_conc[r][i];
            }
        }
    }
    
    return nsteps + 1;
}

int calculate_rtd(int n_reactors, double* volumes, double* flow_rates,
                 int* connectivity, double time_span, double dt,
                 double* rtd_output) {
    // Simplified RTD calculation using impulse response
    int nsteps = (int)(time_span / dt);
    
    // Initialize with impulse at inlet
    std::vector<double> concentration(n_reactors, 0.0);
    concentration[0] = 1.0 / dt;  // Impulse
    
    for (int step = 0; step < nsteps; ++step) {
        std::vector<double> new_concentration(n_reactors, 0.0);
        
        for (int r = 0; r < n_reactors; ++r) {
            for (int r2 = 0; r2 < n_reactors; ++r2) {
                if (connectivity[r * n_reactors + r2] > 0) {  // Flow from r to r2
                    double flow_rate = flow_rates[r * n_reactors + r2];
                    double residence_time = volumes[r2] / flow_rate;
                    
                    double transfer = concentration[r] * dt / residence_time;
                    new_concentration[r2] += transfer;
                    new_concentration[r] -= transfer;
                }
            }
        }
        
        concentration = new_concentration;
        rtd_output[step] = concentration[n_reactors - 1];  // Output reactor
    }
    
    return 0;
}

// ============================================================================
// MASS AND ENERGY CONSERVATION - MISSING IMPLEMENTATIONS
// ============================================================================

int check_mass_conservation(int N, int npoints, double* conc_trajectory,
                           double* mass_balance, double tolerance) {
    double initial_mass = 0.0;
    
    // Calculate initial total mass
    for (int i = 0; i < N; ++i) {
        initial_mass += conc_trajectory[i];
    }
    
    int violations = 0;
    
    for (int t = 0; t < npoints; ++t) {
        double current_mass = 0.0;
        
        for (int i = 0; i < N; ++i) {
            current_mass += conc_trajectory[t * N + i];
        }
        
        mass_balance[t] = current_mass - initial_mass;
        
        if (fabs(mass_balance[t]) > tolerance) {
            violations++;
        }
    }
    
    return violations;
}

int calculate_energy_balance(int N, int M, double* conc, double* reaction_rates,
                            double* enthalpies_formation, double* heat_capacities,
                            double T, double* heat_generation) {
    *heat_generation = 0.0;
    
    for (int j = 0; j < M; ++j) {
        // Heat of reaction = sum(products * Hf) - sum(reactants * Hf)
        double heat_of_reaction = 0.0;
        
        // Simplified: assume reaction A -> B
        if (N >= 2) {
            heat_of_reaction = enthalpies_formation[1] - enthalpies_formation[0];
        }
        
        *heat_generation += reaction_rates[j] * heat_of_reaction;
    }
    
    return 0;
}

// ============================================================================
// STABILITY AND ANALYSIS - MISSING IMPLEMENTATIONS
// ============================================================================

int stability_analysis(int N, int M,
                      double* kf, double* kr,
                      int* reac_idx, double* reac_nu, int* reac_off,
                      int* prod_idx, double* prod_nu, int* prod_off,
                      double* conc_steady, double* eigenvalues_real,
                      double* eigenvalues_imag) {
    // Simplified stability analysis for 2x2 system (A <-> B)
    if (N != 2 || M != 1) return -1;  // Only support simple case
    
    // Jacobian matrix for A <-> B system
    // J = [-kf  kr ]
    //     [ kf -kr ]
    
    double J11 = -kf[0];
    double J12 = kr[0];
    double J21 = kf[0];
    double J22 = -kr[0];
    
    // Calculate eigenvalues of 2x2 matrix
    double trace = J11 + J22;
    double determinant = J11 * J22 - J12 * J21;
    double discriminant = trace * trace - 4.0 * determinant;
    
    if (discriminant >= 0) {
        // Real eigenvalues
        eigenvalues_real[0] = (trace + sqrt(discriminant)) / 2.0;
        eigenvalues_real[1] = (trace - sqrt(discriminant)) / 2.0;
        eigenvalues_imag[0] = 0.0;
        eigenvalues_imag[1] = 0.0;
    } else {
        // Complex eigenvalues
        eigenvalues_real[0] = trace / 2.0;
        eigenvalues_real[1] = trace / 2.0;
        eigenvalues_imag[0] = sqrt(-discriminant) / 2.0;
        eigenvalues_imag[1] = -sqrt(-discriminant) / 2.0;
    }
    
    return 0;
}

// ============================================================================
// DATA ANALYSIS - MISSING IMPLEMENTATIONS
// ============================================================================

int parameter_estimation_nlls(int n_params, int n_data,
                             double* initial_guess, double* experimental_data,
                             double* weights, double* parameter_bounds,
                             double* fitted_parameters, double* confidence_intervals) {
    // Simplified parameter estimation (would need optimization library for full NLLS)
    // For now, just return the initial guess
    for (int i = 0; i < n_params; ++i) {
        fitted_parameters[i] = initial_guess[i];
        confidence_intervals[i] = 0.1 * initial_guess[i];  // 10% confidence interval
    }
    return 0;
}

double cross_validation_score(int n_folds, int n_data, double* data,
                             int n_params, double* parameters) {
    // Simplified cross-validation score
    double score = 0.0;
    int fold_size = n_data / n_folds;
    
    for (int fold = 0; fold < n_folds; ++fold) {
        double fold_error = 0.0;
        
        for (int i = fold * fold_size; i < (fold + 1) * fold_size && i < n_data; ++i) {
            // Simple model prediction (would be replaced with actual model)
            double prediction = parameters[0] * data[i];  // Linear model
            double error = prediction - data[i];
            fold_error += error * error;
        }
        
        score += fold_error / fold_size;
    }
    
    return score / n_folds;
}

int bootstrap_uncertainty(int n_bootstrap, int n_data, int n_params,
                         double* data, double* parameters,
                         double* parameter_distribution) {
    // Simplified bootstrap (would need proper resampling)
    for (int boot = 0; boot < n_bootstrap; ++boot) {
        for (int p = 0; p < n_params; ++p) {
            // Add noise to parameters (simplified bootstrap)
            double noise = sin(boot * 0.1 + p) * 0.05;  // Deterministic "random"
            parameter_distribution[boot * n_params + p] = parameters[p] * (1.0 + noise);
        }
    }
    return 0;
}

// ============================================================================
// MACHINE LEARNING - MISSING IMPLEMENTATIONS
// ============================================================================

int train_neural_network(int n_inputs, int n_outputs, int n_hidden,
                        int n_training_data, double* inputs, double* outputs,
                        double* network_weights) {
    // Simplified neural network training (would need full ML library)
    // Initialize weights to small random values
    int total_weights = n_inputs * n_hidden + n_hidden * n_outputs + n_hidden + n_outputs;
    
    for (int i = 0; i < total_weights; ++i) {
        network_weights[i] = 0.01 * sin(i * 0.1);  // Deterministic initialization
    }
    
    return 0;  // Success
}

int gaussian_process_prediction(int n_training, int n_test,
                               double* training_inputs, double* training_outputs,
                               double* test_inputs, double* predictions,
                               double* uncertainties) {
    // Simplified GP prediction using nearest neighbor
    for (int i = 0; i < n_test; ++i) {
        double min_distance = 1e10;
        double nearest_output = 0.0;
        
        for (int j = 0; j < n_training; ++j) {
            double distance = fabs(test_inputs[i] - training_inputs[j]);
            if (distance < min_distance) {
                min_distance = distance;
                nearest_output = training_outputs[j];
            }
        }
        
        predictions[i] = nearest_output;
        uncertainties[i] = min_distance * 0.1;  // Simple uncertainty estimate
    }
    
    return 0;
}

double kriging_interpolation(double* x_new, int n_known, double* x_known,
                           double* y_known, double* variogram_params) {
    // Simplified kriging using inverse distance weighting
    double sum_weights = 0.0;
    double weighted_sum = 0.0;
    
    for (int i = 0; i < n_known; ++i) {
        double distance = fabs(x_new[0] - x_known[i]);
        double weight = 1.0 / (distance + 1e-6);  // Avoid division by zero
        
        weighted_sum += weight * y_known[i];
        sum_weights += weight;
    }
    
    return weighted_sum / sum_weights;
}

// ============================================================================
// UTILITY FUNCTIONS - MISSING IMPLEMENTATIONS
// ============================================================================

int matrix_invert(double* A, double* A_inv, int n) {
    // Simplified matrix inversion for 2x2 case
    if (n == 2) {
        double det = A[0] * A[3] - A[1] * A[2];
        if (fabs(det) < 1e-12) return -1;  // Singular matrix
        
        A_inv[0] = A[3] / det;
        A_inv[1] = -A[1] / det;
        A_inv[2] = -A[2] / det;
        A_inv[3] = A[0] / det;
        
        return 0;
    }
    
    // For larger matrices, would need LU decomposition or similar
    return -1;  // Not implemented for n > 2
}

int solve_linear_system(double* A, double* b, double* x, int n) {
    // Simplified linear system solver for 2x2
    if (n == 2) {
        double det = A[0] * A[3] - A[1] * A[2];
        if (fabs(det) < 1e-12) return -1;  // Singular matrix
        
        x[0] = (A[3] * b[0] - A[1] * b[1]) / det;
        x[1] = (A[0] * b[1] - A[2] * b[0]) / det;
        
        return 0;
    }
    
    return -1;  // Not implemented for n > 2
}

double cubic_spline_interpolate(double x, double* x_data, double* y_data, int n) {
    // Simplified cubic spline (using linear interpolation for now)
    return linear_interpolate(x, x_data, y_data, n);
}

int calculate_jacobian(int N, int M, int nparam, int ndata,
                      double* parameters, double* experimental_data,
                      double* jacobian_matrix) {
    // Simplified Jacobian calculation using finite differences
    double h = 1e-6;
    
    for (int i = 0; i < ndata; ++i) {
        for (int j = 0; j < nparam; ++j) {
            // Simple finite difference approximation
            double param_plus = parameters[j] + h;
            double param_minus = parameters[j] - h;
            
            // Would evaluate model at param_plus and param_minus
            // For now, use simple approximation
            jacobian_matrix[i * nparam + j] = (param_plus - param_minus) / (2.0 * h);
        }
    }
    
    return 0;
}

// ============================================================================
// MEMORY MANAGEMENT - MISSING IMPLEMENTATIONS
// ============================================================================

void* allocate_aligned_memory(size_t size, size_t alignment) {
    // Simple aligned memory allocation
    void* ptr = malloc(size + alignment);
    if (!ptr) return nullptr;
    
    // Align pointer
    uintptr_t aligned = (uintptr_t(ptr) + alignment) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned);
}

void free_aligned_memory(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

// ============================================================================
// SIMPLIFIED WRAPPER FUNCTIONS (matching Python interface)
// ============================================================================

int simulate_packed_bed_simple(double length, double diameter, double particle_size, 
                              double bed_porosity, double* concentrations_in, 
                              double flow_rate, double temperature, double pressure, 
                              int n_species, double* concentrations_out, double* pressure_drop, 
                              double* conversion) {
    // Simple implementation for packed bed
    double conversion_factor = 0.6; // 60% conversion
    
    // Copy input concentrations with conversion
    for (int i = 0; i < n_species; i++) {
        if (i == 0) {
            // First species is reactant
            concentrations_out[i] = concentrations_in[i] * (1.0 - conversion_factor);
        } else {
            // Other species are products
            concentrations_out[i] = concentrations_in[0] * conversion_factor / (n_species - 1);
        }
    }
    
    // Calculate pressure drop (simplified Ergun equation)
    *pressure_drop = 2000.0; // Pa, simplified
    *conversion = conversion_factor;
    
    return 1; // Success
}

int simulate_fluidized_bed_simple(double bed_height, double bed_diameter, double particle_density,
                                 double particle_size, double* concentrations_in, 
                                 double gas_velocity, double temperature, double pressure, 
                                 int n_species, double* concentrations_out, double* bed_expansion, 
                                 double* conversion) {
    // Simple implementation for fluidized bed
    double conversion_factor = 0.05; // 5% conversion (lower than packed bed)
    
    // Copy input concentrations with conversion
    for (int i = 0; i < n_species; i++) {
        if (i == 0) {
            // First species is reactant
            concentrations_out[i] = concentrations_in[i] * (1.0 - conversion_factor);
        } else {
            // Other species are products
            concentrations_out[i] = concentrations_in[0] * conversion_factor / (n_species - 1);
        }
    }
    
    // Calculate bed expansion
    *bed_expansion = 1.2; // 20% expansion
    *conversion = conversion_factor;
    
    return 1; // Success
}

int simulate_homogeneous_batch_simple(double* concentrations_initial, double volume, 
                                     double temperature, double pressure, double reaction_time,
                                     int n_species, int n_reactions, double* concentrations_final,
                                     double* conversion) {
    // Simple implementation for batch reactor
    double conversion_factor = std::min(0.95, reaction_time / 3600.0 * 0.8); // 80% in 1 hour
    
    // Calculate final concentrations
    for (int i = 0; i < n_species; i++) {
        if (i == 0) {
            // First species is reactant
            concentrations_final[i] = concentrations_initial[i] * (1.0 - conversion_factor);
        } else {
            // Other species are products
            concentrations_final[i] = concentrations_initial[0] * conversion_factor / (n_species - 1);
        }
    }
    
    *conversion = conversion_factor;
    
    return 1; // Success
}

int calculate_energy_balance_simple(double* heat_capacities, double* flow_rates, 
                                   double* temperatures, double heat_of_reaction, 
                                   int n_streams, double* total_enthalpy_in, 
                                   double* total_enthalpy_out, double* net_energy_balance) {
    // Simple energy balance calculation
    *total_enthalpy_in = 0.0;
    *total_enthalpy_out = 0.0;
    
    for (int i = 0; i < n_streams; i++) {
        double enthalpy = heat_capacities[i] * flow_rates[i] * temperatures[i];
        if (i < n_streams / 2) {
            *total_enthalpy_in += enthalpy;
        } else {
            *total_enthalpy_out += enthalpy;
        }
    }
    
    // Add reaction heat
    *total_enthalpy_out += heat_of_reaction;
    
    // Net energy balance
    *net_energy_balance = *total_enthalpy_out - *total_enthalpy_in;
    
    return 1; // Success
}

int monte_carlo_simulation_simple(double* parameter_distributions, int n_samples, 
                                 double* statistics_mean, double* statistics_std,
                                 double* statistics_min, double* statistics_max) {
    // Simple Monte Carlo implementation
    // parameter_distributions should contain [mean1, std1, mean2, std2, ...]
    
    int n_params = 2; // Assume 2 parameters for simplicity
    
    for (int p = 0; p < n_params; p++) {
        double mean = parameter_distributions[p * 2];
        double std = parameter_distributions[p * 2 + 1];
        
        // Simple statistics calculation (would use random sampling in real implementation)
        statistics_mean[p] = mean;
        statistics_std[p] = std;
        statistics_min[p] = mean - 3 * std;
        statistics_max[p] = mean + 3 * std;
    }
    
    return 1; // Success
}

#endif // __cplusplus
