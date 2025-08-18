#include "core.h"
#include <cmath>

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

double enthalpy_c(double cp, double T) {
    return cp * T;
}

double entropy_c(double cp, double T) {
    if (T <= 0) return NAN;
    return cp * std::log(T);
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
