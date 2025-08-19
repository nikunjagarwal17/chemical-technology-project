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
