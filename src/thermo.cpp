// Minimal C++ placeholder for Thermodynamics
#include <cmath>

extern "C" {
    double enthalpy(double cp, double T) {
        return cp * T;
    }
    double entropy(double cp, double T) {
        if (T <= 0) return NAN;
        return cp * log(T);
    }
}
