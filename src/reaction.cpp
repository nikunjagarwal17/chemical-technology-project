// Minimal C++ placeholder for Reaction and Reactor (C interface)
#include <vector>

extern "C" {
    double reaction_rate(double kf, double kr, double A, double B) {
        return kf * A - kr * B;
    }
}
