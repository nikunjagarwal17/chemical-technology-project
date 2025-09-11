// Enhanced C++ implementation for advanced thermodynamic calculations
#include "core.h"
#include <cmath>
#include<string>
#include <vector>
#include <algorithm>

extern "C" {
    // Basic thermodynamic functions
    double enthalpy(double cp, double T) {
        if (T <= 0) return NAN;
        return cp * T;
    }
    
    double entropy(double cp, double T) {
        if (T <= 0) return NAN;
        return cp * log(T);
    }
    
    // Advanced thermodynamic property calculations
    
    // NASA polynomial for heat capacity (7-coefficient form)
    double heat_capacity_nasa(double T, double* coeffs) {
        if (T <= 0) return NAN;
        
        // cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
        const double R = 8.31446261815324; // J/mol/K
        return R * (coeffs[0] + coeffs[1]*T + coeffs[2]*T*T + 
                   coeffs[3]*T*T*T + coeffs[4]*T*T*T*T);
    }
    
    // Enthalpy from NASA polynomials
    double enthalpy_nasa(double T, double* coeffs) {
        if (T <= 0) return NAN;
        
        const double R = 8.31446261815324;
        // H/RT = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T
        return R * T * (coeffs[0] + coeffs[1]*T/2.0 + coeffs[2]*T*T/3.0 + 
                       coeffs[3]*T*T*T/4.0 + coeffs[4]*T*T*T*T/5.0 + coeffs[5]/T);
    }
    
    // Entropy from NASA polynomials
    double entropy_nasa(double T, double* coeffs) {
        if (T <= 0) return NAN;
        
        const double R = 8.31446261815324;
        // S/R = a1*ln(T) + a2*T + a3*T^2/2 + a4*T^3/3 + a5*T^4/4 + a7
        return R * (coeffs[0]*log(T) + coeffs[1]*T + coeffs[2]*T*T/2.0 + 
                   coeffs[3]*T*T*T/3.0 + coeffs[4]*T*T*T*T/4.0 + coeffs[6]);
    }
    
    // Gibbs free energy
    double gibbs_free_energy_nasa(double T, double* coeffs) {
        return enthalpy_nasa(T, coeffs) - T * entropy_nasa(T, coeffs);
    }
    
    // Ideal gas law with compressibility factor
    double pressure_real_gas(double n, double V, double T, double Z) {
        if (V <= 0 || T <= 0 || n <= 0) return NAN;
        const double R = 8.31446261815324;
        return Z * n * R * T / V;
    }
    
    // Van der Waals equation of state
    double pressure_van_der_waals(double n, double V, double T, double a, double b) {
        if (V <= 0 || T <= 0 || n <= 0) return NAN;
        if (V <= n * b) return INFINITY; // Invalid state
        
        const double R = 8.31446261815324;
        return (n * R * T) / (V - n * b) - (a * n * n) / (V * V);
    }
    
    // Peng-Robinson equation of state
    double pressure_peng_robinson(double n, double V, double T, 
                                 double Tc, double Pc, double omega) {
        if (V <= 0 || T <= 0 || n <= 0 || Tc <= 0 || Pc <= 0) return NAN;
        
        const double R = 8.31446261815324;
        double Tr = T / Tc;
        
        // Calculate a and b parameters
        double alpha = pow(1.0 + (0.37464 + 1.54226*omega - 0.26992*omega*omega) * 
                          (1.0 - sqrt(Tr)), 2.0);
        double a = 0.45724 * R*R * Tc*Tc * alpha / Pc;
        double b = 0.07780 * R * Tc / Pc;
        
        return (n * R * T) / (V - n * b) - 
               (a * n * n) / (V * (V + n * b) + n * b * (V - n * b));
    }
    
    // Fugacity coefficient calculation (simplified)
    double fugacity_coefficient(double P, double T, double Tc, double Pc, double omega) {
        if (P <= 0 || T <= 0 || Tc <= 0 || Pc <= 0) return NAN;
        
        double Pr = P / Pc;
        double Tr = T / Tc;
        
        // Simplified correlation for fugacity coefficient
        double Z = 1.0 + Pr * (-0.083 + 0.422 / pow(Tr, 1.6));
        return exp(Z - 1.0 - log(Z));
    }
    
    // Activity coefficient using Wilson model
    double activity_coefficient_wilson(double x1, double x2, double lambda12, double lambda21) {
        if (x1 < 0 || x2 < 0 || x1 + x2 > 1.001) return NAN;
        
        double Lambda12 = exp(-lambda12);
        double Lambda21 = exp(-lambda21);
        
        return exp(-log(x1 + x2 * Lambda12) + x2 * 
                  (Lambda12 / (x1 + x2 * Lambda12) - Lambda21 / (x2 + x1 * Lambda21)));
    }
    
    // Chemical potential
    double chemical_potential(double activity, double T, double mu_standard) {
        if (activity <= 0 || T <= 0) return NAN;
        const double R = 8.31446261815324;
        return mu_standard + R * T * log(activity);
    }
    
    // Reaction equilibrium constant from species data
    double equilibrium_constant_reaction(int n_species, double* nu, double* mu_standard, double T) {
        if (T <= 0) return NAN;
        
        const double R = 8.31446261815324;
        double delta_G = 0.0;
        
        for (int i = 0; i < n_species; ++i) {
            delta_G += nu[i] * mu_standard[i];
        }
        
        return exp(-delta_G / (R * T));
    }
    
    // Phase equilibrium (Raoult's law)
    double vapor_pressure_raoult(double x_liquid, double P_sat) {
        if (x_liquid < 0 || x_liquid > 1 || P_sat < 0) return NAN;
        return x_liquid * P_sat;
    }
    
    // Antoine equation for vapor pressure
    double vapor_pressure_antoine(double T, double A, double B, double C) {
        if (T <= 0) return NAN;
        return exp(A - B / (T + C));
    }
    
    // Heat of vaporization using Clausius-Clapeyron
    double heat_vaporization_clausius_clapeyron(double T1, double P1, double T2, double P2) {
        if (T1 <= 0 || T2 <= 0 || P1 <= 0 || P2 <= 0) return NAN;
        
        const double R = 8.31446261815324;
        return -R * log(P2/P1) / (1.0/T2 - 1.0/T1);
    }
    
    // Mixing rules for properties
    double mixing_rule_linear(int n_components, double* x, double* properties) {
        double result = 0.0;
        for (int i = 0; i < n_components; ++i) {
            result += x[i] * properties[i];
        }
        return result;
    }
    
    double mixing_rule_quadratic(int n_components, double* x, double* properties, double* kij) {
        double result = 0.0;
        for (int i = 0; i < n_components; ++i) {
            for (int j = 0; j < n_components; ++j) {
                double interaction = 1.0;
                if (kij) interaction = (1.0 - kij[i*n_components + j]);
                result += x[i] * x[j] * sqrt(properties[i] * properties[j]) * interaction;
            }
        }
        return result;
    }
    
    // Thermal conductivity correlation
    double thermal_conductivity_gas(double T, double M, double cp, double mu) {
        if (T <= 0 || M <= 0 || cp <= 0 || mu <= 0) return NAN;
        
        // Eucken correlation
        const double R = 8.31446261815324;
        double cv = cp - R/M;  // Convert to mass basis
        return mu * (cv + 1.25 * R/M);
    }
    
    // Viscosity correlation (Sutherland's law)
    double viscosity_sutherland(double T, double T_ref, double mu_ref, double S) {
        if (T <= 0 || T_ref <= 0 || mu_ref <= 0) return NAN;
        return mu_ref * pow(T/T_ref, 1.5) * (T_ref + S) / (T + S);
    }
    
    // Diffusivity in gases (Fuller correlation)
    double diffusivity_fuller(double T, double P, double M_A, double M_B, 
                             double V_A, double V_B) {
        if (T <= 0 || P <= 0 || M_A <= 0 || M_B <= 0) return NAN;
        
        double M_AB = 2.0 / (1.0/M_A + 1.0/M_B);
        double sum_V = pow(V_A, 1.0/3.0) + pow(V_B, 1.0/3.0);
        
        return 1.013e-2 * pow(T, 1.75) * sqrt(M_AB) / (P * sum_V * sum_V);
    }
}

#ifdef __cplusplus

namespace pyroxa {

// Advanced thermodynamic property class
class ThermodynamicProperties {
private:
    std::vector<double> nasa_coeffs_low_;
    std::vector<double> nasa_coeffs_high_;
    double transition_temp_;
    double molecular_weight_;
    std::string species_name_;

public:
    ThermodynamicProperties(const std::string& name, double mw,
                          const std::vector<double>& coeffs_low,
                          const std::vector<double>& coeffs_high,
                          double T_transition = 1000.0)
        : nasa_coeffs_low_(coeffs_low), nasa_coeffs_high_(coeffs_high),
          transition_temp_(T_transition), molecular_weight_(mw),
          species_name_(name) {}
    
    double heat_capacity(double T) const {
        const auto& coeffs = (T < transition_temp_) ? nasa_coeffs_low_ : nasa_coeffs_high_;
        return heat_capacity_nasa(T, const_cast<double*>(coeffs.data()));
    }
    
    double enthalpy(double T) const {
        const auto& coeffs = (T < transition_temp_) ? nasa_coeffs_low_ : nasa_coeffs_high_;
        return enthalpy_nasa(T, const_cast<double*>(coeffs.data()));
    }
    
    double entropy(double T) const {
        const auto& coeffs = (T < transition_temp_) ? nasa_coeffs_low_ : nasa_coeffs_high_;
        return entropy_nasa(T, const_cast<double*>(coeffs.data()));
    }
    
    double gibbs_free_energy(double T) const {
        return enthalpy(T) - T * entropy(T);
    }
    
    double molecular_weight() const { return molecular_weight_; }
    const std::string& name() const { return species_name_; }
};

// Mixture thermodynamics
class MixtureThermodynamics {
private:
    std::vector<ThermodynamicProperties> species_;
    std::vector<double> mole_fractions_;

public:
    MixtureThermodynamics() = default;
    
    void add_species(const ThermodynamicProperties& species, double mole_fraction) {
        species_.push_back(species);
        mole_fractions_.push_back(mole_fraction);
        normalize_mole_fractions();
    }
    
    double mixture_heat_capacity(double T) const {
        double cp_mix = 0.0;
        for (size_t i = 0; i < species_.size(); ++i) {
            cp_mix += mole_fractions_[i] * species_[i].heat_capacity(T);
        }
        return cp_mix;
    }
    
    double mixture_enthalpy(double T) const {
        double h_mix = 0.0;
        for (size_t i = 0; i < species_.size(); ++i) {
            h_mix += mole_fractions_[i] * species_[i].enthalpy(T);
        }
        return h_mix;
    }
    
    double mixture_entropy(double T) const {
        double s_mix = 0.0;
        const double R = 8.31446261815324;
        
        for (size_t i = 0; i < species_.size(); ++i) {
            s_mix += mole_fractions_[i] * species_[i].entropy(T);
            // Add mixing entropy
            if (mole_fractions_[i] > 0) {
                s_mix -= R * mole_fractions_[i] * std::log(mole_fractions_[i]);
            }
        }
        return s_mix;
    }
    
    double mixture_molecular_weight() const {
        double mw_mix = 0.0;
        for (size_t i = 0; i < species_.size(); ++i) {
            mw_mix += mole_fractions_[i] * species_[i].molecular_weight();
        }
        return mw_mix;
    }

private:
    void normalize_mole_fractions() {
        double sum = 0.0;
        for (double x : mole_fractions_) {
            sum += x;
        }
        if (sum > 0) {
            for (double& x : mole_fractions_) {
                x /= sum;
            }
        }
    }
};

// Phase equilibrium calculations
class PhaseEquilibrium {
public:
    static std::vector<double> bubble_point_pressure(
        const std::vector<double>& mole_fractions,
        const std::vector<double>& vapor_pressures) {
        
        std::vector<double> vapor_fractions(mole_fractions.size());
        double total_pressure = 0.0;
        
        for (size_t i = 0; i < mole_fractions.size(); ++i) {
            double partial_pressure = mole_fractions[i] * vapor_pressures[i];
            total_pressure += partial_pressure;
        }
        
        for (size_t i = 0; i < mole_fractions.size(); ++i) {
            vapor_fractions[i] = (mole_fractions[i] * vapor_pressures[i]) / total_pressure;
        }
        
        std::vector<double> result = vapor_fractions;
        result.push_back(total_pressure);
        return result;
    }
    
    static std::vector<double> dew_point_pressure(
        const std::vector<double>& vapor_fractions,
        const std::vector<double>& vapor_pressures) {
        
        std::vector<double> liquid_fractions(vapor_fractions.size());
        double total_pressure_inv = 0.0;
        
        for (size_t i = 0; i < vapor_fractions.size(); ++i) {
            if (vapor_pressures[i] > 0) {
                total_pressure_inv += vapor_fractions[i] / vapor_pressures[i];
            }
        }
        
        double total_pressure = 1.0 / total_pressure_inv;
        
        for (size_t i = 0; i < vapor_fractions.size(); ++i) {
            if (vapor_pressures[i] > 0) {
                liquid_fractions[i] = vapor_fractions[i] * total_pressure / vapor_pressures[i];
            } else {
                liquid_fractions[i] = 0.0;
            }
        }
        
        std::vector<double> result = liquid_fractions;
        result.push_back(total_pressure);
        return result;
    }
};

} // namespace pyroxa

#endif // __cplusplus
