// Enhanced C++ implementation for advanced reaction modeling
#include "core.h"
#include <vector>
#include <cmath>
#include <algorithm>

extern "C" {
    // Basic reaction rate calculation
    double reaction_rate(double kf, double kr, double A, double B) {
        return kf * A - kr * B;
    }
    
    // Enhanced reaction rate with temperature dependence
    double reaction_rate_temperature(double kf_ref, double kr_ref, 
                                   double Ea_f, double Ea_r,
                                   double T, double T_ref,
                                   double A, double B) {
        if (T <= 0 || T_ref <= 0) return 0.0;
        
        const double R = 8.31446261815324; // J/mol/K
        double kf = kf_ref * std::exp(Ea_f / R * (1.0/T_ref - 1.0/T));
        double kr = kr_ref * std::exp(Ea_r / R * (1.0/T_ref - 1.0/T));
        
        return kf * A - kr * B;
    }
    
    // Multi-component reaction rate with arbitrary stoichiometry
    double complex_reaction_rate(int n_reactants, int n_products,
                               int* reactant_indices, double* reactant_stoich,
                               int* product_indices, double* product_stoich,
                               double* concentrations, double kf, double kr) {
        
        // Calculate forward rate term
        double forward_rate = kf;
        for (int i = 0; i < n_reactants; ++i) {
            int idx = reactant_indices[i];
            double stoich = reactant_stoich[i];
            double conc = concentrations[idx];
            if (conc <= 0) {
                forward_rate = 0.0;
                break;
            }
            forward_rate *= std::pow(conc, stoich);
        }
        
        // Calculate reverse rate term
        double reverse_rate = kr;
        for (int i = 0; i < n_products; ++i) {
            int idx = product_indices[i];
            double stoich = product_stoich[i];
            double conc = concentrations[idx];
            if (conc <= 0) {
                reverse_rate = 0.0;
                break;
            }
            reverse_rate *= std::pow(conc, stoich);
        }
        
        return forward_rate - reverse_rate;
    }
    
    // Enzyme kinetics (Michaelis-Menten)
    double michaelis_menten_rate(double Vmax, double Km, double substrate_conc) {
        if (Km <= 0 || Vmax <= 0) return 0.0;
        return (Vmax * substrate_conc) / (Km + substrate_conc);
    }
    
    // Competitive inhibition
    double competitive_inhibition_rate(double Vmax, double Km, 
                                     double substrate_conc, double inhibitor_conc,
                                     double Ki) {
        if (Km <= 0 || Vmax <= 0 || Ki <= 0) return 0.0;
        double apparent_Km = Km * (1.0 + inhibitor_conc / Ki);
        return (Vmax * substrate_conc) / (apparent_Km + substrate_conc);
    }
    
    // Non-competitive inhibition
    double noncompetitive_inhibition_rate(double Vmax, double Km,
                                        double substrate_conc, double inhibitor_conc,
                                        double Ki) {
        if (Km <= 0 || Vmax <= 0 || Ki <= 0) return 0.0;
        double apparent_Vmax = Vmax / (1.0 + inhibitor_conc / Ki);
        return (apparent_Vmax * substrate_conc) / (Km + substrate_conc);
    }
    
    // Autocatalytic reaction rate A + B -> 2B
    double autocatalytic_rate(double k, double A, double B) {
        return k * A * B;
    }
    
    // Hill equation for cooperative binding
    double hill_equation_rate(double Vmax, double K, double n, double substrate_conc) {
        if (K <= 0 || Vmax <= 0 || n <= 0) return 0.0;
        double substrate_n = std::pow(substrate_conc, n);
        double K_n = std::pow(K, n);
        return (Vmax * substrate_n) / (K_n + substrate_n);
    }
    
    // Langmuir-Hinshelwood surface reaction rate
    double langmuir_hinshelwood_rate(double k, double K_A, double K_B,
                                   double conc_A, double conc_B) {
        if (k <= 0 || K_A <= 0 || K_B <= 0) return 0.0;
        double numerator = k * K_A * K_B * conc_A * conc_B;
        double denominator = (1.0 + K_A * conc_A + K_B * conc_B);
        return numerator / (denominator * denominator);
    }
    
    // Mass transfer limited reaction rate
    double mass_transfer_limited_rate(double k_intrinsic, double k_mass_transfer,
                                    double bulk_conc, double surface_conc) {
        // Overall rate is limited by the slower of intrinsic reaction or mass transfer
        double intrinsic_rate = k_intrinsic * surface_conc;
        double mass_transfer_rate = k_mass_transfer * (bulk_conc - surface_conc);
        return std::min(intrinsic_rate, mass_transfer_rate);
    }
    
    // Photochemical reaction rate with light intensity dependence
    double photochemical_rate(double quantum_yield, double molar_absorptivity,
                            double path_length, double light_intensity,
                            double concentration) {
        if (quantum_yield <= 0 || molar_absorptivity <= 0) return 0.0;
        
        // Beer-Lambert law for light absorption
        double absorbed_light = light_intensity * 
            (1.0 - std::exp(-molar_absorptivity * concentration * path_length));
        
        // Rate proportional to absorbed photons
        return quantum_yield * absorbed_light;
    }
    
    // Calculate equilibrium constant from thermodynamics
    double equilibrium_constant_from_thermo(double delta_H, double delta_S,
                                          double T, double R) {
        if (T <= 0 || R <= 0) return 0.0;
        double delta_G = delta_H - T * delta_S;
        return std::exp(-delta_G / (R * T));
    }
    
    // Pressure-dependent reaction rate for gas-phase reactions
    double pressure_dependent_rate(double k0, double k_inf, double pressure,
                                 double F_cent, double temperature) {
        if (pressure <= 0 || temperature <= 0) return 0.0;
        
        // Troe falloff formulation
        double Pr = k0 * pressure / k_inf;
        double log_F_cent = std::log10(F_cent);
        double c = -0.4 - 0.67 * log_F_cent;
        double n = 0.75 - 1.27 * log_F_cent;
        double d = 0.14;
        
        double log_Pr = std::log10(Pr);
        double f = 1.0 / (1.0 + ((log_Pr + c) / (n - d * (log_Pr + c))) * 
                          ((log_Pr + c) / (n - d * (log_Pr + c))));
        
        double F = std::pow(F_cent, f);
        
        return k_inf * (Pr / (1.0 + Pr)) * F;
    }
}

#ifdef __cplusplus

namespace pyroxa {

// Enhanced C++ reaction class with modern features
class AdvancedReaction {
private:
    std::vector<int> reactant_species_;
    std::vector<double> reactant_stoichiometry_;
    std::vector<int> product_species_;
    std::vector<double> product_stoichiometry_;
    
    double k_forward_ref_;
    double k_reverse_ref_;
    double activation_energy_forward_;
    double activation_energy_reverse_;
    double reference_temperature_;
    
    enum ReactionType {
        ELEMENTARY,
        MICHAELIS_MENTEN,
        AUTOCATALYTIC,
        PHOTOCHEMICAL,
        SURFACE_REACTION
    } reaction_type_;

public:
    AdvancedReaction(const std::vector<int>& reactants,
                    const std::vector<double>& reactant_stoich,
                    const std::vector<int>& products,
                    const std::vector<double>& product_stoich,
                    double kf, double kr = 0.0)
        : reactant_species_(reactants), reactant_stoichiometry_(reactant_stoich),
          product_species_(products), product_stoichiometry_(product_stoich),
          k_forward_ref_(kf), k_reverse_ref_(kr),
          activation_energy_forward_(0.0), activation_energy_reverse_(0.0),
          reference_temperature_(298.15), reaction_type_(ELEMENTARY) {}
    
    void set_arrhenius_parameters(double Ea_forward, double Ea_reverse, double T_ref) {
        activation_energy_forward_ = Ea_forward;
        activation_energy_reverse_ = Ea_reverse;
        reference_temperature_ = T_ref;
    }
    
    void set_reaction_type(int type) {
        reaction_type_ = static_cast<ReactionType>(type);
    }
    
    double calculate_rate(const std::vector<double>& concentrations, 
                         double temperature = 298.15) const {
        
        // Calculate temperature-dependent rate constants
        const double R = 8.31446261815324;
        double kf = k_forward_ref_;
        double kr = k_reverse_ref_;
        
        if (temperature != reference_temperature_) {
            kf *= std::exp(activation_energy_forward_ / R * 
                          (1.0/reference_temperature_ - 1.0/temperature));
            kr *= std::exp(activation_energy_reverse_ / R * 
                          (1.0/reference_temperature_ - 1.0/temperature));
        }
        
        switch (reaction_type_) {
            case ELEMENTARY:
                return calculate_elementary_rate(concentrations, kf, kr);
            case MICHAELIS_MENTEN:
                return calculate_mm_rate(concentrations, kf);
            case AUTOCATALYTIC:
                return calculate_autocatalytic_rate(concentrations, kf);
            default:
                return calculate_elementary_rate(concentrations, kf, kr);
        }
    }
    
private:
    double calculate_elementary_rate(const std::vector<double>& conc, 
                                   double kf, double kr) const {
        double forward_rate = kf;
        for (size_t i = 0; i < reactant_species_.size(); ++i) {
            int species = reactant_species_[i];
            double stoich = reactant_stoichiometry_[i];
            if (species >= 0 && species < static_cast<int>(conc.size())) {
                forward_rate *= std::pow(std::max(0.0, conc[species]), stoich);
            }
        }
        
        double reverse_rate = kr;
        for (size_t i = 0; i < product_species_.size(); ++i) {
            int species = product_species_[i];
            double stoich = product_stoichiometry_[i];
            if (species >= 0 && species < static_cast<int>(conc.size())) {
                reverse_rate *= std::pow(std::max(0.0, conc[species]), stoich);
            }
        }
        
        return forward_rate - reverse_rate;
    }
    
    double calculate_mm_rate(const std::vector<double>& conc, double Vmax) const {
        if (reactant_species_.empty()) return 0.0;
        
        int substrate_idx = reactant_species_[0];
        if (substrate_idx < 0 || substrate_idx >= static_cast<int>(conc.size())) return 0.0;
        
        double Km = k_reverse_ref_; // Use kr as Km parameter
        return michaelis_menten_rate(Vmax, Km, conc[substrate_idx]);
    }
    
    double calculate_autocatalytic_rate(const std::vector<double>& conc, double k) const {
        if (reactant_species_.size() < 2) return 0.0;
        
        int A_idx = reactant_species_[0];
        int B_idx = reactant_species_[1];
        
        if (A_idx < 0 || A_idx >= static_cast<int>(conc.size()) ||
            B_idx < 0 || B_idx >= static_cast<int>(conc.size())) return 0.0;
        
        return autocatalytic_rate(k, conc[A_idx], conc[B_idx]);
    }
};

} // namespace pyroxa

#endif // __cplusplus
