"""
Python implementations of newly added C++ functions
These provide the same functionality as the C++ implementations until
Python 3.13 compatibility is fully resolved for the Cython bindings.
"""

import numpy as np
import math


def autocatalytic_rate(k, A, B):
    """Calculate autocatalytic reaction rate"""
    return k * A * B


def michaelis_menten_rate(Vmax, Km, substrate_conc):
    """Calculate Michaelis-Menten enzyme kinetics rate"""
    if substrate_conc < 0:
        return 0.0
    return (Vmax * substrate_conc) / (Km + substrate_conc)


def competitive_inhibition_rate(Vmax, Km, substrate_conc, inhibitor_conc, Ki):
    """Calculate competitive inhibition rate"""
    if substrate_conc < 0 or inhibitor_conc < 0:
        return 0.0
    Km_apparent = Km * (1.0 + inhibitor_conc / Ki)
    return (Vmax * substrate_conc) / (Km_apparent + substrate_conc)


def heat_capacity_nasa(T, coeffs):
    """Calculate heat capacity using NASA polynomial"""
    if T <= 0:
        return 0.0
    R_GAS = 8.314  # J/mol/K
    return R_GAS * (coeffs[0] + coeffs[1]*T + coeffs[2]*T*T + 
                   coeffs[3]*T*T*T + coeffs[4]*T*T*T*T)


def enthalpy_nasa(T, coeffs):
    """Calculate enthalpy using NASA polynomial"""
    if T <= 0:
        return 0.0
    R_GAS = 8.314  # J/mol/K
    return R_GAS * T * (coeffs[0] + coeffs[1]*T/2.0 + coeffs[2]*T*T/3.0 + 
                       coeffs[3]*T*T*T/4.0 + coeffs[4]*T*T*T*T/5.0 + coeffs[5]/T)


def entropy_nasa(T, coeffs):
    """Calculate entropy using NASA polynomial"""
    if T <= 0:
        return 0.0
    R_GAS = 8.314  # J/mol/K
    return R_GAS * (coeffs[0]*math.log(T) + coeffs[1]*T + coeffs[2]*T*T/2.0 + 
                   coeffs[3]*T*T*T/3.0 + coeffs[4]*T*T*T*T/4.0 + coeffs[6])


def mass_transfer_correlation(Re, Sc, geometry_factor):
    """Calculate Sherwood number from Reynolds and Schmidt numbers"""
    if Re <= 0 or Sc <= 0:
        return 0.0
    # Sherwood number correlation: Sh = geometry_factor * Re^0.8 * Sc^0.33
    Sh = geometry_factor * (Re**0.8) * (Sc**(1.0/3.0))
    return Sh


def heat_transfer_correlation(Re, Pr, geometry_factor):
    """Calculate Nusselt number from Reynolds and Prandtl numbers"""
    if Re <= 0 or Pr <= 0:
        return 0.0
    # Nusselt number correlation: Nu = geometry_factor * Re^0.8 * Pr^0.33
    Nu = geometry_factor * (Re**0.8) * (Pr**(1.0/3.0))
    return Nu


def effective_diffusivity(molecular_diff, porosity, tortuosity, constriction_factor):
    """Calculate effective diffusivity in porous media"""
    if molecular_diff <= 0 or porosity <= 0:
        return 0.0
    return molecular_diff * porosity * constriction_factor / tortuosity


def pressure_drop_ergun(velocity, density, viscosity, particle_diameter, bed_porosity, bed_length):
    """Calculate pressure drop using Ergun equation"""
    if velocity <= 0 or density <= 0 or viscosity <= 0 or particle_diameter <= 0:
        return 0.0
    
    epsilon = bed_porosity
    Re_p = density * velocity * particle_diameter / viscosity
    
    # Ergun equation
    friction_factor = 150.0 / Re_p + 1.75
    dp_dz = friction_factor * density * velocity * velocity * (1.0 - epsilon) / \
            (particle_diameter * epsilon * epsilon * epsilon)
    
    return dp_dz * bed_length


class PIDController:
    """PID controller implementation with state preservation"""
    
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_term = 0.0
        self.previous_error = 0.0
    
    def calculate(self, setpoint, process_variable, dt):
        """Calculate PID output"""
        error = setpoint - process_variable
        
        # Proportional term
        proportional = self.Kp * error
        
        # Integral term
        self.integral_term += error * dt
        integral = self.Ki * self.integral_term
        
        # Derivative term
        derivative = self.Kd * (error - self.previous_error) / dt
        self.previous_error = error
        
        return proportional + integral + derivative


def pid_controller(setpoint, process_variable, dt, Kp, Ki, Kd):
    """Simple PID controller function (stateless)"""
    error = setpoint - process_variable
    # Simple proportional control for stateless version
    return Kp * error


# Additional utility functions
def langmuir_hinshelwood_rate(k, K_A, K_B, conc_A, conc_B):
    """Langmuir-Hinshelwood rate expression"""
    if conc_A < 0 or conc_B < 0:
        return 0.0
    denominator = 1.0 + K_A * conc_A + K_B * conc_B
    return (k * K_A * K_B * conc_A * conc_B) / (denominator * denominator)


def photochemical_rate(quantum_yield, molar_absorptivity, path_length, light_intensity, concentration):
    """Photochemical reaction rate"""
    if concentration < 0 or light_intensity < 0:
        return 0.0
    absorbance = molar_absorptivity * concentration * path_length
    absorbed_light = light_intensity * (1.0 - math.exp(-absorbance))
    return quantum_yield * absorbed_light


def pressure_peng_robinson(n, V, T, Tc, Pc, omega):
    """Peng-Robinson equation of state"""
    if V <= 0 or T <= 0:
        return 0.0
    R_GAS = 8.314
    a = 0.45724 * R_GAS * R_GAS * Tc * Tc / Pc
    b = 0.07780 * R_GAS * Tc / Pc
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega * omega
    alpha = (1.0 + kappa * (1.0 - math.sqrt(T/Tc)))**2
    a_T = a * alpha
    
    return (n * R_GAS * T) / (V - n * b) - (n * n * a_T) / (V * V + 2.0 * n * b * V - n * n * b * b)


def fugacity_coefficient(P, T, Tc, Pc, omega):
    """Fugacity coefficient calculation"""
    if P <= 0 or T <= 0:
        return 1.0
    Tr = T / Tc
    Pr = P / Pc
    
    # Simplified correlation for fugacity coefficient
    B0 = 0.083 - 0.422 / (Tr**1.6)
    B1 = 0.139 - 0.172 / (Tr**4.2)
    B = B0 + omega * B1
    
    Z = 1.0 + B * Pr / Tr  # Simplified compressibility factor
    ln_phi = B * Pr / Tr
    
    return math.exp(ln_phi)
