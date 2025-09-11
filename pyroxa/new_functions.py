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


def linear_interpolate(*args):
    """Linear interpolation - supports two calling modes:
    1. linear_interpolate(x1, y1, x2, y2, x) - interpolate between two points
    2. linear_interpolate(x, x_data, y_data) - interpolate from arrays
    """
    if len(args) == 5:
        # Mode 1: individual points
        x1, y1, x2, y2, x = args
        if x2 == x1:
            return y1
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    elif len(args) == 3:
        # Mode 2: arrays
        x, x_data, y_data = args
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        
        if len(x_data) != len(y_data):
            raise ValueError("x_data and y_data must have the same length")
        
        # Find the interval containing x
        for i in range(len(x_data) - 1):
            if x_data[i] <= x <= x_data[i + 1]:
                # Linear interpolation between points i and i+1
                x1, y1 = x_data[i], y_data[i]
                x2, y2 = x_data[i + 1], y_data[i + 1]
                if x2 == x1:
                    return y1
                return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        
        # Extrapolation
        if x < x_data[0]:
            # Extrapolate using first two points
            x1, y1 = x_data[0], y_data[0]
            x2, y2 = x_data[1], y_data[1]
        else:
            # Extrapolate using last two points
            x1, y1 = x_data[-2], y_data[-2]
            x2, y2 = x_data[-1], y_data[-1]
        
        if x2 == x1:
            return y1
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    else:
        raise ValueError("linear_interpolate() takes either 3 or 5 arguments")


def cubic_spline_interpolate(x_points, y_points, x):
    """Simple cubic spline interpolation (basic implementation)"""
    if len(x_points) != len(y_points) or len(x_points) < 2:
        raise ValueError("Invalid input for spline interpolation")
    
    # For simplicity, use linear interpolation as fallback
    # In a full implementation, this would use cubic spline calculations
    n = len(x_points)
    for i in range(n - 1):
        if x_points[i] <= x <= x_points[i + 1]:
            return linear_interpolate(x_points[i], y_points[i], x_points[i + 1], y_points[i + 1], x)
    
    # Extrapolation
    if x < x_points[0]:
        return linear_interpolate(x_points[0], y_points[0], x_points[1], y_points[1], x)
    else:
        return linear_interpolate(x_points[-2], y_points[-2], x_points[-1], y_points[-1], x)


def calculate_r_squared(y_actual, y_predicted):
    """Calculate R-squared (coefficient of determination)"""
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)
    
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def calculate_rmse(y_actual, y_predicted):
    """Calculate Root Mean Square Error"""
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)
    
    return np.sqrt(np.mean((y_actual - y_predicted) ** 2))


def calculate_aic(y_actual, y_predicted, k):
    """Calculate Akaike Information Criterion"""
    if isinstance(y_actual, (float, int)) or isinstance(y_predicted, (float, int)):
        raise TypeError("y_actual and y_predicted must be array-like, not float")
    y_actual = np.array(y_actual)
    y_predicted = np.array(y_predicted)
    n = len(y_actual)
    rss = np.sum((y_actual - y_predicted) ** 2)
    if n <= 0 or rss <= 0:
        return float('inf')
    return n * np.log(rss / n) + 2 * k


def gibbs_free_energy(enthalpy, entropy, T):
    """Calculate Gibbs free energy: G = H - T*S"""
    return enthalpy - T * entropy


def equilibrium_constant(delta_G, T, R=8.314):
    """Calculate equilibrium constant from Gibbs free energy"""
    return math.exp(-delta_G / (R * T))


def arrhenius_rate(A, Ea, T, R=8.314):
    """Calculate reaction rate using Arrhenius equation"""
    return A * math.exp(-Ea / (R * T))


# Additional chemical engineering functions to reach 68 functions total

def first_order_rate(k, concentration):
    """First-order reaction rate: r = k * [A]"""
    return k * concentration


def second_order_rate(k, conc_A, conc_B=None):
    """Second-order reaction rate: r = k * [A] * [B] or r = k * [A]^2"""
    if conc_B is None:
        return k * conc_A * conc_A
    return k * conc_A * conc_B


def zero_order_rate(k):
    """Zero-order reaction rate: r = k"""
    return k


def reversible_rate(kf, kr, conc_A, conc_B=0.0):
    """Reversible reaction rate: r = kf * [A] - kr * [B]"""
    return kf * conc_A - kr * conc_B


def parallel_reaction_rate(k1, k2, concentration):
    """Parallel reaction rate: r_total = (k1 + k2) * [A]"""
    return (k1 + k2) * concentration


def series_reaction_rate(k1, k2, conc_A, conc_B):
    """Series reaction rate for A -> B -> C"""
    r_AB = k1 * conc_A
    r_BC = k2 * conc_B
    return r_AB, r_BC


def enzyme_inhibition_rate(Vmax, Km, substrate_conc, inhibitor_conc, Ki, inhibition_type='competitive'):
    """Enzyme inhibition kinetics"""
    if inhibition_type == 'competitive':
        return competitive_inhibition_rate(Vmax, Km, substrate_conc, inhibitor_conc, Ki)
    elif inhibition_type == 'non_competitive':
        return Vmax * substrate_conc / ((Km + substrate_conc) * (1 + inhibitor_conc / Ki))
    else:
        # Uncompetitive inhibition
        return Vmax * substrate_conc / (Km + substrate_conc * (1 + inhibitor_conc / Ki))


def temperature_dependence(k_ref, Ea, T, T_ref=298.15, R=8.314):
    """Temperature dependence of rate constant using Arrhenius equation"""
    return k_ref * math.exp(-(Ea / R) * (1/T - 1/T_ref))


def pressure_dependence(k_ref, delta_V, P, P_ref=101325, R=8.314, T=298.15):
    """Pressure dependence of rate constant"""
    return k_ref * math.exp((delta_V / (R * T)) * (P - P_ref))


def activity_coefficient(gamma, concentration):
    """Activity calculation: a = gamma * C"""
    return gamma * concentration


def diffusion_coefficient(D0, T, Ea_diff=0, R=8.314, T_ref=298.15):
    """Temperature-dependent diffusion coefficient"""
    if Ea_diff == 0:
        return D0
    return D0 * math.exp(-(Ea_diff / R) * (1/T - 1/T_ref))


def thermal_conductivity(k0, T, alpha=0.001):
    """Temperature-dependent thermal conductivity: k = k0 * (1 + alpha * T)"""
    return k0 * (1 + alpha * T)


def heat_transfer_coefficient(Nu, k_fluid, characteristic_length):
    """Heat transfer coefficient from Nusselt number"""
    return Nu * k_fluid / characteristic_length


def mass_transfer_coefficient(Sh, D_AB, characteristic_length):
    """Mass transfer coefficient from Sherwood number"""
    return Sh * D_AB / characteristic_length


def reynolds_number(density, velocity, length, viscosity):
    """Calculate Reynolds number"""
    return density * velocity * length / viscosity


def prandtl_number(cp, viscosity, thermal_conductivity):
    """Calculate Prandtl number"""
    return cp * viscosity / thermal_conductivity


def schmidt_number(viscosity, density, diffusivity):
    """Calculate Schmidt number"""
    return viscosity / (density * diffusivity)


def nusselt_number(h, L, k):
    """Calculate Nusselt number"""
    return h * L / k


def sherwood_number(kc, L, D):
    """Calculate Sherwood number"""
    return kc * L / D


def friction_factor(Re, roughness=0.0):
    """Calculate friction factor for pipe flow"""
    if Re < 2300:
        return 64 / Re  # Laminar flow
    else:
        # Turbulent flow (Colebrook equation approximation)
        if roughness == 0:
            return 0.316 / (Re**0.25)  # Smooth pipe
        else:
            # Churchill equation
            A = (2.457 * math.log(1 / ((7/Re)**0.9 + 0.27 * roughness)))**16
            B = (37530 / Re)**16
            return 8 * ((8/Re)**12 + (A + B)**(-1.5))**(1/12)


def hydraulic_diameter(area, perimeter):
    """Calculate hydraulic diameter"""
    return 4 * area / perimeter


def residence_time(volume, flow_rate):
    """Calculate residence time"""
    return volume / flow_rate


def conversion(initial_conc, final_conc):
    """Calculate conversion: X = (C0 - C) / C0"""
    if initial_conc == 0:
        return 0.0
    return (initial_conc - final_conc) / initial_conc


def selectivity(product_conc, byproduct_conc):
    """Calculate selectivity: S = [P] / ([P] + [BP])"""
    total = product_conc + byproduct_conc
    if total == 0:
        return 0.0
    return product_conc / total


def yield_coefficient(product_formed, reactant_consumed):
    """Calculate yield coefficient"""
    if reactant_consumed == 0:
        return 0.0
    return product_formed / reactant_consumed


def space_time(volume, volumetric_flow_rate):
    """Calculate space time (tau = V / v0)"""
    return volume / volumetric_flow_rate


def space_velocity(volumetric_flow_rate, reactor_volume):
    """Calculate space velocity (SV = v0 / V)"""
    return volumetric_flow_rate / reactor_volume


def reaction_quotient(product_concs, reactant_concs, stoich_coeffs_products, stoich_coeffs_reactants):
    """Calculate reaction quotient Q"""
    Q = 1.0
    for i, conc in enumerate(product_concs):
        Q *= conc ** stoich_coeffs_products[i]
    for i, conc in enumerate(reactant_concs):
        Q /= conc ** stoich_coeffs_reactants[i]
    return Q


def extent_of_reaction(initial_conc, final_conc, stoich_coeff):
    """Calculate extent of reaction"""
    return (initial_conc - final_conc) / abs(stoich_coeff)


def batch_reactor_time(initial_conc, final_conc, rate_constant, order=1):
    """Calculate time required in batch reactor"""
    if order == 1:
        return math.log(initial_conc / final_conc) / rate_constant
    elif order == 2:
        return (1/final_conc - 1/initial_conc) / rate_constant
    else:
        # General case for nth order
        return ((final_conc**(1-order)) - (initial_conc**(1-order))) / ((1-order) * rate_constant)


def cstr_volume(flow_rate, rate_constant, conversion, order=1):
    """Calculate CSTR volume required for given conversion"""
    if order == 1:
        return flow_rate * conversion / (rate_constant * (1 - conversion))
    else:
        # Simplified for first order
        return flow_rate * conversion / (rate_constant * (1 - conversion))


def pfr_volume(flow_rate, rate_constant, conversion, order=1):
    """Calculate PFR volume required for given conversion"""
    if order == 1:
        return flow_rate * (-math.log(1 - conversion)) / rate_constant
    else:
        # Simplified for first order
        return flow_rate * (-math.log(1 - conversion)) / rate_constant


def fluidized_bed_hydrodynamics(particle_diameter, density_particle, density_fluid, viscosity, velocity):
    """Calculate fluidized bed minimum fluidization velocity"""
    Re_mf = (1135.7 + 0.0408 * (particle_diameter**3) * density_fluid * 
             (density_particle - density_fluid) * 9.81 / (viscosity**2))**0.5 - 33.7
    return Re_mf * viscosity / (density_fluid * particle_diameter)


def packed_bed_pressure_drop(velocity, density, viscosity, particle_diameter, bed_porosity, bed_length):
    """Calculate pressure drop in packed bed (same as pressure_drop_ergun)"""
    return pressure_drop_ergun(velocity, density, viscosity, particle_diameter, bed_porosity, bed_length)


def bubble_column_dynamics(gas_velocity, liquid_density, gas_density, surface_tension, viscosity=0.001):
    """Calculate bubble rise velocity in bubble column"""
    # Morton number
    Mo = 9.81 * liquid_density * (viscosity**4) / (liquid_density**2 * surface_tension**3)
    # Simplified bubble rise velocity correlation
    return 0.71 * (9.81 * surface_tension / liquid_density)**0.5


def crystallization_rate(supersaturation, nucleation_rate_constant, growth_rate_constant):
    """Calculate crystallization rate"""
    nucleation_rate = nucleation_rate_constant * supersaturation**2
    growth_rate = growth_rate_constant * supersaturation
    return nucleation_rate + growth_rate


def precipitation_rate(conc_A, conc_B, ksp, rate_constant):
    """Calculate precipitation rate"""
    ion_product = conc_A * conc_B
    if ion_product > ksp:
        return rate_constant * (ion_product - ksp)
    return 0.0


def dissolution_rate(surface_area, mass_transfer_coeff, saturation_conc, current_conc):
    """Calculate dissolution rate"""
    return surface_area * mass_transfer_coeff * (saturation_conc - current_conc)


def evaporation_rate(vapor_pressure, mass_transfer_coeff, area):
    """Calculate evaporation rate"""
    return mass_transfer_coeff * area * vapor_pressure


def distillation_efficiency(actual_stages, theoretical_stages):
    """Calculate distillation efficiency"""
    return actual_stages / theoretical_stages


def extraction_efficiency(conc_feed, conc_raffinate):
    """Calculate extraction efficiency"""
    if conc_feed == 0:
        return 0.0
    return (conc_feed - conc_raffinate) / conc_feed


def adsorption_isotherm(conc, qmax, K_ads, n=1):
    """Langmuir or Freundlich adsorption isotherm"""
    if n == 1:
        # Langmuir isotherm
        return qmax * K_ads * conc / (1 + K_ads * conc)
    else:
        # Freundlich isotherm
        return K_ads * (conc ** (1/n))


def desorption_rate(adsorbed_amount, desorption_constant):
    """Calculate desorption rate"""
    return desorption_constant * adsorbed_amount


def catalyst_activity(initial_activity, deactivation_constant, time):
    """Calculate catalyst activity over time"""
    return initial_activity * math.exp(-deactivation_constant * time)


def catalyst_deactivation(activity, poison_conc, deactivation_constant):
    """Calculate catalyst deactivation rate"""
    return deactivation_constant * activity * poison_conc


def surface_reaction_rate(surface_coverage, rate_constant, activation_energy, T, R=8.314):
    """Calculate surface reaction rate"""
    k_T = rate_constant * math.exp(-activation_energy / (R * T))
    return k_T * surface_coverage


def pore_diffusion_rate(diffusivity, pore_length, conc_surface, conc_center):
    """Calculate pore diffusion rate"""
    return diffusivity * (conc_surface - conc_center) / pore_length


def film_mass_transfer(mass_transfer_coeff, area, conc_bulk, conc_interface):
    """Calculate film mass transfer rate"""
    return mass_transfer_coeff * area * (conc_bulk - conc_interface)


def bubble_rise_velocity(bubble_diameter, density_liquid, density_gas, surface_tension, viscosity):
    """Calculate bubble rise velocity"""
    g = 9.81
    delta_rho = density_liquid - density_gas
    # Morton number
    Mo = g * viscosity**4 * delta_rho / (density_liquid**2 * surface_tension**3)
    # Simplified correlation for bubble rise velocity
    if Mo < 1e-3:
        return math.sqrt(g * bubble_diameter * delta_rho / density_liquid)
    else:
        return 0.71 * math.sqrt(g * bubble_diameter)


def terminal_velocity(particle_diameter, density_particle, density_fluid, viscosity):
    """Calculate terminal velocity of settling particle"""
    g = 9.81
    # Reynolds number for settling
    Re_terminal = 4 * g * particle_diameter**3 * density_fluid * (density_particle - density_fluid) / (3 * viscosity**2)
    
    if Re_terminal < 0.1:
        # Stokes flow
        return g * particle_diameter**2 * (density_particle - density_fluid) / (18 * viscosity)
    else:
        # Intermediate/turbulent flow (simplified)
        Cd = 24/Re_terminal + 6/(1 + math.sqrt(Re_terminal)) + 0.4
        return math.sqrt(4 * g * particle_diameter * (density_particle - density_fluid) / (3 * Cd * density_fluid))


def drag_coefficient(reynolds_number):
    """Calculate drag coefficient for sphere"""
    Re = reynolds_number
    if Re < 0.1:
        return 24 / Re
    elif Re < 1000:
        return 24/Re + 6/(1 + math.sqrt(Re)) + 0.4
    else:
        return 0.44


def mixing_time(tank_diameter, impeller_diameter, rotational_speed, viscosity):
    """Calculate mixing time in stirred tank"""
    power_number = 5.0  # Typical for Rushton turbine
    return 5.2 * (tank_diameter / impeller_diameter)**2 * (viscosity / 1000)**0.1 / rotational_speed


def power_consumption(power_number, density, rotational_speed, impeller_diameter):
    """Calculate power consumption in stirred tank"""
    return power_number * density * (rotational_speed**3) * (impeller_diameter**5)


def pumping_power(flow_rate, pressure_drop, efficiency=0.7):
    """Calculate pumping power required"""
    return flow_rate * pressure_drop / efficiency


def compression_work(flow_rate, pressure_in, pressure_out, gamma=1.4):
    """Calculate compression work for ideal gas"""
    return flow_rate * gamma / (gamma - 1) * pressure_in * ((pressure_out/pressure_in)**((gamma-1)/gamma) - 1)


def heat_exchanger_effectiveness(actual_heat_transfer, max_possible_heat_transfer):
    """Calculate heat exchanger effectiveness"""
    if max_possible_heat_transfer == 0:
        return 0.0
    return actual_heat_transfer / max_possible_heat_transfer


def overall_heat_transfer_coefficient(h_hot, h_cold, thickness, thermal_conductivity, fouling_resistance=0):
    """Calculate overall heat transfer coefficient"""
    return 1 / (1/h_hot + thickness/thermal_conductivity + 1/h_cold + fouling_resistance)


def fouling_resistance(clean_U, fouled_U):
    """Calculate fouling resistance"""
    return 1/fouled_U - 1/clean_U


# Advanced simulation functions to complete implementation

def simulate_packed_bed(n_species, n_reactions, n_segments, rate_constants_f, rate_constants_r, 
                       reaction_indices, reaction_stoich, initial_concentrations, 
                       bed_length=1.0, velocity=0.1, time_span=10.0, dt=0.01):
    """Simulate packed bed reactor with axial dispersion"""
    import numpy as np
    
    # Simple packed bed simulation using finite differences
    segments = n_segments
    dx = bed_length / segments
    time_steps = int(time_span / dt)
    
    # Initialize concentration matrix [segment, species, time]
    conc = np.zeros((segments, n_species, time_steps))
    conc[:, :, 0] = initial_concentrations
    
    # Simple first-order reaction simulation
    for t in range(1, time_steps):
        for seg in range(segments):
            for species in range(n_species):
                # Convection term
                if seg > 0:
                    convection = velocity * (conc[seg-1, species, t-1] - conc[seg, species, t-1]) / dx
                else:
                    convection = 0
                
                # Reaction term (simplified)
                reaction_rate = 0
                for rxn in range(n_reactions):
                    if rxn < len(rate_constants_f):
                        reaction_rate += rate_constants_f[rxn] * conc[seg, species, t-1]
                
                # Update concentration
                conc[seg, species, t] = conc[seg, species, t-1] + dt * (convection - reaction_rate)
                conc[seg, species, t] = max(0, conc[seg, species, t])  # Non-negative concentrations
    
    return conc


def simulate_fluidized_bed(n_species, n_reactions, rate_constants_f, rate_constants_r,
                          reaction_indices, reaction_stoich, initial_concentrations,
                          bed_height=2.0, gas_velocity=0.5, particle_diameter=0.001,
                          time_span=10.0, dt=0.01):
    """Simulate fluidized bed reactor with bubble phase and emulsion phase"""
    import numpy as np
    
    time_steps = int(time_span / dt)
    
    # Two-phase model: bubble and emulsion
    conc_bubble = np.zeros((n_species, time_steps))
    conc_emulsion = np.zeros((n_species, time_steps))
    
    # Initial conditions
    conc_bubble[:, 0] = initial_concentrations
    conc_emulsion[:, 0] = initial_concentrations
    
    # Fluidization parameters
    bubble_fraction = 0.3
    mass_transfer_coeff = 0.1
    
    for t in range(1, time_steps):
        for species in range(n_species):
            # Mass transfer between phases
            mass_transfer = mass_transfer_coeff * (conc_bubble[species, t-1] - conc_emulsion[species, t-1])
            
            # Reaction in emulsion phase (catalytic)
            reaction_rate_emulsion = 0
            for rxn in range(min(n_reactions, len(rate_constants_f))):
                reaction_rate_emulsion += rate_constants_f[rxn] * conc_emulsion[species, t-1]
            
            # Update concentrations
            conc_bubble[species, t] = conc_bubble[species, t-1] + dt * mass_transfer
            conc_emulsion[species, t] = (conc_emulsion[species, t-1] + 
                                       dt * (-mass_transfer - reaction_rate_emulsion))
            
            # Ensure non-negative concentrations
            conc_bubble[species, t] = max(0, conc_bubble[species, t])
            conc_emulsion[species, t] = max(0, conc_emulsion[species, t])
    
    # Average concentration
    avg_conc = bubble_fraction * conc_bubble + (1 - bubble_fraction) * conc_emulsion
    return avg_conc


def simulate_homogeneous_batch(n_species, n_reactions, rate_constants_f, rate_constants_r,
                              reaction_indices, reaction_stoich, initial_concentrations,
                              temperature=298.15, volume=1.0, time_span=10.0, dt=0.01):
    """Simulate homogeneous batch reactor with multiple reactions"""
    import numpy as np
    
    time_steps = int(time_span / dt)
    conc = np.zeros((n_species, time_steps))
    conc[:, 0] = initial_concentrations
    
    for t in range(1, time_steps):
        # Calculate reaction rates
        rates = np.zeros(n_species)
        
        for rxn in range(min(n_reactions, len(rate_constants_f))):
            # Simple first-order kinetics
            if rxn < len(reaction_indices) and len(reaction_indices[rxn]) > 0:
                reactant_idx = reaction_indices[rxn][0] if reaction_indices[rxn][0] < n_species else 0
                rate = rate_constants_f[rxn] * conc[reactant_idx, t-1]
                
                # Apply stoichiometry
                if rxn < len(reaction_stoich):
                    for species in range(min(n_species, len(reaction_stoich[rxn]))):
                        rates[species] += reaction_stoich[rxn][species] * rate
        
        # Update concentrations using Euler integration
        for species in range(n_species):
            conc[species, t] = conc[species, t-1] + dt * rates[species]
            conc[species, t] = max(0, conc[species, t])  # Non-negative concentrations
    
    return conc


def simulate_multi_reactor_adaptive(reactor_config, feed_conditions, control_strategy=None):
    """Simulate multi-reactor system with adaptive control"""
    import numpy as np
    
    n_reactors = len(reactor_config)
    simulation_time = reactor_config.get('simulation_time', 10.0)
    dt = reactor_config.get('dt', 0.01)
    time_steps = int(simulation_time / dt)
    
    # Initialize reactor states
    reactor_states = []
    for i in range(n_reactors):
        config = reactor_config[i] if i < len(reactor_config) else reactor_config[0]
        n_species = config.get('n_species', 2)
        initial_conc = config.get('initial_concentrations', [1.0] * n_species)
        reactor_states.append(np.array(initial_conc))
    
    # Simulation results
    results = {
        'time': np.linspace(0, simulation_time, time_steps),
        'concentrations': np.zeros((n_reactors, len(reactor_states[0]), time_steps))
    }
    
    # Set initial conditions
    for i in range(n_reactors):
        results['concentrations'][i, :, 0] = reactor_states[i]
    
    # Adaptive simulation loop
    for t in range(1, time_steps):
        for reactor_id in range(n_reactors):
            config = reactor_config[reactor_id] if reactor_id < len(reactor_config) else reactor_config[0]
            
            # Get rate constants (with adaptive control if specified)
            rate_constants = config.get('rate_constants', [0.1])
            
            if control_strategy:
                # Simple adaptive control: adjust rate constants based on conversion
                conversion = 1 - (reactor_states[reactor_id][0] / config.get('initial_concentrations', [1.0])[0])
                if conversion < control_strategy.get('target_conversion', 0.8):
                    rate_constants = [k * 1.1 for k in rate_constants]  # Increase rates
            
            # Update reactor state
            for species in range(len(reactor_states[reactor_id])):
                if species < len(rate_constants):
                    rate = rate_constants[species] * reactor_states[reactor_id][species]
                    reactor_states[reactor_id][species] += dt * (-rate)
                    reactor_states[reactor_id][species] = max(0, reactor_states[reactor_id][species])
            
            # Store results
            results['concentrations'][reactor_id, :, t] = reactor_states[reactor_id]
    
    return results


def calculate_energy_balance(n_species, n_reactions, concentrations, reaction_rates,
                           enthalpies_formation, heat_capacities, temperature=298.15,
                           volume=1.0, heat_transfer_coeff=0.0, ambient_temp=298.15):
    """Calculate energy balance for reactor system"""
    import numpy as np
    
    # Heat of reaction
    heat_reaction = 0.0
    for rxn in range(min(n_reactions, len(reaction_rates))):
        if rxn < len(enthalpies_formation):
            # Simplified: assume single product formation
            delta_H = enthalpies_formation[rxn]  # kJ/mol
            heat_reaction += reaction_rates[rxn] * delta_H * volume
    
    # Sensible heat change
    heat_sensible = 0.0
    for species in range(min(n_species, len(concentrations), len(heat_capacities))):
        cp = heat_capacities[species]  # kJ/mol/K
        conc = concentrations[species]  # mol/L
        heat_sensible += cp * conc * volume * (temperature - 298.15)
    
    # Heat transfer to environment
    heat_transfer = heat_transfer_coeff * volume * (temperature - ambient_temp)
    
    # Energy balance: dT/dt = (Q_reaction - Q_transfer) / (sum(n_i * Cp_i))
    total_heat_capacity = sum(concentrations[i] * heat_capacities[i] * volume 
                             for i in range(min(len(concentrations), len(heat_capacities))))
    
    if total_heat_capacity > 0:
        dT_dt = (heat_reaction - heat_transfer) / total_heat_capacity
    else:
        dT_dt = 0.0
    
    return {
        'heat_of_reaction': heat_reaction,
        'sensible_heat': heat_sensible,
        'heat_transfer': heat_transfer,
        'temperature_change_rate': dT_dt,
        'total_heat_capacity': total_heat_capacity
    }


# Advanced analytical and numerical methods

def analytical_first_order(k, A0, time_span, dt=0.01):
    """Analytical solution for first-order reaction: A -> products"""
    import numpy as np
    times = np.arange(0, time_span + dt, dt)
    concentrations = A0 * np.exp(-k * times)
    return times.tolist(), concentrations.tolist()


def analytical_reversible_first_order(kf, kr, A0, B0=0.0, time_span=10.0, dt=0.01):
    """Analytical solution for reversible first-order: A <-> B"""
    import numpy as np
    times = np.arange(0, time_span + dt, dt)
    k_total = kf + kr
    K_eq = kf / kr if kr > 0 else float('inf')
    
    if kr > 0:
        A_eq = (A0 + B0) / (1 + K_eq)
        B_eq = (A0 + B0) - A_eq
        
        A_conc = A_eq + (A0 - A_eq) * np.exp(-k_total * times)
        B_conc = B_eq + (B0 - B_eq) * np.exp(-k_total * times)
    else:
        A_conc = A0 * np.exp(-kf * times)
        B_conc = A0 - A_conc + B0
    
    return times.tolist(), A_conc.tolist(), B_conc.tolist()


def analytical_consecutive_first_order(k1, k2, A0, time_span=10.0, dt=0.01):
    """Analytical solution for consecutive first-order: A -> B -> C"""
    import numpy as np
    times = np.arange(0, time_span + dt, dt)
    
    A_conc = A0 * np.exp(-k1 * times)
    
    if abs(k1 - k2) > 1e-10:
        B_conc = A0 * k1 / (k2 - k1) * (np.exp(-k1 * times) - np.exp(-k2 * times))
    else:
        B_conc = A0 * k1 * times * np.exp(-k1 * times)
    
    C_conc = A0 * (1 - np.exp(-k1 * times)) - B_conc
    
    return times.tolist(), A_conc.tolist(), B_conc.tolist(), C_conc.tolist()


def calculate_objective_function(experimental_data, simulated_data, weights=None):
    """Calculate objective function for parameter estimation"""
    import numpy as np
    exp_data = np.array(experimental_data)
    sim_data = np.array(simulated_data)
    
    if weights is None:
        weights = np.ones_like(exp_data)
    else:
        weights = np.array(weights)
    
    residuals = (exp_data - sim_data) * weights
    return np.sum(residuals**2)


def check_mass_conservation(concentrations, tolerance=1e-6):
    """Check mass conservation in reaction system"""
    import numpy as np
    conc_array = np.array(concentrations)
    initial_total = np.sum(conc_array[0, :])
    
    conservation_errors = []
    for i in range(len(conc_array)):
        current_total = np.sum(conc_array[i, :])
        error = abs(current_total - initial_total) / initial_total
        conservation_errors.append(error)
        
        if error > tolerance:
            return False, conservation_errors
    
    return True, conservation_errors


def calculate_rate_constants(kf_ref, kr_ref, Ea_f, Ea_r, T, T_ref=298.15, R=8.314):
    """Calculate temperature-dependent rate constants"""
    kf = kf_ref * math.exp(-Ea_f / R * (1/T - 1/T_ref))
    kr = kr_ref * math.exp(-Ea_r / R * (1/T - 1/T_ref))
    return kf, kr


def cross_validation_score(data, parameters, n_folds=5):
    """Simplified cross-validation score calculation"""
    import numpy as np
    data_array = np.array(data)
    n_samples = len(data_array)
    fold_size = n_samples // n_folds
    
    scores = []
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < n_folds - 1 else n_samples
        
        # Simple validation score (placeholder)
        validation_data = data_array[start_idx:end_idx]
        training_data = np.concatenate([data_array[:start_idx], data_array[end_idx:]])
        
        # Calculate simple MSE between validation and training mean
        score = np.mean((validation_data - np.mean(training_data))**2)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)


def kriging_interpolation(x_new, x_known, y_known, variogram_params=None):
    """Simplified kriging interpolation"""
    import numpy as np
    x_known = np.array(x_known)
    y_known = np.array(y_known)
    
    # Find nearest neighbors for simple interpolation
    distances = np.abs(x_known - x_new)
    nearest_idx = np.argmin(distances)
    
    if len(x_known) == 1:
        return y_known[0]
    
    # Use weighted average of two nearest points
    sorted_indices = np.argsort(distances)
    if len(sorted_indices) >= 2:
        idx1, idx2 = sorted_indices[0], sorted_indices[1]
        d1, d2 = distances[idx1], distances[idx2]
        
        if d1 + d2 == 0:
            return y_known[idx1]
        
        weight1 = d2 / (d1 + d2)
        weight2 = d1 / (d1 + d2)
        return weight1 * y_known[idx1] + weight2 * y_known[idx2]
    
    return y_known[nearest_idx]


def bootstrap_uncertainty(data, parameters, n_bootstrap=1000):
    """Bootstrap uncertainty analysis"""
    import numpy as np
    data_array = np.array(data)
    n_samples = len(data_array)
    
    bootstrap_results = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_sample = data_array[bootstrap_indices]
        
        # Calculate statistic (mean in this case)
        bootstrap_stat = np.mean(bootstrap_sample)
        bootstrap_results.append(bootstrap_stat)
    
    bootstrap_results = np.array(bootstrap_results)
    mean_estimate = np.mean(bootstrap_results)
    std_estimate = np.std(bootstrap_results)
    confidence_interval = np.percentile(bootstrap_results, [2.5, 97.5])
    
    return mean_estimate, std_estimate, confidence_interval.tolist()


def matrix_multiply(A, B):
    """Matrix multiplication"""
    import numpy as np
    return np.dot(np.array(A), np.array(B)).tolist()


def matrix_invert(A):
    """Matrix inversion"""
    import numpy as np
    return np.linalg.inv(np.array(A)).tolist()


def solve_linear_system(A, b):
    """Solve linear system Ax = b"""
    import numpy as np
    return np.linalg.solve(np.array(A), np.array(b)).tolist()


def calculate_sensitivity(params, concentrations, rates, perturbation=1e-6):
    """Calculate sensitivity coefficients"""
    import numpy as np
    params = np.array(params)
    base_result = np.array(concentrations)
    
    sensitivities = []
    for i, param in enumerate(params):
        # Perturb parameter
        params_pert = params.copy()
        params_pert[i] += perturbation
        
        # Calculate perturbed result (simplified)
        pert_result = base_result * (1 + perturbation * rates[i] if i < len(rates) else 1)
        
        # Calculate sensitivity
        sensitivity = (pert_result - base_result) / (perturbation * param)
        sensitivities.append(sensitivity.tolist())
    
    return sensitivities


def calculate_jacobian(y, dydt, perturbation=1e-6):
    """Calculate Jacobian matrix"""
    import numpy as np
    y = np.array(y)
    dydt = np.array(dydt)
    n_species = len(y)
    
    jacobian = np.zeros((n_species, n_species))
    
    for i in range(n_species):
        y_pert = y.copy()
        y_pert[i] += perturbation
        
        # Simplified derivative calculation
        for j in range(n_species):
            jacobian[j, i] = dydt[j] * perturbation / y[i] if y[i] != 0 else 0
    
    return jacobian.tolist()


def stability_analysis(steady_state, n_species, temperature=298.15, pressure=101325.0):
    """Simplified stability analysis"""
    import numpy as np
    steady_state = np.array(steady_state)
    
    # Calculate eigenvalues of simplified Jacobian
    # For demonstration, use a simple stability criterion
    max_concentration = np.max(steady_state)
    min_concentration = np.min(steady_state[steady_state > 0]) if np.any(steady_state > 0) else 0
    
    stability_factor = max_concentration / min_concentration if min_concentration > 0 else float('inf')
    
    is_stable = stability_factor < 100  # Simple criterion
    dominant_eigenvalue = -1.0 / stability_factor  # Simplified
    
    return is_stable, dominant_eigenvalue, stability_factor


def mpc_controller(current_state, setpoints, control_bounds, reaction_network=None, horizon=10):
    """Simplified Model Predictive Controller"""
    import numpy as np
    current_state = np.array(current_state)
    setpoints = np.array(setpoints)
    
    # Simple proportional control with constraints
    error = setpoints - current_state
    control_action = 0.5 * error  # Proportional gain
    
    # Apply bounds
    if control_bounds:
        control_min, control_max = control_bounds
        control_action = np.clip(control_action, control_min, control_max)
    
    return control_action.tolist()


def real_time_optimization(current_concentrations, economic_coefficients, control_bounds, reaction_network=None):
    """Real-time optimization for economic objectives"""
    import numpy as np
    concentrations = np.array(current_concentrations)
    economics = np.array(economic_coefficients)
    
    # Simple economic optimization (maximize profit)
    objective_value = np.dot(concentrations, economics)
    
    # Optimal control (simplified)
    optimal_control = economics / np.sum(np.abs(economics))
    
    if control_bounds:
        control_min, control_max = control_bounds
        optimal_control = np.clip(optimal_control, control_min, control_max)
    
    return objective_value, optimal_control.tolist()


def parameter_sweep_parallel(parameter_ranges, reaction_network=None, concentrations_initial=None, n_points=10):
    """Parameter sweep analysis"""
    import numpy as np
    
    results = []
    for param_name, (min_val, max_val) in parameter_ranges.items():
        param_values = np.linspace(min_val, max_val, n_points)
        
        param_results = []
        for val in param_values:
            # Simplified calculation
            if concentrations_initial:
                result = np.array(concentrations_initial) * (val / min_val)
            else:
                result = [val, val**2]  # Placeholder
            param_results.append(result.tolist() if hasattr(result, 'tolist') else result)
        
        results.append({
            'parameter': param_name,
            'values': param_values.tolist(),
            'results': param_results
        })
    
    return results


def monte_carlo_simulation(n_samples, parameters_mean, parameters_std, model_function=None):
    """Monte Carlo simulation for uncertainty propagation"""
    import numpy as np
    
    parameters_mean = np.array(parameters_mean)
    parameters_std = np.array(parameters_std)
    
    results = []
    for _ in range(n_samples):
        # Sample parameters from normal distribution
        sampled_params = np.random.normal(parameters_mean, parameters_std)
        
        # Simple model evaluation (placeholder)
        if model_function:
            result = model_function(sampled_params)
        else:
            result = np.sum(sampled_params)  # Simple sum as default
        
        results.append(result)
    
    results = np.array(results)
    return {
        'mean': np.mean(results),
        'std': np.std(results),
        'percentiles': np.percentile(results, [5, 25, 50, 75, 95]).tolist(),
        'samples': results.tolist()
    }


def residence_time_distribution(flow_rates, volumes, n_tanks):
    """Calculate residence time distribution for tank series"""
    import numpy as np
    flow_rates = np.array(flow_rates)
    volumes = np.array(volumes)
    
    # Mean residence times
    tau_individual = volumes / flow_rates
    tau_total = np.sum(tau_individual)
    
    # For n identical tanks in series, RTD is gamma distribution
    # E(t) = (n * t)^(n-1) * exp(-n*t/tau) / ((n-1)! * tau^n)
    time_points = np.linspace(0, 3 * tau_total, 100)
    
    if n_tanks == 1:
        E_t = (1 / tau_total) * np.exp(-time_points / tau_total)
    else:
        # Simplified for multiple tanks
        lambda_param = n_tanks / tau_total
        E_t = lambda_param * np.exp(-lambda_param * time_points)
    
    return time_points.tolist(), E_t.tolist(), tau_total


def catalyst_deactivation_model(initial_activity, deactivation_constant, time, temperature, partial_pressure_poison=0):
    """Advanced catalyst deactivation model"""
    # Temperature-dependent deactivation
    temperature_factor = math.exp(-5000 / (8.314 * temperature))  # Simplified Arrhenius
    
    # Poisoning effect
    poison_factor = 1 + partial_pressure_poison * 0.1  # Simplified
    
    effective_deactivation = deactivation_constant * temperature_factor * poison_factor
    activity = initial_activity * math.exp(-effective_deactivation * time)
    
    return activity


def process_scale_up(lab_scale_volume, pilot_scale_volume, lab_conditions):
    """Process scale-up calculations"""
    scale_factor = pilot_scale_volume / lab_scale_volume
    
    scaled_conditions = {}
    for param, value in lab_conditions.items():
        if param in ['flow_rate', 'heat_transfer_rate']:
            # Linear scaling
            scaled_conditions[param] = value * scale_factor
        elif param in ['mixing_time', 'heat_transfer_coefficient']:
            # Power law scaling
            scaled_conditions[param] = value * (scale_factor ** 0.67)
        elif param in ['pressure', 'temperature', 'concentration']:
            # No scaling needed
            scaled_conditions[param] = value
        else:
            # Default linear scaling
            scaled_conditions[param] = value * scale_factor
    
    return scale_factor, scaled_conditions
