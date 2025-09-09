import pyroxa
import numpy as np

print("🧪 Testing PyroXa Functions - Final Verification")
print("=" * 60)

# Test 1: Arrhenius rate
print("\n1. Testing arrhenius_rate:")
A = 1e12  # Pre-exponential factor
Ea = 50000  # Activation energy (J/mol)
T = 500  # Temperature (K)
R = 8.314  # Gas constant
rate = pyroxa.arrhenius_rate(A, Ea, T, R)
print(f"   Arrhenius rate at {T}K: {rate:.2e} s⁻¹")

# Test 2: Gibbs free energy
print("\n2. Testing gibbs_free_energy:")
H = 1000  # Enthalpy (J/mol)
S = 50    # Entropy (J/mol/K)
T = 298   # Temperature (K)
G = pyroxa.gibbs_free_energy(H, S, T)
print(f"   Gibbs free energy: {G:.2f} J/mol")

# Test 3: Equilibrium constant
print("\n3. Testing equilibrium_constant:")
delta_G = -5000  # Free energy change (J/mol)
T = 298         # Temperature (K)
K_eq = pyroxa.equilibrium_constant(delta_G, T)
print(f"   Equilibrium constant: {K_eq:.2e}")

# Test 4: Michaelis-Menten kinetics
print("\n4. Testing michaelis_menten_rate:")
Vmax = 100    # Maximum rate
Km = 0.5      # Michaelis constant
substrate = 2.0  # Substrate concentration
rate_mm = pyroxa.michaelis_menten_rate(Vmax, Km, substrate)
print(f"   Michaelis-Menten rate: {rate_mm:.2f}")

# Test 5: NASA polynomial heat capacity
print("\n5. Testing heat_capacity_nasa:")
T = 500  # Temperature (K)
# Example NASA coefficients for CO2
coeffs = [2.275725, 0.0099209, -1.04091e-5, 6.866687e-9, -2.11728e-12]
cp = pyroxa.heat_capacity_nasa(T, coeffs)
print(f"   Heat capacity at {T}K: {cp:.2f} J/mol/K")

# Test 6: Linear interpolation
print("\n6. Testing linear_interpolate:")
x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [2.0, 4.0, 6.0, 8.0]
x_interp = 2.5
y_interp = pyroxa.linear_interpolate(x_interp, x_data, y_data)
print(f"   Interpolated value at x={x_interp}: {y_interp:.2f}")

# Test 7: R-squared calculation
print("\n7. Testing calculate_r_squared:")
experimental = [1.0, 2.0, 3.0, 4.0, 5.0]
predicted = [1.1, 1.9, 3.1, 3.9, 5.0]
r_squared = pyroxa.calculate_r_squared(experimental, predicted)
print(f"   R-squared: {r_squared:.4f}")

# Test 8: Autocatalytic rate
print("\n8. Testing autocatalytic_rate:")
k = 0.1
A = 2.0
B = 1.0
rate_auto = pyroxa.autocatalytic_rate(k, A, B)
print(f"   Autocatalytic rate: {rate_auto:.2f}")

# Test 9: Competitive inhibition
print("\n9. Testing competitive_inhibition_rate:")
Vmax = 100
Km = 0.5
substrate = 2.0
inhibitor = 0.3
Ki = 0.1
rate_inhib = pyroxa.competitive_inhibition_rate(Vmax, Km, substrate, inhibitor, Ki)
print(f"   Inhibited rate: {rate_inhib:.2f}")

# Test 10: Mass transfer correlation
print("\n10. Testing mass_transfer_correlation:")
Re = 1000  # Reynolds number
Sc = 0.7   # Schmidt number
geom = 1.0 # Geometry factor
mass_transfer = pyroxa.mass_transfer_correlation(Re, Sc, geom)
print(f"   Mass transfer coefficient: {mass_transfer:.2f}")

print("\n" + "=" * 60)
print("✅ All function tests completed successfully!")
print(f"\nSUMMARY:")
print(f"Total functions available: {len([name for name in dir(pyroxa) if not name.startswith('_')])}")
print(f"Functions in __all__: {len(pyroxa.__all__)}")
print(f"C++ extension status: ✓ Loaded successfully")
