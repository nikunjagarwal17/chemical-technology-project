#!/usr/bin/env python3
"""
Mass Transfer Problem Solver: Ammonia Diffusion through Stagnant Gas
Solves steady-state diffusion of ammonia through methane-hydrogen mixture
"""

import sys
import os
# Add parent directory to path for PyroXa import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pyroxa

def solve_ammonia_diffusion():
    """
    Solve ammonia diffusion through stagnant gas mixture
    
    Problem: Ammonia diffusing through non-diffusing gas mixture of methane and hydrogen
    Volume ratio CH4:H2 = 2:1
    Total pressure = 1 atm, Temperature = 10°C
    Partial pressures of NH3: 13,000 and 6,500 N/m² at two planes 2 mm apart
    Cross-sectional area = 2 m²
    """
    
    print("🔬 Ammonia Diffusion through Stagnant Gas Mixture")
    print("=" * 60)
    
    # Given data
    T = 10 + 273.15  # Temperature in K (10°C)
    P_total = 1.0 * 101325  # Total pressure in Pa (1 atm)
    L = 2e-3  # Distance between planes in m (2 mm)
    A = 2.0  # Cross-sectional area in m²
    
    # Partial pressures of ammonia at two planes
    P_NH3_1 = 13000  # Pa at plane 1
    P_NH3_2 = 6500   # Pa at plane 2
    
    # Gas mixture composition (volume ratio CH4:H2 = 2:1)
    # Partial pressures of non-diffusing gases
    P_inert_total = P_total - (P_NH3_1 + P_NH3_2) / 2  # Average
    P_CH4 = (2/3) * P_inert_total  # Methane partial pressure
    P_H2 = (1/3) * P_inert_total   # Hydrogen partial pressure
    
    print(f"📊 Problem Parameters:")
    print(f"   Temperature: {T:.1f} K ({T-273.15:.1f}°C)")
    print(f"   Total Pressure: {P_total/101325:.1f} atm")
    print(f"   Distance: {L*1000:.1f} mm")
    print(f"   Area: {A:.1f} m²")
    print(f"   NH₃ pressure at plane 1: {P_NH3_1:.0f} Pa")
    print(f"   NH₃ pressure at plane 2: {P_NH3_2:.0f} Pa")
    print(f"   CH₄ partial pressure: {P_CH4:.0f} Pa")
    print(f"   H₂ partial pressure: {P_H2:.0f} Pa")
    
    # Diffusivity data (need to correct for temperature and pressure)
    # Given: D_NH3-H2 = 7.0 × 10^-5 m²/s at 303 K and 1 atm
    # Given: D_NH3-CH4 = 2.0 × 10^-5 m²/s at 288 K and 1.5 atm
    
    D_NH3_H2_ref = 7.0e-5  # m²/s at 303 K, 1 atm
    T_ref_H2 = 303.15  # K
    P_ref_H2 = 101325  # Pa
    
    D_NH3_CH4_ref = 2.0e-5  # m²/s at 288 K, 1.5 atm
    T_ref_CH4 = 288.15  # K
    P_ref_CH4 = 1.5 * 101325  # Pa
    
    # Correct diffusivities for actual conditions using Chapman-Enskog theory
    # D ∝ T^1.75 / P
    D_NH3_H2 = D_NH3_H2_ref * (T/T_ref_H2)**1.75 * (P_ref_H2/P_total)
    D_NH3_CH4 = D_NH3_CH4_ref * (T/T_ref_CH4)**1.75 * (P_ref_CH4/P_total)
    
    print(f"\n🔍 Corrected Diffusivities at {T:.1f} K:")
    print(f"   D_NH₃-H₂: {D_NH3_H2:.2e} m²/s")
    print(f"   D_NH₃-CH₄: {D_NH3_CH4:.2e} m²/s")
    
    # Calculate effective diffusivity for mixture using Maxwell-Stefan approach
    # For diffusion through stagnant mixture: 1/D_eff = x_CH4/D_NH3-CH4 + x_H2/D_NH3-H2
    x_CH4 = P_CH4 / P_inert_total  # Mole fraction of CH4 in inert mixture
    x_H2 = P_H2 / P_inert_total    # Mole fraction of H2 in inert mixture
    
    D_eff_inv = x_CH4/D_NH3_CH4 + x_H2/D_NH3_H2
    D_eff = 1.0 / D_eff_inv
    
    print(f"\n⚗️ Mixture Analysis:")
    print(f"   CH₄ mole fraction in inert: {x_CH4:.3f}")
    print(f"   H₂ mole fraction in inert: {x_H2:.3f}")
    print(f"   Effective diffusivity: {D_eff:.2e} m²/s")
    
    # Calculate flux using steady-state diffusion equation
    # For stagnant film: N_A = (D_eff * P_total / (R * T * L)) * ln((P_total - P_A2)/(P_total - P_A1))
    R = 8.314  # Gas constant J/(mol·K)
    
    ln_term = np.log((P_total - P_NH3_2) / (P_total - P_NH3_1))
    N_NH3 = (D_eff * P_total / (R * T * L)) * ln_term  # mol/(m²·s)
    
    # Convert to mass flux
    M_NH3 = 17.031e-3  # Molecular weight of NH3 in kg/mol
    flux_mass = N_NH3 * M_NH3  # kg/(m²·s)
    
    # Total diffusion rate
    total_rate = flux_mass * A  # kg/s
    total_rate_per_day = total_rate * 86400  # kg/day
    
    print(f"\n📈 Results:")
    print(f"   Molar flux: {N_NH3:.6f} mol/(m²·s)")
    print(f"   Mass flux: {flux_mass:.6f} kg/(m²·s)")
    print(f"   Total diffusion rate: {total_rate:.6f} kg/s")
    print(f"   Total diffusion rate: {total_rate_per_day:.3f} kg/day")
    
    return {
        'temperature': T,
        'pressure_total': P_total,
        'distance': L,
        'area': A,
        'P_NH3_1': P_NH3_1,
        'P_NH3_2': P_NH3_2,
        'D_eff': D_eff,
        'N_NH3': N_NH3,
        'flux_mass': flux_mass,
        'total_rate': total_rate,
        'total_rate_per_day': total_rate_per_day,
        'D_NH3_H2': D_NH3_H2,
        'D_NH3_CH4': D_NH3_CH4
    }

def create_visualization(results):
    """Create comprehensive visualization of the diffusion problem"""
    print("\n📊 Creating visualization...")
    
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Concentration profile
    ax1 = plt.subplot(2, 3, 1)
    z = np.linspace(0, results['distance']*1000, 100)  # Convert to mm
    
    # For stagnant film, concentration varies exponentially
    P_total = results['pressure_total']
    P1 = results['P_NH3_1']
    P2 = results['P_NH3_2']
    L_mm = results['distance'] * 1000
    
    # Concentration profile: C(z) based on exponential variation in stagnant film
    z_frac = z / L_mm
    P_NH3_z = P1 * ((P_total - P2)/(P_total - P1))**(z_frac)
    C_NH3_z = P_NH3_z / (8.314 * results['temperature'])  # mol/m³
    
    ax1.plot(z, C_NH3_z, 'b-', linewidth=3, label='NH₃ concentration')
    ax1.scatter([0, L_mm], [P1/(8.314*results['temperature']), P2/(8.314*results['temperature'])], 
                color='red', s=100, zorder=5, label='Given points')
    ax1.set_xlabel('Distance (mm)', fontsize=12)
    ax1.set_ylabel('NH₃ Concentration (mol/m³)', fontsize=12)
    ax1.set_title('🔍 NH₃ Concentration Profile', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Pressure profile
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(z, P_NH3_z/1000, 'g-', linewidth=3, label='NH₃ partial pressure')
    ax2.axhline(y=P_total/1000, color='orange', linestyle='--', label='Total pressure')
    ax2.scatter([0, L_mm], [P1/1000, P2/1000], color='red', s=100, zorder=5)
    ax2.set_xlabel('Distance (mm)', fontsize=12)
    ax2.set_ylabel('Pressure (kPa)', fontsize=12)
    ax2.set_title('📊 Pressure Profile', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Driving force
    ax3 = plt.subplot(2, 3, 3)
    driving_force = (P_total - P_NH3_z) / 1000  # kPa
    ax3.plot(z, driving_force, 'purple', linewidth=3)
    ax3.set_xlabel('Distance (mm)', fontsize=12)
    ax3.set_ylabel('Driving Force (P_total - P_NH₃) (kPa)', fontsize=12)
    ax3.set_title('⚡ Mass Transfer Driving Force', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Diffusivity comparison
    ax4 = plt.subplot(2, 3, 4)
    diffusivities = [results['D_NH3_H2']*1e5, results['D_NH3_CH4']*1e5, results['D_eff']*1e5]
    labels = ['NH₃-H₂', 'NH₃-CH₄', 'Effective']
    colors = ['skyblue', 'lightcoral', 'gold']
    
    bars = ax4.bar(labels, diffusivities, color=colors, alpha=0.8)
    ax4.set_ylabel('Diffusivity (×10⁻⁵ m²/s)', fontsize=12)
    ax4.set_title('🔬 Diffusivity Comparison', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, diffusivities):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Flux analysis
    ax5 = plt.subplot(2, 3, 5)
    flux_data = [results['N_NH3']*1e6, results['flux_mass']*1e6]
    flux_labels = ['Molar Flux\n(×10⁻⁶ mol/m²·s)', 'Mass Flux\n(×10⁻⁶ kg/m²·s)']
    colors_flux = ['lightgreen', 'salmon']
    
    bars2 = ax5.bar(flux_labels, flux_data, color=colors_flux, alpha=0.8)
    ax5.set_ylabel('Flux (×10⁻⁶ units)', fontsize=12)
    ax5.set_title('📈 Diffusion Flux', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars2, flux_data):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Total transfer rate
    ax6 = plt.subplot(2, 3, 6)
    time_hours = np.linspace(0, 24, 25)
    cumulative_kg = results['total_rate_per_day'] * time_hours / 24
    
    ax6.plot(time_hours, cumulative_kg, 'b-', linewidth=3, marker='o', markersize=4)
    ax6.set_xlabel('Time (hours)', fontsize=12)
    ax6.set_ylabel('Cumulative NH₃ transferred (kg)', fontsize=12)
    ax6.set_title('⏱️ Daily Transfer Rate', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Add final value annotation
    ax6.annotate(f'{results["total_rate_per_day"]:.3f} kg/day', 
                xy=(24, results['total_rate_per_day']), xytext=(20, results['total_rate_per_day']*0.8),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ammonia_diffusion_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: ammonia_diffusion_analysis.png")
    
    return fig

def main():
    """Main function to solve the diffusion problem"""
    print("🧪 PyroXa Mass Transfer: Ammonia Diffusion Problem")
    print("=" * 70)
    
    # Solve the diffusion problem
    results = solve_ammonia_diffusion()
    
    # Create visualization
    fig = create_visualization(results)
    
    # Summary report
    print(f"\n📋 SOLUTION SUMMARY")
    print("=" * 50)
    print(f"✓ Effective diffusivity: {results['D_eff']:.2e} m²/s")
    print(f"✓ Molar flux of NH₃: {results['N_NH3']:.6f} mol/(m²·s)")
    print(f"✓ Mass flux of NH₃: {results['flux_mass']:.6f} kg/(m²·s)")
    print(f"✓ Total diffusion rate: {results['total_rate']:.6f} kg/s")
    print(f"🎯 Answer: {results['total_rate_per_day']:.3f} kg/day")
    
    print(f"\n💾 Plots saved to 'ammonia_diffusion_analysis.png'")
    print(f"\n✨ Mass transfer analysis complete!")

if __name__ == "__main__":
    main()
