#!/usr/bin/env python3
"""
Pharmaceutical Synthesis Showcase
Demonstrates complex multi-step synthesis with quality control,
impurity tracking, and regulatory compliance analysis.
"""

import sys
import os
# Add parent directory to path for PyroXa import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import yaml
import pyroxa

def load_pharma_spec():
    """Load pharmaceutical synthesis specification"""
    try:
        with open('specs/pharmaceutical_synthesis.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("Pharmaceutical spec not found, using default values")
        return create_default_pharma_spec()

def create_default_pharma_spec():
    """Create default pharmaceutical synthesis specification"""
    return {
        'species': ["Starting_Material", "Intermediate_1", "Intermediate_2", "API", "Impurity_A", "Impurity_B"],
        'initial': {
            'temperature': 323.15,
            'conc': {
                'Starting_Material': 1.0,
                'Intermediate_1': 0.0,
                'Intermediate_2': 0.0,
                'API': 0.0,
                'Impurity_A': 0.0,
                'Impurity_B': 0.0
            }
        },
        'sim': {
            'time_span': 120.0,
            'time_step': 0.2
        }
    }

def run_pharmaceutical_synthesis():
    """Run comprehensive pharmaceutical synthesis simulation"""
    print("💊 Running Pharmaceutical Synthesis Simulation...")
    
    # Load specification
    spec = load_pharma_spec()
    
    # Multi-step synthesis simulation
    time_span = spec['sim']['time_span']
    dt = spec['sim']['time_step']
    times = np.linspace(0, time_span, int(time_span/dt))
    
    # Simulate multi-step synthesis with different process parameters
    process_conditions = [
        {'name': 'Standard', 'temp': 323, 'catalyst': 0.1, 'color': 'blue'},
        {'name': 'Optimized', 'temp': 333, 'catalyst': 0.15, 'color': 'green'},
        {'name': 'High_Temp', 'temp': 343, 'catalyst': 0.1, 'color': 'red'},
        {'name': 'High_Cat', 'temp': 323, 'catalyst': 0.2, 'color': 'purple'}
    ]
    
    results = []
    
    for condition in process_conditions:
        # Rate constants adjusted for temperature and catalyst
        temp_factor = np.exp(-40000/8.314 * (1/condition['temp'] - 1/323))
        cat_factor = condition['catalyst'] / 0.1
        
        # Simulate multi-step kinetics
        SM = np.zeros(len(times))  # Starting Material
        I1 = np.zeros(len(times))  # Intermediate 1
        I2 = np.zeros(len(times))  # Intermediate 2
        API = np.zeros(len(times)) # Active Pharmaceutical Ingredient
        ImpA = np.zeros(len(times)) # Impurity A
        ImpB = np.zeros(len(times)) # Impurity B
        
        # Initial conditions
        SM[0] = 1.0
        
        # Kinetic simulation
        for i in range(1, len(times)):
            dt_sim = times[i] - times[i-1]
            
            # Step 1: SM → I1 (catalytic)
            k1 = 4.0 * temp_factor * cat_factor
            r1 = k1 * SM[i-1]
            
            # Step 2: I1 → I2
            k2 = 2.0 * temp_factor
            r2 = k2 * I1[i-1]
            
            # Step 3: I2 → API
            k3 = 1.5 * temp_factor
            r3 = k3 * I2[i-1]
            
            # Side reactions (impurities)
            k_imp_a = 0.3 * temp_factor
            k_imp_b = 0.2 * temp_factor
            r_imp_a = k_imp_a * I1[i-1]
            r_imp_b = k_imp_b * I2[i-1]
            
            # Update concentrations
            SM[i] = SM[i-1] - r1 * dt_sim
            I1[i] = I1[i-1] + r1 * dt_sim - r2 * dt_sim - r_imp_a * dt_sim
            I2[i] = I2[i-1] + r2 * dt_sim - r3 * dt_sim - r_imp_b * dt_sim
            API[i] = API[i-1] + r3 * dt_sim
            ImpA[i] = ImpA[i-1] + r_imp_a * dt_sim
            ImpB[i] = ImpB[i-1] + r_imp_b * dt_sim
            
            # Ensure non-negative concentrations
            SM[i] = max(0, SM[i])
            I1[i] = max(0, I1[i])
            I2[i] = max(0, I2[i])
        
        # Calculate quality metrics
        final_yield = API[-1] / 1.0 * 100  # % yield
        total_impurities = ImpA[-1] + ImpB[-1]
        purity = API[-1] / (API[-1] + total_impurities) * 100 if (API[-1] + total_impurities) > 0 else 0
        
        results.append({
            'condition': condition,
            'times': times,
            'SM': SM,
            'I1': I1,
            'I2': I2,
            'API': API,
            'ImpA': ImpA,
            'ImpB': ImpB,
            'yield': final_yield,
            'purity': purity,
            'total_impurities': total_impurities
        })
    
    return results

def create_pharmaceutical_plots(results):
    """Create comprehensive pharmaceutical analysis plots"""
    print("📊 Creating pharmaceutical analysis visualizations...")
    
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(18, 14))
    
    # Plot 1: API Formation Kinetics
    ax1 = plt.subplot(3, 3, 1)
    for result in results:
        condition = result['condition']
        ax1.plot(result['times'], result['API'], linewidth=3, 
                label=condition['name'], color=condition['color'], alpha=0.8)
    
    ax1.set_xlabel('Time (min)', fontsize=12)
    ax1.set_ylabel('API Concentration (M)', fontsize=12)
    ax1.set_title('💊 API Formation Kinetics', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Impurity Profile
    ax2 = plt.subplot(3, 3, 2)
    for result in results:
        condition = result['condition']
        total_imp = result['ImpA'] + result['ImpB']
        ax2.plot(result['times'], total_imp * 1000, linewidth=2, 
                linestyle='--', label=condition['name'], color=condition['color'], alpha=0.8)
    
    ax2.set_xlabel('Time (min)', fontsize=12)
    ax2.set_ylabel('Total Impurities (mM)', fontsize=12)
    ax2.set_title('⚠️ Impurity Formation', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Purity vs Yield
    ax3 = plt.subplot(3, 3, 3)
    yields = [r['yield'] for r in results]
    purities = [r['purity'] for r in results]
    colors = [r['condition']['color'] for r in results]
    names = [r['condition']['name'] for r in results]
    
    scatter = ax3.scatter(yields, purities, c=colors, s=200, alpha=0.8, edgecolors='black')
    for i, name in enumerate(names):
        ax3.annotate(name, (yields[i], purities[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax3.set_xlabel('Yield (%)', fontsize=12)
    ax3.set_ylabel('Purity (%)', fontsize=12)
    ax3.set_title('🎯 Quality Trade-off Analysis', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Reaction Progress (best condition)
    ax4 = plt.subplot(3, 3, 4)
    best_result = max(results, key=lambda x: x['yield'] * x['purity'])
    
    ax4.plot(best_result['times'], best_result['SM'], label='Starting Material', linewidth=2, color='orange')
    ax4.plot(best_result['times'], best_result['I1'], label='Intermediate 1', linewidth=2, color='blue')
    ax4.plot(best_result['times'], best_result['I2'], label='Intermediate 2', linewidth=2, color='green')
    ax4.plot(best_result['times'], best_result['API'], label='API', linewidth=3, color='red')
    
    ax4.set_xlabel('Time (min)', fontsize=12)
    ax4.set_ylabel('Concentration (M)', fontsize=12)
    ax4.set_title(f"📈 Reaction Progress - {best_result['condition']['name']}", fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Process Comparison Bar Chart
    ax5 = plt.subplot(3, 3, 5)
    x_pos = np.arange(len(results))
    
    ax5.bar(x_pos - 0.2, yields, 0.4, label='Yield (%)', alpha=0.8, color='skyblue')
    ax5.bar(x_pos + 0.2, purities, 0.4, label='Purity (%)', alpha=0.8, color='lightcoral')
    
    ax5.set_xlabel('Process Condition', fontsize=12)
    ax5.set_ylabel('Performance (%)', fontsize=12)
    ax5.set_title('📊 Process Performance Comparison', fontsize=14, fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([r['condition']['name'] for r in results], rotation=45)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Regulatory Compliance
    ax6 = plt.subplot(3, 3, 6)
    compliance_scores = []
    for result in results:
        # Regulatory scoring (simplified)
        yield_score = min(result['yield'] / 80 * 40, 40)  # Max 40 points for yield
        purity_score = min(result['purity'] / 95 * 50, 50)  # Max 50 points for purity
        impurity_score = max(10 - result['total_impurities'] * 1000, 0)  # Max 10 points for low impurities
        total_score = yield_score + purity_score + impurity_score
        compliance_scores.append(total_score)
    
    bars = ax6.bar(range(len(results)), compliance_scores, 
                  color=[r['condition']['color'] for r in results], alpha=0.8)
    ax6.axhline(y=85, color='red', linestyle='--', linewidth=2, label='Regulatory Threshold')
    ax6.set_xlabel('Process Condition', fontsize=12)
    ax6.set_ylabel('Compliance Score', fontsize=12)
    ax6.set_title('✅ Regulatory Compliance', fontsize=14, fontweight='bold')
    ax6.set_xticks(range(len(results)))
    ax6.set_xticklabels([r['condition']['name'] for r in results], rotation=45)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    # Add score labels
    for bar, score in zip(bars, compliance_scores):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 7: Cost Analysis
    ax7 = plt.subplot(3, 3, 7)
    # Simplified cost model
    raw_material_costs = [100] * len(results)  # Fixed raw material cost
    processing_costs = [r['condition']['temp'] - 300 for r in results]  # Higher temp = higher cost
    catalyst_costs = [r['condition']['catalyst'] * 200 for r in results]  # Catalyst cost
    total_costs = [rm + pc + cc for rm, pc, cc in zip(raw_material_costs, processing_costs, catalyst_costs)]
    
    ax7.bar(range(len(results)), raw_material_costs, label='Raw Materials', alpha=0.8, color='lightblue')
    ax7.bar(range(len(results)), processing_costs, bottom=raw_material_costs, 
           label='Processing', alpha=0.8, color='lightgreen')
    ax7.bar(range(len(results)), catalyst_costs, 
           bottom=[rm + pc for rm, pc in zip(raw_material_costs, processing_costs)],
           label='Catalyst', alpha=0.8, color='lightyellow')
    
    ax7.set_xlabel('Process Condition', fontsize=12)
    ax7.set_ylabel('Cost ($/batch)', fontsize=12)
    ax7.set_title('💰 Cost Breakdown Analysis', fontsize=14, fontweight='bold')
    ax7.set_xticks(range(len(results)))
    ax7.set_xticklabels([r['condition']['name'] for r in results], rotation=45)
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Time-Course Impurity Detail
    ax8 = plt.subplot(3, 3, 8)
    best_result = max(results, key=lambda x: x['purity'])
    
    ax8.plot(best_result['times'], best_result['ImpA'] * 1000, 
            label='Impurity A', linewidth=2, color='red', linestyle='--')
    ax8.plot(best_result['times'], best_result['ImpB'] * 1000, 
            label='Impurity B', linewidth=2, color='orange', linestyle='--')
    ax8.axhline(y=1.0, color='red', linestyle='-', alpha=0.5, label='Spec Limit (1 mM)')
    
    ax8.set_xlabel('Time (min)', fontsize=12)
    ax8.set_ylabel('Impurity Concentration (mM)', fontsize=12)
    ax8.set_title('🔍 Impurity Monitoring', fontsize=14, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Economic Optimization
    ax9 = plt.subplot(3, 3, 9)
    revenues = [r['yield'] * r['purity'] / 100 * 500 for r in results]  # Revenue model
    profits = [rev - cost for rev, cost in zip(revenues, total_costs)]
    
    ax9.plot(range(len(results)), revenues, 'go-', linewidth=3, label='Revenue', alpha=0.8)
    ax9.plot(range(len(results)), total_costs, 'ro-', linewidth=3, label='Total Cost', alpha=0.8)
    ax9.plot(range(len(results)), profits, 'bo-', linewidth=4, label='Profit', alpha=0.8)
    
    ax9.set_xlabel('Process Condition', fontsize=12)
    ax9.set_ylabel('Economic Value ($/batch)', fontsize=12)
    ax9.set_title('📈 Economic Optimization', fontsize=14, fontweight='bold')
    ax9.set_xticks(range(len(results)))
    ax9.set_xticklabels([r['condition']['name'] for r in results], rotation=45)
    ax9.legend(fontsize=10)
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pharmaceutical_synthesis_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: pharmaceutical_synthesis_analysis.png")
    
    return fig

def generate_pharma_report(results):
    """Generate pharmaceutical development report"""
    print("\n💊 PHARMACEUTICAL SYNTHESIS REPORT")
    print("=" * 50)
    
    best_yield = max(results, key=lambda x: x['yield'])
    best_purity = max(results, key=lambda x: x['purity'])
    best_overall = max(results, key=lambda x: x['yield'] * x['purity'])
    
    print(f"🎯 Best Yield: {best_yield['condition']['name']} - {best_yield['yield']:.1f}%")
    print(f"🎯 Best Purity: {best_purity['condition']['name']} - {best_purity['purity']:.1f}%")
    print(f"🎯 Best Overall: {best_overall['condition']['name']} - Combined Score: {best_overall['yield'] * best_overall['purity']:.0f}")
    
    print("\n📊 Process Condition Analysis:")
    for result in results:
        condition = result['condition']
        print(f"   {condition['name']}: Yield={result['yield']:.1f}%, Purity={result['purity']:.1f}%")
        print(f"      Temperature: {condition['temp']}K, Catalyst: {condition['catalyst']*100:.0f}%")
    
    print(f"\n💡 Recommendations:")
    print(f"   • Use {best_overall['condition']['name']} conditions for optimal balance")
    print(f"   • Monitor impurity formation closely during synthesis")
    print(f"   • Target >85% purity for regulatory compliance")
    print(f"   • Consider process optimization for cost reduction")

def main():
    """Main execution function"""
    print("🚀 PyroXa Pharmaceutical Synthesis Showcase")
    print("==========================================")
    
    # Run simulation
    results = run_pharmaceutical_synthesis()
    
    # Create visualizations
    fig = create_pharmaceutical_plots(results)
    
    # Generate report
    generate_pharma_report(results)
    
    # Save plots (don't show interactively to avoid hanging)
    print("💾 Plots saved to 'pharmaceutical_synthesis_analysis.png'")
    
    print("\n✨ Pharmaceutical synthesis analysis complete!")
    print("📊 Check 'pharmaceutical_synthesis_analysis.png' for detailed plots")

if __name__ == "__main__":
    main()
