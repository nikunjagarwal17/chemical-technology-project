# PyroXa Examples

This folder contains comprehensive examples demonstrating PyroXa's advanced capabilities.

## 📁 Examples Overview

### 🌟 Advanced Showcase Examples
- **[industrial_showcase.py](industrial_showcase.py)** - Complete industrial process analysis with temperature optimization, 6-panel visualizations, and real-world catalytic reactions
- **[pharmaceutical_showcase.py](pharmaceutical_showcase.py)** - Multi-step pharmaceutical synthesis with quality control, impurity tracking, and comprehensive analysis

### Interactive Learning
- **[simple_simulation.ipynb](simple_simulation.ipynb)** - Jupyter notebook tutorial for hands-on learning

### 🏭 Industrial Process Specifications
- **[specs/industrial_process.yaml](specs/industrial_process.yaml)** - Complex industrial catalytic process
- **[specs/pharmaceutical_synthesis.yaml](specs/pharmaceutical_synthesis.yaml)** - Multi-step pharmaceutical synthesis
- **[specs/environmental_catalysis.yaml](specs/environmental_catalysis.yaml)** - NOx reduction environmental process
- **[specs/advanced_chain.yaml](specs/advanced_chain.yaml)** - Advanced reaction chain
- **[specs/branching_network.yaml](specs/branching_network.yaml)** - Branching reaction network
- **[specs/simple_reaction.yaml](specs/simple_reaction.yaml)** - Basic reaction setup

### 🧬 Chemical Mechanism Definitions
- **[mechanisms/enzyme_catalysis.yaml](mechanisms/enzyme_catalysis.yaml)** - Michaelis-Menten enzyme kinetics
- **[mechanisms/polymerization_chain.yaml](mechanisms/polymerization_chain.yaml)** - Free radical polymerization
- **[mechanisms/simple_mechanism.yaml](mechanisms/simple_mechanism.yaml)** - Basic reaction mechanism

## 🚀 Quick Start

### Running Advanced Showcase Examples

```bash
# Navigate to examples folder
cd examples

# Run industrial process showcase (generates beautiful plots)
python industrial_showcase.py

# Run pharmaceutical synthesis analysis
python pharmaceutical_showcase.py

# Basic examples
python run_example.py
python run_cstr.py
python sample_display.py
## 🚀 Quick Start

### Running Advanced Showcase Examples

```bash
# Navigate to examples folder
cd examples

# Run industrial process showcase (generates beautiful plots)
python industrial_showcase.py

# Run pharmaceutical synthesis analysis
python pharmaceutical_showcase.py
```

### Jupyter Notebook Tutorial

```bash
# Start Jupyter notebook
jupyter notebook simple_simulation.ipynb
```

## 📚 What's Included

### 🌟 Showcase Examples (Visual Impact)
- **Industrial Process**: Temperature optimization, Arrhenius kinetics, multi-panel analysis with 6 comprehensive plots
- **Pharmaceutical Synthesis**: Multi-step reactions, quality control, impurity tracking with 9-panel visualization suite
- **Real-world Applications**: Catalytic processes, API synthesis, regulatory compliance modeling

### Key Features Demonstrated
- **Reactor Types**: Batch reactors, CSTR, PFR, multi-reactor networks
- **Reaction Systems**: Complex networks, autocatalytic systems, enzyme kinetics, polymerization
- **Analysis Methods**: Sensitivity analysis, optimization, statistical validation, economic evaluation
- **Advanced Visualizations**: Temperature profiles, concentration tracking, yield optimization, cost analysis

## 💡 Usage Patterns

### Basic Workflow
1. Define chemical reactions using YAML specifications
2. Set up reactor configuration with PyroXa
3. Run comprehensive simulations
4. Generate professional visualizations and analysis

### Advanced Features
- Multi-step synthesis pathways
- Temperature-dependent kinetics
- Quality control and impurity tracking
- Economic and regulatory analysis
- Publication-ready visualizations
## 🔧 Configuration Guide

### YAML Specification Format
The examples use sophisticated YAML specifications for complex processes:

```yaml
reactions:
  - equation: "Starting_Material -> Intermediate_1"
    kf: 2.5
    activation_energy: 45000
  - equation: "Intermediate_1 -> API"
    kf: 1.8
    activation_energy: 38000

initial_conditions:
  Starting_Material: 1.0
  temperature: 323.15

reactor:
  type: "BatchReactor"
  volume: 0.001
  temperature_profile: variable

simulation:
  time_span: 120.0
  temperature_range: [298, 373]
```

## 📊 Expected Outputs

The showcase examples generate comprehensive visualizations:
- **Industrial Showcase**: 6-panel analysis including temperature optimization, kinetics, economics
- **Pharmaceutical Showcase**: 9-panel suite covering synthesis pathways, quality control, regulatory compliance
- **Publication-ready plots** with professional styling and detailed annotations

## 🛠️ Customization

To create custom examples:
1. Copy an existing YAML specification from `specs/`
2. Modify reaction parameters and mechanisms
3. Adjust temperature profiles and initial conditions
4. Run the showcase scripts to generate comprehensive analysis

For detailed documentation of all 89 PyroXa functions, see the [`../docs/`](../docs/) folder.

---

*Start with the Jupyter notebook `simple_simulation.ipynb` for an interactive introduction, then explore the advanced showcase examples for comprehensive demonstrations of PyroXa's capabilities.*
