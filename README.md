# Advanced Evolution Simulation

A comprehensive biological evolution simulator that models the progression from simple chemical systems to intelligent multicellular organisms. This simulation implements detailed biophysics, cellular processes, and emergent behaviors with a modular, extensible architecture.

![Evolution Simulation](https://example.com/simulation-preview.png)

## 🧬 Overview

This project simulates the key transitions in biological evolution:

1. **Abiogenesis**: Formation of complex chemicals from simple precursors
2. **Cellular Life**: Emergence of autonomous cells with metabolism and replication
3. **Multicellularity**: Development of specialized tissues and organs through cell cooperation
4. **Intelligence**: Emergence of neural networks and adaptive behaviors

The simulation prioritizes biological realism through detailed implementation of:
- Reaction-diffusion systems for chemical environments
- Physical cell mechanics with deformable membranes
- Gene regulatory networks and protein synthesis
- Cell differentiation and tissue formation
- Neural signaling and sensory processing

## ✨ Key Features

### 📊 Sophisticated Biophysics

- **Advanced Reaction-Diffusion Systems**: Multi-species chemical environments modeled with partial differential equations
- **Deformable Cell Membranes**: Spring-mass systems for realistic cell shapes and mechanical properties
- **Physical Forces & Interactions**: Cell adhesion, collision detection and resolution
- **Chemical Gradients**: Realistic diffusion and signaling molecule propagation

### 🦠 Detailed Cellular Processes

- **Gene Regulatory Networks**: Simulate gene expression controlled by protein interactions
- **Protein Synthesis**: Transcription and translation with regulatory feedback
- **Metabolic Pathways**: Network-based metabolism with energy production and consumption
- **Signaling Cascades**: Internal and external signal processing affecting cell behavior

### 🧠 Emergent Complexity

- **Cell Differentiation**: Spatial and neighbor-dependent specialization into different cell types
- **Tissue Formation**: Cell adhesion properties creating emergent organization
- **Primitive Neural Systems**: Connected networks of signaling nerve cells
- **Adaptive Behaviors**: Environment-responsive movement and resource gathering

### 🎨 Detailed Visualization

- **Cellular Visualization**: Cell membranes, specialization, and internal states
- **Neural Network Display**: Visualization of neural connectivity and signal propagation
- **Environmental Rendering**: Chemical gradients and resource distribution
- **Statistics Tracking**: Population dynamics, complexity metrics, and evolutionary trends

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Core dependencies: NumPy, SciPy, Matplotlib, Numba

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/evolution-simulation.git
cd evolution-simulation

# Install dependencies
pip install -r requirements.txt
```

### Running the Simulation

```bash
# Full evolutionary sequence
python main.py

# Run specific phases
python main.py --phase abiogenesis
python main.py --phase cell
python main.py --phase multicellular
python main.py --phase intelligence

# Adjust simulation length
python main.py --iterations 500

# Run in non-interactive mode (no prompts)
python main.py --non-interactive

# Visualization only
python main.py --phase visualize --vis-type cell
python main.py --phase visualize --vis-type multicell
```

## 📂 Project Structure

```
/simulation/
│
├── abiogenesis/           # Chemical evolution simulation
│   ├── chemical.py        # Chemical compound models
│   └── simulation.py      # Abiogenesis simulation controller
│
├── cellular/              # Single-cell life simulation
│   ├── cell.py            # Advanced cell model with internal processes
│   ├── metabolism.py      # Metabolic pathway simulation
│   ├── genome.py          # Gene regulation models
│   ├── signaling.py       # Cell signaling pathways
│   └── simulation.py      # Single-cell simulation controller
│
├── multicellular/         # Multicellular organism simulation
│   ├── organism.py        # Multicellular organism model
│   ├── differentiation.py # Cell specialization mechanics
│   ├── adhesion.py        # Cell-cell adhesion physics
│   └── simulation.py      # Multicellular simulation controller
│
├── physics/               # Core physics engines
│   ├── diffusion.py       # Reaction-diffusion systems
│   ├── mechanics.py       # Cell mechanics and deformation
│   └── collision.py       # Collision detection and resolution
│
├── visualization/         # Visualization components
│   ├── visualizer.py      # Base visualization utilities
│   ├── cell_vis.py        # Cell visualization
│   └── multicell_vis.py   # Multicellular visualization
│
├── utils/                 # Utility functions
│   └── helpers.py         # Shared helper functions
│
├── config.py              # Simulation configuration parameters
├── main.py                # Main simulation entry point
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## 🔬 Scientific Background

This simulation implements several key scientific models:

- **Gray-Scott Reaction-Diffusion**: Models pattern formation in chemical systems
- **Gene Regulatory Networks**: Boolean and continuous models of gene expression
- **Cellular Potts Model Principles**: For cell shape, adhesion, and movement
- **Signaling Network Models**: Based on systems biology approaches
- **Artificial Neural Networks**: With spatial organization and signal propagation

## 🛠️ Customization

The simulation can be extensively customized through:

1. **Configuration Parameters**: Adjust key settings in `config.py`
2. **Environmental Conditions**: Modify nutrient availability and distribution
3. **Cell Properties**: Customize gene networks, metabolic rates, and signaling sensitivity
4. **Evolutionary Pressures**: Change selection mechanisms and mutation rates

## 📊 Example Results

### Single Cell Simulation

```
Iteration 250: 93 cells, mean energy: 16.64, mean DNA complexity: 9.84
Cell population threshold reached!

Cellular simulation summary:
Total cells: 152
Mean energy: 16.52
Mean DNA complexity: 9.89
Cell types: undifferentiated: 152
```

### Multicellular Simulation

```
Multicellular simulation summary:
Total organisms: 6
Total cells: 10
Mean organism size: 1.67 cells
Mean complexity: 10.16
Organisms with neural systems: 0

Organism 0:
  Cells: 1
  Energy: 24.6
  Cell types: nerve: 1
  Neural cells: 0
```

## 🔭 Future Development

- **Enhanced Abiogenesis Simulation**: Autocatalytic networks and polymerization
- **Spatial Genetics**: Geographic isolation and local adaptation
- **Sexual Reproduction**: Genetic recombination and mating strategies
- **Advanced Neural Systems**: Learning, memory, and behavior optimization
- **Evolutionary Innovations**: Major transitions like photosynthesis or multicellularity

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📚 References

- A. Lindenmayer and P. Prusinkiewicz, "The Algorithmic Beauty of Plants"
- S. Kauffman, "The Origins of Order: Self-Organization and Selection in Evolution"
- J. von Neumann, "Theory of Self-Reproducing Automata"
- I. R. Epstein and J. A. Pojman, "An Introduction to Nonlinear Chemical Dynamics"
- S. A. Newman, "Dynamical Patterning Modules"