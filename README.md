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
- **Complex Environmental Patterns**: Multiple environment types including ecosystems, gradients, patches, and Turing patterns
- **Seasonal Cycles**: Environmental changes that affect diffusion and reaction rates
- **Random Environmental Events**: Occasional disasters and resource booms that challenge adaptation
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
- **Neural Networks**: Advanced neural systems with learning and adaptation
- **Adaptive Behaviors**: Environment-responsive movement and resource gathering
- **Hebbian Learning**: Neural networks that adapt based on experience ("neurons that fire together, wire together")
- **Curiosity-Driven Exploration**: Organisms can develop exploratory behaviors based on novelty
- **Memory and Experience Replay**: Organisms can remember and learn from past experiences

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
├── intelligence/          # Intelligence simulation
│   ├── neural_network.py  # Advanced neural network with learning
│   └── simulation.py      # Intelligence simulation controller
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
- **Turing Patterns**: Self-organizing pattern formation in reaction-diffusion systems
- **Gene Regulatory Networks**: Boolean and continuous models of gene expression
- **Cellular Potts Model Principles**: For cell shape, adhesion, and movement
- **Signaling Network Models**: Based on systems biology approaches
- **Hebbian Learning**: Biologically-inspired learning mechanism for neural networks
- **Neuromodulation**: Chemical signals that modify neural network behavior
- **Artificial Neural Networks**: With spatial organization and signal propagation
- **Curiosity-Driven Learning**: Intrinsic motivation systems for exploration

## 🛠️ Customization

The simulation can be extensively customized through:

1. **Configuration Parameters**: Adjust key settings in `config.py`
2. **Environmental Patterns**: Choose between ecosystem, gradient, patches, or Turing patterns
3. **Neural Network Parameters**: Customize learning rates, network architecture, and memory capacity
4. **Environmental Conditions**: Modify nutrient availability, seasonal effects, and random events
5. **Cell Properties**: Customize gene networks, metabolic rates, and signaling sensitivity
6. **Evolutionary Pressures**: Change selection mechanisms and mutation rates
7. **Organism Traits**: Adjust aggression, sociability, exploration, and metabolism traits

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
Total organisms: 8
Total cells: 34
Mean organism size: 4.25 cells
Mean complexity: 10.00
Organisms with neural systems: 1

Organism 0:
  Cells: 6
  Energy: 52.6
  Cell types: muscle: 4, nerve: 2
  Neural cells: 0

Organism 2:
  Cells: 6
  Energy: 79.9
  Cell types: nerve: 3, skin: 3
  Neural cells: 3
```

### Intelligence Simulation

```
Intelligence Iteration 0: 5 organisms, Mean energy: 99.3, Mean fitness: 0.10
Intelligence Iteration 10: 5 organisms, Mean energy: 91.8, Mean fitness: 0.65
Intelligence Iteration 20: 11 organisms, Mean energy: 37.0, Mean fitness: 0.36

Most fit organism: Fitness=0.32, Age=150, Energy=24.1
Traits: aggression=0.37, sociability=0.77, exploration=0.99, metabolism=0.12
```

## 🔭 Future Development

- **Enhanced Abiogenesis Simulation**: Autocatalytic networks and polymerization
- **Spatial Genetics**: Geographic isolation and local adaptation
- **Sexual Reproduction**: Genetic recombination and mating strategies
- **Reinforcement Learning**: More advanced learning algorithms for neural networks
- **Social Dynamics**: More complex interactions between organisms including cooperation and competition
- **Predator-Prey Relationships**: Food web dynamics and co-evolutionary arms races
- **Genetic Programming**: Evolution of neural network architecture itself
- **Developmental Processes**: Growth and development stages for organisms
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