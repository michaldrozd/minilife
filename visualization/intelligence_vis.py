"""
Visualization for intelligence simulation with neural networks,
organism behaviors, and environmental interactions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge, FancyArrowPatch
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.visualizer import SimulationVisualizer


class IntelligenceVisualizer(SimulationVisualizer):
    """Visualizes intelligent organisms, neural networks, and environmental interactions."""
    def __init__(self, simulation, title="Intelligence Simulation", figsize=(14, 10)):
        super().__init__(title, figsize)
        self.simulation = simulation

        # Create figure with GridSpec for more flexible layout
        self.fig = plt.figure(figsize=figsize)
        self.gs = GridSpec(3, 3, figure=self.fig)

        # Main environment view (larger)
        self.main_ax = self.fig.add_subplot(self.gs[0:2, 0:2])

        # Stats panel
        self.stats_ax = self.fig.add_subplot(self.gs[0, 2])

        # Neural network visualization
        self.network_ax = self.fig.add_subplot(self.gs[1, 2])

        # Organism traits visualization
        self.traits_ax = self.fig.add_subplot(self.gs[2, 0])

        # Learning visualization
        self.learning_ax = self.fig.add_subplot(self.gs[2, 1])

        # Population dynamics
        self.population_ax = self.fig.add_subplot(self.gs[2, 2])

        # Set up main visualization (organisms and environment)
        self.main_ax.set_xlim(0, simulation.grid_size)
        self.main_ax.set_ylim(0, simulation.grid_size)
        self.main_ax.set_title("Organisms and Environment")

        # Initialize environment visualization (nutrient field)
        self.nutrient_im = self.main_ax.imshow(
            np.zeros((simulation.grid_size, simulation.grid_size)),
            cmap='YlGn', origin='lower',
            extent=[0, simulation.grid_size, 0, simulation.grid_size],
            alpha=0.6, vmin=0, vmax=1
        )

        # Initialize organism visualization
        self.organism_patches = []
        self.organism_collection = PatchCollection([], alpha=0.7)
        self.main_ax.add_collection(self.organism_collection)

        # Vision cones for organisms
        self.vision_patches = []
        self.vision_collection = PatchCollection([], alpha=0.3, edgecolor=None)
        self.main_ax.add_collection(self.vision_collection)

        # Movement vectors
        self.movement_arrows = []

        # Neural network visualization
        self.network_ax.set_title("Neural Network")
        self.network_ax.set_xticks([])
        self.network_ax.set_yticks([])

        # For visualizing the neural network of a selected organism
        self.selected_organism_idx = 0
        self.network_nodes = []
        self.network_edges = []
        self.network_edge_collection = LineCollection([], alpha=0.6, linewidths=1)
        self.network_ax.add_collection(self.network_edge_collection)

        # Stats visualization
        self.stats_ax.set_title("Simulation Statistics")
        self.stats_text = self.stats_ax.text(
            0.05, 0.95, "", transform=self.stats_ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        self.stats_ax.set_xticks([])
        self.stats_ax.set_yticks([])

        # Traits visualization
        self.traits_ax.set_title("Organism Traits")
        self.traits_ax.set_xlim(0, 1)
        self.traits_ax.set_ylim(0, 1)
        self.traits_bars = None

        # Learning visualization
        self.learning_ax.set_title("Learning Progress")
        self.learning_ax.set_xlabel("Iteration")
        self.learning_ax.set_ylabel("Neural Complexity")
        self.learning_line, = self.learning_ax.plot([], [], 'r-', linewidth=2)

        # Population dynamics
        self.population_ax.set_title("Population")
        self.population_ax.set_xlabel("Iteration")
        self.population_ax.set_ylabel("Count")
        self.population_line, = self.population_ax.plot([], [], 'b-', linewidth=2)

        # Color maps
        self.trait_colors = {
            'aggression': 'red',
            'sociability': 'green',
            'exploration': 'blue',
            'metabolism': 'orange'
        }

        # Adjust layout
        self.fig.tight_layout()

    def update_visualization(self, frame):
        """Update the visualization for the current frame."""
        # Update environment visualization
        env_data = self.simulation.get_environment_data()
        self.nutrient_im.set_data(env_data['nutrient'])

        # Update organism visualization
        self.update_organism_visualization()

        # Update neural network visualization
        self.update_neural_network_visualization()

        # Update traits visualization
        self.update_traits_visualization()

        # Update learning visualization
        self.update_learning_visualization()

        # Update population visualization
        self.update_population_visualization()

        # Update stats text
        self.update_stats_text(frame)

        # Return all artists that were updated
        artists = [
            self.nutrient_im,
            self.organism_collection,
            self.vision_collection,
            self.stats_text,
            self.learning_line,
            self.population_line
        ]
        artists.extend(self.movement_arrows)

        return artists

    def update_organism_visualization(self):
        """Update the visualization of organisms."""
        # Clear previous patches
        self.organism_patches.clear()
        self.vision_patches.clear()

        # Remove previous movement arrows
        for arrow in self.movement_arrows:
            arrow.remove()
        self.movement_arrows.clear()

        # Get organism data
        organism_data = self.simulation.get_organism_data()
        positions = organism_data['positions']
        energies = organism_data['energies']
        fitnesses = organism_data['fitnesses']

        if len(positions) == 0:
            # No organisms to display
            self.organism_collection.set_paths([])
            self.vision_collection.set_paths([])
            return

        # Create patches for organisms
        colors = []
        for i, pos in enumerate(positions):
            # Create a circle for the organism
            radius = 3 + fitnesses[i] * 3  # Size based on fitness
            circle = Circle(pos, radius=radius)
            self.organism_patches.append(circle)

            # Color based on energy
            energy_normalized = min(1.0, energies[i] / 100.0)
            colors.append(energy_normalized)

            # Add vision cone
            if i < len(self.simulation.organisms):
                organism = self.simulation.organisms[i]

                # Create vision cones (3 directions as in the neural network)
                directions = [
                    [1, 0],      # Forward
                    [0.7, 0.7],  # Forward-right
                    [0.7, -0.7]  # Forward-left
                ]

                for direction in directions:
                    # Normalize direction
                    direction = np.array(direction) / np.linalg.norm(direction)

                    # Create a wedge for the vision cone
                    angle = np.degrees(np.arctan2(direction[1], direction[0]))
                    wedge = Wedge(
                        pos, r=10,  # r is the required radius parameter
                        theta1=angle-15, theta2=angle+15,
                        width=9
                    )
                    self.vision_patches.append(wedge)

                # Add movement vector
                if hasattr(organism, 'action_history') and organism.action_history:
                    last_action = organism.action_history[-1]
                    if len(last_action) >= 2:
                        # Extract movement direction and speed
                        angle = last_action[0] * 2 * np.pi
                        speed = last_action[1] * 5  # Scale for visibility

                        # Calculate movement vector
                        dx = speed * np.cos(angle)
                        dy = speed * np.sin(angle)

                        # Create arrow
                        arrow = FancyArrowPatch(
                            pos, (pos[0] + dx, pos[1] + dy),
                            arrowstyle='->',
                            mutation_scale=10,
                            color='white',
                            linewidth=1.5
                        )
                        self.main_ax.add_patch(arrow)
                        self.movement_arrows.append(arrow)

        # Update the collections
        self.organism_collection.set_paths(self.organism_patches)
        self.organism_collection.set_array(np.array(colors))
        self.organism_collection.set_cmap('plasma')
        self.organism_collection.set_clim(0, 1)

        self.vision_collection.set_paths(self.vision_patches)
        self.vision_collection.set_facecolor('cyan')

    def update_neural_network_visualization(self):
        """Update the neural network visualization."""
        # Clear the axis
        self.network_ax.clear()
        self.network_ax.set_title("Neural Network")
        self.network_ax.set_xticks([])
        self.network_ax.set_yticks([])

        # Get organisms
        organisms = self.simulation.organisms

        # Skip if no organisms
        if not organisms:
            return

        # Update selected organism index if needed
        if self.selected_organism_idx >= len(organisms):
            self.selected_organism_idx = 0

        # Change selected organism periodically
        if self.simulation.iteration % 20 == 0 and len(organisms) > 1:
            self.selected_organism_idx = (self.selected_organism_idx + 1) % len(organisms)

        # Get the selected organism
        organism = organisms[self.selected_organism_idx]

        # Get neural network state
        nn_state = organism.neural_network.get_network_state()

        # Define layer sizes and positions
        input_size = len(nn_state['input_activation'])
        hidden1_size = len(nn_state['hidden1_activation'])
        hidden2_size = len(nn_state['hidden2_activation'])
        output_size = len(nn_state['output_activation'])

        # Calculate positions for each layer
        input_pos = [(0.2, 0.1 + 0.8 * i / max(1, input_size - 1)) for i in range(input_size)]
        hidden1_pos = [(0.4, 0.1 + 0.8 * i / max(1, hidden1_size - 1)) for i in range(hidden1_size)]
        hidden2_pos = [(0.6, 0.1 + 0.8 * i / max(1, hidden2_size - 1)) for i in range(hidden2_size)]
        output_pos = [(0.8, 0.1 + 0.8 * i / max(1, output_size - 1)) for i in range(output_size)]

        # Draw connections between layers
        segments = []
        connection_colors = []

        # Input to Hidden1 connections
        for i in range(input_size):
            for j in range(hidden1_size):
                weight = organism.neural_network.weights_input_hidden1[i, j]
                segments.append([input_pos[i], hidden1_pos[j]])
                connection_colors.append(weight)

        # Hidden1 to Hidden2 connections
        for i in range(hidden1_size):
            for j in range(hidden2_size):
                weight = organism.neural_network.weights_hidden1_hidden2[i, j]
                segments.append([hidden1_pos[i], hidden2_pos[j]])
                connection_colors.append(weight)

        # Hidden2 to Output connections
        for i in range(hidden2_size):
            for j in range(output_size):
                weight = organism.neural_network.weights_hidden2_output[i, j]
                segments.append([hidden2_pos[i], output_pos[j]])
                connection_colors.append(weight)

        # Create line collection for connections
        if segments:
            line_collection = LineCollection(
                segments,
                cmap='coolwarm',
                linewidths=1,
                alpha=0.6
            )
            line_collection.set_array(np.array(connection_colors))
            line_collection.set_clim(-0.5, 0.5)  # Set color limits for weights
            self.network_ax.add_collection(line_collection)

        # Draw nodes
        # Input layer
        for i, pos in enumerate(input_pos):
            activation = nn_state['input_activation'][i]
            circle = plt.Circle(pos, 0.02, color='blue', alpha=0.5 + 0.5 * activation)
            self.network_ax.add_patch(circle)
            if i == 0:
                self.network_ax.text(pos[0] - 0.1, 0.5, "Input", ha='right', va='center')

        # Hidden1 layer
        for i, pos in enumerate(hidden1_pos):
            activation = nn_state['hidden1_activation'][i]
            circle = plt.Circle(pos, 0.02, color='green', alpha=0.5 + 0.5 * activation)
            self.network_ax.add_patch(circle)
            if i == 0:
                self.network_ax.text(pos[0], 0.02, "Hidden 1", ha='center', va='bottom')

        # Hidden2 layer
        for i, pos in enumerate(hidden2_pos):
            activation = nn_state['hidden2_activation'][i]
            circle = plt.Circle(pos, 0.02, color='purple', alpha=0.5 + 0.5 * activation)
            self.network_ax.add_patch(circle)
            if i == 0:
                self.network_ax.text(pos[0], 0.02, "Hidden 2", ha='center', va='bottom')

        # Output layer
        for i, pos in enumerate(output_pos):
            activation = nn_state['output_activation'][i]
            circle = plt.Circle(pos, 0.02, color='red', alpha=0.5 + 0.5 * activation)
            self.network_ax.add_patch(circle)
            if i == 0:
                self.network_ax.text(pos[0] + 0.1, 0.5, "Output", ha='left', va='center')

        # Set axis limits
        self.network_ax.set_xlim(0, 1)
        self.network_ax.set_ylim(0, 1)

        # Add neuromodulator information
        neuromodulators = nn_state['neuromodulators']
        y_pos = 0.95
        for name, value in neuromodulators.items():
            self.network_ax.text(
                0.5, y_pos,
                f"{name.capitalize()}: {value:.2f}",
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
            )
            y_pos -= 0.05

    def update_traits_visualization(self):
        """Update the traits visualization."""
        # Clear the axis
        self.traits_ax.clear()
        self.traits_ax.set_title("Organism Traits")

        # Get organisms
        organisms = self.simulation.organisms

        # Skip if no organisms
        if not organisms:
            self.traits_ax.text(0.5, 0.5, "No organisms", ha='center', va='center')
            return

        # Calculate average traits
        avg_traits = {trait: 0 for trait in self.trait_colors}
        for organism in organisms:
            for trait, value in organism.traits.items():
                avg_traits[trait] += value / len(organisms)

        # Create horizontal bar chart
        traits = list(avg_traits.keys())
        values = [avg_traits[trait] for trait in traits]
        colors = [self.trait_colors[trait] for trait in traits]

        y_pos = np.arange(len(traits))
        self.traits_ax.barh(y_pos, values, color=colors, alpha=0.7)
        self.traits_ax.set_yticks(y_pos)
        self.traits_ax.set_yticklabels([t.capitalize() for t in traits])
        self.traits_ax.set_xlim(0, 1)
        self.traits_ax.set_xlabel("Average Value")

        # Add most extreme organism for each trait
        for i, trait in enumerate(traits):
            # Find organism with highest value for this trait
            max_org = max(organisms, key=lambda org: org.traits[trait])
            max_val = max_org.traits[trait]

            # Add a marker for the max value
            self.traits_ax.plot(max_val, i, 'o', color='black', markersize=6)

            # Add text label
            self.traits_ax.text(
                max_val, i, f" Max: {max_val:.2f}",
                va='center', ha='left', fontsize=8
            )

    def update_learning_visualization(self):
        """Update the learning progress visualization."""
        # Get stats
        stats = self.simulation.stats

        # Skip if no data
        if not stats['neural_complexity']:
            return

        # Plot neural complexity over time
        iterations = list(range(len(stats['neural_complexity'])))
        self.learning_line.set_data(iterations, stats['neural_complexity'])

        # Update axis limits
        if iterations:
            self.learning_ax.set_xlim(0, max(iterations))
            self.learning_ax.set_ylim(0, max(stats['neural_complexity']) * 1.1)

    def update_population_visualization(self):
        """Update the population dynamics visualization."""
        # Get stats
        stats = self.simulation.stats

        # Skip if no data
        if not stats['organism_count']:
            return

        # Plot population over time
        iterations = list(range(len(stats['organism_count'])))
        self.population_line.set_data(iterations, stats['organism_count'])

        # Update axis limits
        if iterations:
            self.population_ax.set_xlim(0, max(iterations))
            self.population_ax.set_ylim(0, max(stats['organism_count']) * 1.1)

    def update_stats_text(self, frame):
        """Update the statistics text."""
        # Get stats
        stats = self.simulation.stats

        # Create stats text
        stats_str = (
            f"Iteration: {self.simulation.iteration}\n"
            f"Organisms: {stats['organism_count'][-1] if stats['organism_count'] else 0}\n"
            f"Births: {stats['births']}\n"
            f"Deaths: {stats['deaths']}\n\n"
        )

        # Add information about selected organism
        organisms = self.simulation.organisms
        if organisms and self.selected_organism_idx < len(organisms):
            org = organisms[self.selected_organism_idx]
            stats_str += (
                f"Selected Organism #{self.selected_organism_idx}:\n"
                f"Energy: {org.energy:.1f}\n"
                f"Fitness: {org.fitness:.2f}\n"
                f"Age: {org.age}\n"
                f"Traits:\n"
            )

            for trait, value in org.traits.items():
                stats_str += f"  {trait.capitalize()}: {value:.2f}\n"

        self.stats_text.set_text(stats_str)

    def animate(self, frames=300, interval=50):
        """Animate the simulation."""
        self.animation = FuncAnimation(
            self.fig, self.update_visualization, frames=frames,
            interval=interval, blit=True
        )
        return self.animation


def intelligence_visualization_callback(simulation, frame):
    """
    Callback function for visualization updates during intelligence simulation.

    Args:
        simulation: The current simulation state
        frame: The current frame number
    """
    # Print periodic updates
    if frame % 10 == 0:
        org_count = len(simulation.organisms)
        if org_count > 0:
            mean_energy = np.mean([org.energy for org in simulation.organisms])
            mean_fitness = np.mean([org.fitness for org in simulation.organisms])
            print(f"Intelligence Iteration {frame}: {org_count} organisms, " +
                  f"Mean energy: {mean_energy:.1f}, Mean fitness: {mean_fitness:.2f}")
        else:
            print(f"Intelligence Iteration {frame}: All organisms have died.")


def visualize_intelligence_simulation(simulation, frames=300, interval=100):
    """
    Set up and display visualization for an intelligence simulation.

    Args:
        simulation: The simulation to visualize
        frames: Number of frames to animate
        interval: Interval between frames in milliseconds

    Returns:
        The visualizer object
    """
    visualizer = IntelligenceVisualizer(simulation)
    visualizer.animate(frames=frames, interval=interval)
    visualizer.show()
    return visualizer
