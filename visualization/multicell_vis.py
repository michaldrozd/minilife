"""
Visualization for multicellular organisms with tissue organization,
nervous systems, and emergent behaviors.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Circle, FancyArrowPatch
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.visualizer import SimulationVisualizer


class MulticellularVisualizer(SimulationVisualizer):
    """Visualizes multicellular organisms, tissue structures, and neural networks."""
    def __init__(self, simulation, title="Multicellular Simulation", figsize=(12, 12)):
        super().__init__(title, figsize)
        self.simulation = simulation
        
        # Create figure and axes
        self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        self.fig.suptitle(title, fontsize=16)
        self.main_ax = self.axes[0, 0]
        self.stats_ax = self.axes[0, 1]
        self.detail_ax = self.axes[1, 0]
        self.network_ax = self.axes[1, 1]
        
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
        
        # Initialize cell visualization
        self.cell_scatter = self.main_ax.scatter([], [], c=[], cmap='viridis', s=30, alpha=0.8)
        
        # Create a colormap for organism IDs
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        self.organism_cmap = LinearSegmentedColormap.from_list('organism_cmap', colors, N=100)
        
        # Setup for cell type colors
        self.cell_type_colors = {
            'undifferentiated': 'blue',
            'nerve': 'yellow',
            'muscle': 'red',
            'skin': 'green'
        }
        
        # Initialize neural network visualization in the network pane
        self.network_ax.set_xlim(0, simulation.grid_size)
        self.network_ax.set_ylim(0, simulation.grid_size)
        self.network_ax.set_title("Neural Networks")
        self.network_lines = LineCollection([], colors='orange', alpha=0.6, linewidths=1)
        self.network_ax.add_collection(self.network_lines)
        self.neuron_scatter = self.network_ax.scatter([], [], c=[], cmap='plasma', s=50, alpha=0.8)
        
        # Stats visualization setup
        self.stats_ax.set_title("Simulation Statistics")
        self.stats_ax.set_xlabel("Iteration")
        self.stats_ax.set_ylabel("Count")
        self.stats_lines = {}
        
        # Detail visualization setup (focus on a single organism)
        self.detail_ax.set_title("Organism Detail View")
        self.detail_ax.set_xlim(0, 50)
        self.detail_ax.set_ylim(0, 50)
        self.detail_patches = []
        self.detail_collection = PatchCollection(self.detail_patches, alpha=0.7)
        self.detail_ax.add_collection(self.detail_collection)
        self.detail_text = self.detail_ax.text(
            0.02, 0.95, "", transform=self.detail_ax.transAxes,
            fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Neural connections for detail view
        self.detail_connections = LineCollection([], colors='magenta', alpha=0.7, linewidths=1.5)
        self.detail_ax.add_collection(self.detail_connections)
        
        # Stats text
        self.stats_text = self.main_ax.text(
            0.02, 0.95, "", transform=self.main_ax.transAxes,
            fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Initialize focus organism index
        self.focus_organism_index = 0
        
        # Animation object
        self.animation = None
    
    def update_visualization(self, frame):
        """Update the visualization for the current frame."""
        # Update environment visualization
        env_data = self.simulation.get_environment_data()
        self.nutrient_im.set_data(env_data['nutrient'])
        
        # Update cell visualization
        cell_data = self.simulation.get_all_cell_data()
        
        if len(cell_data['positions']) > 0:
            self.cell_scatter.set_offsets(cell_data['positions'])
            
            # Color cells by organism ID
            self.cell_scatter.set_array(cell_data['organism_ids'])
            self.cell_scatter.set_cmap(self.organism_cmap)
            
            # Adjust size by energy
            sizes = np.clip(cell_data['energies'] / 10, 5, 50)
            self.cell_scatter.set_sizes(sizes)
            
            # Set marker based on cell type
            markers = []
            for cell_type in cell_data['types']:
                if cell_type == 'nerve':
                    markers.append('^')  # triangle for nerve cells
                elif cell_type == 'muscle':
                    markers.append('s')  # square for muscle cells
                elif cell_type == 'skin':
                    markers.append('h')  # hexagon for skin cells
                else:
                    markers.append('o')  # circle for undifferentiated
            
            # Update marker styles (using a hack since set_marker doesn't support arrays)
            # We'll recreate the scatter with the new markers
            if len(markers) > 0:
                colors = self.cell_scatter.get_array()
                sizes = self.cell_scatter.get_sizes()
                cmap = self.cell_scatter.get_cmap()
                
                # Remove old scatter
                self.cell_scatter.remove()
                
                # Create new scatter with proper markers
                # Since we can't set different markers easily, we'll create separate scatters
                marker_types = set(markers)
                for marker in marker_types:
                    indices = [i for i, m in enumerate(markers) if m == marker]
                    if indices:
                        pos = cell_data['positions'][indices]
                        if len(pos) > 0:
                            # Create scatter for this marker type
                            scatter = self.main_ax.scatter(
                                pos[:, 0], pos[:, 1],
                                c=colors[indices] if len(colors) > 0 else 'blue',
                                s=sizes[indices] if len(sizes) > 0 else 30,
                                cmap=cmap,
                                marker=marker,
                                alpha=0.8
                            )
                
                # Update reference
                #self.cell_scatter = scatter
        else:
            self.cell_scatter.set_offsets(np.zeros((0, 2)))
        
        # Update neural network visualization
        self.update_neural_visualization()
        
        # Update stats visualization
        self.update_stats_visualization()
        
        # Update detail visualization (focus on one organism)
        self.update_detail_visualization()
        
        # Update stats text
        stats = self.simulation.get_stats()
        stats_str = (
            f"Iteration: {self.simulation.iteration}\n"
            f"Organisms: {stats['organism_count'][-1] if stats['organism_count'] else 0}\n"
            f"Total Cells: {stats['total_cells'][-1] if stats['total_cells'] else 0}\n"
            f"Births: {stats['births']}\n"
            f"Deaths: {stats['deaths']}"
        )
        self.stats_text.set_text(stats_str)
        
        # Update title
        self.main_ax.set_title(f"Organisms and Environment - Frame {frame}")
        
        # Return the updated artists
        return [
            self.nutrient_im, self.cell_scatter, 
            self.stats_text, self.detail_collection,
            self.network_lines, self.neuron_scatter,
            self.detail_connections, self.detail_text
        ]
    
    def update_neural_visualization(self):
        """Update the neural network visualization."""
        # Get neural networks from all organisms
        networks = self.simulation.get_neural_networks()
        
        if not networks:
            # No neural networks to display
            self.network_lines.set_segments([])
            self.neuron_scatter.set_offsets(np.zeros((0, 2)))
            return
        
        # Collect segments from all networks
        all_segments = []
        all_neuron_positions = []
        all_activations = []
        
        for network in networks:
            positions = network['positions']
            connections = network['connections']
            activations = network['activations']
            
            # Skip if no data
            if len(positions) == 0 or len(connections) == 0:
                continue
            
            # Add segments for connections
            for i, j in connections:
                all_segments.append([positions[i], positions[j]])
            
            # Add neuron positions and activations
            all_neuron_positions.extend(positions)
            all_activations.extend(activations)
        
        # Update the network line collection
        self.network_lines.set_segments(all_segments)
        
        # Update neuron scatter
        if all_neuron_positions:
            self.neuron_scatter.set_offsets(all_neuron_positions)
            self.neuron_scatter.set_array(np.array(all_activations))
            self.neuron_scatter.set_clim(0, 1)
        else:
            self.neuron_scatter.set_offsets(np.zeros((0, 2)))
    
    def update_stats_visualization(self):
        """Update the statistics visualization."""
        # Get simulation statistics
        stats = self.simulation.get_stats()
        
        # Clear previous plots
        self.stats_ax.clear()
        self.stats_ax.set_title("Simulation Statistics")
        self.stats_ax.set_xlabel("Iteration")
        self.stats_ax.set_ylabel("Count")
        
        # Plot statistics over time
        iterations = range(len(stats['organism_count']))
        if iterations:
            self.stats_ax.plot(iterations, stats['organism_count'], 'b-', label='Organisms')
            self.stats_ax.plot(iterations, stats['mean_organism_size'], 'g-', label='Mean Size')
            if stats['mean_neuron_count'][-1] > 0:
                self.stats_ax.plot(iterations, stats['mean_neuron_count'], 'r-', label='Mean Neurons')
            self.stats_ax.plot(iterations, stats['mean_organism_complexity'], 'y-', label='Mean Complexity')
            
            self.stats_ax.legend(loc='upper left')
            self.stats_ax.grid(True, alpha=0.3)
    
    def update_detail_visualization(self):
        """Update the detailed view of a single organism."""
        # Clear previous content
        self.detail_ax.clear()
        self.detail_ax.set_title("Organism Detail View")
        
        # Get organisms
        organisms = self.simulation.organisms
        
        # Skip if no organisms
        if not organisms:
            return
        
        # Adjust focus organism index if needed
        if self.focus_organism_index >= len(organisms):
            self.focus_organism_index = 0
        
        # Skip to next organism every few frames
        if self.simulation.iteration % 30 == 0 and len(organisms) > 1:
            self.focus_organism_index = (self.focus_organism_index + 1) % len(organisms)
        
        # Get the focus organism
        organism = organisms[self.focus_organism_index]
        
        # Get cell properties
        cell_props = organism.get_cell_properties()
        
        # Skip if no cells
        if len(cell_props['positions']) == 0:
            return
        
        # Get organism's bounding box
        positions = cell_props['positions']
        min_pos = np.min(positions, axis=0) - 5
        max_pos = np.max(positions, axis=0) + 5
        
        # Set axis limits to focus on organism
        self.detail_ax.set_xlim(min_pos[0], max_pos[0])
        self.detail_ax.set_ylim(min_pos[1], max_pos[1])
        
        # Create patches for cells
        for i, pos in enumerate(positions):
            cell_type = cell_props['cell_types'][i]
            energy = cell_props['energies'][i]
            
            # Create a circle for the cell
            color = self.cell_type_colors.get(cell_type, 'blue')
            alpha = min(1.0, energy / 30)
            circle = Circle(pos, radius=2, facecolor=color, edgecolor='black', alpha=alpha)
            self.detail_ax.add_patch(circle)
        
        # Draw neural connections if present
        if organism.nervous_system:
            nerve_positions, connections = organism.get_neural_connections()
            
            # Skip if no connections
            if len(nerve_positions) > 0 and len(connections) > 0:
                segments = []
                for i, j in connections:
                    segments.append([nerve_positions[i], nerve_positions[j]])
                
                # Create line collection for connections
                lines = LineCollection(segments, colors='magenta', alpha=0.7, linewidth=1.5)
                self.detail_ax.add_collection(lines)
                
                # Add activation indicator (circle size proportional to activation)
                for i, pos in enumerate(nerve_positions):
                    activation = organism.nervous_system['activation'][i]
                    radius = 1 + activation * 2
                    circle = Circle(pos, radius=radius, facecolor='magenta', 
                                   edgecolor='white', alpha=0.7)
                    self.detail_ax.add_patch(circle)
        
        # Add organism information text
        org_info = organism.get_state_summary()
        info_text = (
            f"Organism {self.focus_organism_index}\n"
            f"Cells: {len(organism.cells)}\n"
            f"Energy: {org_info['energy']:.1f}\n"
            f"Age: {org_info['age']:.1f}\n"
            f"Complexity: {org_info['complexity']:.2f}\n"
            f"Neural cells: {org_info['neural_cells']}"
        )
        
        # Add cell type breakdown
        if 'cell_types' in org_info:
            info_text += "\n\nCell Types:"
            for cell_type, count in org_info['cell_types'].items():
                info_text += f"\n- {cell_type}: {count}"
        
        self.detail_text = self.detail_ax.text(
            0.02, 0.95, info_text, transform=self.detail_ax.transAxes,
            fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
    
    def animate(self, frames=300, interval=100):
        """Animate the simulation."""
        self.animation = FuncAnimation(
            self.fig, self.update_visualization, frames=frames,
            interval=interval, blit=False
        )
        return self.animation
    
    def save_animation(self, filename, fps=10):
        """Save the animation to a file."""
        if self.animation:
            self.animation.save(filename, fps=fps, extra_args=['-vcodec', 'libx264'])


def multicellular_visualization_callback(simulation, frame):
    """
    Callback function for visualization updates during multicellular simulation.
    
    Args:
        simulation: The current simulation state
        frame: The current frame number
    """
    # Print periodic updates
    if frame % 50 == 0:
        stats = simulation.get_stats()
        print(f"Frame {frame}: "
              f"{stats['organism_count'][-1] if stats['organism_count'] else 0} organisms, "
              f"{stats['total_cells'][-1] if stats['total_cells'] else 0} total cells")


def visualize_multicellular_simulation(simulation, frames=300, interval=100):
    """
    Set up and display visualization for a multicellular simulation.
    
    Args:
        simulation: The simulation to visualize
        frames: Number of frames to animate
        interval: Interval between frames in milliseconds
    
    Returns:
        The visualizer object
    """
    visualizer = MulticellularVisualizer(simulation)
    visualizer.animate(frames=frames, interval=interval)
    visualizer.show()
    return visualizer