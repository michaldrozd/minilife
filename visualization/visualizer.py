"""
Visualization utilities for the evolution simulation.
Handles rendering of cells, environments, and other simulation components.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm


class SimulationVisualizer:
    """
    Base class for simulation visualization.
    Provides common functionality for all simulation phases.
    """
    def __init__(self, title="Simulation", figsize=(8, 8)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_title(title)
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.animation = None
    
    def show(self):
        """Display the visualization."""
        plt.show()
    
    def save(self, filename):
        """Save the visualization to a file."""
        if self.animation:
            self.animation.save(filename, writer='pillow', fps=15)
        else:
            self.fig.savefig(filename)


class CellVisualizer(SimulationVisualizer):
    """
    Visualizes cell simulations with detailed cell membranes, 
    intracellular processes, and environment.
    """
    def __init__(self, simulation, title="Cell Simulation", figsize=(10, 10)):
        super().__init__(title, figsize)
        self.simulation = simulation
        
        # Set up visualization components
        self.ax.set_xlim(0, simulation.grid_size)
        self.ax.set_ylim(0, simulation.grid_size)
        
        # Initialize visualization elements
        self.nutrient_im = self.ax.imshow(
            np.zeros((simulation.grid_size, simulation.grid_size)),
            cmap='YlGn', origin='lower', 
            extent=[0, simulation.grid_size, 0, simulation.grid_size],
            alpha=0.7, vmin=0, vmax=1
        )
        
        # Cell visualization (scatter plot initially, will use polygons for membranes)
        self.cell_scatter = self.ax.scatter([], [], c=[], cmap='viridis', s=30)
        
        # Cell membrane collections (for detailed visualization)
        self.membrane_patches = []
        self.membrane_collection = PatchCollection([], alpha=0.8, facecolor='blue', edgecolor='black')
        self.ax.add_collection(self.membrane_collection)
        
        # Stats text
        self.stats_text = self.ax.text(
            0.02, 0.95, "", transform=self.ax.transAxes,
            fontsize=8, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Color scales
        self.cell_colormap = cm.get_cmap('plasma')
        self.dna_complexity_colormap = cm.get_cmap('viridis')
        self.cell_type_colors = {
            'undifferentiated': 'blue',
            'nerve': 'yellow',
            'muscle': 'red',
            'skin': 'green'
        }
    
    def update_visualization(self, frame):
        """Update the visualization for the current frame."""
        # Update environment visualization
        env_data = self.simulation.get_environment_data()
        self.nutrient_im.set_data(env_data['nutrient'])
        
        # Get cell data
        cell_data = self.simulation.get_cell_properties()
        positions = cell_data['positions']
        
        # Update cell scatter points
        if len(positions) > 0:
            self.cell_scatter.set_offsets(positions)
            self.cell_scatter.set_array(cell_data['energies'])
            
            # Update membrane visualization
            self.update_membrane_visualization()
        else:
            self.cell_scatter.set_offsets(np.zeros((0, 2)))
        
        # Update stats text
        stats = self.simulation.get_stats()
        # Check if stats lists have values before accessing them
        cells = stats['cell_count'][-1] if stats['cell_count'] else 0
        energy = stats['mean_energy'][-1] if stats['mean_energy'] else 0
        dna = stats['mean_dna_complexity'][-1] if stats['mean_dna_complexity'] else 0
        
        stats_str = (
            f"Iteration: {self.simulation.iteration}\n"
            f"Cells: {cells}\n"
            f"Mean Energy: {energy:.2f}\n"
            f"Mean DNA: {dna:.2f}\n"
            f"Divisions: {stats['cell_divisions']}\n"
            f"Deaths: {stats['cell_deaths']}"
        )
        self.stats_text.set_text(stats_str)
        
        # Update title
        self.ax.set_title(f"Cell Simulation - Frame {frame}, Cells: {len(positions)}")
        
        return [self.nutrient_im, self.cell_scatter, self.membrane_collection, self.stats_text]
    
    def update_membrane_visualization(self):
        """Update the visualization of cell membranes."""
        # Clear previous patches
        self.membrane_patches.clear()
        
        # Get membrane points for all cells
        membrane_points_list = self.simulation.get_membrane_points()
        cell_data = self.simulation.get_cell_properties()
        
        # Create polygon patches for cell membranes
        for i, points in enumerate(membrane_points_list):
            # Skip if insufficient points
            if len(points) < 3:
                continue
            
            # Get cell properties for coloring
            energy = cell_data['energies'][i]
            dna = cell_data['dna_complexities'][i]
            cell_type = cell_data['cell_types'][i]
            
            # Create polygon from points
            polygon = Polygon(
                points, closed=True, 
                facecolor=self.cell_type_colors.get(cell_type, 'blue'),
                alpha=min(1.0, energy / 30.0),
                edgecolor='black', linewidth=0.5
            )
            self.membrane_patches.append(polygon)
        
        # Update the patch collection
        self.membrane_collection.set_paths(self.membrane_patches)
    
    def animate(self, frames=300, interval=50):
        """Animate the simulation."""
        self.animation = FuncAnimation(
            self.fig, self.update_visualization, frames=frames,
            interval=interval, blit=True
        )
        return self.animation


def cellular_visualization_callback(simulation, frame):
    """
    Callback function for visualization updates during cellular simulation.
    Useful for saving snapshots or collecting visualization data.
    
    Args:
        simulation: The current simulation state
        frame: The current frame number
    """
    # This is just a sample implementation that could be expanded
    if frame % 50 == 0:
        print(f"Frame {frame}: {len(simulation.cells)} cells, "
              f"Mean energy: {np.mean([cell.energy for cell in simulation.cells]) if simulation.cells else 0:.2f}")


def visualize_cellular_simulation(simulation, frames=300, interval=50):
    """
    Set up and display visualization for a cellular simulation.
    
    Args:
        simulation: The simulation to visualize
        frames: Number of frames to animate
        interval: Interval between frames in milliseconds
    
    Returns:
        The visualizer object
    """
    visualizer = CellVisualizer(simulation)
    visualizer.animate(frames=frames, interval=interval)
    visualizer.show()
    return visualizer