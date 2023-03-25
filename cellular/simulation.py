"""
Advanced single-cell simulation with detailed biophysics and cellular processes.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.diffusion import MultiSpeciesReactionDiffusion
from physics.mechanics import CollisionResolver
from cellular.cell import AdvancedCell
import config


class AdvancedCellularSimulation:
    """
    Simulates a population of cells with detailed internal processes
    and physical interactions in a complex environment.
    """
    def __init__(self, grid_size=config.GRID_SIZE, initial_cells=config.INITIAL_CELLS):
        self.grid_size = grid_size
        self.cells = []
        
        # Initialize cells
        for _ in range(initial_cells):
            pos = np.random.rand(2) * grid_size
            energy = config.CELL_INITIAL_ENERGY * (0.8 + 0.4 * np.random.rand())
            dna = config.CELL_INITIAL_DNA_COMPLEXITY * (0.8 + 0.4 * np.random.rand())
            self.cells.append(AdvancedCell(pos, energy=energy, dna_complexity=dna))
        
        # Create environment with multiple chemical species
        self.environment = MultiSpeciesReactionDiffusion(grid_size, num_species=5)
        
        # Set up physics engine for cell interactions
        self.collision_resolver = CollisionResolver()
        
        # Statistics and tracking
        self.iteration = 0
        self.stats = {
            'cell_count': [],
            'mean_energy': [],
            'mean_dna_complexity': [],
            'cell_deaths': 0,
            'cell_divisions': 0
        }
    
    def update(self):
        """Update the simulation for one timestep."""
        self.iteration += 1
        
        # Update environment (reaction-diffusion system)
        self.environment.update(dt=0.2)
        
        # Get environment state for cell sensing
        nutrient_field = self.environment.get_nutrient_field()
        signal_fields = self.environment.get_signal_fields()
        
        # Cell sensing, movement, and internal updates
        for cell in self.cells:
            # Sense environment
            cell.sense_environment(nutrient_field, self.grid_size, signal_fields=signal_fields)
            
            # Update intracellular processes
            cell.update_intracellular()
            
            # Move cell
            cell.move(self.grid_size)
            
            # Consume nutrients at cell location
            pos = cell.get_position()
            i, j = int(pos[0]) % self.grid_size, int(pos[1]) % self.grid_size
            self.environment.consume_at(i, j, species_index=0, amount=0.1)
            
            # Update cell metabolism
            cell.metabolize()
        
        # Resolve collisions between cells
        physical_cells = [cell.physical_cell for cell in self.cells]
        self.collision_resolver.resolve_collisions(physical_cells)
        
        # Cell division
        new_cells = []
        for cell in self.cells:
            offspring = cell.replicate()
            if offspring:
                new_cells.append(offspring)
                self.stats['cell_divisions'] += 1
        
        # Add new cells
        self.cells.extend(new_cells)
        
        # Remove dead cells
        old_count = len(self.cells)
        self.cells = [cell for cell in self.cells if cell.energy > 0 and cell.age < config.CELL_MAX_AGE]
        self.stats['cell_deaths'] += (old_count - len(self.cells) + len(new_cells))
        
        # Update statistics
        self.stats['cell_count'].append(len(self.cells))
        if len(self.cells) > 0:
            self.stats['mean_energy'].append(np.mean([cell.energy for cell in self.cells]))
            self.stats['mean_dna_complexity'].append(np.mean([cell.dna_complexity for cell in self.cells]))
        else:
            self.stats['mean_energy'].append(0)
            self.stats['mean_dna_complexity'].append(0)
    
    def get_cell_positions(self):
        """Get positions of all cells for visualization."""
        return np.array([cell.get_position() for cell in self.cells])
    
    def get_cell_properties(self):
        """Get cell properties for visualization."""
        positions = []
        energies = []
        dna_complexities = []
        ages = []
        cell_types = []
        
        for cell in self.cells:
            positions.append(cell.get_position())
            energies.append(cell.energy)
            dna_complexities.append(cell.dna_complexity)
            ages.append(cell.age)
            cell_type, _ = cell.get_cell_type()
            cell_types.append(cell_type)
        
        return {
            'positions': np.array(positions) if positions else np.zeros((0, 2)),
            'energies': np.array(energies),
            'dna_complexities': np.array(dna_complexities),
            'ages': np.array(ages),
            'cell_types': cell_types
        }
    
    def get_environment_data(self):
        """Get environment data for visualization."""
        return {
            'nutrient': self.environment.get_nutrient_field(),
            'signals': self.environment.get_signal_fields()
        }
    
    def get_membrane_points(self):
        """Get membrane points for all cells for detailed visualization."""
        membranes = []
        for cell in self.cells:
            membranes.append(cell.get_membrane_points())
        return membranes
    
    def get_stats(self):
        """Get simulation statistics."""
        return self.stats
    
    def add_nutrient(self, amount=0.5):
        """Add nutrients to the environment."""
        # Get the primary nutrient field and add nutrients
        field = self.environment.get_nutrient_field()
        
        # Create a nutrient source pattern
        center = self.grid_size // 2
        radius = self.grid_size // 4
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if dist < radius:
                    self.environment.concentrations[0, i, j] += amount * (1 - dist / radius)


def run_cellular_simulation(max_iterations=500, visualization_callback=None, stop_condition=None):
    """
    Run the cellular simulation for a specified number of iterations or until a stop condition is met.
    
    Args:
        max_iterations: Maximum number of simulation iterations
        visualization_callback: Optional callback for visualization updates
        stop_condition: Optional function that takes the simulation object and returns True when simulation should stop
    
    Returns:
        The final simulation state
    """
    # Create the simulation
    simulation = AdvancedCellularSimulation()
    
    # Run simulation loop
    for iteration in range(max_iterations):
        # Update the simulation
        simulation.update()
        
        # Call visualization callback if provided
        if visualization_callback:
            visualization_callback(simulation, iteration)
        
        # Check stop condition if provided
        if stop_condition and stop_condition(simulation):
            break
        
        # Add nutrients occasionally to prevent extinction
        if iteration % 50 == 0:
            simulation.add_nutrient()
    
    return simulation