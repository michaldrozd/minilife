"""
Simulation of multicellular organisms with tissue differentiation,
emergent behaviors, and evolution.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.diffusion import MultiSpeciesReactionDiffusion
from physics.mechanics import CollisionResolver
from multicellular.organism import MulticellularOrganism
import config


class MulticellularSimulation:
    """
    Simulates a population of multicellular organisms with tissue formation,
    signaling, and emergent behaviors in a complex environment.
    """
    def __init__(self, grid_size=config.GRID_SIZE, initial_organisms=None, cells=None):
        self.grid_size = grid_size
        self.organisms = []
        
        # Initialize either from organisms or cells
        if initial_organisms:
            self.organisms = initial_organisms
        elif cells and len(cells) > 0:
            # Group cells into initial organisms based on proximity
            self.organisms = self._group_cells_into_organisms(cells)
        else:
            # If no cells or organisms provided, create a test organism
            # This is helpful for demonstration purposes
            from utils.helpers import create_test_organism
            self.organisms = [create_test_organism()]
        
        # Create environment with multiple chemical species
        self.environment = MultiSpeciesReactionDiffusion(grid_size, num_species=5)
        
        # Set up physics engine for collision resolution
        self.collision_resolver = CollisionResolver()
        
        # Statistics and tracking
        self.iteration = 0
        self.stats = {
            'organism_count': [],
            'total_cells': [],
            'mean_organism_size': [],
            'mean_organism_complexity': [],
            'mean_neuron_count': [],
            'births': 0,
            'deaths': 0
        }
    
    def _group_cells_into_organisms(self, cells, proximity_threshold=15):
        """Group cells into organisms based on proximity."""
        # Skip if no cells
        if not cells:
            return []
        
        # Initialize with all cells unclustered
        unclustered = set(cells)
        clusters = []
        
        # Simple clustering algorithm
        while unclustered:
            # Start a new cluster with a random cell
            current = unclustered.pop()
            current_cluster = [current]
            
            # Find all cells within proximity
            queue = [current]
            while queue:
                cell = queue.pop(0)
                pos = cell.get_position()
                
                # Check all unclustered cells
                to_remove = []
                for other in unclustered:
                    other_pos = other.get_position()
                    distance = np.linalg.norm(pos - other_pos)
                    
                    if distance < proximity_threshold:
                        # Add to cluster
                        current_cluster.append(other)
                        queue.append(other)
                        to_remove.append(other)
                
                # Remove newly clustered cells from unclustered set
                for cell in to_remove:
                    unclustered.remove(cell)
            
            # Create organism from cluster
            if len(current_cluster) > 0:
                clusters.append(MulticellularOrganism(current_cluster))
        
        return clusters
    
    def update(self):
        """Update the simulation for one timestep."""
        self.iteration += 1
        
        # Update environment (reaction-diffusion system)
        self.environment.update(dt=0.2)
        
        # Update each organism
        for organism in self.organisms:
            # Sense environment
            organism.sense(self.environment)
            
            # Update internal processes
            organism.update(dt=0.1, grid_size=self.grid_size)
            
            # Try to grow (cell division)
            organism.grow(self.grid_size)
            
            # Prune dead cells
            organism.prune()
            
            # Move organism
            organism.move(self.grid_size)
            
            # Consume nutrients at organism position
            for cell in organism.cells:
                pos = cell.get_position()
                i, j = int(pos[0]) % self.grid_size, int(pos[1]) % self.grid_size
                self.environment.consume_at(i, j, species_index=0, amount=0.05)
        
        # Organism reproduction
        new_organisms = []
        for organism in self.organisms:
            offspring = organism.replicate(self.grid_size)
            if offspring:
                new_organisms.append(offspring)
                self.stats['births'] += 1
        
        # Add new organisms
        self.organisms.extend(new_organisms)
        
        # Remove dead organisms (those with no cells)
        old_count = len(self.organisms)
        self.organisms = [org for org in self.organisms if org.cells]
        self.stats['deaths'] += (old_count - len(self.organisms) + len(new_organisms))
        
        # Update statistics
        self.stats['organism_count'].append(len(self.organisms))
        total_cells = sum(len(org.cells) for org in self.organisms)
        self.stats['total_cells'].append(total_cells)
        
        if self.organisms:
            self.stats['mean_organism_size'].append(total_cells / len(self.organisms))
            self.stats['mean_organism_complexity'].append(
                np.mean([org.complexity for org in self.organisms]))
            self.stats['mean_neuron_count'].append(
                np.mean([
                    len(org.nervous_system['cells']) if org.nervous_system else 0
                    for org in self.organisms
                ]))
        else:
            self.stats['mean_organism_size'].append(0)
            self.stats['mean_organism_complexity'].append(0)
            self.stats['mean_neuron_count'].append(0)
        
        # Add nutrients occasionally
        if self.iteration % 20 == 0:
            self.add_nutrients()
    
    def add_nutrients(self, amount=0.5):
        """Add nutrients to the environment."""
        # Get the primary nutrient field
        
        # Create random nutrient patches
        for _ in range(3):
            center_x = np.random.randint(0, self.grid_size)
            center_y = np.random.randint(0, self.grid_size)
            radius = np.random.randint(5, 15)
            
            for i in range(max(0, center_x - radius), min(self.grid_size, center_x + radius)):
                for j in range(max(0, center_y - radius), min(self.grid_size, center_y + radius)):
                    dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if dist < radius:
                        self.environment.concentrations[0, i, j] += amount * (1 - dist / radius)
    
    def get_organism_positions(self):
        """Get positions of all organisms."""
        return np.array([org.position for org in self.organisms]) if self.organisms else np.array([])
    
    def get_all_cell_data(self):
        """Get data for all cells across all organisms."""
        all_positions = []
        all_types = []
        all_energies = []
        all_organism_ids = []
        
        for i, organism in enumerate(self.organisms):
            cell_props = organism.get_cell_properties()
            
            # Skip empty organisms
            if len(cell_props['positions']) == 0:
                continue
                
            all_positions.extend(cell_props['positions'])
            all_types.extend(cell_props['cell_types'])
            all_energies.extend(cell_props['energies'])
            all_organism_ids.extend([i] * len(cell_props['positions']))
        
        return {
            'positions': np.array(all_positions) if all_positions else np.zeros((0, 2)),
            'types': all_types,
            'energies': np.array(all_energies) if all_energies else np.array([]),
            'organism_ids': np.array(all_organism_ids) if all_organism_ids else np.array([])
        }
    
    def get_neural_networks(self):
        """Get neural networks for all organisms."""
        networks = []
        
        for organism in self.organisms:
            if organism.nervous_system:
                positions, connections = organism.get_neural_connections()
                networks.append({
                    'positions': positions,
                    'connections': connections,
                    'activations': organism.nervous_system['activation']
                })
        
        return networks
    
    def get_environment_data(self):
        """Get environment data for visualization."""
        return {
            'nutrient': self.environment.get_nutrient_field(),
            'signals': self.environment.get_signal_fields()
        }
    
    def get_stats(self):
        """Get simulation statistics."""
        return self.stats
    
    def get_organism_summaries(self):
        """Get summaries of all organisms' states."""
        return [org.get_state_summary() for org in self.organisms]


def run_multicellular_simulation(cells=None, max_iterations=500, visualization_callback=None, stop_condition=None):
    """
    Run the multicellular simulation for a specified number of iterations
    or until a stop condition is met.
    
    Args:
        cells: Initial cells from cellular simulation (optional)
        max_iterations: Maximum number of simulation iterations
        visualization_callback: Optional callback for visualization updates
        stop_condition: Optional function that takes the simulation object and returns True when simulation should stop
    
    Returns:
        The final simulation state
    """
    # Create the simulation
    simulation = MulticellularSimulation(cells=cells)
    
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
        
        # Add nutrients occasionally
        if iteration % 50 == 0:
            simulation.add_nutrients()
    
    return simulation