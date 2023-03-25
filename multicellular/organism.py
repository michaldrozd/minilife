"""
Models multicellular organisms with cell differentiation, tissue formation,
and emergent behaviors.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellular.cell import AdvancedCell
import config


class CellDifferentiation:
    """Handles differentiation of cells into specialized types."""
    def __init__(self, num_cell_types=4):
        self.num_cell_types = num_cell_types
        self.cell_types = ['undifferentiated', 'nerve', 'muscle', 'skin']
        
        # Spatial biases for cell types (determines likelihood of differentiation based on position)
        # e.g., nerve cells more likely to form in the "head", muscle in the "body", etc.
        self.spatial_bias = {
            'nerve': lambda x, y: 0.7 if y > 0.7 else 0.1,  # More likely at "top"
            'muscle': lambda x, y: 0.6 if (0.3 < y < 0.7) else 0.1,  # More likely in "middle"
            'skin': lambda x, y: 0.8 if (x < 0.2 or x > 0.8 or y < 0.2 or y > 0.8) else 0.1  # More likely on edges
        }
        
        # Neighboring cell type influences
        # Positive: cell type i promotes differentiation to type j
        # Negative: cell type i inhibits differentiation to type j
        self.interaction_matrix = np.zeros((num_cell_types, num_cell_types))
        
        # Set up some reasonable interaction rules
        # Undifferentiated cells can become any type
        self.interaction_matrix[0, 1:] = 0.1
        
        # Nerve cells promote nearby nerve cell differentiation
        self.interaction_matrix[1, 1] = 0.3
        
        # Muscle cells promote nearby muscle differentiation
        self.interaction_matrix[2, 2] = 0.4
        
        # Skin cells strongly promote nearby skin differentiation
        self.interaction_matrix[3, 3] = 0.5
        
        # Nerve cells slightly inhibit muscle differentiation
        self.interaction_matrix[1, 2] = -0.1
        
        # Skin cells inhibit nerve differentiation at the periphery
        self.interaction_matrix[3, 1] = -0.2
    
    def calculate_differentiation_probabilities(self, position, neighbors):
        """
        Calculate differentiation probabilities based on position and neighbors.
        
        Args:
            position: Normalized position (x, y) in organism (0-1 range)
            neighbors: List of (cell_type, distance) tuples for nearby cells
        
        Returns:
            Dictionary mapping cell types to differentiation probabilities
        """
        x, y = position
        probs = {}
        
        # Base probabilities based on spatial position
        for cell_type in self.cell_types[1:]:  # Skip 'undifferentiated'
            probs[cell_type] = self.spatial_bias[cell_type](x, y)
        
        # Adjust based on neighboring cells
        if neighbors:
            for neighbor_type, distance in neighbors:
                # Skip undifferentiated neighbors
                if neighbor_type == 'undifferentiated':
                    continue
                
                # Get indices in interaction matrix
                i = self.cell_types.index(neighbor_type)
                
                # Apply influence to each cell type
                for j, target_type in enumerate(self.cell_types[1:], 1):
                    # Influence decreases with distance
                    influence = self.interaction_matrix[i, j] / (1 + distance)
                    probs[target_type] = probs.get(target_type, 0) + influence
        
        # Ensure probabilities are in valid range
        for cell_type in probs:
            probs[cell_type] = max(0.01, min(0.99, probs[cell_type]))
        
        return probs
    
    def differentiate_cell(self, cell, position, neighbors):
        """
        Differentiate a cell based on its position and neighbors.
        
        Args:
            cell: The cell to differentiate
            position: Normalized position in the organism
            neighbors: List of (cell_type, distance) tuples for nearby cells
        
        Returns:
            The differentiated cell
        """
        # Skip already differentiated cells
        if cell.cell_type != 'undifferentiated':
            return cell
        
        # Calculate differentiation probabilities
        probs = self.calculate_differentiation_probabilities(position, neighbors)
        
        # Randomly select a cell type based on probabilities
        cell_types = list(probs.keys())
        probabilities = list(probs.values())
        
        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        # Choose a cell type
        chosen_type = np.random.choice(cell_types, p=probabilities)
        
        # Calculate specialization factor (how specialized the cell becomes)
        specialization = 0.5 + 0.5 * np.random.random()
        
        # Apply differentiation
        cell.differentiate(chosen_type, specialization)
        
        return cell


class CellularAdhesion:
    """Models adhesion forces between cells in a multicellular organism."""
    def __init__(self):
        # Adhesion strength matrix between different cell types
        # Higher values mean stronger adhesion
        self.cell_types = ['undifferentiated', 'nerve', 'muscle', 'skin']
        self.adhesion_strength = np.ones((len(self.cell_types), len(self.cell_types))) * 0.2
        
        # Set specific adhesion strengths between cell types
        # Same cell types adhere more strongly to each other
        for i in range(len(self.cell_types)):
            self.adhesion_strength[i, i] = 0.5
        
        # Skin cells have strong adhesion with each other (epithelial tissue)
        skin_idx = self.cell_types.index('skin')
        self.adhesion_strength[skin_idx, skin_idx] = 0.8
        
        # Muscle cells have strong adhesion with each other
        muscle_idx = self.cell_types.index('muscle')
        self.adhesion_strength[muscle_idx, muscle_idx] = 0.7
        
        # Nerve cells have medium adhesion with each other
        nerve_idx = self.cell_types.index('nerve')
        self.adhesion_strength[nerve_idx, nerve_idx] = 0.6
    
    def calculate_adhesion(self, cell1, cell2, distance):
        """
        Calculate adhesion force between two cells.
        
        Args:
            cell1: First cell
            cell2: Second cell
            distance: Distance between cells
        
        Returns:
            Adhesion force magnitude
        """
        # Get cell types
        type1, spec1 = cell1.get_cell_type()
        type2, spec2 = cell2.get_cell_type()
        
        # Get indices in adhesion matrix
        i = self.cell_types.index(type1)
        j = self.cell_types.index(type2)
        
        # Base adhesion strength between these cell types
        base_strength = self.adhesion_strength[i, j]
        
        # Adjust by specialization factors
        adjusted_strength = base_strength * (spec1 + spec2) / 2
        
        # Adhesion falls off with distance
        optimal_distance = cell1.physical_cell.radius + cell2.physical_cell.radius
        
        if distance < optimal_distance:
            # Repulsive component when too close
            force = adjusted_strength * (distance - optimal_distance) / optimal_distance
        else:
            # Attractive component at longer distances
            max_range = 2.5 * optimal_distance
            if distance > max_range:
                force = 0
            else:
                # Force decreases with distance from optimal
                force = adjusted_strength * (1.0 - (distance - optimal_distance) / 
                                          (max_range - optimal_distance))
        
        return force


class MulticellularOrganism:
    """
    Models a multicellular organism composed of specialized cell types
    with tissue formation, signaling, and emergent behaviors.
    """
    def __init__(self, cells=None, position=None):
        # Initialize with cells or create a single undifferentiated cell
        self.cells = cells if cells is not None else []
        
        # Set organism position
        if position is not None:
            self.position = np.array(position)
        elif cells:
            # Calculate center of mass
            self.position = np.mean([cell.get_position() for cell in self.cells], axis=0)
        else:
            self.position = np.array([50.0, 50.0])  # Default position
        
        # Organism properties
        self.age = 0
        self.energy = sum(cell.energy for cell in self.cells) if cells else 100
        self.complexity = np.mean([cell.dna_complexity for cell in self.cells]) if cells else 10
        
        # Differentiation manager
        self.differentiation = CellDifferentiation()
        
        # Adhesion manager
        self.adhesion = CellularAdhesion()
        
        # Organization and structure
        self.develop_structure()
        
        # Signaling system across the organism
        self.signaling_molecules = {
            'growth': 0.5,  # Promotes cell division
            'apoptosis': 0.1,  # Promotes cell death for remodeling
            'metabolism': 0.5  # Controls energy usage
        }
        
        # Neural cells and signaling
        self.nervous_system = None
        if len(self.cells) > 5:
            self.develop_nervous_system()
    
    def develop_structure(self):
        """Establish initial structural organization of the organism."""
        # Skip if no cells yet
        if not self.cells:
            return
        
        # Calculate relative positions of cells within the organism
        min_pos = np.min([cell.get_position() for cell in self.cells], axis=0)
        max_pos = np.max([cell.get_position() for cell in self.cells], axis=0)
        size = max_pos - min_pos
        size = np.maximum(size, 1)  # Avoid division by zero
        
        # Normalize cell positions to 0-1 range relative to organism
        normalized_positions = {}
        for cell in self.cells:
            pos = cell.get_position()
            norm_pos = (pos - min_pos) / size
            normalized_positions[cell] = norm_pos
        
        # Differentiate cells based on position
        for i, cell in enumerate(self.cells):
            # Find neighboring cells
            neighbors = []
            for other_cell in self.cells:
                if other_cell is not cell:
                    dist = np.linalg.norm(cell.get_position() - other_cell.get_position())
                    other_type, _ = other_cell.get_cell_type()
                    neighbors.append((other_type, dist))
            
            # Differentiate cell based on position and neighbors
            self.differentiation.differentiate_cell(cell, normalized_positions[cell], neighbors)
    
    def develop_nervous_system(self):
        """Develop a simple nervous system from nerve cells."""
        # Find all nerve cells
        nerve_cells = [cell for cell in self.cells 
                       if cell.get_cell_type()[0] == 'nerve' and 
                       cell.get_cell_type()[1] > 0.5]  # Only well-differentiated nerve cells
        
        if len(nerve_cells) < 3:
            return  # Not enough nerve cells
        
        # Create a simple graph representation of connected nerve cells
        connections = {}
        for i, cell1 in enumerate(nerve_cells):
            connections[cell1] = []
            pos1 = cell1.get_position()
            
            for j, cell2 in enumerate(nerve_cells):
                if cell1 is not cell2:
                    pos2 = cell2.get_position()
                    dist = np.linalg.norm(pos1 - pos2)
                    
                    # Connect nearby nerve cells
                    if dist < 15:
                        connections[cell1].append((cell2, dist))
        
        # Store nerve cells and their connections
        self.nervous_system = {
            'cells': nerve_cells,
            'connections': connections,
            'activation': np.zeros(len(nerve_cells))
        }
    
    def update_internal_systems(self, dt=0.1):
        """Update all internal systems and signaling."""
        # Update signaling molecules based on organism state
        self.signaling_molecules['growth'] = max(0.1, min(0.9, 0.5 + 0.3 * (
            self.energy / (100 * len(self.cells)) - 0.5)))
        
        self.signaling_molecules['apoptosis'] = max(0.05, min(0.5, 0.1 + 0.2 * (
            self.age / 100 - len(self.cells) / 200)))
        
        self.signaling_molecules['metabolism'] = max(0.1, min(0.9, 0.5 + 0.3 * (
            np.mean([cell.energy for cell in self.cells]) / 30 - 0.5)))
        
        # Update nervous system if present
        if self.nervous_system:
            self._update_nervous_system(dt)
    
    def _update_nervous_system(self, dt=0.1):
        """Update neural signaling in the organism."""
        # Skip if no nervous system
        if not self.nervous_system:
            return
        
        # Get environmental inputs from sensor cells (nerve cells near the edge)
        edge_threshold = 0.8  # How far from center to be considered an edge
        
        # Calculate center of mass
        center = np.mean([cell.get_position() for cell in self.nervous_system['cells']], axis=0)
        
        # Process signals through the neural network
        for i, cell in enumerate(self.nervous_system['cells']):
            pos = cell.get_position()
            dist_from_center = np.linalg.norm(pos - center)
            
            # Input signals from environment for edge cells
            if dist_from_center > edge_threshold * np.sqrt(len(self.cells)):
                # "Sensory" nerve cells
                # Get the cell's sensory data
                sensor_data = np.sum(list(cell.sensors.values()))
                
                # Update activation
                activation = max(0, min(1, sensor_data))
                self.nervous_system['activation'][i] = activation
        
        # Propagate signals through connections
        new_activation = np.zeros_like(self.nervous_system['activation'])
        
        for i, cell in enumerate(self.nervous_system['cells']):
            # Start with cell's current activation
            new_activation[i] = self.nervous_system['activation'][i] * 0.9  # Decay factor
            
            # Add inputs from connected cells
            for connected_cell, dist in self.nervous_system['connections'].get(cell, []):
                j = self.nervous_system['cells'].index(connected_cell)
                signal_strength = self.nervous_system['activation'][j] / (1 + dist * 0.1)
                new_activation[i] += signal_strength * 0.3
        
        # Apply non-linearity (sigmoid) to activations
        self.nervous_system['activation'] = 1.0 / (1.0 + np.exp(-new_activation))
    
    def update(self, dt=0.1, grid_size=100):
        """Update the organism for one timestep."""
        # Update age
        self.age += dt
        
        # Update internal systems and signaling
        self.update_internal_systems(dt)
        
        # Update individual cells
        energy_delta = 0
        
        for cell in self.cells:
            # Transfer organism signals to cell
            cell.sensors['density'] = len(self.cells) / 100
            
            # Update intracellular processes
            cell.update_intracellular(dt)
            
            # Move cell
            cell.move(grid_size)
            
            # Calculate energy change
            energy_delta += (cell.energy - cell.energy) * 0.1
        
        # Apply adhesion forces between cells
        self._apply_adhesion_forces()
        
        # Update organism energy
        self.energy += energy_delta
        
        # Update organism position to center of mass
        if self.cells:
            self.position = np.mean([cell.get_position() for cell in self.cells], axis=0)
        
        # Update complexity
        if self.cells:
            self.complexity = np.mean([cell.dna_complexity for cell in self.cells])
    
    def _apply_adhesion_forces(self):
        """Apply adhesion forces between cells."""
        # Skip if fewer than 2 cells
        if len(self.cells) < 2:
            return
        
        # Calculate and apply forces between all cell pairs
        for i, cell1 in enumerate(self.cells):
            for j, cell2 in enumerate(self.cells[i+1:], i+1):
                pos1 = cell1.get_position()
                pos2 = cell2.get_position()
                
                # Calculate distance
                direction = pos2 - pos1
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    # Normalize direction
                    direction = direction / distance
                    
                    # Calculate adhesion force
                    force_magnitude = self.adhesion.calculate_adhesion(cell1, cell2, distance)
                    
                    # Apply force to both cells in opposite directions
                    if force_magnitude != 0:
                        force = direction * force_magnitude
                        cell1.physical_cell.apply_external_force(force)
                        cell2.physical_cell.apply_external_force(-force)
    
    def grow(self, grid_size=100):
        """Attempt cell division to grow the organism."""
        # Skip if no cells
        if not self.cells:
            return False
        
        # Calculate growth probability based on energy
        growth_probability = self.signaling_molecules['growth'] * min(1.0, self.energy / (50 * len(self.cells)))
        
        # Cap growth to prevent unlimited expansion
        if len(self.cells) > 100:
            growth_probability *= 0.5
        
        if len(self.cells) > 200:
            growth_probability *= 0.2
        
        # Check for growth
        if np.random.random() > growth_probability:
            return False
        
        # Select a cell to divide
        weighted_energies = [max(0, cell.energy - 10) for cell in self.cells]
        total_energy = sum(weighted_energies)
        
        # Skip if no cells have enough energy
        if total_energy <= 0:
            return False
        
        # Randomly select a cell, weighted by energy
        probabilities = [e / total_energy for e in weighted_energies]
        parent_index = np.random.choice(len(self.cells), p=probabilities)
        parent_cell = self.cells[parent_index]
        
        # Attempt division
        offspring = parent_cell.replicate()
        if offspring:
            # Add new cell to organism
            self.cells.append(offspring)
            
            # Apply differentiation based on position
            min_pos = np.min([cell.get_position() for cell in self.cells], axis=0)
            max_pos = np.max([cell.get_position() for cell in self.cells], axis=0)
            size = max_pos - min_pos
            size = np.maximum(size, 1)  # Avoid division by zero
            
            # Get normalized position
            pos = offspring.get_position()
            norm_pos = (pos - min_pos) / size
            
            # Find neighboring cells
            neighbors = []
            for other_cell in self.cells:
                if other_cell is not offspring:
                    dist = np.linalg.norm(pos - other_cell.get_position())
                    other_type, _ = other_cell.get_cell_type()
                    neighbors.append((other_type, dist))
            
            # Differentiate new cell
            self.differentiation.differentiate_cell(offspring, norm_pos, neighbors)
            
            # Update nervous system
            if offspring.get_cell_type()[0] == 'nerve' and offspring.get_cell_type()[1] > 0.5:
                if self.nervous_system:
                    self.nervous_system['cells'].append(offspring)
                    
                    # Create connections to existing nerve cells
                    self.nervous_system['connections'][offspring] = []
                    for cell in self.nervous_system['cells'][:-1]:  # All except the new one
                        pos1 = offspring.get_position()
                        pos2 = cell.get_position()
                        dist = np.linalg.norm(pos1 - pos2)
                        
                        if dist < 15:
                            self.nervous_system['connections'][offspring].append((cell, dist))
                            self.nervous_system['connections'][cell].append((offspring, dist))
                    
                    # Extend activation array
                    self.nervous_system['activation'] = np.append(self.nervous_system['activation'], 0)
                else:
                    # If we have enough nerve cells now, initialize the nervous system
                    self.develop_nervous_system()
            
            return True
        
        return False
    
    def prune(self):
        """Remove dead or aging cells."""
        if not self.cells:
            return 0
        
        # Calculate apoptosis probability
        apoptosis_probability = self.signaling_molecules['apoptosis']
        
        # Track removed cells
        removed_count = 0
        
        # Check each cell
        cells_to_keep = []
        for cell in self.cells:
            # Cells die if energy is too low
            if cell.energy <= 0:
                removed_count += 1
                continue
            
            # Check for programmed cell death (apoptosis)
            # More likely for older cells and with higher apoptosis signal
            age_factor = min(1.0, cell.age / 100)
            if np.random.random() < apoptosis_probability * age_factor:
                removed_count += 1
                continue
            
            # Keep this cell
            cells_to_keep.append(cell)
        
        # Update nervous system if cells were removed
        if removed_count > 0 and self.nervous_system:
            # Check which nerve cells were removed
            old_nerve_cells = set(self.nervous_system['cells'])
            new_nerve_cells = [cell for cell in cells_to_keep 
                              if cell in old_nerve_cells]
            
            if len(new_nerve_cells) < 3:
                # Too few nerve cells left, deactivate nervous system
                self.nervous_system = None
            else:
                # Rebuild nervous system with remaining cells
                self.nervous_system['cells'] = new_nerve_cells
                
                # Rebuild connections and activation array
                new_connections = {}
                new_activation = []
                
                for cell in new_nerve_cells:
                    new_connections[cell] = []
                    old_idx = list(old_nerve_cells).index(cell)
                    new_activation.append(self.nervous_system['activation'][old_idx])
                    
                    for connected_cell, dist in self.nervous_system['connections'].get(cell, []):
                        if connected_cell in new_nerve_cells:
                            new_connections[cell].append((connected_cell, dist))
                
                self.nervous_system['connections'] = new_connections
                self.nervous_system['activation'] = np.array(new_activation)
        
        # Update cells list
        self.cells = cells_to_keep
        
        return removed_count
    
    def sense(self, environment):
        """
        Let the organism sense its environment.
        Updates sensory cells (nerve cells at the periphery).
        """
        # Skip if no cells
        if not self.cells:
            return
        
        # Update each cell's environmental sensing
        for cell in self.cells:
            cell.sense_environment(
                environment.get_nutrient_field(),
                environment.grid_size,
                signal_fields=environment.get_signal_fields()
            )
    
    def get_movement_direction(self):
        """
        Calculate overall movement direction for the organism.
        Based on cell forces and neural signals.
        
        Returns:
            Direction vector [dx, dy]
        """
        # Default: no movement
        direction = np.zeros(2)
        
        # Skip if no cells
        if not self.cells:
            return direction
        
        # Aggregate cell movement tendencies
        cell_directions = []
        for cell in self.cells:
            # Get the cell's last nutrient gradient as its preferred direction
            cell_dir = cell.last_nutrient_gradient
            # Add to list if non-zero
            if np.linalg.norm(cell_dir) > 0:
                cell_directions.append(cell_dir)
        
        # If neural system is developed, incorporate neural direction
        if self.nervous_system and len(self.nervous_system['cells']) >= 3:
            # Use neural activations to generate a direction vector
            neural_direction = np.zeros(2)
            
            # Get activations of peripheral (sensor) cells
            center = np.mean([cell.get_position() for cell in self.cells], axis=0)
            for i, cell in enumerate(self.nervous_system['cells']):
                pos = cell.get_position()
                
                # Direction from center to cell
                direction_to_cell = pos - center
                if np.linalg.norm(direction_to_cell) > 0:
                    direction_to_cell = direction_to_cell / np.linalg.norm(direction_to_cell)
                    
                    # Weight by activation
                    activation = self.nervous_system['activation'][i]
                    neural_direction += direction_to_cell * activation
            
            # Normalize neural direction
            if np.linalg.norm(neural_direction) > 0:
                neural_direction = neural_direction / np.linalg.norm(neural_direction)
                
                # Add to cell directions, weighted by neural development
                neural_weight = min(1.0, len(self.nervous_system['cells']) / 20)
                cell_directions.append(neural_direction * neural_weight * 2)
        
        # Calculate average direction
        if cell_directions:
            direction = np.mean(cell_directions, axis=0)
            
            # Normalize
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
        
        return direction
    
    def move(self, grid_size=100):
        """Move the organism as a whole."""
        # Skip if no cells
        if not self.cells:
            return
        
        # Get movement direction
        direction = self.get_movement_direction()
        
        # Skip if no movement
        if np.linalg.norm(direction) == 0:
            return
        
        # Calculate movement magnitude based on energy and organism size
        magnitude = 0.2 * min(1.0, self.energy / (50 * len(self.cells)))
        
        # Move each cell in the organism
        for cell in self.cells:
            # Apply force in the movement direction
            force = direction * magnitude
            cell.physical_cell.apply_external_force(force)
    
    def replicate(self, grid_size=100):
        """
        Attempt to replicate the entire organism.
        
        Returns:
            A new organism or None if replication failed
        """
        # Skip if too few cells or not enough energy
        min_cells = 20
        if len(self.cells) < min_cells or self.energy < 50 * len(self.cells):
            return None
        
        # Replication chance increases with size and energy
        replication_chance = min(0.01, len(self.cells) / 5000 * self.energy / (100 * len(self.cells)))
        
        if np.random.random() > replication_chance:
            return None
        
        # Create new cells for the offspring
        new_cells = []
        
        # Select half of the parent cells for the offspring, biased toward higher energy
        parent_weights = [max(0, cell.energy) for cell in self.cells]
        
        # Skip if no viable parents
        if sum(parent_weights) <= 0:
            return None
        
        # Normalize weights
        parent_weights = [w / sum(parent_weights) for w in parent_weights]
        
        # Select cells for offspring
        new_cell_count = len(self.cells) // 2
        selected_indices = np.random.choice(
            len(self.cells), 
            size=new_cell_count, 
            replace=False, 
            p=parent_weights
        )
        
        # Create offspring cells
        offset_angle = np.random.uniform(0, 2*np.pi)
        offset_distance = 20
        offset = np.array([
            np.cos(offset_angle) * offset_distance,
            np.sin(offset_angle) * offset_distance
        ])
        
        for idx in selected_indices:
            parent_cell = self.cells[idx]
            
            # Clone the parent cell
            energy = parent_cell.energy / 2
            parent_cell.energy /= 2
            
            # Create new position with offset
            new_pos = parent_cell.get_position() + offset
            new_pos = new_pos % grid_size
            
            # Create offspring cell with possible mutation
            mutation = np.random.choice([-1, 0, 0, 0, 1])
            new_dna = max(1, parent_cell.dna_complexity + mutation)
            
            # Create new cell
            new_cell = AdvancedCell(new_pos, energy=energy, dna_complexity=new_dna)
            
            # Inherit cell type
            cell_type, specialization = parent_cell.get_cell_type()
            if cell_type != 'undifferentiated':
                new_cell.differentiate(cell_type, specialization)
            
            new_cells.append(new_cell)
        
        # Create new organism
        new_organism = MulticellularOrganism(new_cells)
        
        # Offspring inherits some energy
        energy_transfer = self.energy * 0.3
        self.energy -= energy_transfer
        new_organism.energy += energy_transfer
        
        return new_organism
    
    def get_cell_positions(self):
        """Get positions of all cells."""
        return np.array([cell.get_position() for cell in self.cells]) if self.cells else np.array([])
    
    def get_cell_types(self):
        """Get types of all cells."""
        return [cell.get_cell_type()[0] for cell in self.cells]
    
    def get_cell_properties(self):
        """Get detailed properties of all cells."""
        positions = []
        energies = []
        dna_complexities = []
        cell_types = []
        specializations = []
        
        for cell in self.cells:
            positions.append(cell.get_position())
            energies.append(cell.energy)
            dna_complexities.append(cell.dna_complexity)
            cell_type, specialization = cell.get_cell_type()
            cell_types.append(cell_type)
            specializations.append(specialization)
        
        return {
            'positions': np.array(positions) if positions else np.array([]),
            'energies': np.array(energies),
            'dna_complexities': np.array(dna_complexities),
            'cell_types': cell_types,
            'specializations': np.array(specializations)
        }
    
    def get_membrane_points(self):
        """Get membrane points for all cells."""
        return [cell.get_membrane_points() for cell in self.cells]
    
    def get_neural_connections(self):
        """Get neural connections for visualization."""
        if not self.nervous_system:
            return [], []
        
        # Get positions of nerve cells
        positions = [cell.get_position() for cell in self.nervous_system['cells']]
        
        # Get connections
        connections = []
        for i, cell in enumerate(self.nervous_system['cells']):
            for connected_cell, _ in self.nervous_system['connections'].get(cell, []):
                j = self.nervous_system['cells'].index(connected_cell)
                connections.append((i, j))
        
        return np.array(positions), connections
    
    def get_state_summary(self):
        """Get a summary of the organism's state."""
        cell_types = {}
        for cell in self.cells:
            cell_type, _ = cell.get_cell_type()
            cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
        
        return {
            'position': self.position,
            'age': self.age,
            'energy': self.energy,
            'complexity': self.complexity,
            'cell_count': len(self.cells),
            'cell_types': cell_types,
            'has_nervous_system': self.nervous_system is not None,
            'neural_cells': len(self.nervous_system['cells']) if self.nervous_system else 0
        }