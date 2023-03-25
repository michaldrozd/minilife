"""
Realistic cell model with detailed internal processes and gene regulatory networks.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from physics.mechanics import DeformableCell, Vector2D
import config


class ProteinSynthesis:
    """Models protein synthesis from gene expression to functional proteins."""
    def __init__(self, num_genes=10):
        self.num_genes = num_genes
        
        # Gene expression levels (0-1 range)
        self.gene_expression = np.random.rand(num_genes) * 0.5
        
        # Gene to protein conversion rates
        self.translation_rates = np.random.rand(num_genes) * 0.2 + 0.1
        
        # Protein degradation rates
        self.degradation_rates = np.random.rand(num_genes) * 0.1 + 0.05
        
        # Current protein levels
        self.protein_levels = np.zeros(num_genes)
        
        # Gene regulatory network (matrix of influences)
        # Positive value: gene i activates gene j
        # Negative value: gene i represses gene j
        self.regulatory_network = np.random.randn(num_genes, num_genes) * 0.2
        # Make diagonal negative for self-regulation (to prevent runaway activation)
        np.fill_diagonal(self.regulatory_network, -0.3 - np.random.rand(num_genes) * 0.2)
    
    def update(self, dt=0.1, external_signals=None):
        """Update gene expression and protein levels."""
        # Update gene expression based on regulatory network
        gene_influence = np.dot(self.protein_levels, self.regulatory_network)
        
        # Add external signals if provided
        if external_signals is not None:
            # Make sure the signals match the gene count
            signal_array = np.zeros(self.num_genes)
            signal_array[:min(len(external_signals), self.num_genes)] = external_signals[:min(len(external_signals), self.num_genes)]
            gene_influence += signal_array
        
        # Update gene expression using a sigmoid activation function
        self.gene_expression += dt * gene_influence
        self.gene_expression = 1.0 / (1.0 + np.exp(-self.gene_expression))
        
        # Update protein levels based on gene expression and degradation
        synthesis_rate = self.gene_expression * self.translation_rates
        degradation = self.protein_levels * self.degradation_rates
        
        self.protein_levels += dt * (synthesis_rate - degradation)
        self.protein_levels = np.clip(self.protein_levels, 0, None)
    
    def get_protein_level(self, protein_index):
        """Get the level of a specific protein."""
        return self.protein_levels[protein_index]
    
    def get_expression_profile(self):
        """Get the entire gene expression profile."""
        return self.gene_expression.copy()
    
    def get_protein_profile(self):
        """Get the entire protein profile."""
        return self.protein_levels.copy()
    
    def set_external_signal(self, gene_index, signal_strength):
        """Set an external signal affecting a specific gene."""
        temp_signals = np.zeros(self.num_genes)
        temp_signals[gene_index] = signal_strength
        self.update(external_signals=temp_signals)


class MetabolicNetwork:
    """Models cellular metabolism with interdependent reactions."""
    def __init__(self, num_metabolites=8):
        self.num_metabolites = num_metabolites
        
        # Metabolite concentrations
        self.metabolite_levels = np.random.rand(num_metabolites) * 0.5
        
        # Define metabolic pathways as a directed graph
        # Positive: reaction converts metabolite i to j
        # Negative: reaction consumes metabolite i
        self.reaction_matrix = np.zeros((num_metabolites, num_metabolites))
        
        # Create simple metabolic pathways
        for i in range(num_metabolites - 1):
            # Forward reaction rate (i -> i+1)
            self.reaction_matrix[i, i+1] = 0.1 + np.random.rand() * 0.1
            # Consumption rate of i
            self.reaction_matrix[i, i] = -0.15 - np.random.rand() * 0.1
        
        # Last metabolite loops back or is excreted
        if np.random.rand() > 0.5:
            # Loop back to first metabolite
            self.reaction_matrix[num_metabolites-1, 0] = 0.1 + np.random.rand() * 0.1
        self.reaction_matrix[num_metabolites-1, num_metabolites-1] = -0.15 - np.random.rand() * 0.1
        
        # Enzyme efficiencies (affected by proteins)
        self.enzyme_efficiency = np.ones(num_metabolites)
    
    def update(self, dt=0.1, nutrients=None, protein_levels=None):
        """Update metabolite concentrations."""
        # Add nutrients if provided
        if nutrients is not None:
            self.metabolite_levels[0] += nutrients * 0.5
        
        # Adjust enzyme efficiencies if proteins provided
        if protein_levels is not None:
            # Use first few proteins as enzyme modifiers
            num_enzymes = min(len(protein_levels), self.num_metabolites)
            self.enzyme_efficiency[:num_enzymes] = 0.5 + protein_levels[:num_enzymes]
        
        # Calculate metabolic fluxes
        fluxes = np.zeros(self.num_metabolites)
        for i in range(self.num_metabolites):
            for j in range(self.num_metabolites):
                # Apply reaction rate * substrate level * enzyme efficiency
                if self.reaction_matrix[i, j] > 0 and self.metabolite_levels[i] > 0:
                    reaction_rate = self.reaction_matrix[i, j] * self.metabolite_levels[i] * self.enzyme_efficiency[i]
                    fluxes[j] += reaction_rate  # Product formation
                    fluxes[i] -= reaction_rate  # Substrate consumption
        
        # Update metabolite levels
        self.metabolite_levels += dt * fluxes
        self.metabolite_levels = np.clip(self.metabolite_levels, 0, None)
    
    def get_energy_production(self):
        """Calculate energy production as a function of metabolite levels."""
        # Consider the last metabolites as energy-yielding
        energy_metabolites = self.metabolite_levels[-3:]
        return np.sum(energy_metabolites) * 2.0
    
    def get_metabolite_levels(self):
        """Get all metabolite levels."""
        return self.metabolite_levels.copy()


class SignalingPathway:
    """Models cell signaling pathways for environmental response."""
    def __init__(self, num_receptors=3, num_signals=5):
        self.num_receptors = num_receptors
        self.num_signals = num_signals
        
        # Receptor states (0-1 range, 0=inactive, 1=fully activated)
        self.receptor_states = np.zeros(num_receptors)
        
        # Intracellular signaling molecules
        self.signal_molecules = np.zeros(num_signals)
        
        # Receptor to signal transduction matrix
        self.transduction_matrix = np.random.rand(num_receptors, num_signals) * 0.3
        
        # Signal propagation matrix
        self.signal_propagation = np.random.randn(num_signals, num_signals) * 0.2
        np.fill_diagonal(self.signal_propagation, -0.1 - np.random.rand(num_signals) * 0.1)
        
        # Signal to gene expression influence matrix (variable size)
        self.num_genes = 10  # Default gene count
        self.signal_to_gene = np.random.randn(num_signals, self.num_genes) * 0.2
    
    def activate_receptors(self, signals):
        """Activate receptors based on environmental signals."""
        # Clamp signals to valid range for each receptor
        signals = np.clip(signals[:self.num_receptors], 0, 1)
        
        # Update receptor states (with some delay)
        self.receptor_states = 0.8 * self.receptor_states + 0.2 * signals
    
    def update(self, dt=0.1):
        """Update signaling molecule concentrations."""
        # Receptor to signal transduction
        receptor_input = np.dot(self.receptor_states, self.transduction_matrix)
        
        # Signal propagation
        signal_crosstalk = np.dot(self.signal_molecules, self.signal_propagation)
        
        # Update signal molecules
        self.signal_molecules += dt * (receptor_input + signal_crosstalk)
        
        # Add baseline deactivation
        self.signal_molecules -= dt * 0.1 * self.signal_molecules
        
        # Clamp to valid range
        self.signal_molecules = np.clip(self.signal_molecules, 0, 1)
    
    def get_gene_signals(self):
        """Convert signaling state to gene expression signals."""
        return np.dot(self.signal_molecules, self.signal_to_gene)
    
    def get_signaling_state(self):
        """Get the current state of all signaling molecules."""
        return self.signal_molecules.copy()


class AdvancedCell:
    """
    An advanced cell model with detailed internal processes.
    Combines mechanical properties with intracellular processes.
    """
    def __init__(self, position, energy=20, dna_complexity=10, radius=3.0):
        # Initialize mechanical cell properties
        self.physical_cell = DeformableCell(position, radius=radius)
        
        # Cell state
        self.energy = energy
        self.dna_complexity = dna_complexity
        self.age = 0
        
        # Calculate number of genes based on DNA complexity
        self.num_genes = int(10 + dna_complexity // 2)
        
        # Cellular components
        self.protein_synthesis = ProteinSynthesis(num_genes=self.num_genes)
        self.metabolism = MetabolicNetwork(num_metabolites=8)
        self.signaling = SignalingPathway(num_receptors=3, num_signals=5)
        
        # Update signaling pathway to match gene count
        self.signaling.num_genes = self.num_genes
        self.signaling.signal_to_gene = np.random.randn(5, self.num_genes) * 0.2
        
        # Cell specialization (for multicellular organisms)
        self.cell_type = 'undifferentiated'
        self.specialization_factor = 0.0  # 0 = undifferentiated, 1 = fully specialized
        
        # Environmental sensing
        self.last_nutrient_gradient = np.zeros(2)
        self.sensors = {
            'nutrient': 0.0,
            'toxin': 0.0,
            'signal1': 0.0,
            'signal2': 0.0,
            'density': 0.0
        }
    
    def sense_environment(self, nutrient_field, grid_size, toxin_field=None, signal_fields=None):
        """Sense the local environment and update receptors."""
        # Get cell position
        i, j = int(self.physical_cell.position.x) % grid_size, int(self.physical_cell.position.y) % grid_size
        
        # Sense nutrient level
        self.sensors['nutrient'] = nutrient_field[i, j]
        
        # Sense toxin if available
        if toxin_field is not None:
            self.sensors['toxin'] = toxin_field[i, j]
        
        # Sense signaling molecules if available
        if signal_fields is not None:
            if len(signal_fields) > 0:
                self.sensors['signal1'] = signal_fields[0][i, j]
            if len(signal_fields) > 1:
                self.sensors['signal2'] = signal_fields[1][i, j]
        
        # Compute nutrient gradient for movement
        # Using central differences
        left = nutrient_field[(i - 1) % grid_size, j]
        right = nutrient_field[(i + 1) % grid_size, j]
        up = nutrient_field[i, (j + 1) % grid_size]
        down = nutrient_field[i, (j - 1) % grid_size]
        grad_x = (right - left) / 2.0
        grad_y = (up - down) / 2.0
        self.last_nutrient_gradient = np.array([grad_x, grad_y])
        
        # Activate signaling pathways based on environment
        receptor_signals = [
            self.sensors['nutrient'],
            1.0 - self.sensors['toxin'],  # Invert toxin - high toxin = low signal
            self.sensors['signal1']
        ]
        self.signaling.activate_receptors(receptor_signals)
    
    def update_intracellular(self, dt=0.1):
        """Update all intracellular processes."""
        # Update signaling pathways
        self.signaling.update(dt)
        
        # Get signals to pass to gene expression
        gene_signals = self.signaling.get_gene_signals()
        
        # Update protein synthesis with signals
        self.protein_synthesis.update(dt, external_signals=gene_signals)
        
        # Update metabolism with nutrients and proteins
        self.metabolism.update(dt, nutrients=self.sensors['nutrient'], 
                              protein_levels=self.protein_synthesis.get_protein_profile())
        
        # Update energy based on metabolism
        energy_production = self.metabolism.get_energy_production()
        energy_consumption = 1.0 + self.dna_complexity * 0.05
        
        self.energy += dt * (energy_production - energy_consumption)
    
    def move(self, grid_size):
        """Move the cell based on sensory input and internal state."""
        # Calculate force based on nutrient gradient and hunger
        hunger = max(0, 20 - self.energy)
        force = self.last_nutrient_gradient * hunger * 0.1
        
        # Also influenced by signaling and protein levels
        signal_state = self.signaling.get_signaling_state()
        if len(signal_state) > 1:
            # First signal can modify movement direction
            signal_angle = signal_state[0] * 2 * np.pi
            signal_force = np.array([np.cos(signal_angle), np.sin(signal_angle)]) * 0.05
            force += signal_force
        
        # Apply force to physical cell
        self.physical_cell.apply_external_force(force)
        
        # Update physical cell
        self.physical_cell.update(dt=0.1, grid_size=grid_size)
    
    def metabolize(self, dt=0.1):
        """Update metabolism and age."""
        # Energy dynamics handled in update_intracellular
        self.age += dt
    
    def replicate(self):
        """Attempt to replicate the cell."""
        # Replication requires sufficient energy and age
        if self.energy > config.CELL_REPLICATION_THRESHOLD and self.age > config.CELL_REPLICATION_AGE:
            # Mutation chance scales with DNA complexity
            mutation_range = [-1, 0, 0, 0, 1]  # Bias toward no change or positive change
            mutation = np.random.choice(mutation_range)
            new_dna = max(1, self.dna_complexity + mutation)
            
            # Create offspring with half the parent's energy
            offspring_energy = self.energy / 2
            self.energy /= 2
            
            # Create an offset position for the offspring
            angle = np.random.uniform(0, 2*np.pi)
            offset = np.array([np.cos(angle), np.sin(angle)]) * self.physical_cell.radius * 1.2
            new_pos = self.physical_cell.get_position() + offset
            
            return AdvancedCell(new_pos, offspring_energy, new_dna, self.physical_cell.radius)
        
        return None
    
    def differentiate(self, cell_type, specialization_factor=0.5):
        """Differentiate cell into a specialized type."""
        self.cell_type = cell_type
        self.specialization_factor = specialization_factor
        
        # Modify gene expression based on cell type
        expression_profile = self.protein_synthesis.get_expression_profile()
        
        if cell_type == 'nerve':
            # Enhance signaling-related genes
            gene_mask = np.random.choice([0, 1], size=len(expression_profile), p=[0.7, 0.3])
            expression_profile = expression_profile * (1 - gene_mask) + gene_mask * (0.8 + 0.2 * np.random.rand(len(expression_profile)))
        
        elif cell_type == 'muscle':
            # Enhance energy-producing genes
            gene_mask = np.random.choice([0, 1], size=len(expression_profile), p=[0.7, 0.3])
            expression_profile = expression_profile * (1 - gene_mask) + gene_mask * (0.8 + 0.2 * np.random.rand(len(expression_profile)))
        
        elif cell_type == 'skin':
            # Enhance structural genes
            gene_mask = np.random.choice([0, 1], size=len(expression_profile), p=[0.7, 0.3])
            expression_profile = expression_profile * (1 - gene_mask) + gene_mask * (0.8 + 0.2 * np.random.rand(len(expression_profile)))
        
        # Apply the modified expression profile
        self.protein_synthesis.gene_expression = expression_profile
    
    def get_position(self):
        """Get the cell's current position."""
        return self.physical_cell.get_position()
    
    def get_cell_type(self):
        """Get the cell's type and specialization level."""
        return self.cell_type, self.specialization_factor
    
    def get_membrane_points(self):
        """Get points for rendering the cell membrane."""
        return self.physical_cell.get_membrane_points()
    
    def get_state_summary(self):
        """Get a summary of the cell's state."""
        return {
            'position': self.get_position(),
            'energy': self.energy,
            'age': self.age,
            'dna_complexity': self.dna_complexity,
            'cell_type': self.cell_type,
            'specialization': self.specialization_factor,
            'metabolism': sum(self.metabolism.get_metabolite_levels()),
            'signaling': sum(self.signaling.get_signaling_state()),
            'protein_count': sum(self.protein_synthesis.get_protein_profile())
        }