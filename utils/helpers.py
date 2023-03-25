"""
Helper functions for the simulation.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellular.cell import AdvancedCell
from multicellular.organism import MulticellularOrganism


def create_test_organism():
    """Create a test organism with some differentiated cells and a nervous system."""
    # Create cells in a small cluster
    cells = []
    center = np.array([50.0, 50.0])
    
    # Create 20 cells in a cluster
    for i in range(20):
        # Position in a rough circle around center
        angle = 2 * np.pi * i / 20
        distance = 5 + np.random.rand() * 3
        offset = np.array([np.cos(angle), np.sin(angle)]) * distance
        pos = center + offset
        
        # Create cell with random energy and DNA complexity
        energy = 20 + np.random.rand() * 10
        dna = 10 + np.random.rand() * 5
        cells.append(AdvancedCell(pos, energy=energy, dna_complexity=dna))
    
    # Create organism with these cells
    organism = MulticellularOrganism(cells)
    
    # Force development of structure (this normally happens automatically)
    organism.develop_structure()
    
    # Ensure we have some nerve cells (at least 5)
    nerve_count = sum(1 for cell in organism.cells if cell.get_cell_type()[0] == 'nerve')
    
    if nerve_count < 5:
        # Convert some cells to nerve cells
        non_nerve_cells = [cell for cell in organism.cells if cell.get_cell_type()[0] != 'nerve']
        cells_to_convert = min(5 - nerve_count, len(non_nerve_cells))
        
        for i in range(cells_to_convert):
            non_nerve_cells[i].differentiate('nerve', specialization_factor=0.9)
    
    # Force development of nervous system
    organism.develop_nervous_system()
    
    return organism