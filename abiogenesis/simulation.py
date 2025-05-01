"""
Simulation of abiogenesis: the origin of complex organic molecules.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from abiogenesis.chemical import Chemical
import config


def run_abiogenesis_simulation(max_iter=config.ABIOGENESIS_MAX_ITER, complexity_threshold=config.CHEMICAL_COMPLEXITY_THRESHOLD):
    """
    Simulate the origin of complex organic molecules.
    This is a basic placeholder implementation.
    """
    print("Simulating abiogenesis...")
    
    chemicals = [Chemical(1) for _ in range(config.INITIAL_CHEMICALS)] # Assuming INITIAL_CHEMICALS is added to config later
    
    for iteration in range(max_iter):
        # Basic replication and potential complexity increase
        new_chemicals = []
        for chemical in chemicals:
            new_chemicals.append(chemical.replicate())
            
        chemicals.extend(new_chemicals)
        
        # Remove duplicates and low complexity chemicals
        unique_chemicals = {}
        for chemical in chemicals:
            # Use complexity as a simple identifier for now
            if chemical.complexity not in unique_chemicals:
                unique_chemicals[chemical.complexity] = chemical
        
        # Filter for chemicals above threshold
        chemicals = [chem for chem in unique_chemicals.values() if chem.complexity >= complexity_threshold]
        
        if iteration % 1000 == 0:
             print(f"Abiogenesis Iteration {iteration}: Found {len(chemicals)} chemicals above complexity {complexity_threshold}")
        
        # Stop condition: find a certain number of complex chemicals
        if len(chemicals) > 5: # Arbitrary stop condition for placeholder
             print("Sufficient complex chemicals formed!")
             break

    print("Abiogenesis simulation finished.")
    
    return chemicals
