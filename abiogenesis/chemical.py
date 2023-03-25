"""
Basic chemical model for abiogenesis simulation.
Placeholder to be expanded in future versions.
"""

import numpy as np


class Chemical:
    """
    Represents a chemical compound in the abiogenesis simulation.
    """
    def __init__(self, complexity):
        """
        Initialize a chemical with a given complexity.
        
        Args:
            complexity: The complexity of the chemical (higher = more complex)
        """
        self.complexity = complexity
    
    def replicate(self):
        """
        Attempt to replicate the chemical.
        With a low probability, the replication increases complexity.
        
        Returns:
            A new Chemical instance
        """
        # With a low probability, the replication "upgrades" complexity.
        if np.random.rand() < 0.001:
            return Chemical(self.complexity + 1)
        return Chemical(self.complexity)