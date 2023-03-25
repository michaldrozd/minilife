"""
Advanced reaction-diffusion models for chemical dynamics in the simulation.
Implements more realistic spatial and temporal dynamics for nutrients and chemicals.
"""

import numpy as np
from scipy import ndimage
import numba


class ReactionDiffusionSystem:
    """
    Implements a reaction-diffusion system using the Gray-Scott model.
    This provides more realistic chemical dynamics than simple diffusion.
    """
    def __init__(self, grid_size, Du=0.16, Dv=0.08, f=0.035, k=0.065):
        self.grid_size = grid_size
        # Diffusion rates
        self.Du = Du  # Diffusion rate of U (nutrients)
        self.Dv = Dv  # Diffusion rate of V (consumed nutrients/byproducts)
        # Reaction parameters
        self.f = f    # Feed rate
        self.k = k    # Kill rate
        
        # Initialize concentration fields
        self.U = np.ones((grid_size, grid_size))
        self.V = np.zeros((grid_size, grid_size))
        
        # Add some random initial concentrations to V
        center = grid_size // 2
        r = grid_size // 10
        y, x = np.ogrid[-center:grid_size-center, -center:grid_size-center]
        mask = x*x + y*y <= r*r
        self.V[mask] = 0.5
        self.V += np.random.random((grid_size, grid_size)) * 0.1
        self.U -= self.V

    def laplacian(self, X):
        """
        Compute the Laplacian of X using convolution.
        This is faster and more accurate than manual neighbor calculations.
        """
        laplacian_kernel = np.array([[0.05, 0.2, 0.05], 
                                     [0.2, -1.0, 0.2], 
                                     [0.05, 0.2, 0.05]])
        return ndimage.convolve(X, laplacian_kernel, mode='wrap')

    def update(self, dt=1.0):
        """
        Update the reaction-diffusion system for one timestep.
        """
        # Compute Laplacians
        laplacian_U = self.laplacian(self.U)
        laplacian_V = self.laplacian(self.V)
        
        # Compute reaction terms
        uvv = self.U * self.V * self.V
        
        # Update concentrations
        self.U += dt * (self.Du * laplacian_U - uvv + self.f * (1 - self.U))
        self.V += dt * (self.Dv * laplacian_V + uvv - (self.f + self.k) * self.V)
        
        # Ensure concentrations stay in valid range
        self.U = np.clip(self.U, 0, 1)
        self.V = np.clip(self.V, 0, 1)
    
    def add_nutrient(self, amount=0.1):
        """
        Add nutrient uniformly across the system.
        """
        self.U += amount
        self.U = np.clip(self.U, 0, 1)
    
    def consume_at(self, x, y, amount=0.1):
        """
        Consume nutrient at a specific location, simulating cellular consumption.
        Also increases V as a byproduct.
        """
        # Ensure coordinates are within grid
        x, y = int(x) % self.grid_size, int(y) % self.grid_size
        
        # Create a consumption mask (a small area around the point)
        radius = 2
        y_grid, x_grid = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x_grid**2 + y_grid**2 <= radius**2
        
        # Apply mask to grid coordinates
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if i**2 + j**2 <= radius**2:
                    grid_x = (x + i) % self.grid_size
                    grid_y = (y + j) % self.grid_size
                    
                    # Consumption falls off with distance from center
                    dist_factor = 1.0 - np.sqrt(i**2 + j**2) / (radius + 1)
                    consumption = min(amount * dist_factor, self.U[grid_y, grid_x])
                    
                    self.U[grid_y, grid_x] -= consumption
                    self.V[grid_y, grid_x] += consumption * 0.8  # 80% conversion to byproduct


class MultiSpeciesReactionDiffusion:
    """
    A more advanced reaction-diffusion system with multiple chemical species.
    Suitable for modeling complex environments with multiple nutrients and signals.
    """
    def __init__(self, grid_size, num_species=3):
        self.grid_size = grid_size
        self.num_species = num_species
        
        # Initialize concentration arrays for each species
        self.concentrations = np.zeros((num_species, grid_size, grid_size))
        
        # Default: first species is primary nutrient, fully available
        self.concentrations[0] = np.ones((grid_size, grid_size))
        
        # Add some random initial concentrations to other species
        for i in range(1, num_species):
            center = grid_size // 2
            r = grid_size // (5 * (i + 1))  # Different size patches
            y, x = np.ogrid[-center:grid_size-center, -center:grid_size-center]
            mask = x*x + y*y <= r*r
            self.concentrations[i, mask] = 0.5
            self.concentrations[i] += np.random.random((grid_size, grid_size)) * 0.05
        
        # Random diffusion and reaction parameters for each species
        self.diffusion_rates = np.random.uniform(0.05, 0.2, num_species)
        
        # Reaction matrix: how species affect each other
        # Positive: species i increases species j
        # Negative: species i decreases species j
        # We want realistic cycles, so construct food web-like interactions
        self.reaction_matrix = np.zeros((num_species, num_species))
        for i in range(num_species):
            for j in range(num_species):
                if i == j:
                    # Self-limitation
                    self.reaction_matrix[i, j] = -0.05 - 0.05 * np.random.random()
                elif (i + 1) % num_species == j:
                    # Species i feeds species j (conversion)
                    self.reaction_matrix[i, j] = 0.03 + 0.03 * np.random.random()
                    self.reaction_matrix[j, i] = -0.04 - 0.04 * np.random.random()
    
    def laplacian(self, X):
        """Compute the Laplacian of X using convolution."""
        laplacian_kernel = np.array([[0.05, 0.2, 0.05], 
                                     [0.2, -1.0, 0.2], 
                                     [0.05, 0.2, 0.05]])
        return ndimage.convolve(X, laplacian_kernel, mode='wrap')
    
    def update(self, dt=0.2):
        """Update all chemical species for one timestep."""
        # Compute diffusion for each species
        diffusion_terms = np.zeros_like(self.concentrations)
        for i in range(self.num_species):
            diffusion_terms[i] = self.diffusion_rates[i] * self.laplacian(self.concentrations[i])
        
        # Compute reaction terms
        reaction_terms = np.zeros_like(self.concentrations)
        for i in range(self.num_species):
            for j in range(self.num_species):
                reaction_terms[i] += self.reaction_matrix[j, i] * self.concentrations[j] * self.concentrations[i]
        
        # Update concentrations
        self.concentrations += dt * (diffusion_terms + reaction_terms)
        
        # Add small constant replenishment to first species (main nutrient)
        self.concentrations[0] += dt * 0.01
        
        # Ensure concentrations stay in valid range
        self.concentrations = np.clip(self.concentrations, 0, 1)
    
    def get_nutrient_field(self):
        """Return the primary nutrient concentration field (first species)."""
        return self.concentrations[0]
    
    def get_signal_fields(self):
        """Return all non-primary-nutrient concentration fields."""
        return self.concentrations[1:]
    
    def consume_at(self, x, y, species_index=0, amount=0.1):
        """Consume a specific species at location (x,y)."""
        # Ensure coordinates are within grid
        x, y = int(x) % self.grid_size, int(y) % self.grid_size
        
        # Create a consumption mask (a small area around the point)
        radius = 2
        
        # Apply consumption to grid coordinates
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if i**2 + j**2 <= radius**2:
                    grid_x = (x + i) % self.grid_size
                    grid_y = (y + j) % self.grid_size
                    
                    # Consumption falls off with distance from center
                    dist_factor = 1.0 - np.sqrt(i**2 + j**2) / (radius + 1)
                    consumption = min(amount * dist_factor, self.concentrations[species_index, grid_y, grid_x])
                    
                    self.concentrations[species_index, grid_y, grid_x] -= consumption
                    
                    # Add some to the next species in the chain (conversion/metabolism)
                    next_species = (species_index + 1) % self.num_species
                    self.concentrations[next_species, grid_y, grid_x] += consumption * 0.7