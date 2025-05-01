"""
Advanced reaction-diffusion models for chemical dynamics in the simulation.
Implements more realistic spatial and temporal dynamics for nutrients and chemicals.
"""

import numpy as np
from scipy import ndimage # Keep for potential future use, though not used by Numba Laplacian
import numba
from numba import prange # Import prange for parallel loops


@numba.jit(nopython=True, parallel=True)
def numba_laplacian(grid, grid_size):
    """
    Calculate the Laplacian of a 2D grid using Numba with periodic boundary conditions.
    Uses a 5-point stencil: L(u) = u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]
    """
    laplacian = np.zeros_like(grid)
    for i in prange(grid_size): # Parallel over rows
        for j in range(grid_size):
            # Get neighbor indices with periodic boundaries
            i_prev = (i - 1 + grid_size) % grid_size
            i_next = (i + 1) % grid_size
            j_prev = (j - 1 + grid_size) % grid_size
            j_next = (j + 1) % grid_size

            # Calculate Laplacian using 5-point stencil
            laplacian[i, j] = (grid[i_next, j] + grid[i_prev, j] +
                               grid[i, j_next] + grid[i, j_prev] -
                               4.0 * grid[i, j])
    return laplacian


@numba.jit(nopython=True, parallel=True)
def numba_calculate_reactions(concentrations, reaction_matrix, dt, num_species, grid_size):
    """
    Numba-jitted function to calculate reactions between species.
    Modifies concentrations in-place.
    """
    # Create a temporary array to store changes to avoid race conditions in parallel updates
    delta_concentrations = np.zeros_like(concentrations)

    for i in prange(grid_size): # Parallel over rows
        for j in range(grid_size):
            # Get current concentrations at this grid point
            current_concs = concentrations[:, i, j] # Shape (num_species,)

            # Calculate reaction changes for each species
            # dC/dt = C * R^T (where C is row vector, R is reaction matrix)
            # Or element-wise: dC_k/dt = sum(C_m * R_{m,k})
            # This represents how species m affects species k
            reaction_changes = np.zeros(num_species)
            for k in range(num_species): # Target species
                for m in range(num_species): # Source species
                    # Reaction rate depends on source concentration
                    rate = reaction_matrix[m, k] * current_concs[m]
                    reaction_changes[k] += rate

            # Store changes for this grid point
            delta_concentrations[:, i, j] = reaction_changes

    # Apply changes and clip
    concentrations += dt * delta_concentrations
    # Clipping needs to be done carefully in parallel or after the parallel loop
    # Let's do it after the main calculation loop
    for s in prange(num_species): # Parallel over species
        for i in range(grid_size):
            for j in range(grid_size):
                if concentrations[s, i, j] < 0: concentrations[s, i, j] = 0
                if concentrations[s, i, j] > 1: concentrations[s, i, j] = 1


@numba.jit(nopython=True) # No parallelism needed for single point consumption
def numba_consume_at(concentrations, species_index, x, y, grid_size, amount, radius=2):
    """
    Numba-jitted function to consume a species at a specific location (x, y).
    Modifies concentrations in-place.
    """
    # Apply consumption in a small radius around the point
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            # Check if within circular radius
            if i*i + j*j <= radius*radius:
                # Calculate grid coordinates with periodic boundaries
                grid_x = (x + i + grid_size) % grid_size
                grid_y = (y + j + grid_size) % grid_size

                # Consumption falls off with distance from center
                dist_sq = i*i + j*j
                # Use max(1e-9, radius + 1.0) to prevent division by zero if radius is -1 or less (unlikely but safe)
                dist_factor = 1.0 - np.sqrt(dist_sq) / max(1e-9, radius + 1.0)

                # Calculate consumption amount, ensuring it doesn't go below zero
                consumption = min(amount * dist_factor, concentrations[species_index, grid_y, grid_x])

                # Apply consumption
                concentrations[species_index, grid_y, grid_x] -= consumption
                # Note: Byproduct generation (increasing V in Gray-Scott) is handled separately
                # in the ReactionDiffusionSystem class or specific reaction logic if needed.


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
        self.U = np.clip(self.U, 0, 1) # Ensure U doesn't go below 0 initially
        self.V = np.clip(self.V, 0, 1) # Ensure V stays in range

    # Removed the old laplacian method

    @numba.jit(nopython=True, parallel=True)
    def _update_numba(self, dt, U, V, Du, Dv, f, k, grid_size):
        """Numba-jitted core update logic for Gray-Scott."""
        # Create temporary arrays for Laplacians
        laplacian_U = np.zeros_like(U)
        laplacian_V = np.zeros_like(V)

        # Compute Laplacians using the jitted function
        # We cannot call numba_laplacian directly from a jitted method on instance attributes
        # Let's move the laplacian calculation logic directly here or pass the grid to the function.
        # Passing attributes is okay, let's call the external jitted laplacian function.
        laplacian_U[:] = numba_laplacian(U, grid_size) # Use [:] to update in-place
        laplacian_V[:] = numba_laplacian(V, grid_size)

        # Compute reaction terms and update concentrations
        # This loop can be parallelized over grid cells
        for i in prange(grid_size): # Parallel over rows
            for j in range(grid_size):
                uvv = U[i, j] * V[i, j] * V[i, j]

                # Update concentrations at (i, j)
                U[i, j] += dt * (Du * laplacian_U[i, j] - uvv + f * (1 - U[i, j]))
                V[i, j] += dt * (Dv * laplacian_V[i, j] + uvv - (f + k) * V[i, j])

        # Ensure concentrations stay in valid range (0 to 1)
        # This loop can also be parallelized
        for i in prange(grid_size): # Parallel over rows
            for j in range(grid_size):
                if U[i, j] < 0: U[i, j] = 0
                if U[i, j] > 1: U[i, j] = 1
                if V[i, j] < 0: V[i, j] = 0
                if V[i, j] > 1: V[i, j] = 1


    def update(self, dt=1.0):
        """
        Update the reaction-diffusion system for one timestep.
        Calls the Numba-jitted update function.
        """
        # Pass mutable numpy arrays and simple types to the jitted function
        self._update_numba(dt, self.U, self.V, self.Du, self.Dv, self.f, self.k, self.grid_size)

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

        # Apply consumption to grid coordinates
        # This loop can be potentially jitted if needed, but likely less impactful
        # than the main update loop.
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if i**2 + j**2 <= radius**2:
                    grid_x = (x + i + self.grid_size) % self.grid_size
                    grid_y = (y + j + self.grid_size) % self.grid_size

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
    def __init__(self, grid_size, num_species=5, pattern_type='ecosystem'):
        """
        Initialize a multi-species reaction-diffusion system.

        Args:
            grid_size: Size of the grid
            num_species: Number of chemical species to simulate
            pattern_type: Type of initial pattern ('ecosystem', 'gradient', 'patches', 'turing')
        """
        self.grid_size = grid_size
        self.num_species = num_species
        self.pattern_type = pattern_type
        self.time = 0  # Track simulation time

        # Initialize concentration arrays for each species
        self.concentrations = np.zeros((num_species, grid_size, grid_size))

        # Set up initial patterns based on pattern_type
        if pattern_type == 'ecosystem':
            self._init_ecosystem_pattern()
        elif pattern_type == 'gradient':
            self._init_gradient_pattern()
        elif pattern_type == 'patches':
            self._init_patches_pattern()
        elif pattern_type == 'turing':
            self._init_turing_pattern()
        else:
            # Default to ecosystem pattern
            self._init_ecosystem_pattern()

        # Ensure initial concentrations are within valid range
        self.concentrations = np.clip(self.concentrations, 0, 1)

        # Set up diffusion rates based on pattern type
        if pattern_type == 'turing':
            # For Turing patterns, we need specific diffusion rate relationships
            self.diffusion_rates = np.zeros(num_species, dtype=np.float64)
            # First species (activator) diffuses slowly
            self.diffusion_rates[0] = 0.05
            # Second species (inhibitor) diffuses faster
            if num_species > 1:
                self.diffusion_rates[1] = 0.2
            # Other species have random diffusion rates
            for i in range(2, num_species):
                self.diffusion_rates[i] = np.random.uniform(0.05, 0.2)
        else:
            # Random diffusion rates for other pattern types
            self.diffusion_rates = np.random.uniform(0.05, 0.2, num_species).astype(np.float64)

        # Set up reaction matrix based on pattern type
        self._init_reaction_matrix()

        # Add environmental variability
        self.variability = 0.01  # Random fluctuations in environment
        self.seasonal_cycle = 0.0  # Seasonal cycle phase
        self.seasonal_strength = 0.1  # Strength of seasonal effects

    def _init_ecosystem_pattern(self):
        """Initialize with ecosystem-like patterns (food web)."""
        # First species is primary nutrient, available in most places
        self.concentrations[0] = 0.8 + 0.2 * np.random.random((self.grid_size, self.grid_size))

        # Create patches of other species
        for i in range(1, self.num_species):
            # Create multiple patches for each species
            num_patches = np.random.randint(2, 5)
            for _ in range(num_patches):
                center_x = np.random.randint(0, self.grid_size)
                center_y = np.random.randint(0, self.grid_size)
                r = self.grid_size // (5 * (i + 1))  # Different size patches

                # Create a patch
                for x in range(self.grid_size):
                    for y in range(self.grid_size):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if dist < r:
                            # Concentration falls off with distance from center
                            self.concentrations[i, y, x] += 0.5 * (1 - dist/r)

            # Add some random noise
            self.concentrations[i] += np.random.random((self.grid_size, self.grid_size)) * 0.05

    def _init_gradient_pattern(self):
        """Initialize with gradient patterns."""
        # First species has a horizontal gradient
        for x in range(self.grid_size):
            gradient = x / self.grid_size
            self.concentrations[0, :, x] = gradient

        # Second species has a vertical gradient
        if self.num_species > 1:
            for y in range(self.grid_size):
                gradient = y / self.grid_size
                self.concentrations[1, y, :] = gradient

        # Third species has a radial gradient from center
        if self.num_species > 2:
            center = self.grid_size // 2
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    dist = np.sqrt((x - center)**2 + (y - center)**2)
                    max_dist = np.sqrt(2) * center
                    self.concentrations[2, y, x] = 1 - min(1, dist / max_dist)

        # Other species have random patterns
        for i in range(3, self.num_species):
            self.concentrations[i] = np.random.random((self.grid_size, self.grid_size))

    def _init_patches_pattern(self):
        """Initialize with distinct patches of different species."""
        # Divide the grid into regions
        region_size = self.grid_size // int(np.sqrt(self.num_species))

        for i in range(self.num_species):
            # Calculate region boundaries
            row = i // int(np.sqrt(self.num_species))
            col = i % int(np.sqrt(self.num_species))

            x_start = col * region_size
            x_end = min(self.grid_size, (col + 1) * region_size)
            y_start = row * region_size
            y_end = min(self.grid_size, (row + 1) * region_size)

            # Fill region with this species
            self.concentrations[i, y_start:y_end, x_start:x_end] = 0.8

            # Add some random variation
            self.concentrations[i] += np.random.random((self.grid_size, self.grid_size)) * 0.1

    def _init_turing_pattern(self):
        """Initialize for Turing pattern formation."""
        # Start with uniform concentrations plus small random perturbations
        for i in range(self.num_species):
            if i == 0:  # Activator
                self.concentrations[i] = 0.5 + np.random.random((self.grid_size, self.grid_size)) * 0.01
            elif i == 1:  # Inhibitor
                self.concentrations[i] = 0.25 + np.random.random((self.grid_size, self.grid_size)) * 0.01
            else:
                self.concentrations[i] = 0.1 * i + np.random.random((self.grid_size, self.grid_size)) * 0.01

    def _init_reaction_matrix(self):
        """Initialize the reaction matrix based on pattern type."""
        self.reaction_matrix = np.zeros((self.num_species, self.num_species), dtype=np.float64)

        if self.pattern_type == 'turing':
            # Set up reaction matrix for Turing pattern formation
            # Activator (species 0) auto-catalyzes itself and inhibits inhibitor
            self.reaction_matrix[0, 0] = 0.05  # Auto-catalysis
            if self.num_species > 1:
                self.reaction_matrix[0, 1] = 0.08  # Activator produces inhibitor
                self.reaction_matrix[1, 0] = -0.1  # Inhibitor inhibits activator
                self.reaction_matrix[1, 1] = -0.05  # Inhibitor decay

        elif self.pattern_type == 'ecosystem':
            # Set up food web-like interactions
            for i in range(self.num_species):
                for j in range(self.num_species):
                    if i == j:
                        # Self-limitation
                        self.reaction_matrix[i, j] = -0.05 - 0.05 * np.random.random()
                    elif (i + 1) % self.num_species == j:
                        # Species i feeds species j (conversion)
                        self.reaction_matrix[i, j] = 0.03 + 0.03 * np.random.random()
                        self.reaction_matrix[j, i] = -0.04 - 0.04 * np.random.random()

        else:
            # Default reaction matrix with some random interactions
            for i in range(self.num_species):
                for j in range(self.num_species):
                    if i == j:
                        # Self-limitation
                        self.reaction_matrix[i, j] = -0.05 - 0.05 * np.random.random()
                    else:
                        # Random interactions
                        self.reaction_matrix[i, j] = 0.1 * np.random.normal()

    # Removed the old laplacian method

    def update(self, dt=0.2):
        """
        Update all chemical species for one timestep.
        Uses Numba-jitted functions for diffusion and reactions.
        Includes environmental variability and seasonal changes.
        """
        # Update simulation time
        self.time += dt

        # Update seasonal cycle
        self.seasonal_cycle = (self.seasonal_cycle + dt * 0.01) % (2 * np.pi)
        seasonal_factor = self.seasonal_strength * np.sin(self.seasonal_cycle)

        # Compute diffusion for each species using the jitted function
        # Create a temporary array to store diffusion terms
        diffusion_terms = np.zeros_like(self.concentrations)

        # Loop over species, applying the parallel Laplacian to each species' grid
        # This loop itself is not parallelized by numba, but the called numba_laplacian is.
        for i in range(self.num_species):
             diffusion_terms[i] = numba_laplacian(self.concentrations[i], self.grid_size)

        # Apply diffusion terms to concentrations
        # Use broadcasting for diffusion rates
        # Modify diffusion rates based on seasonal changes
        modified_diffusion_rates = self.diffusion_rates.copy()
        if self.num_species > 1:
            # First species diffuses faster in "summer", slower in "winter"
            modified_diffusion_rates[0] *= (1.0 + seasonal_factor)
            # Second species has opposite pattern
            modified_diffusion_rates[1] *= (1.0 - seasonal_factor)

        self.concentrations += dt * modified_diffusion_rates[:, np.newaxis, np.newaxis] * diffusion_terms

        # Apply reactions and replenishment using the jitted function
        # Create a temporary reaction matrix with seasonal modifications
        modified_reaction_matrix = self.reaction_matrix.copy()

        # Modify reaction rates based on seasonal changes
        if self.pattern_type == 'ecosystem':
            # In "summer" (positive seasonal_factor), growth rates increase
            # In "winter" (negative seasonal_factor), decay rates increase
            for i in range(self.num_species):
                for j in range(self.num_species):
                    if modified_reaction_matrix[i, j] > 0:
                        # Growth rates affected by season
                        modified_reaction_matrix[i, j] *= (1.0 + seasonal_factor)
                    elif modified_reaction_matrix[i, j] < 0:
                        # Decay rates affected by season (opposite direction)
                        modified_reaction_matrix[i, j] *= (1.0 - seasonal_factor)

        # Apply reactions with modified rates
        numba_calculate_reactions(self.concentrations, modified_reaction_matrix, dt, self.num_species, self.grid_size)

        # Add environmental variability (random fluctuations)
        if self.variability > 0:
            random_fluctuations = np.random.normal(0, self.variability, self.concentrations.shape)
            self.concentrations += random_fluctuations
            # Ensure concentrations stay in valid range
            self.concentrations = np.clip(self.concentrations, 0, 1)

        # Occasionally add random events (e.g., "disasters" or "bonanzas")
        if np.random.random() < 0.001:  # 0.1% chance per timestep
            event_type = np.random.choice(['disaster', 'bonanza'])
            species_affected = np.random.randint(0, self.num_species)

            if event_type == 'disaster':
                # Create a localized disaster (e.g., toxin spill)
                center_x = np.random.randint(0, self.grid_size)
                center_y = np.random.randint(0, self.grid_size)
                radius = self.grid_size // 10

                for x in range(max(0, center_x - radius), min(self.grid_size, center_x + radius)):
                    for y in range(max(0, center_y - radius), min(self.grid_size, center_y + radius)):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if dist < radius:
                            # Reduce concentration based on distance from center
                            reduction = 0.5 * (1 - dist/radius)
                            self.concentrations[species_affected, y, x] = max(0,
                                self.concentrations[species_affected, y, x] - reduction)

            elif event_type == 'bonanza':
                # Create a localized resource boom
                center_x = np.random.randint(0, self.grid_size)
                center_y = np.random.randint(0, self.grid_size)
                radius = self.grid_size // 10

                for x in range(max(0, center_x - radius), min(self.grid_size, center_x + radius)):
                    for y in range(max(0, center_y - radius), min(self.grid_size, center_y + radius)):
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        if dist < radius:
                            # Increase concentration based on distance from center
                            increase = 0.5 * (1 - dist/radius)
                            self.concentrations[species_affected, y, x] = min(1,
                                self.concentrations[species_affected, y, x] + increase)


    def get_nutrient_field(self):
        """Return the primary nutrient concentration field (first species)."""
        return self.concentrations[0]

    def get_signal_fields(self):
        """Return all non-primary-nutrient concentration fields."""
        return self.concentrations[1:]

    def consume_at(self, x, y, species_index=0, amount=0.1):
        """Consume a specific species at location (x,y) using Numba."""
        # Ensure coordinates are within grid
        x_int, y_int = int(x) % self.grid_size, int(y) % self.grid_size

        # Call the jitted kernel
        numba_consume_at(self.concentrations, species_index, x_int, y_int, self.grid_size, amount)
