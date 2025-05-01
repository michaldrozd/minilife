"""
Physics-based cell mechanics including deformable cells, adhesion forces,
and collision detection/resolution.
"""

import numpy as np
import numba
from numba import float64, int64 # Import specific types for signatures
from numba import prange # Import prange for parallel loops


# Define a Numba-compatible struct-like type for Vector2D data?
# Numba structured arrays are better. Let's just use NumPy arrays of shape (N, 2).


class Vector2D:
    """
    A simple 2D vector class with basic operations.
    Note: This class is NOT Numba compatible in nopython mode directly.
    Operations involving Vector2D objects should be done in Python or
    using NumPy arrays within Numba functions.
    """
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def length(self):
        return np.sqrt(self.x**2 + self.y**2)
    
    def normalized(self):
        length = self.length()
        if length > 0:
            return Vector2D(self.x / length, self.y / length)
        return Vector2D(0, 0)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y
    
    def to_array(self):
        return np.array([self.x, self.y])
    
    @classmethod
    def from_array(cls, arr):
        return cls(arr[0], arr[1])


class DeformableCell:
    """
    A physics-based cell model with deformable membrane.
    Uses a spring-mass system to model the cell membrane.
    """
    def __init__(self, position, radius=3.0, num_vertices=16):
        self.position = Vector2D.from_array(position)
        self.radius = radius
        self.num_vertices = num_vertices
        
        # Generate vertices around the perimeter
        self.vertices = []
        for i in range(num_vertices):
            angle = 2 * np.pi * i / num_vertices
            x = self.position.x + radius * np.cos(angle)
            y = self.position.y + radius * np.sin(angle)
            self.vertices.append(Vector2D(x, y))
        
        # Physical properties
        self.mass = radius * radius * np.pi  # Cell mass proportional to area
        self.velocity = Vector2D(0, 0)
        self.forces = Vector2D(0, 0)
        
        # Spring properties (for vertex connections)
        self.spring_constant = 0.8
        self.damping = 0.2
        self.rest_lengths = self._calculate_rest_lengths()
        
        # Pressure properties (outward force to maintain volume)
        self.pressure_constant = 0.05
        self.target_area = np.pi * radius * radius
        
        # Vertex velocities
        self.vertex_velocities = [Vector2D(0, 0) for _ in range(num_vertices)]
        
    def apply_external_force(self, force_vector):
        """Apply an external force to the cell."""
        # force = Vector2D.from_array(force_vector) # Removed this line as force_vector is used directly
            
        # Distribute force to all vertices by adding to velocities
        # Convert vertex velocities to numpy array, add force, convert back
        vertex_velocities_array = self.get_vertex_velocities_array()
            
        # Add force / mass * dt to velocity (integrating force over a conceptual dt, or impulse)
        # Or, simpler, just add a velocity change proportional to force magnitude
        # The current code adds a fixed proportion (0.1) of the force
        # Let's apply it as an impulse scaled by a factor and mass
        # Velocity change = force_vector * scale / mass
        velocity_change = force_vector * (0.1 / self.mass) # shape (2,)
    
        # Apply this velocity change to all vertices
        vertex_velocities_array += velocity_change[np.newaxis, :] # Broadcast velocity_change across all vertices
    
        # Update internal Vector2D list
        self._update_vertices_from_arrays(self.get_vertex_positions_array(), vertex_velocities_array)
    
    def _calculate_rest_lengths(self):
        """Calculate the rest lengths between adjacent vertices."""
        rest_lengths = []
        for i in range(self.num_vertices):
            j = (i + 1) % self.num_vertices
            dx = self.vertices[i].x - self.vertices[j].x
            dy = self.vertices[i].y - self.vertices[j].y
            distance = np.sqrt(dx*dx + dy*dy)
            rest_lengths.append(distance)
        return rest_lengths
    
    def calculate_area(self):
        """Calculate the current area of the cell using polygon area formula."""
        area = 0
        for i in range(self.num_vertices):
            j = (i + 1) % self.num_vertices
            area += self.vertices[i].x * self.vertices[j].y
            area -= self.vertices[j].x * self.vertices[i].y
        return abs(area) / 2
    
    def calculate_centroid(self):
        """Calculate the centroid of the cell."""
        cx, cy = 0, 0
        for vertex in self.vertices:
            cx += vertex.x
            cy += vertex.y
        return Vector2D(cx / self.num_vertices, cy / self.num_vertices)
    
    def apply_spring_forces(self):
        """Apply spring forces between adjacent vertices."""
        forces = [Vector2D(0, 0) for _ in range(self.num_vertices)]
        
        for i in range(self.num_vertices):
            j = (i + 1) % self.num_vertices
            
            # Vector from i to j
            dx = self.vertices[j].x - self.vertices[i].x
            dy = self.vertices[j].y - self.vertices[i].y
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Spring force (F = k * (|x| - L) * x/|x|)
            if distance > 0:
                displacement = distance - self.rest_lengths[i]
                force_magnitude = self.spring_constant * displacement
                
                force_x = force_magnitude * dx / distance
                force_y = force_magnitude * dy / distance
                
                # Apply equal and opposite forces to both vertices
                forces[i].x += force_x
                forces[i].y += force_y
                forces[j].x -= force_x
                forces[j].y -= force_y
        
        """
        Apply spring forces between adjacent vertices.
        Returns a NumPy array of forces (num_vertices, 2).
        """
        forces = np.zeros((self.num_vertices, 2), dtype=np.float64)
        
        # Convert vertices to numpy array for easier calculation
        vertices_array = self.get_vertex_positions_array()
        
        for i in range(self.num_vertices):
            j = (i + 1) % self.num_vertices
            
            # Vector from i to j
            dx = vertices_array[j, 0] - vertices_array[i, 0]
            dy = vertices_array[j, 1] - vertices_array[i, 1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Spring force (F = k * (|x| - L) * x/|x|)
            if distance > 1e-9: # Avoid division by zero
                displacement = distance - self.rest_lengths[i]
                force_magnitude = self.spring_constant * displacement
                
                force_x = force_magnitude * dx / distance
                force_y = force_magnitude * dy / distance
                
                # Apply equal and opposite forces to both vertices
                forces[i, 0] += force_x
                forces[i, 1] += force_y
                forces[j, 0] -= force_x
                forces[j, 1] -= force_y
        
        return forces
    
    def apply_pressure_forces(self):
        """
        Apply pressure forces to maintain cell volume.
        Returns a NumPy array of forces (num_vertices, 2).
        """
        forces = np.zeros((self.num_vertices, 2), dtype=np.float64)
        area = self.calculate_area()
        centroid = self.calculate_centroid() # Returns Vector2D
        
        # Convert vertices to numpy array
        vertices_array = self.get_vertex_positions_array()
        
        # Force proportional to difference from target area
        area_difference = self.target_area - area
        pressure_force_magnitude = self.pressure_constant * area_difference
        
        for i in range(self.num_vertices):
            # Direction from centroid to vertex
            dx = vertices_array[i, 0] - centroid.x
            dy = vertices_array[i, 1] - centroid.y
            length = np.sqrt(dx*dx + dy*dy)
            
            if length > 1e-9: # Avoid division by zero
                # Normalize and scale by pressure
                forces[i, 0] += pressure_force_magnitude * dx / length
                forces[i, 1] += pressure_force_magnitude * dy / length
        
        return forces
    
    # Helper function to get vertex positions as a NumPy array
    def get_vertex_positions_array(self):
         """Convert list of Vector2D vertices to a NumPy array."""
         return np.array([[v.x, v.y] for v in self.vertices], dtype=np.float64)

    # Helper function to get vertex velocities as a NumPy array
    def get_vertex_velocities_array(self):
         """Convert list of Vector2D vertex velocities to a NumPy array."""
         return np.array([[v.x, v.y] for v in self.vertex_velocities], dtype=np.float64)

    # Helper function to update Vector2D lists from NumPy arrays
    def _update_vertices_from_arrays(self, vertex_positions_array, vertex_velocities_array):
        """Update internal Vector2D lists from NumPy arrays."""
        for i in range(self.num_vertices):
            self.vertices[i].x = vertex_positions_array[i, 0]
            self.vertices[i].y = vertex_positions_array[i, 1]
            self.vertex_velocities[i].x = vertex_velocities_array[i, 0]
            self.vertex_velocities[i].y = vertex_velocities_array[i, 1]
            
    def update(self, dt=0.1, grid_size=100):
        """
        Update the cell for one timestep.
        Uses Numba-jitted function for vertex updates.
        """
        # Calculate spring forces (returns NumPy array)
        spring_forces = self.apply_spring_forces()
        
        # Calculate pressure forces (returns NumPy array)
        pressure_forces = self.apply_pressure_forces()
        
        # Sum all forces for each vertex
        total_forces = spring_forces + pressure_forces
        
        # Get vertex positions and velocities as NumPy arrays
        vertex_positions_array = self.get_vertex_positions_array()
        vertex_velocities_array = self.get_vertex_velocities_array()

        # Call the standalone Numba-jitted kernel to update vertex positions and velocities
        # The kernel modifies the arrays in-place
        _update_cell_vertices_numba(
            vertex_positions_array, vertex_velocities_array, total_forces,
            self.damping, self.mass, dt, grid_size
        )

        # Update internal Vector2D lists from the modified NumPy arrays
        self._update_vertices_from_arrays(vertex_positions_array, vertex_velocities_array)

        # Update cell position to the centroid
        self.position = self.calculate_centroid()
        
    def get_membrane_points(self):
        """Get points representing the cell membrane for visualization."""
        return [(v.x, v.y) for v in self.vertices]
    
    def get_position(self):
        """Get the cell's current position as an array."""
        return np.array([self.position.x, self.position.y])


# Standalone Numba-jitted function for updating cell vertices
@numba.jit(nopython=True) # Not parallel yet, will parallelize the outer loop over cells
def _update_cell_vertices_numba(vertex_positions, vertex_velocities, total_forces, damping, mass, dt, grid_size):
    """Numba-jitted kernel to update vertex positions and velocities for a single cell."""
    num_vertices = vertex_positions.shape[0]
    
    for i in range(num_vertices):
        # Sum forces (already done before calling this function)
        total_force_x = total_forces[i, 0]
        total_force_y = total_forces[i, 1]

        # Apply damping to velocity (F = -c*v)
        damping_x = -damping * vertex_velocities[i, 0]
        damping_y = -damping * vertex_velocities[i, 1]
        total_force_x += damping_x
        total_force_y += damping_y

        # Update velocity (v = v + a*dt = v + F/m*dt)
        vertex_velocities[i, 0] += total_force_x * dt / mass
        vertex_velocities[i, 1] += total_force_y * dt / mass

        # Update position (x = x + v*dt)
        vertex_positions[i, 0] += vertex_velocities[i, 0] * dt
        vertex_positions[i, 1] += vertex_velocities[i, 1] * dt

        # Apply periodic boundary conditions
        vertex_positions[i, 0] = vertex_positions[i, 0] % grid_size
        vertex_positions[i, 1] = vertex_positions[i, 1] % grid_size

    # Return updated arrays (or modify in-place, Numba does both)
    # Modifying in-place is typical for performance
    # return vertex_positions, vertex_velocities # Not needed if modifying in-place


class CellAdhesion:
    """Handles adhesion forces between cells."""
    def __init__(self, adhesion_strength=0.2, adhesion_range=2.0):
        self.adhesion_strength = adhesion_strength
        self.adhesion_range = adhesion_range
    
    def calculate_adhesion(self, cell1, cell2):
        """Calculate adhesion force between two cells."""
        # For simplicity, we use cell centers for the calculation
        dx = cell2.position.x - cell1.position.x
        dy = cell2.position.y - cell1.position.y
        
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Adhesion only applies within a certain range
        if distance > 0 and distance < self.adhesion_range * (cell1.radius + cell2.radius):
            # Force increases when cells approach optimal adhesion distance,
            # then decreases as they get too close
            optimal_distance = 0.9 * (cell1.radius + cell2.radius)
            
            if distance < optimal_distance:
                # Repulsive component when too close
                force_magnitude = self.adhesion_strength * (distance - optimal_distance) / optimal_distance
            else:
                # Attractive component at longer distances
                force_magnitude = self.adhesion_strength * (1.0 - (distance - optimal_distance) / 
                                                   (self.adhesion_range * (cell1.radius + cell2.radius) - optimal_distance))
            
            force_x = force_magnitude * dx / distance
            force_y = force_magnitude * dy / distance
            
            return np.array([force_x, force_y])
        
        return np.array([0, 0], dtype=np.float64) # Ensure adhesion returns float64 array


@numba.jit(nopython=True) # Not parallel itself, called within parallel outer loop
def _apply_repulsion_force_numba(vertex_positions, vertex_velocities, other_cell_pos, other_cell_radius, repulsion_strength):
    """
    Numba-jitted kernel to check vertex-sphere collision and apply repulsion force.
    Modifies vertex_velocities in-place.
    """
    num_vertices = vertex_positions.shape[0]

    for i in range(num_vertices):
        vx = vertex_positions[i, 0]
        vy = vertex_positions[i, 1]
        other_x = other_cell_pos[0]
        other_y = other_cell_pos[1]

        # Check if vertex is inside the other cell's circle approximation
        dx = vx - other_x
        dy = vy - other_y
        distance = np.sqrt(dx*dx + dy*dy)

        if distance < other_cell_radius:
            # Calculate penetration depth
            penetration = other_cell_radius - distance

            # Calculate normal vector (direction from other cell center to vertex)
            if distance > 1e-9: # Avoid division by zero
                normal_x = dx / distance
                normal_y = dy / distance
            else:
                # If centers coincide, use a fixed small vector if distance is zero
                normal_x = 0.1 # Small non-zero vector
                normal_y = 0.1

            # Apply repulsion force: force = normal * penetration * strength
            # We apply this as an impulse, directly changing velocity
            impulse_x = normal_x * penetration * repulsion_strength
            impulse_y = normal_y * penetration * repulsion_strength

            # Add impulse to vertex velocity
            vertex_velocities[i, 0] += impulse_x
            vertex_velocities[i, 1] += impulse_y


class CollisionResolver:
    """Resolves collisions between deformable cells."""
    def __init__(self, repulsion_strength=0.5):
        self.repulsion_strength = repulsion_strength
    
    # Removed check_vertex_collision as its logic is integrated into the jitted function

# Standalone function for collision resolution
@numba.jit(nopython=True) # Remove parallel for now to simplify debugging
def _resolve_collisions_numba(cell_positions, cell_radii, all_vertex_positions, all_vertex_velocities, cell_vertex_offsets, repulsion_strength):
    """
    Numba-jitted kernel to resolve collisions between all cells.
    Operates on flattened arrays of vertex data.
    Modifies all_vertex_velocities in-place.
    """
    num_cells = cell_positions.shape[0]

    # Use range instead of prange since we're not using parallel processing
    for i in range(num_cells):
        # Get cell i data
        cell1_pos = cell_positions[i]
        cell1_radius = cell_radii[i]
        # Get the slice of vertex data for cell i
        v_start_i = cell_vertex_offsets[i]
        v_end_i = cell_vertex_offsets[i+1] # offset[num_cells] is total vertices
        
        # Need to make a writable copy of the slice if modifying it directly?
        # Numba usually handles slices of mutable arrays correctly in nopython mode.
        # Let's assume all_vertex_velocities is C-contiguous which makes slices writable.
        cell1_vertex_positions = all_vertex_positions[v_start_i : v_end_i]
        cell1_vertex_velocities = all_vertex_velocities[v_start_i : v_end_i] # This slice needs to be writable

        for j in range(i + 1, num_cells):
            # Get cell j data
            cell2_pos = cell_positions[j]
            cell2_radius = cell_radii[j]
            # Get the slice of vertex data for cell j
            v_start_j = cell_vertex_offsets[j]
            v_end_j = cell_vertex_offsets[j+1]
            cell2_vertex_positions = all_vertex_positions[v_start_j : v_end_j]
            cell2_vertex_velocities = all_vertex_velocities[v_start_j : v_end_j] # This slice needs to be writable!

            # Quick check using bounding circles
            dx = cell2_pos[0] - cell1_pos[0]
            dy = cell2_pos[1] - cell1_pos[1]
            distance = np.sqrt(dx*dx + dy*dy)

            if distance < cell1_radius + cell2_radius:
                # Detailed check using vertex-sphere collision and apply repulsion

                # Apply repulsion from cell2 to cell1's vertices
                _apply_repulsion_force_numba(
                    cell1_vertex_positions, cell1_vertex_velocities,
                    cell2_pos, cell2_radius, repulsion_strength
                )

                # Apply repulsion from cell1 to cell2's vertices (note reversed effect implicitly handled by _apply_repulsion_force_numba)
                _apply_repulsion_force_numba(
                    cell2_vertex_positions, cell2_vertex_velocities,
                    cell1_pos, cell1_radius, repulsion_strength
                )


class CollisionResolver:
    """Resolves collisions between deformable cells."""
    def __init__(self, repulsion_strength=0.5):
        self.repulsion_strength = repulsion_strength
    
    # Removed check_vertex_collision as its logic is integrated into the jitted function

    def resolve_collisions(self, cells):
        """
        Resolves collisions between deformable cells using a Numba-jitted kernel.
        """
        num_cells = len(cells)
        if num_cells < 2:
            return # Nothing to collide

        # Extract data into NumPy arrays for Numba
        cell_positions = np.array([cell.get_position() for cell in cells], dtype=np.float64)
        cell_radii = np.array([cell.physical_cell.radius if hasattr(cell, 'physical_cell') else cell.radius for cell in cells], dtype=np.float64)
        
        # Collect all vertex positions and velocities into flattened arrays
        all_vertex_positions = []
        all_vertex_velocities = []
        # Store offsets to know which vertices belong to which cell
        cell_vertex_offsets = [0] # Offset for the start of each cell's vertices
        
        for cell in cells:
             actual_cell = cell.physical_cell if hasattr(cell, 'physical_cell') else cell
             vp_array = actual_cell.get_vertex_positions_array()
             vv_array = actual_cell.get_vertex_velocities_array()
             all_vertex_positions.append(vp_array)
             all_vertex_velocities.append(vv_array)
             cell_vertex_offsets.append(cell_vertex_offsets[-1] + vp_array.shape[0])

        # Concatenate into large flat arrays
        all_vertex_positions = np.concatenate(all_vertex_positions, axis=0)
        all_vertex_velocities = np.concatenate(all_vertex_velocities, axis=0) # This array will be modified
        cell_vertex_offsets = np.array(cell_vertex_offsets, dtype=np.int64)

        # Call the Numba-jitted collision resolution kernel
        # The kernel modifies all_vertex_velocities in-place
        _resolve_collisions_numba(
            cell_positions, cell_radii,
            all_vertex_positions, all_vertex_velocities, # Pass the velocities array to be modified
            cell_vertex_offsets, self.repulsion_strength
        )

        # Update the Vector2D velocities back in the original DeformableCell objects
        current_offset = 0
        for cell in cells:
            actual_cell = cell.physical_cell if hasattr(cell, 'physical_cell') else cell
            num_vertices = actual_cell.num_vertices
            # Get the updated velocities slice for this cell
            updated_velocities = all_vertex_velocities[current_offset : current_offset + num_vertices]
            
            # Update the internal Vector2D velocity list for the cell
            for i in range(num_vertices):
                 actual_cell.vertex_velocities[i].x = updated_velocities[i, 0]
                 actual_cell.vertex_velocities[i].y = updated_velocities[i, 1]
            
            current_offset += num_vertices

        # Note: Vertex positions were not changed by the collision resolver, only velocities.
        # The positions will be updated in the cell's own update() method.
        # If collisions *did* need to change positions (e.g. pushing apart), that logic
        # would also need to be included in the jitted function and positions updated back here.
