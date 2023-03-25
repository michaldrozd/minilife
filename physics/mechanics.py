"""
Physics-based cell mechanics including deformable cells, adhesion forces,
and collision detection/resolution.
"""

import numpy as np


class Vector2D:
    """A simple 2D vector class with basic operations."""
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
        
        return forces
    
    def apply_pressure_forces(self):
        """Apply pressure forces to maintain cell volume."""
        forces = [Vector2D(0, 0) for _ in range(self.num_vertices)]
        area = self.calculate_area()
        centroid = self.calculate_centroid()
        
        # Force proportional to difference from target area
        area_difference = self.target_area - area
        pressure_force = self.pressure_constant * area_difference
        
        for i in range(self.num_vertices):
            # Direction from centroid to vertex
            dx = self.vertices[i].x - centroid.x
            dy = self.vertices[i].y - centroid.y
            length = np.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                # Normalize and scale by pressure
                forces[i].x += pressure_force * dx / length
                forces[i].y += pressure_force * dy / length
        
        return forces
    
    def update(self, dt=0.1, grid_size=100):
        """Update the cell for one timestep."""
        # Calculate spring forces
        spring_forces = self.apply_spring_forces()
        
        # Calculate pressure forces
        pressure_forces = self.apply_pressure_forces()
        
        # Update vertex positions based on forces
        for i in range(self.num_vertices):
            # Sum all forces
            total_force = spring_forces[i] + pressure_forces[i]
            
            # Apply damping to velocity (F = -c*v)
            damping_x = -self.damping * self.vertex_velocities[i].x
            damping_y = -self.damping * self.vertex_velocities[i].y
            total_force.x += damping_x
            total_force.y += damping_y
            
            # Update velocity (v = v + a*dt = v + F/m*dt)
            self.vertex_velocities[i].x += total_force.x * dt / self.mass
            self.vertex_velocities[i].y += total_force.y * dt / self.mass
            
            # Update position (x = x + v*dt)
            self.vertices[i].x += self.vertex_velocities[i].x * dt
            self.vertices[i].y += self.vertex_velocities[i].y * dt
            
            # Apply periodic boundary conditions
            self.vertices[i].x %= grid_size
            self.vertices[i].y %= grid_size
        
        # Update cell position to the centroid
        self.position = self.calculate_centroid()
    
    def apply_external_force(self, force_vector):
        """Apply an external force to the cell."""
        force = Vector2D.from_array(force_vector)
        
        # Distribute force to all vertices
        for i in range(self.num_vertices):
            self.vertex_velocities[i].x += force.x * 0.1 / self.mass
            self.vertex_velocities[i].y += force.y * 0.1 / self.mass
    
    def get_membrane_points(self):
        """Get points representing the cell membrane for visualization."""
        return [(v.x, v.y) for v in self.vertices]
    
    def get_position(self):
        """Get the cell's current position as an array."""
        return np.array([self.position.x, self.position.y])


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
        
        return np.array([0, 0])


class CollisionResolver:
    """Resolves collisions between deformable cells."""
    def __init__(self, repulsion_strength=0.5):
        self.repulsion_strength = repulsion_strength
    
    def check_vertex_collision(self, vertex, other_cell):
        """Check if a vertex is inside another cell."""
        # For simplicity, we approximate the other cell as a circle
        dx = vertex.x - other_cell.position.x
        dy = vertex.y - other_cell.position.y
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance < other_cell.radius:
            # Calculate penetration depth
            penetration = other_cell.radius - distance
            
            # Calculate normal vector
            if distance > 0:
                normal_x = dx / distance
                normal_y = dy / distance
            else:
                # If centers coincide, use a random direction
                angle = np.random.uniform(0, 2*np.pi)
                normal_x = np.cos(angle)
                normal_y = np.sin(angle)
            
            return True, penetration, Vector2D(normal_x, normal_y)
        
        return False, 0, Vector2D(0, 0)
    
    def resolve_collisions(self, cells):
        """Resolve collisions between all cells."""
        num_cells = len(cells)
        
        for i in range(num_cells):
            for j in range(i+1, num_cells):
                cell1 = cells[i]
                cell2 = cells[j]
                
                # Quick check using bounding circles
                dx = cell2.position.x - cell1.position.x
                dy = cell2.position.y - cell1.position.y
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < cell1.radius + cell2.radius:
                    # Check vertex penetrations both ways
                    for vertex in cell1.vertices:
                        collision, penetration, normal = self.check_vertex_collision(vertex, cell2)
                        if collision:
                            # Apply repulsion force
                            repulsion_force = normal * penetration * self.repulsion_strength
                            
                            # Add to vertex velocity
                            index = cell1.vertices.index(vertex)
                            cell1.vertex_velocities[index].x += repulsion_force.x
                            cell1.vertex_velocities[index].y += repulsion_force.y
                    
                    for vertex in cell2.vertices:
                        collision, penetration, normal = self.check_vertex_collision(vertex, cell1)
                        if collision:
                            # Apply repulsion force (note reversed normal)
                            repulsion_force = normal * -penetration * self.repulsion_strength
                            
                            # Add to vertex velocity
                            index = cell2.vertices.index(vertex)
                            cell2.vertex_velocities[index].x += repulsion_force.x
                            cell2.vertex_velocities[index].y += repulsion_force.y