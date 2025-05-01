"""
Simulation of the emergence of intelligence in multicellular organisms.
This module implements a more advanced simulation with learning and adaptation.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multicellular.organism import MulticellularOrganism
from intelligence.neural_network import IntelligentOrganism, EvolvableNeuralNetwork
from physics.diffusion import MultiSpeciesReactionDiffusion
import config


class IntelligenceSimulation:
    """
    Simulates the emergence of intelligence in a population of organisms.
    """
    def __init__(self, initial_organisms=None, grid_size=config.GRID_SIZE):
        """
        Initialize the intelligence simulation.

        Args:
            initial_organisms: List of multicellular organisms to start with
            grid_size: Size of the environment grid
        """
        self.grid_size = grid_size
        self.organisms = []

        # Convert multicellular organisms to intelligent organisms
        if initial_organisms:
            for org in initial_organisms:
                # Only use organisms with neural systems
                if org.nervous_system and len(org.nervous_system['cells']) >= 3:
                    intelligent_org = IntelligentOrganism(base_organism=org)
                    intelligent_org.position = org.position
                    self.organisms.append(intelligent_org)

        # If no valid organisms provided, create some from scratch
        if not self.organisms:
            for _ in range(5):
                org = IntelligentOrganism()
                org.position = np.random.rand(2) * grid_size
                self.organisms.append(org)

        # Create a complex environment with multiple chemical species
        self.environment = MultiSpeciesReactionDiffusion(
            grid_size,
            num_species=5,
            pattern_type=config.ENVIRONMENT_PATTERN_TYPE
        )

        # Set environmental variability
        self.environment.variability = config.ENVIRONMENT_VARIABILITY
        self.environment.seasonal_strength = config.ENVIRONMENT_SEASONAL_STRENGTH

        # Add some initial nutrients
        self.add_nutrients(pattern='random')

        # Statistics tracking
        self.iteration = 0
        self.stats = {
            'organism_count': [],
            'mean_energy': [],
            'mean_fitness': [],
            'mean_age': [],
            'births': 0,
            'deaths': 0,
            'neural_complexity': []
        }

    def update(self):
        """Update the simulation for one timestep."""
        self.iteration += 1

        # Update environment
        self.environment.update(dt=0.2)

        # Update each organism
        rewards = []
        for organism in self.organisms:
            reward = organism.update(self.environment, self.organisms)
            rewards.append(reward)

        # Handle reproduction
        new_organisms = []
        for organism in self.organisms:
            offspring = organism.reproduce()
            if offspring:
                new_organisms.append(offspring)
                self.stats['births'] += 1

        # Add new organisms
        self.organisms.extend(new_organisms)

        # Remove dead organisms
        old_count = len(self.organisms)
        self.organisms = [org for org in self.organisms if org.energy > 0]
        self.stats['deaths'] += old_count - len(self.organisms)

        # Add nutrients even more frequently
        if self.iteration % 5 == 0:
            self.add_nutrients(pattern='random', amount=1.0)

        # Update statistics
        self.stats['organism_count'].append(len(self.organisms))
        if self.organisms:
            self.stats['mean_energy'].append(np.mean([org.energy for org in self.organisms]))
            self.stats['mean_fitness'].append(np.mean([org.fitness for org in self.organisms]))
            self.stats['mean_age'].append(np.mean([org.age for org in self.organisms]))

            # Calculate neural complexity (sum of absolute weights)
            complexities = []
            for org in self.organisms:
                weights_sum = (
                    np.sum(np.abs(org.neural_network.weights_input_hidden1)) +
                    np.sum(np.abs(org.neural_network.weights_hidden1_hidden2)) +
                    np.sum(np.abs(org.neural_network.weights_hidden2_output))
                )
                complexities.append(weights_sum)
            self.stats['neural_complexity'].append(np.mean(complexities))
        else:
            self.stats['mean_energy'].append(0)
            self.stats['mean_fitness'].append(0)
            self.stats['mean_age'].append(0)
            self.stats['neural_complexity'].append(0)

    def add_nutrients(self, amount=0.5, pattern='center'):
        """
        Add nutrients to the environment.

        Args:
            amount: Amount of nutrients to add
            pattern: Pattern of nutrient distribution ('center', 'gradient', 'random')
        """
        if pattern == 'center':
            # Create a circular nutrient source in the center
            center = self.grid_size // 2
            radius = self.grid_size // 4

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    dist = np.sqrt((i - center)**2 + (j - center)**2)
                    if dist < radius:
                        self.environment.concentrations[0, i, j] += amount * (1 - dist / radius)

        elif pattern == 'gradient':
            # Create a gradient of nutrients
            for i in range(self.grid_size):
                gradient = i / self.grid_size
                for j in range(self.grid_size):
                    self.environment.concentrations[0, i, j] += amount * gradient

        elif pattern == 'random':
            # Create random patches of nutrients
            for _ in range(5):
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                radius = np.random.randint(5, 15)

                for i in range(max(0, x - radius), min(self.grid_size, x + radius)):
                    for j in range(max(0, y - radius), min(self.grid_size, y + radius)):
                        dist = np.sqrt((i - x)**2 + (j - y)**2)
                        if dist < radius:
                            self.environment.concentrations[0, i, j] += amount * (1 - dist / radius)

    def get_organism_data(self):
        """Get data about all organisms for visualization."""
        positions = []
        energies = []
        ages = []
        fitnesses = []

        for organism in self.organisms:
            positions.append(organism.position)
            energies.append(organism.energy)
            ages.append(organism.age)
            fitnesses.append(organism.fitness)

        return {
            'positions': np.array(positions) if positions else np.zeros((0, 2)),
            'energies': np.array(energies),
            'ages': np.array(ages),
            'fitnesses': np.array(fitnesses)
        }

    def get_environment_data(self):
        """Get environment data for visualization."""
        return {
            'nutrient': self.environment.get_nutrient_field(),
            'signals': self.environment.get_signal_fields()
        }

    def get_neural_networks(self):
        """Get neural network data for visualization."""
        networks = []

        for organism in self.organisms:
            networks.append(organism.neural_network.get_network_state())

        return networks

    def get_stats(self):
        """Get simulation statistics."""
        return self.stats


def run_intelligence_simulation(organisms, max_iterations=config.INTELLIGENCE_MAX_FRAMES):
    """
    Run the intelligence simulation for a specified number of iterations.

    Args:
        organisms: List of multicellular organisms to start with
        max_iterations: Maximum number of simulation iterations

    Returns:
        The final simulation state
    """
    print("Simulating intelligence...")

    # Create the simulation
    simulation = IntelligenceSimulation(initial_organisms=organisms)

    if not organisms:
        print("No organisms provided for intelligence simulation.")

    print(f"Starting Intelligence simulation with {len(simulation.organisms)} organisms for {max_iterations} iterations.")

    # Run simulation loop
    for iteration in range(max_iterations):
        # Update the simulation
        simulation.update()

        # Print progress occasionally
        if iteration % 10 == 0:
            org_count = len(simulation.organisms)
            if org_count > 0:
                mean_energy = np.mean([org.energy for org in simulation.organisms])
                mean_fitness = np.mean([org.fitness for org in simulation.organisms])
                print(f"Intelligence Iteration {iteration}: {org_count} organisms, " +
                      f"Mean energy: {mean_energy:.1f}, Mean fitness: {mean_fitness:.2f}")
            else:
                print(f"Intelligence Iteration {iteration}: All organisms have died.")
                break

    # Print summary
    if simulation.organisms:
        print(f"Intelligence simulation finished with {len(simulation.organisms)} surviving organisms.")

        # Find the most fit organism
        most_fit = max(simulation.organisms, key=lambda org: org.fitness)
        print(f"Most fit organism: Fitness={most_fit.fitness:.2f}, Age={most_fit.age}, Energy={most_fit.energy:.1f}")

        # Print some traits of the most fit organism
        print(f"Traits: {', '.join([f'{k}={v:.2f}' for k, v in most_fit.traits.items()])}")
    else:
        print("Intelligence simulation finished with no surviving organisms.")

    # Return the final organisms
    return simulation.organisms
