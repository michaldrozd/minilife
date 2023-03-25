import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# --------------------------- Phase 1: Abiogenesis ---------------------------
class Chemical:
    def __init__(self, complexity):
        self.complexity = complexity

    def replicate(self):
        # With a low probability, the replication "upgrades" complexity.
        if np.random.rand() < 0.001:
            return Chemical(self.complexity + 1)
        return Chemical(self.complexity)


def simulate_abiogenesis(max_iter=10000, complexity_threshold=10):
    chemicals = [Chemical(1)]
    iteration = 0

    fig, ax = plt.subplots()
    ax.set_title("Abiogenesis: Chemical Complexity Distribution")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Count")

    def update(frame):
        nonlocal chemicals, iteration
        iteration += 1
        new_chems = [chem.replicate() for chem in chemicals]
        chemicals.extend(new_chems)
        complexities = [chem.complexity for chem in chemicals]
        ax.cla()
        ax.set_title(f"Iteration {iteration}, Mean Complexity: {np.mean(complexities):.2f}")
        ax.set_xlabel("Complexity")
        ax.set_ylabel("Count")
        ax.hist(complexities, bins=np.arange(1, max(complexities) + 2), color='blue', alpha=0.7)
        # Stop if any chemical reaches the threshold.
        if any(c.complexity >= complexity_threshold for c in chemicals):
            ani.event_source.stop()

    ani = FuncAnimation(fig, update, interval=10)
    plt.show()
    print(f"[Abiogenesis] Complete at iteration {iteration} with {len(chemicals)} chemicals.")
    return chemicals


# --------------------------- Phase 2: Single–Cell Life with Detailed Physics ---------------------------
class Cell:
    def __init__(self, pos, energy, dna_complexity):
        self.pos = np.array(pos, dtype=float)
        self.energy = energy
        self.dna_complexity = dna_complexity
        self.velocity = np.zeros(2)  # New: velocity vector for momentum.
        self.age = 0  # New: track age.

    def compute_nutrient_gradient(self, nutrient_field, grid_size):
        # Approximate the gradient using central differences.
        i = int(self.pos[0]) % grid_size
        j = int(self.pos[1]) % grid_size
        left = nutrient_field[(i - 1) % grid_size, j]
        right = nutrient_field[(i + 1) % grid_size, j]
        up = nutrient_field[i, (j + 1) % grid_size]
        down = nutrient_field[i, (j - 1) % grid_size]
        grad_x = (right - left) / 2.0
        grad_y = (up - down) / 2.0
        return np.array([grad_x, grad_y])

    def move(self, nutrient_field, grid_size):
        # Compute the local nutrient gradient.
        grad = self.compute_nutrient_gradient(nutrient_field, grid_size)
        # Compute "hunger" as energy deficit (if energy < 20, the cell is hungry).
        hunger = max(0, 20 - self.energy)
        # Force is proportional to both the gradient and the hunger.
        force = grad * hunger * 0.1
        # Update velocity (assuming unit mass) and apply friction.
        self.velocity += force
        self.velocity *= 0.9  # friction/damping
        # Update position with the current velocity.
        self.pos += self.velocity
        self.pos = np.mod(self.pos, grid_size)

    def metabolize(self):
        # Energy gain scaled by genetic complexity minus a base cost.
        self.energy += self.dna_complexity * 0.2 - 1.0
        self.age += 1  # Increase age.

    def replicate(self):
        # Replication occurs if energy is high enough and cell is not too young.
        if self.energy > 30 and self.age > 5:
            mutation = np.random.choice([-1, 0, 1])
            new_dna = max(1, self.dna_complexity + mutation)
            offspring_energy = self.energy / 2
            self.energy /= 2
            return Cell(self.pos.copy(), offspring_energy, new_dna)
        return None


class SingleCellSimulation:
    def __init__(self, grid_size=100, initial_cells=10):
        self.grid_size = grid_size
        self.cells = []
        for _ in range(initial_cells):
            pos = np.random.rand(2) * grid_size
            self.cells.append(Cell(pos, energy=20, dna_complexity=10))
        # Create a nutrient field over the grid.
        self.nutrient = np.full((grid_size, grid_size), 5.0, dtype=float)

    def diffuse_nutrient(self):
        # Use a simple averaging of neighbors as a diffusion step.
        new_nutrient = self.nutrient.copy()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                total = 0
                count = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni = (i + di) % self.grid_size
                        nj = (j + dj) % self.grid_size
                        total += self.nutrient[ni, nj]
                        count += 1
                new_nutrient[i, j] = total / count
        new_nutrient += 0.2  # Add a small constant nutrient source.
        self.nutrient = new_nutrient

    def update(self):
        self.diffuse_nutrient()
        new_cells = []
        for cell in self.cells:
            cell.move(self.nutrient, self.grid_size)
            # Cell consumes nutrient at its current grid location.
            i, j = int(cell.pos[0]) % self.grid_size, int(cell.pos[1]) % self.grid_size
            consumption = min(0.8, self.nutrient[i, j])
            cell.energy += consumption
            self.nutrient[i, j] -= consumption
            cell.metabolize()
            offspring = cell.replicate()
            if offspring:
                new_cells.append(offspring)
        self.cells.extend(new_cells)
        # Remove cells that run out of energy or get too old.
        self.cells = [cell for cell in self.cells if cell.energy > 0 and cell.age < 100]

    def get_cell_positions(self):
        return np.array([cell.pos for cell in self.cells])


def simulate_single_cell():
    sim = SingleCellSimulation(grid_size=100, initial_cells=10)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Single-Cell Life Simulation with Detailed Physics")
    nutrient_im = ax.imshow(sim.nutrient, cmap='YlGn', origin='lower', extent=[0, sim.grid_size, 0, sim.grid_size])
    scatter = ax.scatter([], [], c='red')

    def update(frame):
        sim.update()
        nutrient_im.set_data(sim.nutrient)
        positions = sim.get_cell_positions()
        scatter.set_offsets(positions)
        ax.set_title(f"Single-Cell Life: Frame {frame}, Cells: {len(sim.cells)}")
        if len(sim.cells) > 150:
            ani.event_source.stop()

    ani = FuncAnimation(fig, update, interval=100)
    plt.show()
    print(f"[Single-Cell Life] Complete with {len(sim.cells)} cells.")
    return sim.cells


# --------------------------- Phase 3: Multicellularity ---------------------------
class MulticellularOrganism:
    def __init__(self, cells):
        self.cells = cells
        self.pos = np.mean([cell.pos for cell in cells], axis=0)
        self.brain_complexity = np.mean([cell.dna_complexity for cell in cells])
        self.energy = sum(cell.energy for cell in cells)
        self.age = 0

    def update(self, grid_size):
        # Organism movement: a simple random shift scaled by brain complexity.
        angle = np.random.rand() * 2 * np.pi
        step = (self.brain_complexity / 20.0)
        self.pos += np.array([np.cos(angle), np.sin(angle)]) * step
        self.pos = np.mod(self.pos, grid_size)
        self.energy -= step * 0.5
        if self.energy > 50:
            self.energy += np.log(self.brain_complexity + 1)
        self.age += 1

    def replicate(self):
        if self.energy > 150 and self.age > 10:
            mutation = np.random.choice([-1, 0, 1])
            new_brain = max(1, self.brain_complexity + mutation)
            self.energy /= 2
            new_cells = [Cell(self.pos.copy(), energy=self.energy / len(self.cells),
                              dna_complexity=cell.dna_complexity + mutation)
                         for cell in self.cells]
            return MulticellularOrganism(new_cells)
        return None


class MulticellularSimulation:
    def __init__(self, cells, grid_size=100):
        self.organisms = [MulticellularOrganism([cell]) for cell in cells]
        self.grid_size = grid_size

    def update(self):
        new_orgs = []
        for org in self.organisms:
            org.update(self.grid_size)
            offspring = org.replicate()
            if offspring:
                new_orgs.append(offspring)
        self.organisms.extend(new_orgs)
        self.organisms = [org for org in self.organisms if org.energy > 0 and org.age < 200]

    def get_positions_and_complexity(self):
        positions = np.array([org.pos for org in self.organisms])
        complexities = np.array([org.brain_complexity for org in self.organisms])
        return positions, complexities


def simulate_multicellularity(cells):
    sim = MulticellularSimulation(cells, grid_size=100)
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter([], [], c='blue', cmap='viridis', s=50)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Multicellular Organisms Simulation")

    def update(frame):
        sim.update()
        positions, complexities = sim.get_positions_and_complexity()
        scatter.set_offsets(positions)
        scatter.set_array(complexities)
        ax.set_title(f"Multicellularity: Frame {frame}, Organisms: {len(sim.organisms)}")
        if len(sim.organisms) > 70:
            ani.event_source.stop()

    ani = FuncAnimation(fig, update, interval=200)
    plt.show()
    print(f"[Multicellularity] Complete with {len(sim.organisms)} organisms.")
    return sim.organisms


# --------------------------- Phase 4: Emergence of Intelligence ---------------------------
# A deeper neural network with two hidden layers is used.
class DeepNeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        self.W1 = np.random.randn(input_size, hidden1_size)
        self.b1 = np.random.randn(hidden1_size)
        self.W2 = np.random.randn(hidden1_size, hidden2_size)
        self.b2 = np.random.randn(hidden2_size)
        self.W3 = np.random.randn(hidden2_size, output_size)
        self.b3 = np.random.randn(output_size)

    def forward(self, x):
        h1 = np.tanh(np.dot(x, self.W1) + self.b1)
        h2 = np.tanh(np.dot(h1, self.W2) + self.b2)
        out = np.tanh(np.dot(h2, self.W3) + self.b3)
        return out

    def mutate(self):
        self.W1 += np.random.randn(*self.W1.shape) * 0.05
        self.b1 += np.random.randn(*self.b1.shape) * 0.05
        self.W2 += np.random.randn(*self.W2.shape) * 0.05
        self.b2 += np.random.randn(*self.b2.shape) * 0.05
        self.W3 += np.random.randn(*self.W3.shape) * 0.05
        self.b3 += np.random.randn(*self.b3.shape) * 0.05


class IntelligentOrganism:
    def __init__(self, multicell_org, neural_net):
        self.org = multicell_org
        self.nn = neural_net
        self.pos = multicell_org.pos.copy()
        self.brain_complexity = multicell_org.brain_complexity
        self.energy = multicell_org.energy
        self.hunger = 0  # New state variable.

    def sense_and_act(self):
        # Simulate sensing the environment: random features plus the hunger state.
        env_input = np.random.randn(5)
        input_vector = np.concatenate([env_input, [self.hunger * 0.1]])
        output = self.nn.forward(input_vector)
        # First two outputs adjust position.
        self.pos += output[:2]
        self.pos = np.mod(self.pos, 100)
        # Third output adjusts energy (as a proxy for decision-making success).
        self.energy += output[2] * 0.5
        # Update hunger: if energy is low, set hunger flag.
        self.hunger = 1 if self.energy < 50 else 0

    def update(self):
        self.sense_and_act()
        self.energy -= 1  # Baseline energy cost for neural activity.
        self.org.energy = self.energy  # Sync with the underlying organism.

    def replicate(self):
        if self.energy > 250:
            new_nn = DeepNeuralNetwork(6, 12, 8, 3)
            new_nn.W1 = self.nn.W1.copy()
            new_nn.b1 = self.nn.b1.copy()
            new_nn.W2 = self.nn.W2.copy()
            new_nn.b2 = self.nn.b2.copy()
            new_nn.W3 = self.nn.W3.copy()
            new_nn.b3 = self.nn.b3.copy()
            new_nn.mutate()
            self.energy /= 2
            return IntelligentOrganism(self.org, new_nn)
        return None


class IntelligenceSimulation:
    def __init__(self, multicell_orgs):
        self.intelligent_orgs = [IntelligentOrganism(org, DeepNeuralNetwork(6, 12, 8, 3)) for org in multicell_orgs]

    def update(self):
        new_intels = []
        for intel in self.intelligent_orgs:
            intel.update()
            offspring = intel.replicate()
            if offspring:
                new_intels.append(offspring)
        self.intelligent_orgs.extend(new_intels)
        self.intelligent_orgs = [intel for intel in self.intelligent_orgs if intel.energy > 0]

    def get_positions_and_energy(self):
        positions = np.array([intel.pos for intel in self.intelligent_orgs])
        energies = np.array([intel.energy for intel in self.intelligent_orgs])
        return positions, energies


def simulate_intelligence(multicell_orgs):
    sim = IntelligenceSimulation(multicell_orgs)
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter([], [], c='magenta', s=50)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Emergence of Intelligence Simulation")

    def update(frame):
        sim.update()
        positions, energies = sim.get_positions_and_energy()
        scatter.set_offsets(positions)
        ax.set_title(f"Intelligence: Frame {frame}, Organisms: {len(sim.intelligent_orgs)}")
        if frame % 10 == 0:
            print(f"Frame {frame}: Avg Energy = {np.mean(energies) if len(energies) > 0 else 0:.2f}")
        if frame > 300:
            ani.event_source.stop()

    ani = FuncAnimation(fig, update, interval=100)
    plt.show()
    print(f"[Intelligence] Complete with {len(sim.intelligent_orgs)} intelligent organisms.")
    return sim.intelligent_orgs


# --------------------------- Main Simulation ---------------------------
def main():
    print("Starting Advanced Evolutionary Simulation")

    # Phase 1: Abiogenesis
    chemicals = simulate_abiogenesis(max_iter=10000, complexity_threshold=10)
    input("Press Enter to proceed to Single-Cell Life simulation...")

    # Phase 2: Single-Cell Life
    cells = simulate_single_cell()
    input("Press Enter to proceed to Multicellularity simulation...")

    # Phase 3: Multicellularity
    multicell_orgs = simulate_multicellularity(cells)
    input("Press Enter to proceed to Intelligence simulation...")

    # Phase 4: Emergence of Intelligence
    intelligent_orgs = simulate_intelligence(multicell_orgs)

    print("Advanced Evolutionary Simulation Complete.")


if __name__ == "__main__":
    main()
