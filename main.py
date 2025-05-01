"""
Main entry point for the evolution simulation.
Orchestrates the simulation phases from abiogenesis to intelligence.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import sys

# Import simulation phases
from abiogenesis.chemical import Chemical  # Placeholder for now
from abiogenesis.chemical import Chemical
from abiogenesis.simulation import run_abiogenesis_simulation # Import the new abiogenesis simulation function
from cellular.simulation import run_cellular_simulation, AdvancedCellularSimulation
from cellular.cell import AdvancedCell  # For creating test cells
from multicellular.simulation import run_multicellular_simulation
from intelligence.simulation import run_intelligence_simulation # Import the new intelligence simulation function
import config # Import the config module

def visualize_only(vis_type=None):
    """
    Run just the visualization on a pre-initialized simulation.

    Args:
        vis_type: Optional type of visualization ("cell", "multicell", or "intelligence")
    """
    # If vis_type not provided, ask for it
    if vis_type is None:
        print("Which simulation phase would you like to visualize?")
        print("1. Cellular Life")
        print("2. Multicellular Life")
        print("3. Intelligence")

        try:
            choice = input("Enter your choice (1/2/3): ")
            if choice == "1":
                vis_type = "cell"
            elif choice == "2":
                vis_type = "multicell"
            elif choice == "3":
                vis_type = "intelligence"
            else:
                print("Invalid choice. Running cellular simulation by default.")
                vis_type = "cell"
        except Exception as e:
            print(f"Error: {e}")
            print("Running cellular simulation by default.")
            vis_type = "cell"

    # Create and run a simulation without visualization (since it doesn't work well in this environment)
    # Instead, we'll print out information about the simulation progress
    if vis_type == "intelligence":
        print("Initializing intelligence simulation...")
        from intelligence.simulation import IntelligenceSimulation
        from visualization.intelligence_vis import visualize_intelligence_simulation

        # Create a simulation with initial organisms
        simulation = IntelligenceSimulation()

        # Run the simulation with visualization
        print("Running intelligence simulation with visualization...")
        visualize_intelligence_simulation(simulation, frames=200, interval=50)

    elif vis_type == "multicell":
        print("Initializing multicellular simulation...")
        from multicellular.simulation import MulticellularSimulation
        from visualization.multicell_vis import visualize_multicellular_simulation

        # Create a simulation with a single initial organism
        initial_org = create_test_organism()
        simulation = MulticellularSimulation(initial_organisms=[initial_org])

        # Run the simulation with visualization
        print("Running multicellular simulation with visualization...")
        visualize_multicellular_simulation(simulation, frames=200, interval=50)

    else:
        # Default to cellular simulation
        print("Initializing cellular simulation...")
        from visualization.visualizer import visualize_cellular_simulation

        simulation = AdvancedCellularSimulation()
        for _ in range(5):  # Add some initial cells
            pos = np.random.rand(2) * 100
            simulation.cells.append(AdvancedCell(pos, energy=20, dna_complexity=10))

        # Run the simulation with visualization
        print("Running cellular simulation with visualization...")
        visualize_cellular_simulation(simulation, frames=200, interval=50)


from utils.helpers import create_test_organism


def main():
    """Main entry point for the simulation."""
    parser = argparse.ArgumentParser(description="Evolution Simulation")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["abiogenesis", "cell", "multicellular", "intelligence", "all", "visualize"],
                        help="Which simulation phase to run (default: all)")
    parser.add_argument("--vis-type", type=str, choices=["cell", "multicell", "intelligence"],
                        help="Type of visualization to use with --phase visualize")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Maximum number of iterations per phase (default: 500)")
    parser.add_argument("--no-vis", action="store_true",
                        help="Disable visualization (faster simulation)")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Run without interactive prompts (for batch/CI processing)")

    args = parser.parse_args()

    print("Starting Advanced Evolutionary Simulation")
    np.random.seed(42)  # For reproducibility

    if args.phase == "visualize":
        visualize_only(vis_type=args.vis_type)
        return

    # Phase 1: Abiogenesis
    # Phase 1: Abiogenesis
    chemicals = []
    if args.phase in ["abiogenesis", "all"]:
        chemicals = run_abiogenesis_simulation(max_iter=args.iterations, complexity_threshold=config.CHEMICAL_COMPLEXITY_THRESHOLD)
        if not args.non_interactive:
            try:
                input("Press Enter to proceed to Single-Cell Life simulation...")
            except EOFError:
                print("Running in non-interactive mode, continuing automatically...")

    # Phase 2: Single-Cell Life
    cells = []
    if args.phase in ["cell", "all"]:
        # Use chemicals from previous phase as potential starting point (basic placeholder)
        # In a real scenario, this would involve seeding the environment/simulation based on chemicals

        # For now, still use the initial cells from config/helper, but acknowledge chemicals
        print(f"Starting Single-Cell Life simulation, potentially influenced by {len(chemicals)} complex chemicals...")

        cells = run_cellular_simulation(max_iterations=args.iterations)

        # Instead of visualization, print summary of cells
        if cells and cells.cells: # Check if simulation object exists and has cells
            print("\nCellular simulation summary:")
            cell_count = len(cells.cells)
            cell_types = {}
            for cell in cells.cells:
                cell_type, _ = cell.get_cell_type()
                cell_types[cell_type] = cell_types.get(cell_type, 0) + 1

            mean_energy = np.mean([cell.energy for cell in cells.cells]) if cells.cells else 0
            mean_dna = np.mean([cell.dna_complexity for cell in cells.cells]) if cells.cells else 0

            print(f"Total cells: {cell_count}")
            print(f"Mean energy: {mean_energy:.2f}")
            print(f"Mean DNA complexity: {mean_dna:.2f}")
            print(f"Cell types: {', '.join(f'{t}: {c}' for t, c in cell_types.items() if c > 0)}")

        if not args.non_interactive:
            try:
                input("Press Enter to proceed to Multicellularity simulation...")
            except EOFError:
                print("Running in non-interactive mode, continuing automatically...")

    # Phase 3: Multicellularity
    multicell_simulation = None
    if args.phase in ["multicellular", "all"]:
        # Use cells from previous phase as initial input
        # Pass the list of cells from the cellular simulation object
        multicell_simulation = run_multicellular_simulation(cells=cells.cells if cells else [], max_iterations=args.iterations)

        # Instead of visualization, print summary of organisms
        if multicell_simulation and multicell_simulation.organisms: # Check if simulation object exists and has organisms
            print("\nMulticellular simulation summary:")
            org_count = len(multicell_simulation.organisms)
            total_cells = sum(len(org.cells) for org in multicell_simulation.organisms)
            mean_size = total_cells / org_count if org_count > 0 else 0
            mean_complexity = np.mean([org.complexity for org in multicell_simulation.organisms]) if multicell_simulation.organisms else 0
            neural_orgs = sum(1 for org in multicell_simulation.organisms if org.nervous_system is not None)

            print(f"Total organisms: {org_count}")
            print(f"Total cells: {total_cells}")
            print(f"Mean organism size: {mean_size:.2f} cells")
            print(f"Mean complexity: {mean_complexity:.2f}")
            print(f"Organisms with neural systems: {neural_orgs}")

            # Print details of a few organisms
            for i, org in enumerate(multicell_simulation.organisms[:3]):  # Show details for up to 3 organisms
                cell_types = {}
                for cell in org.cells:
                    cell_type, _ = cell.get_cell_type()
                    cell_types[cell_type] = cell_types.get(cell_type, 0) + 1

                print(f"\nOrganism {i}:")
                print(f"  Cells: {len(org.cells)}")
                print(f"  Energy: {org.energy:.1f}")
                print(f"  Cell types: {', '.join(f'{t}: {c}' for t, c in cell_types.items() if c > 0)}")
                print(f"  Neural cells: {len(org.nervous_system['cells']) if org.nervous_system else 0}")

        if not args.non_interactive:
            try:
                input("Press Enter to proceed to Intelligence simulation...")
            except EOFError:
                print("Running in non-interactive mode, continuing automatically...")

    # Phase 4: Intelligence
    if args.phase in ["intelligence", "all"]:
        # Use organisms from previous phase as initial input
        # Pass the list of organisms from the simulation object
        organisms_list = multicell_simulation.organisms if multicell_simulation else []

        if args.no_vis:
            # Run without visualization
            run_intelligence_simulation(organisms_list, max_iterations=args.iterations)
        else:
            # Run with visualization
            from intelligence.simulation import IntelligenceSimulation
            from visualization.intelligence_vis import visualize_intelligence_simulation

            # Create the simulation
            simulation = IntelligenceSimulation(initial_organisms=organisms_list)

            # Run with visualization
            print("Running intelligence simulation with visualization...")
            visualize_intelligence_simulation(simulation, frames=args.iterations, interval=50)

    print("Advanced Evolutionary Simulation Complete.")


if __name__ == "__main__":
    main()
