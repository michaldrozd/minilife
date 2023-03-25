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
from cellular.simulation import run_cellular_simulation, AdvancedCellularSimulation
from cellular.cell import AdvancedCell  # For creating test cells


def simulate_abiogenesis(max_iter=1000, complexity_threshold=10):
    """
    Simulate the origin of complex organic molecules.
    This is a placeholder until we implement the full abiogenesis module.
    """
    print("Simulating abiogenesis...")
    chemicals = [Chemical(1) for _ in range(10)]  # Placeholder
    # Add abiogenesis simulation code here
    return chemicals


def simulate_single_cell_life(max_iterations=500):
    """Simulate single-cell life with detailed biophysics."""
    print("Simulating single-cell life...")
    
    # Set up callback for visualization and reporting
    def visualization_callback(simulation, iteration):
        if iteration % 50 == 0:
            stats = simulation.get_stats()
            print(f"Iteration {iteration}: {stats['cell_count'][-1]} cells, "
                  f"mean energy: {stats['mean_energy'][-1]:.2f}, "
                  f"mean DNA complexity: {stats['mean_dna_complexity'][-1]:.2f}")
    
    # Define stop condition (stop if cell count exceeds threshold or drops to zero)
    def stop_condition(simulation):
        if len(simulation.cells) > 150:
            print("Cell population threshold reached!")
            return True
        if len(simulation.cells) == 0:
            print("All cells have died. Simulation ended.")
            return True
        return False
    
    # Run the simulation
    simulation = run_cellular_simulation(
        max_iterations=max_iterations,
        visualization_callback=visualization_callback,
        stop_condition=stop_condition
    )
    
    # Return the cells for the next simulation phase
    return simulation.cells


def simulate_multicellularity(cells, max_iterations=300):
    """Simulate the emergence of multicellular organisms."""
    print("Simulating multicellularity...")
    
    # Set up callback for visualization and reporting
    def visualization_callback(simulation, iteration):
        if iteration % 50 == 0:
            stats = simulation.get_stats()
            org_count = stats['organism_count'][-1] if stats['organism_count'] else 0
            cell_count = stats['total_cells'][-1] if stats['total_cells'] else 0
            neuron_count = stats['mean_neuron_count'][-1] if stats['mean_neuron_count'] else 0
            print(f"Iteration {iteration}: {org_count} organisms, "
                  f"{cell_count} total cells, {neuron_count:.2f} avg neurons")
    
    # Define stop condition (stop if organization count exceeds threshold or drops to zero)
    def stop_condition(simulation):
        if len(simulation.organisms) > 20:
            print("Organism population threshold reached!")
            return True
        if len(simulation.organisms) == 0 and simulation.iteration > 50:
            print("All organisms have died. Simulation ended.")
            return True
        return False
    
    # Run the multicellular simulation
    from multicellular.simulation import run_multicellular_simulation
    simulation = run_multicellular_simulation(
        cells=cells,
        max_iterations=max_iterations,
        visualization_callback=visualization_callback,
        stop_condition=stop_condition
    )
    
    # Return the organisms for the next simulation phase
    return simulation.organisms


def simulate_intelligence(organisms, max_iterations=300):
    """Simulate the emergence of intelligence."""
    print("Simulating intelligence...")
    
    # Placeholder for intelligence simulation
    print("Intelligence simulation not yet implemented in this version.")
    # Will be implemented in the next phase
    
    return []  # Return placeholder intelligent organisms


def visualize_only(vis_type=None):
    """
    Run just the visualization on a pre-initialized simulation.
    
    Args:
        vis_type: Optional type of visualization ("cell" or "multicell")
    """
    # If vis_type not provided, ask for it
    if vis_type is None:
        print("Which simulation phase would you like to visualize?")
        print("1. Cellular Life")
        print("2. Multicellular Life")
        
        try:
            choice = input("Enter your choice (1/2): ")
            if choice == "1":
                vis_type = "cell"
            elif choice == "2":
                vis_type = "multicell"
            else:
                print("Invalid choice. Running cellular simulation by default.")
                vis_type = "cell"
        except Exception as e:
            print(f"Error: {e}")
            print("Running cellular simulation by default.")
            vis_type = "cell"
    
    # Create and run a simulation without visualization (since it doesn't work well in this environment)
    # Instead, we'll print out information about the simulation progress
    if vis_type == "multicell":
        print("Initializing multicellular simulation...")
        from multicellular.simulation import MulticellularSimulation
        
        # Create a simulation with a single initial organism
        initial_org = create_test_organism()
        simulation = MulticellularSimulation(initial_organisms=[initial_org])
        
        # Run the simulation for a few iterations and print results
        print("Running simulation for 50 iterations...")
        for i in range(50):
            simulation.update()
            
            if i % 10 == 0:
                # Print summary statistics
                org_count = len(simulation.organisms)
                total_cells = sum(len(org.cells) for org in simulation.organisms)
                mean_neurons = np.mean([
                    len(org.nervous_system['cells']) if org.nervous_system else 0
                    for org in simulation.organisms
                ]) if simulation.organisms else 0
                
                print(f"Iteration {i}: {org_count} organisms, {total_cells} total cells, {mean_neurons:.2f} avg neurons")
                
                # Print details of the first organism
                if simulation.organisms:
                    org = simulation.organisms[0]
                    cell_types = {}
                    for cell in org.cells:
                        cell_type, _ = cell.get_cell_type()
                        cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
                    
                    print(f"  Organism 0: {len(org.cells)} cells, {org.energy:.1f} energy")
                    print(f"  Cell types: {', '.join(f'{t}: {c}' for t, c in cell_types.items())}")
                    print(f"  Has nervous system: {org.nervous_system is not None}")
                    if org.nervous_system:
                        print(f"  Neural cells: {len(org.nervous_system['cells'])}")
        
        print("Simulation complete.")
        
    else:
        # Default to cellular simulation
        print("Initializing cellular simulation...")
        simulation = AdvancedCellularSimulation()
        for _ in range(5):  # Add some initial cells
            pos = np.random.rand(2) * 100
            simulation.cells.append(AdvancedCell(pos, energy=20, dna_complexity=10))
        
        # Run the simulation for a few iterations and print results
        print("Running simulation for 50 iterations...")
        for i in range(50):
            simulation.update()
            
            if i % 10 == 0:
                # Print summary statistics
                cell_count = len(simulation.cells)
                mean_energy = np.mean([cell.energy for cell in simulation.cells]) if simulation.cells else 0
                mean_dna = np.mean([cell.dna_complexity for cell in simulation.cells]) if simulation.cells else 0
                
                print(f"Iteration {i}: {cell_count} cells, {mean_energy:.2f} mean energy, {mean_dna:.2f} mean DNA complexity")
        
        print("Simulation complete.")


from utils.helpers import create_test_organism


def main():
    """Main entry point for the simulation."""
    parser = argparse.ArgumentParser(description="Evolution Simulation")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["abiogenesis", "cell", "multicellular", "intelligence", "all", "visualize"],
                        help="Which simulation phase to run (default: all)")
    parser.add_argument("--vis-type", type=str, choices=["cell", "multicell"],
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
    if args.phase in ["abiogenesis", "all"]:
        chemicals = simulate_abiogenesis(max_iter=args.iterations)
        if not args.non_interactive:
            try:
                input("Press Enter to proceed to Single-Cell Life simulation...")
            except EOFError:
                print("Running in non-interactive mode, continuing automatically...")
    else:
        chemicals = []  # Placeholder
    
    # Phase 2: Single-Cell Life
    if args.phase in ["cell", "all"]:
        cells = simulate_single_cell_life(max_iterations=args.iterations)
        
        # Instead of visualization, print summary of cells
        if cells:
            print("\nCellular simulation summary:")
            cell_count = len(cells)
            cell_types = {}
            for cell in cells:
                cell_type, _ = cell.get_cell_type()
                cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
            
            mean_energy = np.mean([cell.energy for cell in cells])
            mean_dna = np.mean([cell.dna_complexity for cell in cells])
            
            print(f"Total cells: {cell_count}")
            print(f"Mean energy: {mean_energy:.2f}")
            print(f"Mean DNA complexity: {mean_dna:.2f}")
            print(f"Cell types: {', '.join(f'{t}: {c}' for t, c in cell_types.items() if c > 0)}")
        
        if not args.non_interactive:
            try:
                input("Press Enter to proceed to Multicellularity simulation...")
            except EOFError:
                print("Running in non-interactive mode, continuing automatically...")
    else:
        cells = []  # Placeholder
    
    # Phase 3: Multicellularity
    if args.phase in ["multicellular", "all"]:
        organisms = simulate_multicellularity(cells, max_iterations=args.iterations)
        
        # Instead of visualization, print summary of organisms
        if organisms:
            print("\nMulticellular simulation summary:")
            org_count = len(organisms)
            total_cells = sum(len(org.cells) for org in organisms)
            mean_size = total_cells / org_count if org_count > 0 else 0
            mean_complexity = np.mean([org.complexity for org in organisms]) if organisms else 0
            neural_orgs = sum(1 for org in organisms if org.nervous_system is not None)
            
            print(f"Total organisms: {org_count}")
            print(f"Total cells: {total_cells}")
            print(f"Mean organism size: {mean_size:.2f} cells")
            print(f"Mean complexity: {mean_complexity:.2f}")
            print(f"Organisms with neural systems: {neural_orgs}")
            
            # Print details of a few organisms
            for i, org in enumerate(organisms[:3]):  # Show details for up to 3 organisms
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
    else:
        organisms = []  # Placeholder
    
    # Phase 4: Intelligence
    if args.phase in ["intelligence", "all"]:
        intelligent_organisms = simulate_intelligence(organisms, max_iterations=args.iterations)
    
    print("Advanced Evolutionary Simulation Complete.")


if __name__ == "__main__":
    main()