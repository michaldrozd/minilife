"""
Configuration parameters for the evolution simulation.
"""

# General simulation parameters
GRID_SIZE = 100
RANDOM_SEED = None  # Set to an integer for reproducible results

# Abiogenesis parameters
ABIOGENESIS_MAX_ITER = 10000
CHEMICAL_COMPLEXITY_THRESHOLD = 10
CHEMICAL_MUTATION_RATE = 0.001
CHEMICAL_MUTATION_STEP = 1
INITIAL_CHEMICALS = 20 # Added initial chemicals for abiogenesis sim

# Cellular parameters
INITIAL_CELLS = 10
CELL_INITIAL_ENERGY = 20
CELL_INITIAL_DNA_COMPLEXITY = 10
CELL_FRICTION = 0.9
CELL_METABOLIC_COST = 1.0
CELL_REPLICATION_THRESHOLD = 30
CELL_REPLICATION_AGE = 5
CELL_MAX_AGE = 100

# Nutrient and diffusion parameters
NUTRIENT_INITIAL_VALUE = 5.0
NUTRIENT_REGENERATION_RATE = 0.2
NUTRIENT_CONSUMPTION_MAX = 0.8

# Multicellular parameters
MULTICELL_MOVEMENT_SCALE = 0.05
MULTICELL_ENERGY_THRESHOLD = 150
MULTICELL_REPLICATION_AGE = 10
MULTICELL_MAX_AGE = 200

# Intelligence parameters
NN_INPUT_SIZE = 9  # Increased input size for more sensory data
NN_HIDDEN1_SIZE = 16  # Larger hidden layer for more complex processing
NN_HIDDEN2_SIZE = 12  # Larger second hidden layer
NN_OUTPUT_SIZE = 4  # Added output for more complex behaviors
NN_MUTATION_RATE = 0.08  # Increased mutation rate for faster evolution
NN_LEARNING_RATE = 0.01  # Learning rate for Hebbian learning
INTELLIGENCE_ENERGY_THRESHOLD = 80  # Significantly reduced threshold for reproduction
INTELLIGENCE_ENERGY_COST = 0.5  # Reduced energy cost for survival
INTELLIGENCE_MAX_FRAMES = 500  # Increased simulation length
INTELLIGENCE_MEMORY_CAPACITY = 15  # Memory capacity for experiences
INTELLIGENCE_CURIOSITY_FACTOR = 0.3  # Factor for curiosity-driven exploration

# Environment parameters
ENVIRONMENT_PATTERN_TYPE = 'gradient'  # Options: 'ecosystem', 'gradient', 'patches', 'turing'
ENVIRONMENT_VARIABILITY = 0.01  # Random environmental fluctuations
ENVIRONMENT_SEASONAL_STRENGTH = 0.2  # Strength of seasonal effects
