"""
Advanced neural network implementation for the intelligence simulation.
This module provides a more sophisticated neural network model that can
evolve and learn from experience.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class EvolvableNeuralNetwork:
    """
    A neural network that can evolve over time through mutations and
    can learn through a simple form of Hebbian learning.
    """
    def __init__(self, input_size=config.NN_INPUT_SIZE,
                 hidden1_size=config.NN_HIDDEN1_SIZE,
                 hidden2_size=config.NN_HIDDEN2_SIZE,
                 output_size=config.NN_OUTPUT_SIZE):
        """
        Initialize a neural network with configurable architecture.

        Args:
            input_size: Number of input neurons
            hidden1_size: Number of neurons in first hidden layer
            hidden2_size: Number of neurons in second hidden layer
            output_size: Number of output neurons
        """
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size

        # Initialize weights with small random values
        self.weights_input_hidden1 = np.random.randn(input_size, hidden1_size) * 0.1
        self.weights_hidden1_hidden2 = np.random.randn(hidden1_size, hidden2_size) * 0.1
        self.weights_hidden2_output = np.random.randn(hidden2_size, output_size) * 0.1

        # Bias terms
        self.bias_hidden1 = np.zeros(hidden1_size)
        self.bias_hidden2 = np.zeros(hidden2_size)
        self.bias_output = np.zeros(output_size)

        # Activation values for each layer
        self.input_activation = np.zeros(input_size)
        self.hidden1_activation = np.zeros(hidden1_size)
        self.hidden2_activation = np.zeros(hidden2_size)
        self.output_activation = np.zeros(output_size)

        # Memory of previous activations for Hebbian learning
        self.prev_hidden1_activation = np.zeros(hidden1_size)
        self.prev_hidden2_activation = np.zeros(hidden2_size)

        # Learning rate for Hebbian learning
        self.learning_rate = config.NN_LEARNING_RATE

        # Neuron properties for more complex behavior
        self.neuron_adaptation = np.zeros(hidden1_size + hidden2_size + output_size)
        self.neuron_fatigue = np.zeros(hidden1_size + hidden2_size + output_size)

        # Experience memory (simple form of memory)
        self.experience_memory = []
        self.memory_capacity = config.INTELLIGENCE_MEMORY_CAPACITY

        # Neuromodulators that affect learning and behavior
        self.neuromodulators = {
            'reward': 0.0,  # Positive experiences
            'stress': 0.0,  # Negative experiences
            'curiosity': 0.5,  # Exploration drive
            'focus': 0.5  # Exploitation vs exploration balance
        }

    def forward(self, inputs):
        """
        Forward pass through the neural network.

        Args:
            inputs: Input values for the network

        Returns:
            Output activations
        """
        # Ensure inputs are the right size
        if len(inputs) != self.input_size:
            # Pad or truncate inputs to match expected size
            if len(inputs) < self.input_size:
                inputs = np.pad(inputs, (0, self.input_size - len(inputs)))
            else:
                inputs = inputs[:self.input_size]

        # Store input activation
        self.input_activation = np.array(inputs)

        # First hidden layer
        self.hidden1_activation = np.dot(self.input_activation, self.weights_input_hidden1) + self.bias_hidden1
        # Apply activation function (tanh) and neuron adaptation/fatigue
        self.hidden1_activation = np.tanh(self.hidden1_activation)
        self._apply_neuron_dynamics(self.hidden1_activation, 0, self.hidden1_size)

        # Second hidden layer
        self.hidden2_activation = np.dot(self.hidden1_activation, self.weights_hidden1_hidden2) + self.bias_hidden2
        # Apply activation function (tanh) and neuron adaptation/fatigue
        self.hidden2_activation = np.tanh(self.hidden2_activation)
        self._apply_neuron_dynamics(self.hidden2_activation, self.hidden1_size, self.hidden1_size + self.hidden2_size)

        # Output layer
        self.output_activation = np.dot(self.hidden2_activation, self.weights_hidden2_output) + self.bias_output
        # Apply activation function (sigmoid for outputs between 0-1)
        self.output_activation = 1.0 / (1.0 + np.exp(-self.output_activation))
        self._apply_neuron_dynamics(self.output_activation, self.hidden1_size + self.hidden2_size,
                                   self.hidden1_size + self.hidden2_size + self.output_size)

        return self.output_activation

    def _apply_neuron_dynamics(self, activations, start_idx, end_idx):
        """Apply neuron adaptation and fatigue to activations."""
        # Update neuron adaptation (neurons adapt to continuous stimulation)
        self.neuron_adaptation[start_idx:end_idx] += 0.01 * (activations - self.neuron_adaptation[start_idx:end_idx])

        # Update neuron fatigue (neurons get tired with continuous activation)
        self.neuron_fatigue[start_idx:end_idx] += 0.005 * activations
        self.neuron_fatigue[start_idx:end_idx] *= 0.95  # Fatigue recovery

        # Apply adaptation and fatigue to activations
        adaptation_factor = 1.0 - 0.2 * self.neuron_adaptation[start_idx:end_idx]
        fatigue_factor = 1.0 - 0.3 * self.neuron_fatigue[start_idx:end_idx]

        # Modify activations in-place
        activations *= adaptation_factor * fatigue_factor

    def hebbian_learning(self, reward=0.0):
        """
        Apply Hebbian learning: neurons that fire together, wire together.
        Modulated by reward signal.
        """
        # Skip if this is the first activation (no previous values)
        if np.all(self.prev_hidden1_activation == 0):
            self.prev_hidden1_activation = self.hidden1_activation.copy()
            self.prev_hidden2_activation = self.hidden2_activation.copy()
            return

        # Calculate effective learning rate based on neuromodulators
        effective_lr = self.learning_rate * (1.0 + 2.0 * self.neuromodulators['focus'])

        # Adjust learning based on reward signal
        if reward > 0:
            effective_lr *= (1.0 + reward)
            self.neuromodulators['reward'] = min(1.0, self.neuromodulators['reward'] + 0.1 * reward)
        elif reward < 0:
            self.neuromodulators['stress'] = min(1.0, self.neuromodulators['stress'] - 0.1 * reward)

        # Hebbian update for input to hidden1 weights
        for i in range(self.input_size):
            for j in range(self.hidden1_size):
                # Correlation between input and hidden1 activation
                correlation = self.input_activation[i] * self.hidden1_activation[j]
                # Update weight based on correlation
                self.weights_input_hidden1[i, j] += effective_lr * correlation

        # Hebbian update for hidden1 to hidden2 weights
        for i in range(self.hidden1_size):
            for j in range(self.hidden2_size):
                # Correlation between current and previous activations
                correlation = self.hidden1_activation[i] * self.hidden2_activation[j]
                prev_correlation = self.prev_hidden1_activation[i] * self.prev_hidden2_activation[j]
                # Update weight based on correlation difference
                self.weights_hidden1_hidden2[i, j] += effective_lr * (correlation - prev_correlation)

        # Hebbian update for hidden2 to output weights
        for i in range(self.hidden2_size):
            for j in range(self.output_size):
                # Correlation between hidden2 and output activation
                correlation = self.hidden2_activation[i] * self.output_activation[j]
                # Update weight based on correlation and reward
                self.weights_hidden2_output[i, j] += effective_lr * correlation * (1.0 + reward)

        # Store current activations for next update
        self.prev_hidden1_activation = self.hidden1_activation.copy()
        self.prev_hidden2_activation = self.hidden2_activation.copy()

        # Apply weight normalization to prevent runaway weights
        self._normalize_weights()

    def _normalize_weights(self):
        """Normalize weights to prevent them from growing too large."""
        # L2 normalization for each neuron's incoming weights

        # Normalize input to hidden1 weights
        for j in range(self.hidden1_size):
            weights = self.weights_input_hidden1[:, j]
            norm = np.sqrt(np.sum(weights**2))
            if norm > 1.0:
                self.weights_input_hidden1[:, j] = weights / norm

        # Normalize hidden1 to hidden2 weights
        for j in range(self.hidden2_size):
            weights = self.weights_hidden1_hidden2[:, j]
            norm = np.sqrt(np.sum(weights**2))
            if norm > 1.0:
                self.weights_hidden1_hidden2[:, j] = weights / norm

        # Normalize hidden2 to output weights
        for j in range(self.output_size):
            weights = self.weights_hidden2_output[:, j]
            norm = np.sqrt(np.sum(weights**2))
            if norm > 1.0:
                self.weights_hidden2_output[:, j] = weights / norm

    def mutate(self, mutation_rate=config.NN_MUTATION_RATE):
        """
        Randomly mutate the neural network weights and biases.

        Args:
            mutation_rate: Probability of each weight/bias being mutated
        """
        # Mutate weights
        self.weights_input_hidden1 = self._mutate_array(self.weights_input_hidden1, mutation_rate)
        self.weights_hidden1_hidden2 = self._mutate_array(self.weights_hidden1_hidden2, mutation_rate)
        self.weights_hidden2_output = self._mutate_array(self.weights_hidden2_output, mutation_rate)

        # Mutate biases
        self.bias_hidden1 = self._mutate_array(self.bias_hidden1, mutation_rate)
        self.bias_hidden2 = self._mutate_array(self.bias_hidden2, mutation_rate)
        self.bias_output = self._mutate_array(self.bias_output, mutation_rate)

        # Occasionally mutate architecture (add or remove neurons)
        if np.random.random() < mutation_rate * 0.1:
            self._mutate_architecture()

    def _mutate_array(self, array, mutation_rate):
        """Apply random mutations to an array of weights or biases."""
        # Create a mask of values to mutate
        mutation_mask = np.random.random(array.shape) < mutation_rate

        # Generate random mutations
        mutations = np.random.randn(*array.shape) * 0.2

        # Apply mutations where mask is True
        array = array.copy()  # Create a copy to avoid modifying the original
        array[mutation_mask] += mutations[mutation_mask]

        return array

    def _mutate_architecture(self):
        """Occasionally mutate the network architecture."""
        # This is a simplified version - in a real implementation,
        # this would be more complex to handle adding/removing neurons

        # For now, just adjust the learning rate and neuromodulators
        self.learning_rate *= np.random.uniform(0.8, 1.2)
        self.learning_rate = max(0.001, min(0.1, self.learning_rate))

        for key in self.neuromodulators:
            self.neuromodulators[key] *= np.random.uniform(0.9, 1.1)
            self.neuromodulators[key] = max(0.0, min(1.0, self.neuromodulators[key]))

    def store_experience(self, inputs, outputs, reward):
        """Store an experience in memory for later learning."""
        # Add experience to memory
        experience = {
            'inputs': inputs.copy(),
            'outputs': outputs.copy(),
            'reward': reward
        }

        self.experience_memory.append(experience)

        # Limit memory size
        if len(self.experience_memory) > self.memory_capacity:
            self.experience_memory.pop(0)

    def replay_experiences(self):
        """Replay stored experiences to reinforce learning."""
        if not self.experience_memory:
            return

        # Choose experiences to replay, biased toward high reward
        rewards = np.array([exp['reward'] for exp in self.experience_memory])
        # Convert rewards to probabilities (add small constant to avoid division by zero)
        probs = np.abs(rewards) / (np.sum(np.abs(rewards)) + 1e-10)

        # Select experiences to replay
        num_replays = min(3, len(self.experience_memory))
        indices = np.random.choice(len(self.experience_memory), size=num_replays, p=probs, replace=False)

        # Replay selected experiences
        for idx in indices:
            exp = self.experience_memory[idx]
            # Forward pass with stored inputs
            self.forward(exp['inputs'])
            # Apply Hebbian learning with stored reward
            self.hebbian_learning(exp['reward'])

    def get_curiosity_signal(self, inputs):
        """
        Calculate a curiosity signal based on prediction error.
        High when inputs are novel, low when inputs are familiar.
        """
        # Simple implementation: check if similar inputs exist in memory
        curiosity = 1.0  # Start with high curiosity

        for exp in self.experience_memory:
            # Calculate similarity with stored experiences
            similarity = np.mean(np.abs(inputs - exp['inputs']))
            # Reduce curiosity for similar inputs
            if similarity < 0.3:
                curiosity *= 0.5

        # Update curiosity neuromodulator
        self.neuromodulators['curiosity'] = 0.9 * self.neuromodulators['curiosity'] + 0.1 * curiosity

        return curiosity

    def get_network_state(self):
        """Get the current state of the neural network for visualization."""
        return {
            'input_activation': self.input_activation,
            'hidden1_activation': self.hidden1_activation,
            'hidden2_activation': self.hidden2_activation,
            'output_activation': self.output_activation,
            'neuromodulators': self.neuromodulators
        }

    def clone(self):
        """Create a copy of this neural network with the same weights."""
        clone = EvolvableNeuralNetwork(
            input_size=self.input_size,
            hidden1_size=self.hidden1_size,
            hidden2_size=self.hidden2_size,
            output_size=self.output_size
        )

        # Copy weights and biases
        clone.weights_input_hidden1 = self.weights_input_hidden1.copy()
        clone.weights_hidden1_hidden2 = self.weights_hidden1_hidden2.copy()
        clone.weights_hidden2_output = self.weights_hidden2_output.copy()
        clone.bias_hidden1 = self.bias_hidden1.copy()
        clone.bias_hidden2 = self.bias_hidden2.copy()
        clone.bias_output = self.bias_output.copy()

        # Copy learning parameters
        clone.learning_rate = self.learning_rate
        clone.neuromodulators = self.neuromodulators.copy()

        return clone


class IntelligentOrganism:
    """
    An organism with an evolvable neural network that can learn from experience.
    This extends the multicellular organism with more advanced cognitive abilities.
    """
    def __init__(self, base_organism=None):
        """
        Initialize an intelligent organism, optionally based on an existing multicellular organism.

        Args:
            base_organism: Optional multicellular organism to use as a base
        """
        self.base_organism = base_organism

        # Create neural network
        self.neural_network = EvolvableNeuralNetwork()

        # Organism state
        self.age = 0
        self.energy = 100 if base_organism is None else base_organism.energy
        self.position = np.zeros(2) if base_organism is None else base_organism.position
        self.fitness = 0

        # Memory of recent states and actions
        self.state_history = []
        self.action_history = []
        self.reward_history = []

        # Behavioral traits
        self.traits = {
            'aggression': np.random.random(),
            'sociability': np.random.random(),
            'exploration': np.random.random(),
            'metabolism': np.random.random()
        }

        # Sensory inputs
        self.sensors = {
            'vision': np.zeros(3),  # RGB-like vision in 3 directions
            'chemical': np.zeros(2),  # Chemical sensing (nutrients, toxins)
            'touch': 0.0,  # Contact sensing
            'internal': np.zeros(3)  # Internal state (energy, age, etc.)
        }

    def sense_environment(self, environment, other_organisms=None):
        """
        Sense the environment and other organisms.

        Args:
            environment: The environment object
            other_organisms: List of other organisms in the environment
        """
        # Get position
        x, y = int(self.position[0]) % environment.grid_size, int(self.position[1]) % environment.grid_size

        # Sense nutrients (chemical sensing)
        nutrient_field = environment.get_nutrient_field()
        self.sensors['chemical'][0] = nutrient_field[x, y]

        # Sense signals (chemical sensing)
        signal_fields = environment.get_signal_fields()
        if isinstance(signal_fields, np.ndarray) and signal_fields.size > 0:
            self.sensors['chemical'][1] = signal_fields[0][x, y]

        # Simple vision in three directions (forward, left, right)
        directions = [
            [1, 0],  # Forward
            [0.7, 0.7],  # Forward-right
            [0.7, -0.7]  # Forward-left
        ]

        for i, direction in enumerate(directions):
            # Normalize direction
            direction = np.array(direction) / np.linalg.norm(direction)

            # Look in this direction
            vision_x = int((x + direction[0] * 10)) % environment.grid_size
            vision_y = int((y + direction[1] * 10)) % environment.grid_size

            # What do we see? (simplified to nutrient level for now)
            self.sensors['vision'][i] = nutrient_field[vision_x, vision_y]

        # Touch sensing - detect nearby organisms
        self.sensors['touch'] = 0.0
        if other_organisms:
            for org in other_organisms:
                if org is not self:
                    dist = np.linalg.norm(self.position - org.position)
                    if dist < 15:  # Touch detection radius
                        self.sensors['touch'] = 1.0 - dist / 15

        # Internal state sensing
        self.sensors['internal'][0] = self.energy / 100  # Normalized energy
        self.sensors['internal'][1] = min(1.0, self.age / 100)  # Normalized age
        self.sensors['internal'][2] = self.fitness / 100  # Normalized fitness

    def think(self):
        """
        Process sensory information through the neural network.

        Returns:
            Action outputs from the neural network
        """
        # Prepare inputs for neural network
        inputs = np.concatenate([
            self.sensors['vision'],
            self.sensors['chemical'],
            [self.sensors['touch']],
            self.sensors['internal']
        ])

        # Calculate curiosity signal
        curiosity = self.neural_network.get_curiosity_signal(inputs)

        # Add curiosity to inputs to encourage exploration of novel environments
        if self.traits['exploration'] > 0.5:
            # Only high-exploration organisms use curiosity
            curiosity_input = curiosity * config.INTELLIGENCE_CURIOSITY_FACTOR
            # Add curiosity as an input to the neural network
            if len(inputs) > 0:
                inputs[-1] = 0.7 * inputs[-1] + 0.3 * curiosity_input

        # Process through neural network
        outputs = self.neural_network.forward(inputs)

        # Store current state and action
        self.state_history.append(inputs.copy())
        self.action_history.append(outputs.copy())

        # Limit history length
        if len(self.state_history) > 20:
            self.state_history.pop(0)
            self.action_history.pop(0)
            if self.reward_history:
                self.reward_history.pop(0)

        return outputs

    def act(self, outputs, environment, other_organisms=None):
        """
        Take actions based on neural network outputs.

        Args:
            outputs: Output values from neural network
            environment: The environment object
            other_organisms: List of other organisms in the environment

        Returns:
            Reward signal based on action outcomes
        """
        # Interpret outputs as:
        # [0]: Movement direction (angle)
        # [1]: Movement speed
        # [2]: Interaction type (feed, communicate, etc.)

        # Calculate movement
        angle = outputs[0] * 2 * np.pi  # Convert to radians
        speed = outputs[1] * 2.0  # Scale speed

        # Calculate movement vector
        move_x = speed * np.cos(angle)
        move_y = speed * np.sin(angle)

        # Apply movement
        self.position[0] = (self.position[0] + move_x) % environment.grid_size
        self.position[1] = (self.position[1] + move_y) % environment.grid_size

        # Energy cost of movement
        movement_cost = speed * config.INTELLIGENCE_ENERGY_COST
        self.energy -= movement_cost

        # Interaction with environment
        interaction_type = outputs[2]
        reward = 0.0

        # Feeding behavior
        if interaction_type < 0.33:
            # Try to consume nutrients
            x, y = int(self.position[0]) % environment.grid_size, int(self.position[1]) % environment.grid_size
            nutrient_level = environment.get_nutrient_field()[x, y]

            if nutrient_level > 0.1:
                # Consume nutrients
                consumed = min(nutrient_level, 0.3)
                environment.consume_at(x, y, amount=consumed)

                # Energy gain - further increased for survival and reproduction
                energy_gain = consumed * 30
                self.energy += energy_gain

                # Positive reward for successful feeding
                reward += 0.5 * consumed

        # Social/communication behavior
        elif interaction_type < 0.66:
            # Try to communicate with nearby organisms
            if other_organisms:
                for org in other_organisms:
                    if org is not self:
                        dist = np.linalg.norm(self.position - org.position)
                        if dist < 20:  # Communication range
                            # Simple communication: share some energy if both are social
                            if self.traits['sociability'] > 0.5 and org.traits['sociability'] > 0.5:
                                # Both organisms gain a small benefit
                                energy_share = min(5, self.energy * 0.05)
                                self.energy -= energy_share
                                org.energy += energy_share * 1.2  # Net positive interaction

                                # Positive reward for successful social interaction
                                reward += 0.3

        # Exploration behavior
        else:
            # Reward for exploring new areas
            # Check if current position is far from recent positions
            is_new_area = True
            for past_state in self.state_history[:-1]:  # Exclude current state
                past_pos = past_state[-3:-1]  # Extract position from state
                if np.linalg.norm(self.position - past_pos) < 10:
                    is_new_area = False
                    break

            if is_new_area:
                # Reward exploration
                reward += 0.2 * self.traits['exploration']

        # Penalize low energy
        if self.energy < 20:
            reward -= 0.3

        # Reward energy efficiency
        energy_efficiency = 1.0 - movement_cost / max(1.0, self.energy)
        reward += 0.1 * energy_efficiency

        # Store reward
        self.reward_history.append(reward)

        # Update fitness based on survival and energy
        self.fitness = 0.9 * self.fitness + 0.1 * (self.energy / 100)

        # Learn from this experience
        self.neural_network.hebbian_learning(reward)

        # Store experience in memory
        if len(self.state_history) > 0 and len(self.action_history) > 0:
            last_state = self.state_history[-1]
            last_action = self.action_history[-1]
            self.neural_network.store_experience(last_state, last_action, reward)

        # Occasionally replay experiences
        if np.random.random() < 0.1:
            self.neural_network.replay_experiences()

        return reward

    def update(self, environment, other_organisms=None):
        """
        Update the organism for one timestep.

        Args:
            environment: The environment object
            other_organisms: List of other organisms in the environment

        Returns:
            Reward from this update step
        """
        # Increase age
        self.age += 1

        # Basic metabolism cost - reduced for longer survival
        self.energy -= config.INTELLIGENCE_ENERGY_COST * (0.3 + 0.3 * self.traits['metabolism'])

        # Sense environment
        self.sense_environment(environment, other_organisms)

        # Think (process through neural network)
        outputs = self.think()

        # Act based on outputs
        reward = self.act(outputs, environment, other_organisms)

        # Die if out of energy
        if self.energy <= 0:
            return -1.0  # Death is a strong negative reward

        return reward

    def reproduce(self, mutation_rate=config.NN_MUTATION_RATE):
        """
        Attempt to reproduce, creating a new organism.

        Args:
            mutation_rate: Rate of mutation for the offspring

        Returns:
            New organism if reproduction successful, None otherwise
        """
        # Check if enough energy to reproduce - with random chance for lower threshold
        reproduction_threshold = config.INTELLIGENCE_ENERGY_THRESHOLD
        if np.random.random() < 0.1:  # 10% chance of reproducing at lower energy
            reproduction_threshold *= 0.7

        # Check energy and age requirements
        if self.energy < reproduction_threshold or self.age < 15:  # Reduced age requirement
            return None

        # Create offspring
        offspring = IntelligentOrganism()

        # Inherit neural network with mutations
        offspring.neural_network = self.neural_network.clone()
        offspring.neural_network.mutate(mutation_rate)

        # Inherit traits with mutations
        for trait in self.traits:
            offspring.traits[trait] = max(0, min(1,
                self.traits[trait] + np.random.normal(0, 0.1)))

        # Transfer energy
        offspring_energy = self.energy * 0.3
        self.energy -= offspring_energy
        offspring.energy = offspring_energy

        # Position offspring nearby
        angle = np.random.uniform(0, 2*np.pi)
        distance = np.random.uniform(5, 10)
        offspring.position = np.array([
            (self.position[0] + distance * np.cos(angle)),
            (self.position[1] + distance * np.sin(angle))
        ])

        return offspring

    def get_state(self):
        """Get the current state of the organism for visualization."""
        return {
            'position': self.position,
            'energy': self.energy,
            'age': self.age,
            'fitness': self.fitness,
            'traits': self.traits,
            'neural_state': self.neural_network.get_network_state()
        }
