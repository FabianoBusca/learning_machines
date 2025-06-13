import json
import os
import random
from collections import deque
from datetime import time, datetime

import numpy as np
import torch
import torch.nn as nn

from robobo_interface import SimulationRobobo, HardwareRobobo, IRobobo
from learning_machines.helpers.sensor_handler import SensorInputHandler
from learning_machines.exploration_learning import ExplorationLearner


class SensorInputHandler:
    """Processes sensor data from the robot."""

    def __init__(self, robot: IRobobo):
        self.robot = robot
        self.sensor_history = deque(maxlen=10)  # Store recent sensor readings

    def get_current_state(self):
        """Get current sensor state as normalized values."""
        try:
            ir_readings = self.robot.read_irs()
            if ir_readings is None or len(ir_readings) < 5:
                print("Warning: Invalid IR sensor data")
                return None

            # Normalize sensor values (0-1 range)
            normalized = [min(1.0, max(0.0, val / 1000.0)) for val in ir_readings]
            self.sensor_history.append(normalized)

            # Return the current state representation
            return normalized

        except Exception as e:
            print(f"Error reading sensors: {e}")
            return None

    def detect_collision(self):
        """
        Detect if the robot is about to collide with an obstacle.
        Returns True if any front sensors report high values (close to obstacle).
        """
        state = self.get_current_state()
        if state is None:
            return False

        # Check front sensors (indices around the front)
        # Focus on center sensors which are most important for collision detection
        front_center_indices = [3, 4, 5]  # Front-facing sensors
        for idx in front_center_indices:
            if idx < len(state) and state[idx] > 0.25:
                return True

        return False

    def is_path_clear(self):
        """
        Check if there's a clear path forward with no walls.
        Returns True if the front sensors indicate an open path.
        """
        state = self.get_current_state()
        if state is None or len(state) < 5:
            return False

        # Check front-facing sensors
        # Values close to 0 mean no obstacles detected
        front_indices = [3, 4, 5]  # Adjust these indices based on your sensor configuration
        front_readings = [state[i] for i in front_indices if i < len(state)]

        # Consider path clear if all front sensors show low values
        return all(val < 0.15 for val in front_readings)

ACTIONS = [
    [100, 100],  # Forward
    [80, 80],  # Slow forward
    [-50, 100],  # Turn left
    [-70, 100],  # Sharp left
    [100, -50],  # Turn right
    [100, -70],  # Sharp right
    [-100, -100],  # Backward
    [0, 0]  # Stop
]

# Movement duration in milliseconds
MOVEMENT_DURATION = 500


class QNetwork(nn.Module):
    """Simple Q-network for reinforcement learning."""

    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ExplorationLearner:
    """Reinforcement learning agent that learns through exploration and trial-and-error."""

    def __init__(self, robot, sensor_handler, max_memory=10000, lr=0.001):
        self.robot = robot
        self.sensor_handler = sensor_handler
        self.state_size = 8  # Number of IR sensors
        self.action_size = len(ACTIONS)
        self.memory = []
        self.max_memory = max_memory

        # Learning parameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate (start with 100% exploration)
        self.epsilon_min = 0.05  # Minimum exploration rate
        self.epsilon_decay = 0.99  # Slower epsilon decay to maintain some exploration
        self.learning_rate = lr

        # Set up the neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = QNetwork(self.state_size, self.action_size).to(self.device)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Tracking metrics
        self.episode_rewards = []
        self.collisions = []
        self.steps_per_episode = []

        # Keep track of consecutive non-forward actions for basic reward shaping
        self.consecutive_non_forward = 0

        # Probability of forcing forward movement when path is clear - reduced to prioritize safety
        self.forward_bias = 0.5  # Reduced from 0.85

        # Obstacle avoidance parameters
        self.obstacle_threshold = 0.3  # Threshold for detecting obstacles
        self.close_obstacle_threshold = 0.8  # Threshold for detecting close obstacles
        self.critical_obstacle_threshold = 0.7  # Threshold for detecting very close obstacles
        self.sensor_weights = [0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 0.8, 0.6]  # Weighting for different sensors

        # Experience replay prioritization
        self.priority_alpha = 0.6  # Prioritization exponent
        self.priority_beta = 0.4  # Importance sampling exponent
        self.priority_epsilon = 0.01  # Small constant to prevent zero priority
        self.episode_experiences = []  # To store experiences from the current episode
        self.collision_states = []  # Store states that led to collisions for learning

        # Episodic learning transfer
        self.safe_paths = []  # Store successful navigation paths
        self.hazard_areas = []  # Store areas with collision risks
        self.current_path = []  # Track the current path
        self.first_episode = True  # Flag to indicate if this is the first episode
        self.best_score = float("-inf")  # or +inf if "lower is better"

    def update_target_model(self):
        """Update the target model with the weights from the current model."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done, is_collision=False):
        """Store experience in memory with prioritization."""
        # Calculate priority based on reward magnitude
        priority = abs(reward) + self.priority_epsilon

        # Increase priority for collision experiences to learn from mistakes
        if is_collision:
            priority *= 2.0
            # Store collision states separately for focused learning
            self.collision_states.append((state, next_state))
            if len(self.collision_states) > 100:  # Keep only the most recent collision states
                self.collision_states.pop(0)

        # Store in episode-specific memory first
        self.episode_experiences.append((state, action, reward, next_state, done, priority))

        # Store in main memory with priority
        if len(self.memory) > self.max_memory:
            # Remove lowest priority experience instead of oldest
            if len(self.memory) > 0:
                min_priority_idx = min(range(len(self.memory)), key=lambda i: self.memory[i][5])
                self.memory.pop(min_priority_idx)

        self.memory.append((state, action, reward, next_state, done, priority))

    def store_path_data(self, state, is_safe=True):
        """Store information about safe paths and hazard areas."""
        if state is None:
            return

        # Convert state to tuple for hashability
        state_key = tuple(round(x, 2) for x in state)  # Round to reduce sensitivity

        if is_safe:
            self.current_path.append(state_key)
            # Add to safe paths if not already present
            if state_key not in self.safe_paths:
                self.safe_paths.append(state_key)
        else:
            # Add to hazard areas if not already present
            if state_key not in self.hazard_areas:
                self.hazard_areas.append(state_key)

        # Limit the size of these collections
        if len(self.safe_paths) > 1000:
            self.safe_paths.pop(0)
        if len(self.hazard_areas) > 500:
            self.hazard_areas.pop(0)

    def is_path_clear(self, state):
        """
        Check if there's a clear path forward with no walls based on the current state.
        This is a convenience method that uses the sensor values directly 
        without calling the sensor handler again.
        """
        if state is None or len(state) < 8:
            return False

        # Check front-facing sensors (indices 3, 4, 5)
        # Values close to 0 mean no obstacles detected
        front_readings = [state[i] for i in [3, 4, 5]]
        return all(val < self.obstacle_threshold for val in front_readings)  # All sensors show clear path

    def get_obstacle_direction(self, state):
        """
        Determine which direction has obstacles and how close they are.
        Returns:
        - left_blocked: Boolean indicating if left path is blocked
        - right_blocked: Boolean indicating if right path is blocked
        - front_blocked: Boolean indicating if front path is blocked
        - closest_direction: String indicating the direction with the closest obstacle ('left', 'right', 'front', or 'none')
        """
        if state is None or len(state) < 8:
            return True, True, True, 'none'

        # Left sensors (0, 1, 2)
        left_readings = [state[i] * self.sensor_weights[i] for i in range(3)]
        left_blocked = any(val > self.obstacle_threshold for val in left_readings)
        left_max = max(left_readings) if left_readings else 0

        # Front sensors (3, 4)
        front_readings = [state[i] * self.sensor_weights[i] for i in [3, 4]]
        front_blocked = any(val > self.obstacle_threshold for val in front_readings)
        front_max = max(front_readings) if front_readings else 0

        # Right sensors (5, 6, 7)
        right_readings = [state[i] * self.sensor_weights[i] for i in range(5, 8)]
        right_blocked = any(val > self.obstacle_threshold for val in right_readings)
        right_max = max(right_readings) if right_readings else 0

        # Determine closest obstacle direction
        max_readings = [left_max, front_max, right_max]
        max_index = max_readings.index(max(max_readings))
        closest_direction = ['left', 'front', 'right'][max_index]

        return left_blocked, right_blocked, front_blocked, closest_direction

    def is_state_similar(self, state1, state2, threshold=0.15):
        """Check if two states are similar within a threshold."""
        if state1 is None or state2 is None:
            return False

        # Calculate Euclidean distance between states
        distance = sum((s1 - s2) ** 2 for s1, s2 in zip(state1, state2)) ** 0.5
        return distance < threshold

    def is_state_risky(self, state):
        """Check if current state is close to previously identified hazard areas."""
        if state is None or not self.hazard_areas:
            return False

        # Check if current state is similar to any hazard area
        state_tuple = tuple(round(x, 2) for x in state)

        # Direct match
        if state_tuple in self.hazard_areas:
            return True

        # Similarity check for approximate matches
        for hazard_state in self.hazard_areas:
            if self.is_state_similar(state, hazard_state):
                return True

        return False

    def act(self, state, explore=True):
        """
        Choose an action based on the current state.
        Uses sensor data to avoid obstacles and make intelligent navigation decisions.
        Prioritizes safety over forward movement.
        """
        if state is None:
            return random.randrange(self.action_size)

        # Get obstacle information
        left_blocked, right_blocked, front_blocked, closest_obstacle = self.get_obstacle_direction(state)

        # Check if current state is in a known hazard area
        state_is_risky = self.is_state_risky(state)

        # Emergency avoidance for very close obstacles or known hazard areas
        front_critical = any(state[i] > self.critical_obstacle_threshold for i in [3, 4])
        if front_critical or state_is_risky:
            # Immediate safety action
            if not left_blocked and not right_blocked:
                return random.choice([2, 4])  # Turn either direction
            elif not left_blocked:
                return 2  # Turn left
            elif not right_blocked:
                return 4  # Turn right
            else:
                return 6  # Backward action

        # Safety-first navigation
        # Only consider forward movement if path is clear and it's not risky
        if not front_blocked and self.is_path_clear(state) and not state_is_risky:
            # Even with clear path, only go forward some of the time (based on forward_bias)
            if random.random() < self.forward_bias:
                # Slow down if obstacles are somewhat near but not blocking
                if any(state[i] > self.obstacle_threshold / 2 for i in [2, 3, 4, 5]):
                    return 1  # Slow forward
                return 0  # Fast forward

        # Normal exploration/exploitation with intelligent obstacle avoidance
        if explore and np.random.rand() <= self.epsilon:
            # Intelligent random action based on obstacle detection
            if front_blocked:
                if not left_blocked and not right_blocked:
                    # Both sides clear, choose either direction
                    return random.choice([2, 3, 4, 5])  # No bias, just avoid front
                elif not left_blocked:
                    # Only left is clear
                    return random.choice([2, 3])
                elif not right_blocked:
                    # Only right is clear
                    return random.choice([4, 5])
                else:
                    # All directions blocked, back up
                    return 6
            elif left_blocked:
                # Left blocked but front clear
                return random.choice([0, 1, 4, 5])
            elif right_blocked:
                # Right blocked but front clear
                return random.choice([0, 1, 2, 3])
            else:
                # All clear, balance forward with turns for exploration
                return random.choice([0, 1, 2, 4])

        # Model-based action (exploitation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.model.eval()
            q_values = self.model(state_tensor)
            self.model.train()

        # Get the best predicted action
        best_action = torch.argmax(q_values).item()

        # Override if necessary for safety (even during exploitation)
        if best_action <= 1 and (front_critical or state_is_risky):  # If trying to go forward in risky situation
            if not left_blocked and not right_blocked:
                return random.choice([2, 4])  # Turn either direction
            elif not left_blocked:
                return 2  # Turn left
            elif not right_blocked:
                return 4  # Turn right
            else:
                return 6  # Back up

        return best_action

    def replay(self, batch_size=32):
        """Train the model using experience replay with prioritization."""
        if len(self.memory) < batch_size:
            return

        # Calculate total priority for sampling
        priorities = np.array([exp[5] for exp in self.memory])
        sampling_probs = priorities ** self.priority_alpha
        sampling_probs /= np.sum(sampling_probs)

        # Sample based on priorities
        indices = np.random.choice(len(self.memory), batch_size, p=sampling_probs, replace=False)
        minibatch = [self.memory[i] for i in indices]

        # Calculate importance sampling weights
        weights = (len(self.memory) * sampling_probs[indices]) ** (-self.priority_beta)
        weights /= np.max(weights)  # Normalize weights

        states = []
        targets = []

        for i, (state, action, reward, next_state, done, _) in enumerate(minibatch):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

            # Current Q-value
            target = self.model(state_tensor).detach().clone()

            if done:
                # If episode is done, just use the reward
                target[0][action] = reward
            else:
                # Q-learning formula: Q(s,a) = r + gamma * max(Q(s',a'))
                with torch.no_grad():
                    next_q = self.target_model(next_state_tensor)
                target[0][action] = reward + self.gamma * torch.max(next_q).item()

            # Apply importance sampling weight
            target[0][action] *= weights[i]

            states.append(state)
            targets.append(target.squeeze().cpu().numpy())

        # Convert to tensors and train
        states_tensor = torch.FloatTensor(states).to(self.device)
        targets_tensor = torch.FloatTensor(targets).to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(states_tensor)
        loss = self.criterion(outputs, targets_tensor)
        loss.backward()
        self.optimizer.step()

        # Decay epsilon more slowly to maintain some exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay_collision_experiences(self, batch_size=16):
        """Special replay focusing on collision experiences for better learning."""
        if not self.collision_states or len(self.memory) < batch_size:
            return

        # Focus learning on collision states and their neighbors
        states_to_learn = []

        # First, add some collision states
        collision_samples = min(batch_size // 2, len(self.collision_states))
        if collision_samples > 0:
            collision_indices = np.random.choice(len(self.collision_states), collision_samples, replace=False)
            for i in collision_indices:
                states_to_learn.append(self.collision_states[i])

        # Then, find experiences in memory that led to these states
        remaining_samples = batch_size - len(states_to_learn)
        if remaining_samples > 0 and len(self.memory) > 0:
            # Get random experiences from memory
            memory_indices = np.random.choice(len(self.memory), min(remaining_samples, len(self.memory)), replace=False)
            for i in memory_indices:
                state, action, reward, next_state, done, _ = self.memory[i]
                states_to_learn.append((state, next_state))

        # Now train on these selected states
        for pre_state, post_state in states_to_learn:
            if pre_state is None or post_state is None:
                continue

            state_tensor = torch.FloatTensor(pre_state).unsqueeze(0).to(self.device)

            # Get current Q-values
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(state_tensor)
            self.model.train()

            # Focus on punishing actions that led to collision
            target = q_values.clone()

            # Update all action values to discourage this state
            for a in range(self.action_size):
                # More negative values for forward actions, less for turning/backing
                if a <= 1:  # Forward actions
                    target[0][a] -= 2.0  # Strong discouragement
                else:
                    target[0][a] -= 0.5  # Mild discouragement

            # Train on this focused example
            self.optimizer.zero_grad()
            outputs = self.model(state_tensor)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()

    def transfer_learning(self):
        """
        Transfer learning from previous episodes to improve future episodes.
        This is called at the end of each episode to consolidate learning.
        """
        if self.first_episode:
            self.first_episode = False
            return

        # Clear the current path for the next episode
        self.current_path = []

        # Process episodic experiences to enhance learning
        if self.episode_experiences:
            # Add extra replay passes focusing on the current episode's experiences
            episode_batch_size = min(32, len(self.episode_experiences))
            for _ in range(3):  # Extra training passes on episode data
                if len(self.episode_experiences) >= episode_batch_size:
                    # Sample batch from episode experiences
                    batch = random.sample(self.episode_experiences, episode_batch_size)

                    states = []
                    targets = []

                    for state, action, reward, next_state, done, _ in batch:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

                        # Current Q-value
                        target = self.model(state_tensor).detach().clone()

                        if done:
                            target[0][action] = reward
                        else:
                            with torch.no_grad():
                                next_q = self.target_model(next_state_tensor)
                            target[0][action] = reward + self.gamma * torch.max(next_q).item()

                        states.append(state)
                        targets.append(target.squeeze().cpu().numpy())

                    if states:
                        # Convert to tensors and train
                        states_tensor = torch.FloatTensor(states).to(self.device)
                        targets_tensor = torch.FloatTensor(targets).to(self.device)

                        self.optimizer.zero_grad()
                        outputs = self.model(states_tensor)
                        loss = self.criterion(outputs, targets_tensor)
                        loss.backward()
                        self.optimizer.step()

            # Focus learning on collision experiences
            self.replay_collision_experiences()

            # Clear episode experiences after processing
            self.episode_experiences = []

    def train(self, episodes=100, max_steps=200, update_target_every=5, save_every=10):
        """Train the agent using reinforcement learning with improved inter-episode learning."""
        print(f"Starting training for {episodes} episodes...")
        print(f"Using device: {self.device}")

        for episode in range(episodes):
            # Reset environment
            if isinstance(self.robot, SimulationRobobo):
                if self.robot.is_running():
                    self.robot.stop_simulation()
                    time.sleep(1)
                self.robot.play_simulation()
                time.sleep(2)

            # Get initial state
            state = self.sensor_handler.get_current_state()
            if state is None:
                print("Could not get initial state. Skipping episode.")
                continue

            total_reward = 0
            episode_collisions = 0
            self.consecutive_non_forward = 0  # Reset counter at beginning of episode
            last_collision_step = -10  # Track when the last collision happened

            # Clear current path for new episode
            self.current_path = []

            # If not the first episode, reduce initial exploration to leverage previous learning
            initial_steps = min(30, max_steps // 4) if episode == 0 else min(15, max_steps // 8)

            # Initial exploration phase for the first episode, shorter for subsequent episodes
            print(f"Starting episode {episode + 1} with initial exploration ({initial_steps} steps)...")

            # Try to find a safe position to start exploration
            continuous_safe_count = 0
            target_safe_streak = 5  # We want the robot to move safely for this many steps

            for step in range(initial_steps):
                # Use improved obstacle detection
                left_blocked, right_blocked, front_blocked, closest_obstacle = self.get_obstacle_direction(state)

                # Check if the current state is risky based on previous episodes
                state_is_risky = self.is_state_risky(state)

                # For initial exploration, prioritize safety completely
                if front_blocked or state_is_risky:
                    continuous_safe_count = 0

                    # Intelligent turning based on obstacle detection
                    if front_blocked:
                        if left_blocked and not right_blocked:
                            action_idx = 4  # Turn right
                            print(f"Initial exploration step {step + 1}: Front and left blocked, turning right")
                        elif right_blocked and not left_blocked:
                            action_idx = 2  # Turn left
                            print(f"Initial exploration step {step + 1}: Front and right blocked, turning left")
                        elif left_blocked and right_blocked:
                            action_idx = 6  # Back up
                            print(f"Initial exploration step {step + 1}: All directions blocked, backing up")
                        else:
                            # Both sides clear, choose randomly
                            action_idx = random.choice([2, 4])
                            print(f"Initial exploration step {step + 1}: Front blocked, turning to avoid")
                    elif left_blocked:
                        action_idx = 4  # Turn right
                        print(f"Initial exploration step {step + 1}: Left blocked, turning right")
                    elif right_blocked:
                        action_idx = 2  # Turn left
                        print(f"Initial exploration step {step + 1}: Right blocked, turning left")
                    elif state_is_risky:
                        # Known risky area, make a turn
                        action_idx = random.choice([2, 4])
                        print(f"Initial exploration step {step + 1}: In risky area, making a turn")
                    else:
                        # Nothing immediately blocked but not fully clear - slight turn
                        action_idx = random.choice([2, 4])
                        print(f"Initial exploration step {step + 1}: Minor adjustment")
                else:
                    # Safe to move forward
                    action_idx = 1  # Slow forward - prioritize safety
                    continuous_safe_count += 1
                    print(f"Initial exploration step {step + 1}: Moving safely (streak: {continuous_safe_count})")

                    # If we've found a safe path to follow, we can end initial exploration early
                    if continuous_safe_count >= target_safe_streak:
                        print(f"Found a safe path after {step + 1} steps. Moving to learning phase.")
                        break

                # Execute the action
                action = ACTIONS[action_idx]
                self.robot.move_blocking(action[0], action[1], MOVEMENT_DURATION)
                time.sleep(0.2)

                # Get the next state
                next_state = self.sensor_handler.get_current_state()
                if next_state is None:
                    break

                # Store the current state as part of the path
                self.store_path_data(state, is_safe=True)

                # Check for collision
                collision = self.sensor_handler.detect_collision()
                if collision:
                    # Mark this state as a hazard
                    self.store_path_data(state, is_safe=False)

                    # Back up and try a different direction
                    episode_collisions += 1
                    continuous_safe_count = 0  # Reset safe streak
                    print(f"Collision detected during initial exploration. Backing up...")
                    self.robot.move_blocking(-100, -100, MOVEMENT_DURATION * 2)
                    time.sleep(0.3)

                    # After backing up, make an intelligent turn
                    left_blocked, right_blocked, _, _ = self.get_obstacle_direction(next_state)
                    if left_blocked and not right_blocked:
                        action_idx = 5  # Sharp right
                    elif right_blocked and not left_blocked:
                        action_idx = 3  # Sharp left
                    else:
                        action_idx = random.choice([3, 5])  # Random sharp turn

                    action = ACTIONS[action_idx]
                    self.robot.move_blocking(action[0], action[1], MOVEMENT_DURATION)
                    time.sleep(0.2)

                    # Exit the initial exploration phase early if we encounter too many collisions
                    if episode_collisions > 2:  # Reduced threshold to prioritize safety
                        print("Too many collisions in initial exploration. Moving to learning phase.")
                        break

                # Update the state
                state = next_state

            print(f"Beginning learning phase of episode {episode + 1}...")

            # Main training loop
            for step in range(max_steps - initial_steps):
                # Choose and take action using improved obstacle detection
                action_idx = self.act(state)
                action = ACTIONS[action_idx]

                self.robot.move_blocking(action[0], action[1], MOVEMENT_DURATION)
                time.sleep(0.2)  # Allow time for action to have effect

                # Observe new state
                next_state = self.sensor_handler.get_current_state()
                if next_state is None:
                    print("Could not get next state. Ending episode.")
                    break

                # Store the current state as part of the path if it's not a collision
                collision = self.sensor_handler.detect_collision()
                self.store_path_data(state, is_safe=not collision)

                # Handle collision - count it, but don't end the episode
                if collision:
                    episode_collisions += 1

                    # More sophisticated collision recovery
                    left_blocked, right_blocked, front_blocked, _ = self.get_obstacle_direction(next_state)

                    # First back up
                    self.robot.move_blocking(-100, -100, MOVEMENT_DURATION * 2)
                    time.sleep(0.3)

                    # Then make an intelligent turn based on sensor data
                    if left_blocked and not right_blocked:
                        self.robot.move_blocking(100, -70, MOVEMENT_DURATION)  # Sharp right
                    elif right_blocked and not left_blocked:
                        self.robot.move_blocking(-70, 100, MOVEMENT_DURATION)  # Sharp left
                    elif front_blocked:
                        # Random sharp turn if front is blocked
                        turn_action = random.choice([[-70, 100], [100, -70]])
                        self.robot.move_blocking(turn_action[0], turn_action[1], MOVEMENT_DURATION)
                    else:
                        # If unsure, make a random turn
                        turn_action = random.choice([[-50, 100], [100, -50]])
                        self.robot.move_blocking(turn_action[0], turn_action[1], MOVEMENT_DURATION)

                    time.sleep(0.2)

                    # Update the last collision step
                    last_collision_step = step

                # Enhanced reward structure based on sensor readings and obstacle proximity
                # Base reward - strongly penalize collisions
                reward = -30.0 if collision else 0.0

                # Get obstacle information for reward calculation
                left_blocked, right_blocked, front_blocked, closest_obstacle = self.get_obstacle_direction(next_state)

                # Calculate proximity penalty - penalize being close to obstacles
                proximity_penalty = 0
                if next_state:
                    # Higher penalty for being close to obstacles, especially in front
                    front_proximity = max([next_state[i] for i in [3, 4]])
                    side_proximity = max([next_state[i] for i in [0, 1, 2, 5, 6, 7]])

                    if front_proximity > self.close_obstacle_threshold:
                        proximity_penalty -= front_proximity * 2.0  # Higher penalty for front obstacles
                    if side_proximity > self.close_obstacle_threshold:
                        proximity_penalty -= side_proximity * 1  # Lower penalty for side obstacles

                # Add proximity penalty to reward
                reward += proximity_penalty

                # Safe navigation reward - high reward for maintaining safe distance
                if not collision and next_state:
                    # Check for safe navigation - being in open space away from obstacles
                    if all(val < self.obstacle_threshold for val in next_state):
                        reward += 2.0  # Strong bonus for safe navigation
                    # Smaller bonus for maintaining some distance
                    elif all(val < self.close_obstacle_threshold for val in next_state):
                        reward += 1.0  # Bonus for reasonable safety

                # Action-specific rewards - now focused on safety rather than forward progress
                if action_idx in [0, 1]:  # Forward movements
                    # Only reward forward movement if it's safe
                    if not front_blocked and self.is_path_clear(next_state):
                        reward += 1.0  # Moderate reward for safe forward movement
                    elif front_blocked:
                        reward -= 1.0  # Penalty for moving forward toward obstacles
                elif action_idx in [2, 3]:  # Left turns
                    # Reward turning away from obstacles
                    if right_blocked and not left_blocked:
                        reward += 1.0  # Good reward for appropriate turn
                    elif front_blocked and not left_blocked:
                        reward += 0.8  # Reward for avoiding front obstacle
                elif action_idx in [4, 5]:  # Right turns
                    # Reward turning away from obstacles
                    if left_blocked and not right_blocked:
                        reward += 1.0  # Good reward for appropriate turn
                    elif front_blocked and not right_blocked:
                        reward += 0.8  # Reward for avoiding front obstacle
                elif action_idx == 6:  # Backward
                    # Reward backing up when necessary
                    if front_blocked and (left_blocked or right_blocked):
                        reward += 1.5  # Strong reward for appropriate backup
                    elif front_blocked:
                        reward += 0.5  # Moderate reward for caution
                    elif not collision and step - last_collision_step > 3:
                        reward -= 1.0  # Smaller penalty for unnecessary backup
                elif action_idx == 7:  # Stop
                    reward -= 1.0  # Penalty for stopping

                # Check if episode is done - only end the episode when max steps is reached
                done = step == max_steps - initial_steps - 1

                # Store experience with collision flag
                self.remember(state, action_idx, reward, next_state, done, is_collision=collision)

                # Learn from experience
                self.replay()

                # Update state and tracking
                state = next_state
                total_reward += reward

                # Print status
                if step % 10 == 0:
                    print(
                        f"Episode {episode + 1}/{episodes}, Step {step}: Reward {total_reward:.2f}, Epsilon {self.epsilon:.4f}, Collisions: {episode_collisions}")

                if done:
                    break

            # Update target model more frequently to improve learning
            if episode % update_target_every == 0:
                self.update_target_model()
                print("Target model updated")

            # Track episode stats
            self.episode_rewards.append(total_reward)
            self.collisions.append(episode_collisions)
            self.steps_per_episode.append(step + 1)

            print(
                f"Episode {episode + 1} completed: Reward {total_reward:.2f}, Steps {step + 1}, Collisions {episode_collisions}")

            # Transfer learning from this episode to improve future episodes
            self.transfer_learning()

            #  track & save the “best” 
            score = total_reward
            if score > self.best_score:
                self.best_score = score
                self.save_model("best_model")
                print(f"  New best model (score {score:.2f}) saved")

            # Save model periodically
            if (episode + 1) % save_every == 0:
                self.save_model(f"model_ep{episode + 1}")
                self.save_metrics(f"metrics_ep{episode + 1}")

        # Save final model
        self.save_model("final_model")
        self.save_metrics("final_metrics")

        print("Training complete!")

    def save_model(self, name):
        """Save the model to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = "/root/results/models/" / f"{name}_{timestamp}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, path):
        """Load the model from disk."""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.update_target_model()
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")

    def save_metrics(self, name):
        """Save training metrics to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = "/root/results/rl/" / f"{name}_{timestamp}.json"
        metrics = {
            'rewards': self.episode_rewards,
            'collisions': self.collisions,
            'steps_per_episode': self.steps_per_episode,
            'final_epsilon': self.epsilon
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")


def run_exploration_learning(rob: IRobobo):
    """
    Main entry point that starts the reinforcement learning robot.
    This uses pure trial-and-error learning without manual data collection.

    Args:
        rob: An instance of IRobobo (either SimulationRobobo or HardwareRobobo)
    """
    print("Starting Trial-and-Error Learning for Robobo Navigation")

    # Initialize components
    sensor_handler = SensorInputHandler(rob)

    # Check if we should run in simulation or hardware mode
    if isinstance(rob, SimulationRobobo):
        print("Running in simulation mode")
        try:
            if not rob.is_running():
                print("Starting simulation...")
                rob.play_simulation()
                rob.sleep(2)
        except Exception as e:
            print(f"Simulation init error: {e}")
            return
    else:
        print("Running on hardware robot")

    # Create the learning agent
    agent = ExplorationLearner(rob, sensor_handler)

    # Train the agent with trial and error
    print("Beginning trial-and-error learning...")
    print("The robot will explore the environment and learn from its mistakes.")
    agent.train(episodes=50, max_steps=200)

    # Clean up
    if isinstance(rob, SimulationRobobo):
        try:
            rob.stop_simulation()
            print("Simulation stopped.")
        except Exception as e:
            print(f"Error stopping simulation: {e}")

def learn(rob: IRobobo):
    """
    Main entry point that starts the reinforcement learning navigation system.
    This function uses pure exploration learning without manual data collection.
    
    Args:
        rob: An instance of IRobobo (either SimulationRobobo or HardwareRobobo)
    """
    print("Starting Learning Machines Robobo RL Navigation System")
    
    # Initialize the sensor handler with the provided robot
    sensor_handler = SensorInputHandler(rob)
    
    # Check if we should run in simulation or hardware mode
    if isinstance(rob, SimulationRobobo):
        print("Running in simulation mode")
        try:
            if not rob.is_running():
                print("Starting simulation...")
                rob.play_simulation()
                rob.sleep(2)
        except Exception as e:
            print(f"Simulation init error: {e}")
            return
    else:
        print("Running on hardware robot")
    
    # Create the learning agent
    agent = ExplorationLearner(rob, sensor_handler)
    
    # Train the agent with trial and error
    print("Beginning trial-and-error learning...")
    print("The robot will explore the environment and learn from its mistakes.")
    agent.train(episodes=50, max_steps=200)
    
    # Clean up
    if isinstance(rob, SimulationRobobo):
        try:
            rob.stop_simulation()
            print("Simulation stopped.")
        except Exception as e:
            print(f"Error stopping simulation: {e}")
