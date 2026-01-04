"""
Deep Q-Network (DQN) — Learning to Balance a Pole

This module implements DQN to solve CartPole-v1, demonstrating the transition from
tabular Q-Learning to deep reinforcement learning with neural network function
approximation.

Why DQN matters for robotics:
- Handles continuous state spaces (joint angles, velocities, sensor readings)
- Learns from raw sensory input → actions mapping
- Foundation for more advanced methods (DDPG, SAC, PPO)

Key concepts:
    1. Function Approximation: Neural network replaces Q-table
       Q-table: Q[s, a] → single value
       DQN: Q_network(s) → vector of Q-values for all actions

    2. Experience Replay: Store experiences, train on random batches
       - Breaks correlation between consecutive samples
       - Reuses data efficiently

    3. Target Network: Separate network for computing TD targets
       - Provides stable targets during training
       - Updated slowly via soft updates (polyak averaging)

    4. TD Loss (same Bellman idea as Q-Learning, but with gradients):
       target = r + γ * max_a' Q_target(s', a')
       loss = MSE(Q_online(s, a), target)

CartPole Environment:
    State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Actions: 0 = push left, 1 = push right
    Reward: +1 for each timestep the pole stays upright
    Done: pole angle > 12°, cart position > 2.4, or 500 steps
    Solved: Average reward >= 195 over 100 consecutive episodes
"""

import re
import matplotlib
from torch._prims_common import Dim

matplotlib.use("TkAgg")

import random
from collections import deque
from typing import Tuple, List, Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

BUFFER_SIZE = 10000  # Replay buffer capacity
BATCH_SIZE = 64  # Training batch size
GAMMA = 0.99  # Discount factor (same as Q-Learning!)
LR = 1e-3  # Learning rate for Adam optimizer
TAU = 0.005  # Soft update rate for target network
EPSILON_START = 1.0  # Initial exploration rate
EPSILON_END = 0.01  # Minimum exploration rate
EPSILON_DECAY = 0.995  # Decay rate per episode
MAX_EPISODES = 500  # Maximum training episodes
TARGET_SCORE = 195  # CartPole "solved" threshold


# =============================================================================
# REPLAY BUFFER — TODO(human)
# =============================================================================


class ReplayBuffer:
    """
    Circular buffer to store and sample experiences for training.

    Experience replay is crucial for DQN stability:
    - Breaks temporal correlation between consecutive experiences
    - Allows reusing experiences multiple times (data efficiency)
    - Provides diverse mini-batches for stable gradient updates

    Each experience is a tuple: (state, action, reward, next_state, done)
    """

    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.

        TODO(human): Create a buffer that can hold up to `capacity` experiences.
        Hint: collections.deque(maxlen=capacity) automatically removes oldest
              items when full — perfect for a circular buffer!

        Args:
            capacity: Maximum number of experiences to store
        """
        # TODO(human): Initialize your buffer here (~1 line)
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add an experience to the buffer.

        TODO(human): Store the experience tuple in your buffer.

        Args:
            state: Current state (numpy array)
            action: Action taken (int)
            reward: Reward received (float)
            next_state: Resulting state (numpy array)
            done: Whether episode ended (bool)
        """
        # TODO(human): Add the experience to your buffer (~1 line)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a random batch of experiences.

        TODO(human): Randomly select `batch_size` experiences and return them
        as separate numpy arrays.

        Steps:
        1. Use random.sample() to get batch_size random experiences
        2. Unzip the batch into separate lists: states, actions, rewards, etc.
        3. Convert each list to a numpy array with appropriate dtype:
           - states, next_states: float32
           - actions: int64
           - rewards, dones: float32

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            Shapes: (batch, state_dim), (batch,), (batch,), (batch, state_dim), (batch,)
        """
        # TODO(human): Sample and return batch (~5-8 lines)

        chosen = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*chosen)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        """
        Return the current number of experiences in the buffer.

        TODO(human): Return the length of your buffer (~1 line)
        """
        # TODO(human): Return buffer length
        return len(self.buffer)


# =============================================================================
# Q-NETWORK — Neural network to approximate Q-values
# =============================================================================


class QNetwork(nn.Module):
    """
    Neural network that approximates Q(s, a) for all actions.

    Architecture: state → 128 → 128 → n_actions
    This replaces the Q-table from tabular Q-Learning.

    Input: state vector (e.g., [position, velocity, angle, angular_velocity])
    Output: Q-values for each action (e.g., [Q(s, left), Q(s, right)])
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        """
        Initialize the Q-network.

        Args:
            state_dim: Dimension of state space (4 for CartPole)
            n_actions: Number of possible actions (2 for CartPole)
            hidden_dim: Size of hidden layers
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state → Q-values for all actions.

        Args:
            state: State tensor, shape (batch, state_dim)

        Returns:
            Q-values tensor, shape (batch, n_actions)
        """
        return self.network(state)


# =============================================================================
# DQN AGENT
# =============================================================================


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.

    Key components:
    - q_network: Online network for action selection and learning
    - target_network: Frozen network for stable TD targets
    - replay_buffer: Experience storage for batch training
    - optimizer: Adam for gradient updates
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        buffer_size: int = BUFFER_SIZE,
        batch_size: int = BATCH_SIZE,
        gamma: float = GAMMA,
        lr: float = LR,
        tau: float = TAU,
        device: str = "cpu",
    ):
        """
        Initialize DQN agent with two networks and replay buffer.

        Args:
            state_dim: Dimension of state space
            n_actions: Number of possible actions
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            gamma: Discount factor for future rewards
            lr: Learning rate for optimizer
            tau: Soft update rate for target network
            device: 'cpu' or 'cuda'
        """
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Create Q-network and target network (same architecture)
        self.q_network = QNetwork(state_dim, n_actions).to(device)
        self.target_network = QNetwork(state_dim, n_actions).to(device)

        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer for q_network only (target network is updated via soft update)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Select action using epsilon-greedy policy with neural network.

        This is similar to your Q-Learning choose_action(), but instead of
        looking up Q[state, :] in a table, we pass state through the neural
        network to get Q-values.

        TODO(human): Implement epsilon-greedy action selection:

        1. With probability epsilon:
           - Return a random action (exploration)
           - Hint: random.randrange(self.n_actions)

        2. Otherwise (exploitation):
           - Convert state to tensor: torch.FloatTensor(state).to(self.device)
           - Add batch dimension: state_tensor.unsqueeze(0) → shape (1, state_dim)
           - Get Q-values from network (no gradients needed!)
             with torch.no_grad():
                 q_values = self.q_network(state_tensor)
           - Return action with highest Q-value: q_values.argmax(dim=1).item()

        Args:
            state: Current state (numpy array)
            epsilon: Exploration probability (0 to 1)

        Returns:
            Action to take (int)
        """
        # TODO(human): Implement epsilon-greedy with neural network (~8-10 lines)
        chance = np.random.random()
        if chance < epsilon:
            return random.randrange(self.n_actions)
        else:
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            # dim=1: argmax over actions (columns), not batch (rows)
            # .item(): convert single-element tensor to Python int for gym
            return q_values.argmax(dim=1).item()

    def compute_td_loss(self, batch: Tuple[np.ndarray, ...]) -> torch.Tensor:
        """
        Compute TD loss for a batch of experiences — THE CORE OF DQN.

        This implements the DQN Bellman equation:
            target = r + γ * max_a' Q_target(s', a')    (if not done)
                   = r                                   (if done)
            loss = MSE(Q_online(s, a), target)

        TODO(human): Implement the TD loss computation:

        1. Unpack batch into: states, actions, rewards, next_states, dones

        2. Convert to tensors on correct device:
           - states, next_states: torch.FloatTensor(...).to(self.device)
           - actions: torch.LongTensor(...).to(self.device)
           - rewards, dones: torch.FloatTensor(...).to(self.device)

        3. Compute current Q-values Q(s, a):
           - q_values = self.q_network(states)  # shape: (batch, n_actions)
           - q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
             ^ This selects Q(s, a) for the action that was actually taken

        4. Compute target Q-values (NO GRADIENTS — use target network):
           with torch.no_grad():
               next_q_values = self.target_network(next_states)
               next_q_value = next_q_values.max(dim=1)[0]  # max over actions

        5. Compute TD target:
           target = rewards + self.gamma * next_q_value * (1 - dones)
           ^ Note: (1 - dones) zeros out future rewards for terminal states!

        6. Return MSE loss:
           return F.mse_loss(q_value, target)

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones)

        Returns:
            TD loss as a scalar tensor
        """
        # TODO(human): Implement TD loss (~15-20 lines)

        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_network(states)
        # gather(dim, index): selects elements along dim using index
        # q_values: (batch, n_actions), actions: (batch,)
        # unsqueeze(1): (batch,) → (batch, 1) for gather
        # gather picks Q(s, action_taken) for each row → (batch, 1)
        # squeeze(1): (batch, 1) → (batch,)
        # We use gather (not max) because we want Q for action TAKEN, not best
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            # .max(dim=1) returns (values, indices), already removes the dim
            # Result is (batch,) directly — no squeeze needed
            # We use max (not gather) because Bellman assumes optimal future action
            next_q_value = next_q_values.max(dim=1)[0]

        # Bellman target: r + γ * max Q(s')
        # gamma (0.99): discount factor — how much we value future vs immediate
        # (1 - dones): zeros out future term if episode ended (no next state)
        target = rewards + self.gamma * next_q_value * (1 - dones)

        return F.mse_loss(q_value, target)

    def soft_update_target(self) -> None:
        """
        Soft update target network weights: θ⁻ = τ*θ + (1-τ)*θ⁻

        This provides stable targets for TD learning. Instead of copying
        weights completely (hard update), we slowly blend in the new weights.

        With τ = 0.005:
        - target_weight = 0.005 * online_weight + 0.995 * target_weight
        - Target network changes very slowly → stable TD targets

        TODO(human): Update each parameter pair from online → target:

        for online_param, target_param in zip(
            self.q_network.parameters(),
            self.target_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1 - self.tau) * target_param.data
            )

        Or equivalently using in-place operations:
            target_param.data.mul_(1 - self.tau)
            target_param.data.add_(self.tau * online_param.data)
        """
        # TODO(human): Implement soft update (~3-5 lines)

        for online_param, target_param in zip(
            self.q_network.parameters(), self.target_network.parameters()
        ):
            # Equivalent to: target = (1-tau)*target + tau*online
            # In-place ops (mul_, add_) are more memory efficient
            target_param.data.mul_(1 - self.tau)
            target_param.data.add_(self.tau * online_param.data)

    def update(self) -> Optional[float]:
        """
        Perform one training step if enough experiences are available.

        This orchestrates the learning:
        1. Check if we have enough experiences
        2. Sample a batch from replay buffer
        3. Compute TD loss (your implementation!)
        4. Backpropagate and update weights
        5. Soft update target network

        Returns:
            Loss value if update was performed, None otherwise
        """
        # Need enough experiences to form a batch
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)

        # Compute loss using your implementation
        loss = self.compute_td_loss(batch)

        if loss is None:
            return None

        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network (your implementation!)
        self.soft_update_target()

        return loss.item()

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)


# =============================================================================
# TRAINING LOOP
# =============================================================================


def train_dqn(
    env_name: str = "CartPole-v1",
    max_episodes: int = MAX_EPISODES,
    target_score: float = TARGET_SCORE,
    render_freq: int = 0,
) -> Tuple[List[float], List[float], DQNAgent]:
    """
    Train DQN agent on CartPole environment.

    Args:
        env_name: Gymnasium environment name
        max_episodes: Maximum training episodes
        target_score: Score threshold to consider solved
        render_freq: Render every N episodes (0 = never)

    Returns:
        Tuple of (episode_rewards, epsilon_history, trained_agent)
    """
    print("=" * 60)
    print("DEEP Q-NETWORK (DQN) TRAINING")
    print("=" * 60)
    print(f"Environment: {env_name}")
    print(f"Target Score: {target_score} (avg over 100 episodes)")
    print()

    # Create environment
    env = gym.make(env_name, render_mode="human" if render_freq > 0 else None)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    print(f"State dimension: {state_dim}")
    print(f"Number of actions: {n_actions}")
    print()

    # Check if implementations are complete
    agent = DQNAgent(state_dim, n_actions)

    # Test if methods are implemented
    test_state = np.zeros(state_dim, dtype=np.float32)
    test_action = agent.select_action(test_state, epsilon=0.5)

    if test_action is None:
        print("!" * 60)
        print("NOTE: select_action() not yet implemented!")
        print("Complete the TODO(human) sections in DQNAgent, then run again.")
        print("!" * 60)
        env.close()
        return [], [], agent

    # Initialize agent
    agent = DQNAgent(state_dim, n_actions)

    # Training tracking
    episode_rewards: List[float] = []
    epsilon_history: List[float] = []
    epsilon = EPSILON_START
    recent_rewards: deque = deque(maxlen=100)

    print("Starting training...")
    print("-" * 60)

    for episode in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, epsilon)
            if action is None:
                break

            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store experience
            agent.store_experience(state, action, reward, next_state, done)

            # Train on a batch
            agent.update()

            # Update state and reward
            state = next_state
            total_reward += reward

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Track progress
        episode_rewards.append(total_reward)
        epsilon_history.append(epsilon)
        recent_rewards.append(total_reward)
        avg_reward = np.mean(recent_rewards)

        # Print progress
        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1:4d}: "
                f"Reward={total_reward:6.1f}, "
                f"Avg(100)={avg_reward:6.1f}, "
                f"Epsilon={epsilon:.3f}"
            )

        # Check if solved
        if len(recent_rewards) >= 100 and avg_reward >= target_score:
            print()
            print("=" * 60)
            print(f"SOLVED in {episode + 1} episodes!")
            print(f"Average reward over last 100 episodes: {avg_reward:.1f}")
            print("=" * 60)
            break

    env.close()

    return episode_rewards, epsilon_history, agent


# =============================================================================
# VISUALIZATION
# =============================================================================


def visualize_training(
    episode_rewards: List[float],
    epsilon_history: List[float],
    target_score: float = TARGET_SCORE,
) -> None:
    """
    Visualize training progress: rewards and epsilon decay.

    Args:
        episode_rewards: List of total rewards per episode
        epsilon_history: List of epsilon values per episode
        target_score: Score threshold to mark as "solved"
    """
    if not episode_rewards:
        print("No training data to visualize.")
        return

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("DQN Training on CartPole-v1", fontsize=14)

    # Plot rewards
    episodes = range(1, len(episode_rewards) + 1)
    ax1.plot(episodes, episode_rewards, alpha=0.6, label="Episode Reward")

    # Running average
    window = 100
    if len(episode_rewards) >= window:
        running_avg = [
            np.mean(episode_rewards[max(0, i - window) : i + 1])
            for i in range(len(episode_rewards))
        ]
        ax1.plot(
            episodes, running_avg, color="red", linewidth=2, label="100-ep Average"
        )

    ax1.axhline(
        y=target_score, color="green", linestyle="--", label=f"Target ({target_score})"
    )
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Training Rewards")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot epsilon decay
    ax2.plot(episodes, epsilon_history, color="orange", linewidth=2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon")
    ax2.set_title("Exploration Rate (Epsilon) Decay")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/dqn_training.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved to output/dqn_training.png")
    plt.ioff()
    plt.show()


def demo_trained_agent(
    agent: DQNAgent, env_name: str = "CartPole-v1", episodes: int = 3
) -> None:
    """
    Demonstrate trained agent with rendering.

    Args:
        agent: Trained DQN agent
        env_name: Environment name
        episodes: Number of demo episodes
    """
    print()
    print("=" * 60)
    print("DEMONSTRATING TRAINED AGENT")
    print("=" * 60)

    env = gym.make(env_name, render_mode="human")

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Use greedy policy (epsilon=0)
            action = agent.select_action(state, epsilon=0)
            if action is None:
                break
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Demo Episode {ep + 1}: Reward = {total_reward:.0f}")

    env.close()


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Main entry point — train DQN on CartPole."""
    print()
    print("=" * 60)
    print("DQN: FROM Q-TABLE TO NEURAL NETWORK")
    print("=" * 60)
    print()
    print("Remember your Q-Learning update?")
    print("  Q[s,a] = Q[s,a] + α * (r + γ * max(Q[s',:]) - Q[s,a])")
    print()
    print("DQN does the same thing, but:")
    print("  - Q-table → Neural network")
    print("  - Single updates → Batch gradient descent")
    print("  - One Q-table → Two networks (online + target)")
    print()
    print("Your implementations connect these ideas:")
    print("  - ReplayBuffer: Store experiences for batch training")
    print("  - select_action: Same epsilon-greedy, but query network")
    print("  - compute_td_loss: Bellman equation + MSE loss")
    print("  - soft_update_target: Slowly update target network")
    print()
    input("Press Enter to start training...")
    print()

    # Train the agent
    rewards, epsilons, agent = train_dqn()

    if rewards:
        # Visualize results
        visualize_training(rewards, epsilons)

        # Demo the trained agent
        response = input("\nWatch trained agent? (y/n): ")
        if response.lower() == "y":
            demo_trained_agent(agent, episodes=3)


if __name__ == "__main__":
    main()
