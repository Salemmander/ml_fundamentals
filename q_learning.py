"""
Q-Learning: Teaching a Robot to Navigate
=========================================
A robot learns to navigate a grid world through trial and error.

Key concepts:
- Q-table: Maps (state, action) → expected future reward
- Bellman update: Learn from experience
- ε-greedy: Balance exploration vs exploitation
"""

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import time


# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_ARROWS = ["↑", "↓", "←", "→"]

# Movement deltas: (row_delta, col_delta)
ACTION_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}


class GridWorld:
    """
    A simple grid world environment for the robot to navigate.

    Grid cell types:
    - 0: Empty (passable)
    - 1: Wall (impassable)
    - 2: Goal (terminal state, big reward)
    - 3: Pit (terminal state, big penalty)
    """

    def __init__(self, grid: Optional[np.ndarray] = None):
        """Initialize the grid world."""
        if grid is None:
            # Default 5x5 grid with obstacles
            self.grid = np.array(
                [
                    [0, 0, 0, 1, 2],  # Goal at (0, 4)
                    [0, 1, 0, 1, 0],
                    [0, 1, 0, 0, 0],
                    [0, 3, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                ]
            )
        else:
            self.grid = grid.copy()

        self.n_rows, self.n_cols = self.grid.shape
        self.start_pos = (4, 0)  # Bottom-left
        self.robot_pos = self.start_pos

        # Find goal position
        goal_positions = np.where(self.grid == 2)
        self.goal_pos = (goal_positions[0][0], goal_positions[1][0])

        # Rewards
        self.step_reward = -0.1  # Small penalty per step (encourages efficiency)
        self.wall_reward = -0.5  # Penalty for hitting a wall
        self.goal_reward = 10.0  # Big reward for reaching goal
        self.pit_reward = -10.0  # Big penalty for falling in pit

    def reset(self, random_start: bool = False) -> Tuple[int, int]:
        """Reset robot to starting position. Returns initial state."""
        if random_start:
            # Find all empty cells
            empty_cells = list(zip(*np.where(self.grid == 0)))
            self.robot_pos = empty_cells[np.random.randint(len(empty_cells))]
        else:
            self.robot_pos = self.start_pos
        return self.robot_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Take an action in the environment.

        Args:
            action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

        Returns:
            (new_state, reward, done)
        """
        row, col = self.robot_pos
        d_row, d_col = ACTION_DELTAS[action]
        new_row, new_col = row + d_row, col + d_col

        # Check boundaries
        if not (0 <= new_row < self.n_rows and 0 <= new_col < self.n_cols):
            # Hit boundary wall
            return self.robot_pos, self.wall_reward, False

        # Check for wall
        if self.grid[new_row, new_col] == 1:
            # Hit internal wall
            return self.robot_pos, self.wall_reward, False

        # Move is valid
        self.robot_pos = (new_row, new_col)
        cell_type = self.grid[new_row, new_col]

        if cell_type == 2:  # Goal
            return self.robot_pos, self.goal_reward, True
        elif cell_type == 3:  # Pit
            return self.robot_pos, self.pit_reward, True
        else:  # Empty
            return self.robot_pos, self.step_reward, False

    def get_valid_actions(self, state: Tuple[int, int]) -> list:
        """Get list of valid actions from a state (for reference, not used in basic Q-learning)."""
        valid = []
        row, col = state
        for action, (d_row, d_col) in ACTION_DELTAS.items():
            new_row, new_col = row + d_row, col + d_col
            if (
                0 <= new_row < self.n_rows
                and 0 <= new_col < self.n_cols
                and self.grid[new_row, new_col] != 1
            ):
                valid.append(action)
        return valid


class QLearningAgent:
    """
    Q-Learning agent that learns to navigate the grid world.
    """

    def __init__(
        self,
        n_states: Tuple[int, int],
        n_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        """
        Initialize the Q-Learning agent.

        Args:
            n_states: (n_rows, n_cols) of the grid
            n_actions: Number of possible actions
            learning_rate: α - how fast to learn from new experiences
            discount_factor: γ - how much to value future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Multiply epsilon by this after each episode
            epsilon_min: Minimum epsilon value
        """
        self.n_rows, self.n_cols = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate  # α
        self.gamma = discount_factor  # γ
        self.epsilon = epsilon  # ε
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table with zeros
        # Shape: (n_rows, n_cols, n_actions)
        self.Q = np.zeros((self.n_rows, self.n_cols, n_actions))

        # Training history
        self.episode_rewards = []
        self.episode_lengths = []

    def choose_action(self, state: Tuple[int, int]) -> int:
        """
        Choose an action using ε-greedy policy.

        Args:
            state: Current (row, col) position

        Returns:
            Action to take (0-3)
        """
        # TODO(human): Implement epsilon-greedy action selection
        #
        # The ε-greedy policy balances exploration and exploitation:
        # - With probability ε: choose a RANDOM action (explore)
        # - With probability 1-ε: choose the BEST action from Q-table (exploit)
        #
        # Steps:
        # 1. Generate a random number between 0 and 1 using np.random.random()
        # 2. If random number < self.epsilon: return a random action
        #    - Use np.random.randint(self.n_actions) for random action
        # 3. Otherwise: return the action with highest Q-value for this state
        #    - Use np.argmax(self.Q[state[0], state[1], :]) to find best action
        #
        # This is crucial for learning! Pure exploitation gets stuck in local optima.
        # Pure exploration never uses what it learned.

        random = np.random.random()
        if random < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            # gets max Q value for moving in any direction from the current state
            return int(np.argmax(self.Q[state[0], state[1], :]))

    def update(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int],
        done: bool,
    ) -> float:
        """
        Update Q-value based on experience.

        This is the heart of Q-Learning: the Bellman update equation.

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether episode ended

        Returns:
            The TD error (for monitoring learning)
        """
        row, col = state
        next_row, next_col = next_state

        # Current Q-value estimate
        current_q = self.Q[row, col, action]

        # Calculate target Q-value
        if done:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Bellman equation: reward + discounted future value
            # The "max" is key: we assume we'll act optimally in the future
            best_next_q = np.max(self.Q[next_row, next_col, :])
            target_q = reward + self.gamma * best_next_q

        # TD (Temporal Difference) error: how wrong were we?
        td_error = target_q - current_q

        # Update Q-value: move towards target
        # Q(s,a) = Q(s,a) + α * (target - Q(s,a))
        self.Q[row, col, action] = current_q + self.alpha * td_error

        return td_error

    def decay_epsilon(self):
        """Reduce exploration rate over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self) -> np.ndarray:
        """Get the current policy (best action for each state)."""
        return np.argmax(self.Q, axis=2)

    def get_values(self) -> np.ndarray:
        """Get the current value function (max Q for each state)."""
        return np.max(self.Q, axis=2)


def visualize_training(
    env: GridWorld,
    agent: QLearningAgent,
    n_episodes: int = 500,
    animate_every: int = 50,
    delay: float = 0.05,
):
    """
    Train the agent and visualize the learning process.
    """
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ((ax_grid, ax_policy), (ax_rewards, ax_values)) = axes

    # Training loop
    for episode in range(n_episodes):
        state = env.reset(random_start=True)
        total_reward = 0
        steps = 0
        max_steps = 100  # Prevent infinite loops

        while steps < max_steps:
            # Choose action
            action = agent.choose_action(state)

            # Take action
            next_state, reward, done = env.step(action)

            # Learn from experience
            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        # Record episode stats
        agent.episode_rewards.append(total_reward)
        agent.episode_lengths.append(steps)

        # Decay exploration rate
        agent.decay_epsilon()

        # Visualize periodically
        if episode % animate_every == 0 or episode == n_episodes - 1:
            visualize_state(fig, axes, env, agent, episode, n_episodes, delay)

    plt.ioff()
    print("\nTraining complete! Close the plot window to exit.")
    plt.show()


def visualize_state(fig, axes, env, agent, episode, n_episodes, delay):
    """Update visualization with current learning state."""
    ((ax_grid, ax_policy), (ax_rewards, ax_values)) = axes

    for ax in axes.flat:
        ax.clear()

    # 1. Grid World with Policy Arrows
    ax_grid.set_title(f"Grid World & Policy (Episode {episode}/{n_episodes})")
    draw_grid(ax_grid, env, agent)

    # 2. Policy visualization
    ax_policy.set_title(f"Q-Values Heatmap (ε = {agent.epsilon:.3f})")
    values = agent.get_values()
    im = ax_policy.imshow(values, cmap="RdYlGn", aspect="equal")
    ax_policy.set_xticks(range(env.n_cols))
    ax_policy.set_yticks(range(env.n_rows))
    for i in range(env.n_rows):
        for j in range(env.n_cols):
            if env.grid[i, j] != 1:  # Not a wall
                ax_policy.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center")

    # 3. Reward history
    ax_rewards.set_title("Episode Rewards (Moving Average)")
    if len(agent.episode_rewards) > 0:
        rewards = agent.episode_rewards
        ax_rewards.plot(rewards, alpha=0.3, color="blue")
        # Moving average
        window = min(50, len(rewards))
        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax_rewards.plot(
                range(window - 1, len(rewards)),
                ma,
                color="red",
                linewidth=2,
                label=f"{window}-ep MA",
            )
        ax_rewards.set_xlabel("Episode")
        ax_rewards.set_ylabel("Total Reward")
        ax_rewards.legend()
        ax_rewards.grid(True, alpha=0.3)

    # 4. Value function
    ax_values.set_title("Episode Length (steps to goal)")
    if len(agent.episode_lengths) > 0:
        lengths = agent.episode_lengths
        ax_values.plot(lengths, alpha=0.3, color="green")
        window = min(50, len(lengths))
        if len(lengths) >= window:
            ma = np.convolve(lengths, np.ones(window) / window, mode="valid")
            ax_values.plot(
                range(window - 1, len(lengths)), ma, color="red", linewidth=2
            )
        ax_values.set_xlabel("Episode")
        ax_values.set_ylabel("Steps")
        ax_values.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(delay)

    # Print progress
    if len(agent.episode_rewards) > 0:
        recent_reward = (
            np.mean(agent.episode_rewards[-50:])
            if len(agent.episode_rewards) >= 50
            else np.mean(agent.episode_rewards)
        )
        recent_length = (
            np.mean(agent.episode_lengths[-50:])
            if len(agent.episode_lengths) >= 50
            else np.mean(agent.episode_lengths)
        )
        print(
            f"Episode {episode:4d} | Avg Reward: {recent_reward:7.2f} | Avg Steps: {recent_length:5.1f} | ε: {agent.epsilon:.3f}"
        )


def draw_grid(ax, env: GridWorld, agent: QLearningAgent):
    """Draw the grid world with policy arrows."""
    n_rows, n_cols = env.n_rows, env.n_cols

    # Draw cells
    for i in range(n_rows):
        for j in range(n_cols):
            cell = env.grid[i, j]

            if cell == 1:  # Wall
                color = "black"
            elif cell == 2:  # Goal
                color = "gold"
            elif cell == 3:  # Pit
                color = "red"
            else:  # Empty
                color = "white"

            rect = plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="gray", linewidth=2
            )
            ax.add_patch(rect)

            # Draw policy arrow for non-terminal, non-wall cells
            if cell == 0:
                best_action = np.argmax(agent.Q[i, j, :])
                dx, dy = 0, 0
                if best_action == UP:
                    dx, dy = 0, -0.3
                elif best_action == DOWN:
                    dx, dy = 0, 0.3
                elif best_action == LEFT:
                    dx, dy = -0.3, 0
                elif best_action == RIGHT:
                    dx, dy = 0.3, 0

                if np.max(agent.Q[i, j, :]) != 0:  # Only show if learned something
                    ax.arrow(
                        j,
                        i,
                        dx,
                        dy,
                        head_width=0.15,
                        head_length=0.1,
                        fc="blue",
                        ec="blue",
                    )

    # Mark start and goal
    ax.plot(env.start_pos[1], env.start_pos[0], "go", markersize=15, label="Start")
    ax.plot(env.goal_pos[1], env.goal_pos[0], "r*", markersize=20, label="Goal")

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)  # Flip y-axis
    ax.set_aspect("equal")
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.legend(loc="upper right")


def demo_learned_policy(env: GridWorld, agent: QLearningAgent, delay: float = 0.3):
    """Show the learned policy in action."""
    print("\n" + "=" * 50)
    print("DEMONSTRATING LEARNED POLICY")
    print("=" * 50)

    state = env.reset()
    total_reward = 0
    steps = 0
    path = [state]

    print(f"Start: {state}")

    while steps < 50:
        # Use greedy policy (no exploration)
        action = np.argmax(agent.Q[state[0], state[1], :])
        next_state, reward, done = env.step(action)

        total_reward += reward
        steps += 1
        path.append(next_state)

        print(f"  {ACTION_NAMES[action]} → {next_state} (reward: {reward:.1f})")

        if done:
            break
        state = next_state

    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Steps taken: {steps}")
    print("Path:", " → ".join([f"({r},{c})" for r, c in path]))


if __name__ == "__main__":
    print("=" * 60)
    print("Q-LEARNING: ROBOT NAVIGATION")
    print("=" * 60)
    print("\nThe robot (green dot) must learn to reach the goal (gold star)")
    print("while avoiding walls (black squares).")
    print("\nWatch the arrows evolve as it learns the optimal policy!")
    print("=" * 60 + "\n")

    # Create environment and agent
    env = GridWorld()
    agent = QLearningAgent(
        n_states=(env.n_rows, env.n_cols),
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    # Train with visualization
    visualize_training(env, agent, n_episodes=500, animate_every=25)

    # Demo the learned policy
    demo_learned_policy(env, agent)
