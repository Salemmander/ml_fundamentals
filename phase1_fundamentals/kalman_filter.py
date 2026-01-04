"""
1D Kalman Filter Implementation
================================
Track a robot moving in 1D with noisy position measurements.

This is our first ML algorithm implementation - understanding state estimation
is fundamental for robotics (localization, tracking, SLAM).
"""

import numpy as np
import matplotlib

matplotlib.use("TkAgg")  # Interactive backend - must be before pyplot import
import matplotlib.pyplot as plt
from typing import Tuple, List


class KalmanFilter1D:
    """
    A 1D Kalman Filter for tracking position with noisy measurements.

    State: position (x)
    Measurement: noisy position reading from sensor
    """

    def __init__(
        self,
        initial_position: float,
        initial_uncertainty: float,
        process_noise: float,
        measurement_noise: float,
    ):
        """
        Initialize the Kalman Filter.

        Args:
            initial_position: Starting estimate of position
            initial_uncertainty: How uncertain we are about initial position (variance)
            process_noise: Q - How much uncertainty the motion model adds per step
            measurement_noise: R - How noisy our sensor is (variance)
        """
        self.x = initial_position  # State estimate
        self.P = initial_uncertainty  # Estimate uncertainty (variance)
        self.Q = process_noise  # Process noise variance
        self.R = measurement_noise  # Measurement noise variance

        # History for visualization
        self.history = {
            "estimates": [initial_position],
            "uncertainties": [initial_uncertainty],
            "kalman_gains": [],
        }

    def predict(self, velocity: float, dt: float) -> Tuple[float, float]:
        """
        Prediction step: Estimate where the robot will be based on motion model.

        Args:
            velocity: Commanded velocity (m/s)
            dt: Time step (seconds)

        Returns:
            Tuple of (predicted_position, predicted_uncertainty)
        """
        # Predict new position based on motion model: x_new = x + v * dt
        self.x = self.x + velocity * dt

        # Uncertainty grows during prediction (we're less sure after movement)
        # This is because our motion model isn't perfect (wheels slip, etc.)
        self.P = self.P + self.Q

        return self.x, self.P

    def update(self, measurement: float) -> Tuple[float, float, float]:
        """
        Update step: Incorporate a new sensor measurement to refine our estimate.

        This is where the "magic" happens - we optimally blend our prediction
        with the noisy measurement based on their respective uncertainties.

        Args:
            measurement: Noisy position reading from sensor

        Returns:
            Tuple of (updated_position, updated_uncertainty, kalman_gain)
        """
        # TODO(human): Implement the Kalman Filter update equations
        #
        # You need to:
        # 1. Calculate the Kalman Gain: K = P / (P + R)
        #    - K tells us how much to trust the measurement vs our prediction
        #    - When R is small (accurate sensor), K approaches 1 (trust measurement)
        #    - When R is large (noisy sensor), K approaches 0 (trust prediction)
        #
        # 2. Update the state estimate: x = x + K * (measurement - x)
        #    - (measurement - x) is the "innovation" or "residual"
        #    - We adjust our estimate by K times the difference
        #
        # 3. Update the uncertainty: P = (1 - K) * P
        #    - After incorporating a measurement, we become MORE certain
        #    - Our uncertainty shrinks!
        #
        # Return: (self.x, self.P, K)
        #
        # Available instance variables:
        #   self.x = current position estimate
        #   self.P = current uncertainty
        #   self.R = measurement noise variance

        K = self.P / (self.P + self.R)  # Kalman Gain
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P

        # After implementing, uncomment these lines to track history:
        self.history["estimates"].append(self.x)
        self.history["uncertainties"].append(self.P)
        self.history["kalman_gains"].append(K)

        return self.x, self.P, K


def simulate_robot_movement(
    n_steps: int, true_velocity: float, dt: float, start_position: float = 0.0
) -> np.ndarray:
    """Simulate true robot positions (ground truth)."""
    positions = [start_position]
    for _ in range(n_steps):
        # Add small random perturbation to simulate real-world motion
        noise = np.random.normal(0, 0.1)
        positions.append(positions[-1] + true_velocity * dt + noise)
    return np.array(positions)


def generate_noisy_measurements(
    true_positions: np.ndarray, measurement_noise_std: float
) -> np.ndarray:
    """Generate noisy sensor readings from true positions."""
    noise = np.random.normal(0, measurement_noise_std, len(true_positions))
    return true_positions + noise


def run_filter_demo(animate: bool = True, delay: float = 0.3):
    """
    Run a demonstration of the 1D Kalman Filter.

    Args:
        animate: If True, show step-by-step animation. If False, show final result.
        delay: Seconds between animation frames (default 0.15s)
    """

    # Simulation parameters
    n_steps = 50
    dt = 0.1  # 100ms time steps
    true_velocity = 2.0  # m/s

    # Noise parameters
    process_noise = 0.1  # Q: motion model uncertainty
    measurement_noise = 1.0  # R: sensor noise variance (std = 1m)

    # Generate ground truth and measurements
    np.random.seed(42)  # For reproducibility
    true_positions = simulate_robot_movement(n_steps, true_velocity, dt)
    measurements = generate_noisy_measurements(
        true_positions, np.sqrt(measurement_noise)
    )

    # Initialize Kalman Filter
    kf = KalmanFilter1D(
        initial_position=0.0,
        initial_uncertainty=1.0,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
    )

    # Set up the plot
    plt.ion()  # Interactive mode for live updates
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax1, ax2 = axes

    # Pre-plot the ground truth (faded, revealed progressively)
    all_time = np.arange(n_steps + 1) * dt
    ax1.set_xlim(-0.2, all_time[-1] + 0.2)
    ax1.set_ylim(-3, 12)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position (m)")
    ax1.set_title("1D Kalman Filter: Robot Position Tracking")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlim(-0.2, all_time[-1] + 0.2)
    ax2.set_ylim(0, 0.6)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Kalman Gain")
    ax2.set_title("Kalman Gain Over Time (shows how much we trust measurements)")
    ax2.grid(True, alpha=0.3)

    # Initialize plot elements
    (true_line,) = ax1.plot([], [], "g-", linewidth=2, label="True Position")
    meas_scatter = ax1.scatter(
        [], [], c="red", s=40, alpha=0.6, label="Noisy Measurement", zorder=5
    )
    (estimate_line,) = ax1.plot([], [], "b-", linewidth=2, label="Kalman Estimate")
    (predict_marker,) = ax1.plot(
        [], [], "co", markersize=10, label="Prediction", zorder=6
    )
    uncertainty_fill = None

    (kalman_line,) = ax2.plot([], [], "purple", linewidth=2)

    ax1.legend(loc="upper left")
    plt.tight_layout()

    # Storage for animation
    time_history = [0]
    true_history = [true_positions[0]]
    meas_history = [measurements[0]]
    estimate_history = [kf.x]
    uncertainty_history = [kf.P]
    kalman_history = []

    print("\n" + "=" * 60)
    print("KALMAN FILTER LIVE DEMO")
    print("=" * 60)
    print("Watch the blue line (estimate) track the green line (truth)")
    print("while filtering out the red noise from measurements!")
    print("=" * 60 + "\n")

    # Run filter with animation
    for i in range(1, n_steps + 1):
        current_time = i * dt

        # === PREDICT STEP ===
        predicted_x, predicted_P = kf.predict(velocity=true_velocity, dt=dt)

        if animate:
            # Show prediction (before update)
            predict_marker.set_data([current_time], [predicted_x])
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(delay / 2)

        # === UPDATE STEP ===
        updated_x, updated_P, K = kf.update(measurements[i])

        # Store history
        time_history.append(current_time)
        true_history.append(true_positions[i])
        meas_history.append(measurements[i])
        estimate_history.append(updated_x)
        uncertainty_history.append(updated_P)
        kalman_history.append(K)

        # Update plot data
        true_line.set_data(time_history, true_history)
        meas_scatter.set_offsets(np.column_stack([time_history, meas_history]))
        estimate_line.set_data(time_history, estimate_history)
        predict_marker.set_data([], [])  # Hide prediction marker after update

        # Update uncertainty band
        est_arr = np.array(estimate_history)
        unc_arr = np.array(uncertainty_history)
        if uncertainty_fill is not None:
            uncertainty_fill.remove()
        uncertainty_fill = ax1.fill_between(
            time_history,
            est_arr - 2 * np.sqrt(unc_arr),
            est_arr + 2 * np.sqrt(unc_arr),
            alpha=0.3,
            color="blue",
        )

        # Update Kalman gain plot
        kalman_line.set_data(time_history[1:], kalman_history)

        if animate:
            # Print step info
            print(
                f"Step {i:2d} | Time: {current_time:.1f}s | "
                f"Measurement: {measurements[i]:6.2f} | "
                f"Estimate: {updated_x:6.2f} | "
                f"K: {K:.3f}"
            )

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(delay)

    plt.ioff()  # Turn off interactive mode

    # Save final result
    plt.savefig("output/kalman_demo.png", dpi=150)

    # Print summary statistics
    estimates = np.array(estimate_history)
    print("\n" + "=" * 60)
    print("KALMAN FILTER PERFORMANCE SUMMARY")
    print("=" * 60)
    print(
        f"Mean Absolute Error (measurements): {np.mean(np.abs(measurements - true_positions)):.3f} m"
    )
    print(
        f"Mean Absolute Error (Kalman):       {np.mean(np.abs(estimates - true_positions)):.3f} m"
    )
    improvement = (
        1
        - np.mean(np.abs(estimates - true_positions))
        / np.mean(np.abs(measurements - true_positions))
    ) * 100
    print(f"Improvement: {improvement:.1f}%")
    print("=" * 60)

    print("\nClose the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    run_filter_demo()
