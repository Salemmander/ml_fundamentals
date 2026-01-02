"""
2D Kalman Filter Implementation
================================
Track a robot's position AND velocity using only noisy position measurements.

This demonstrates the power of the Kalman Filter: estimating hidden state
variables (velocity) from indirect measurements (position).

Key concepts:
- State vector: [position, velocity]
- We only MEASURE position, but can ESTIMATE velocity
- Matrix formulation generalizes to any number of state variables
"""

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import Tuple


class KalmanFilter2D:
    """
    A 2D Kalman Filter tracking position and velocity.

    State: [position, velocity]^T
    Measurement: position only (we can't directly measure velocity)
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        process_noise: np.ndarray,
        measurement_noise: float,
        dt: float,
    ):
        """
        Initialize the 2D Kalman Filter.

        Args:
            initial_state: Initial [position, velocity] estimate (2,)
            initial_covariance: Initial uncertainty matrix P (2x2)
            process_noise: Process noise covariance Q (2x2)
            measurement_noise: Measurement noise variance R (scalar, since we only measure position)
            dt: Time step between predictions
        """
        self.x = initial_state.reshape(2, 1)  # State vector [pos, vel]^T
        self.P = initial_covariance  # State covariance (2x2)
        self.Q = process_noise  # Process noise covariance (2x2)
        self.R = np.array([[measurement_noise]])  # Measurement noise (1x1)
        self.dt = dt

        # State transition matrix: x_new = F @ x
        # [pos_new]   [1  dt] [pos]     pos_new = pos + vel*dt
        # [vel_new] = [0   1] [vel]     vel_new = vel
        self.F = np.array([[1, dt], [0, 1]])

        # Observation matrix: z = H @ x
        # We only measure position, not velocity
        self.H = np.array([[1, 0]])  # Shape (1, 2)

        # Identity matrix for updates
        self.I = np.eye(2)

        # History for visualization
        self.history = {
            "position_estimates": [self.x[0, 0]],
            "velocity_estimates": [self.x[1, 0]],
            "position_variance": [self.P[0, 0]],
            "velocity_variance": [self.P[1, 1]],
        }

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction step: Project state and covariance forward in time.

        Returns:
            Tuple of (predicted_state, predicted_covariance)
        """
        # TODO(human): Implement the matrix prediction equations
        #
        # The prediction step projects our state estimate forward using the motion model.
        #
        # You need to implement:
        # 1. State prediction:      x = F @ x
        #    - F is the state transition matrix (self.F)
        #    - This applies the physics: new_pos = old_pos + velocity * dt
        #
        # 2. Covariance prediction: P = F @ P @ F.T + Q
        #    - F.T is the transpose of F (use self.F.T)
        #    - Q is process noise (self.Q) - accounts for model uncertainty
        #    - The F @ P @ F.T part propagates uncertainty through the motion model
        #
        # After implementing, return (self.x, self.P)
        #
        # Hint: Use @ for matrix multiplication in numpy (not *)
        #
        # Available instance variables:
        #   self.x = state vector (2x1): [[position], [velocity]]
        #   self.P = covariance matrix (2x2)
        #   self.F = state transition matrix (2x2)
        #   self.Q = process noise covariance (2x2)

        self.x = self.F @ self.x

        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x, self.P

    def update(self, measurement: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update step: Incorporate a position measurement.

        Args:
            measurement: Noisy position reading

        Returns:
            Tuple of (updated_state, updated_covariance, kalman_gain)
        """
        z = np.array([[measurement]])  # Measurement vector (1x1)

        # Innovation (measurement residual): difference between measurement and prediction
        # y = z - H @ x  (what we measured vs what we expected to measure)
        # How much we are off by on the measurement vs our prediction
        y = z - self.H @ self.x

        # Innovation covariance: uncertainty in the innovation
        # S = H @ P @ H.T + R
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman Gain: how much to trust the measurement
        # K = P @ H.T @ inv(S)
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update: blend prediction with measurement
        # x = x + K @ y
        self.x = self.x + K @ y

        # Covariance update: we're now more certain
        # P = (I - K @ H) @ P
        self.P = (self.I - K @ self.H) @ self.P

        # Store history
        self.history["position_estimates"].append(self.x[0, 0])
        self.history["velocity_estimates"].append(self.x[1, 0])
        self.history["position_variance"].append(self.P[0, 0])
        self.history["velocity_variance"].append(self.P[1, 1])

        return self.x, self.P, K


def simulate_robot_motion(
    n_steps: int,
    dt: float,
    initial_velocity: float = 2.0,
    acceleration_noise_std: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate robot motion with slight random accelerations.

    Returns:
        Tuple of (true_positions, true_velocities)
    """
    positions = [0.0]
    velocities = [initial_velocity]

    for _ in range(n_steps):
        # Add small random acceleration (simulates real-world perturbations)
        acceleration = np.random.normal(0, acceleration_noise_std)
        new_velocity = velocities[-1] + acceleration * dt
        new_position = positions[-1] + velocities[-1] * dt + 0.5 * acceleration * dt**2

        velocities.append(new_velocity)
        positions.append(new_position)

    return np.array(positions), np.array(velocities)


def generate_position_measurements(
    true_positions: np.ndarray, noise_std: float
) -> np.ndarray:
    """Generate noisy position measurements."""
    return true_positions + np.random.normal(0, noise_std, len(true_positions))


def run_filter_demo(animate: bool = True, delay: float = 0.2):
    """Run a demonstration of the 2D Kalman Filter."""

    # Simulation parameters
    n_steps = 50
    dt = 0.1
    initial_velocity = 2.0

    # Noise parameters
    measurement_noise_std = 1.0  # Position measurement noise
    process_noise_std = 0.5  # Model uncertainty

    # Generate ground truth
    np.random.seed(42)
    true_positions, true_velocities = simulate_robot_motion(
        n_steps, dt, initial_velocity
    )
    measurements = generate_position_measurements(true_positions, measurement_noise_std)

    # Initialize filter
    initial_state = np.array([0.0, 1.0])  # Start with wrong velocity guess!
    initial_covariance = np.array(
        [
            [1.0, 0.0],  # Position variance
            [0.0, 4.0],  # Velocity variance (high - we're uncertain)
        ]
    )
    process_noise = (
        np.array(
            [
                [0.25 * dt**4, 0.5 * dt**3],
                [0.5 * dt**3, dt**2],
            ]
        )
        * process_noise_std**2
    )

    kf = KalmanFilter2D(
        initial_state=initial_state,
        initial_covariance=initial_covariance,
        process_noise=process_noise,
        measurement_noise=measurement_noise_std**2,
        dt=dt,
    )

    # Set up animation
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ((ax_pos, ax_vel), (ax_pos_err, ax_vel_err)) = axes

    all_time = np.arange(n_steps + 1) * dt

    # Position plot setup
    ax_pos.set_xlim(-0.2, all_time[-1] + 0.2)
    ax_pos.set_ylim(-3, 12)
    ax_pos.set_xlabel("Time (s)")
    ax_pos.set_ylabel("Position (m)")
    ax_pos.set_title("Position Tracking")
    ax_pos.grid(True, alpha=0.3)

    # Velocity plot setup
    ax_vel.set_xlim(-0.2, all_time[-1] + 0.2)
    ax_vel.set_ylim(0, 4)
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Velocity (m/s)")
    ax_vel.set_title("Velocity Estimation (from position measurements only!)")
    ax_vel.grid(True, alpha=0.3)

    # Error plots setup
    ax_pos_err.set_xlim(-0.2, all_time[-1] + 0.2)
    ax_pos_err.set_ylim(-3, 3)
    ax_pos_err.set_xlabel("Time (s)")
    ax_pos_err.set_ylabel("Position Error (m)")
    ax_pos_err.set_title("Position Estimation Error")
    ax_pos_err.grid(True, alpha=0.3)
    ax_pos_err.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    ax_vel_err.set_xlim(-0.2, all_time[-1] + 0.2)
    ax_vel_err.set_ylim(-2, 2)
    ax_vel_err.set_xlabel("Time (s)")
    ax_vel_err.set_ylabel("Velocity Error (m/s)")
    ax_vel_err.set_title("Velocity Estimation Error")
    ax_vel_err.grid(True, alpha=0.3)
    ax_vel_err.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    # Plot elements
    (true_pos_line,) = ax_pos.plot([], [], "g-", linewidth=2, label="True Position")
    meas_scatter = ax_pos.scatter(
        [], [], c="red", s=30, alpha=0.5, label="Measurements"
    )
    (est_pos_line,) = ax_pos.plot([], [], "b-", linewidth=2, label="Estimated Position")

    (true_vel_line,) = ax_vel.plot([], [], "g-", linewidth=2, label="True Velocity")
    (est_vel_line,) = ax_vel.plot([], [], "b-", linewidth=2, label="Estimated Velocity")
    ax_vel.axhline(
        y=initial_velocity,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Initial Guess",
    )

    (pos_err_line,) = ax_pos_err.plot([], [], "b-", linewidth=2)
    (vel_err_line,) = ax_vel_err.plot([], [], "b-", linewidth=2)

    ax_pos.legend(loc="upper left")
    ax_vel.legend(loc="upper left")

    plt.tight_layout()

    # Storage
    time_hist = [0]
    true_pos_hist = [true_positions[0]]
    true_vel_hist = [true_velocities[0]]
    meas_hist = [measurements[0]]
    est_pos_hist = [kf.x[0, 0]]
    est_vel_hist = [kf.x[1, 0]]

    print("\n" + "=" * 70)
    print("2D KALMAN FILTER: ESTIMATING VELOCITY FROM POSITION MEASUREMENTS")
    print("=" * 70)
    print("Watch the velocity estimate (blue) converge to true velocity (green)")
    print("even though we NEVER directly measure velocity!")
    print("=" * 70 + "\n")

    # Run filter
    for i in range(1, n_steps + 1):
        current_time = i * dt

        # Predict
        kf.predict()

        # Update
        kf.update(measurements[i])

        # Store
        time_hist.append(current_time)
        true_pos_hist.append(true_positions[i])
        true_vel_hist.append(true_velocities[i])
        meas_hist.append(measurements[i])
        est_pos_hist.append(kf.x[0, 0])
        est_vel_hist.append(kf.x[1, 0])

        # Update plots
        true_pos_line.set_data(time_hist, true_pos_hist)
        meas_scatter.set_offsets(np.column_stack([time_hist, meas_hist]))
        est_pos_line.set_data(time_hist, est_pos_hist)

        true_vel_line.set_data(time_hist, true_vel_hist)
        est_vel_line.set_data(time_hist, est_vel_hist)

        pos_errors = np.array(est_pos_hist) - np.array(true_pos_hist)
        vel_errors = np.array(est_vel_hist) - np.array(true_vel_hist)
        pos_err_line.set_data(time_hist, pos_errors)
        vel_err_line.set_data(time_hist, vel_errors)

        if animate:
            print(
                f"Step {i:2d} | "
                f"True vel: {true_velocities[i]:.2f} | "
                f"Est vel: {kf.x[1, 0]:.2f} | "
                f"Vel error: {kf.x[1, 0] - true_velocities[i]:+.3f}"
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(delay)

    plt.ioff()

    # Summary
    est_pos = np.array(est_pos_hist)
    est_vel = np.array(est_vel_hist)

    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(
        f"Position MAE (measurements): {np.mean(np.abs(measurements - true_positions)):.3f} m"
    )
    print(
        f"Position MAE (Kalman):       {np.mean(np.abs(est_pos - true_positions)):.3f} m"
    )
    print(
        f"Velocity MAE (Kalman):       {np.mean(np.abs(est_vel - true_velocities)):.3f} m/s"
    )
    print(f"\nFinal velocity estimate: {est_vel[-1]:.3f} m/s")
    print(f"Final true velocity:     {true_velocities[-1]:.3f} m/s")
    print("=" * 70)

    plt.savefig(
        "/home/salem/Documents/Projects/Learning/ml_fundamentals/kalman_2d_demo.png",
        dpi=150,
    )
    print("\nClose the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    run_filter_demo()
