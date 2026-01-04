"""
Linear Regression with Gradient Descent — Learning Optimization Fundamentals

This module implements linear regression from scratch using gradient descent,
the foundational optimization algorithm that powers all of deep learning.

Why this matters for robotics:
- Sensor calibration: fitting measurement models to ground truth
- Trajectory fitting: learning motion models from data
- System identification: estimating physical parameters from observations

Key concepts:
- Loss function (MSE): measures how wrong our predictions are
- Gradient: direction of steepest increase in loss
- Gradient descent: iteratively step opposite to gradient to minimize loss

The gradient descent pattern here is IDENTICAL to how neural networks learn,
just without the chain rule complexity of backpropagation.
"""

import matplotlib

matplotlib.use("TkAgg")  # Must come before pyplot import for Linux

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


class LinearRegression:
    """
    Linear regression model trained with gradient descent.

    Model: y = X @ W + b
    - X: input features (n_samples, n_features)
    - W: weight vector (n_features,)
    - b: bias scalar

    Training uses gradient descent to minimize Mean Squared Error (MSE):
    L = (1/n) * sum((predictions - targets)^2)

    Attributes:
        weights (np.ndarray): Weight vector, shape (n_features,)
        bias (float): Bias term
        learning_rate (float): Step size for gradient descent
        history (dict): Training history for visualization
            - 'loss': MSE at each iteration
            - 'weights': weight values over time
            - 'bias': bias values over time
            - 'grad_magnitude': gradient norm at each step
    """

    def __init__(
        self,
        n_features: int = 1,
        learning_rate: float = 0.01,
        random_seed: int | None = None,
    ):
        """
        Initialize linear regression model with random weights.

        Args:
            n_features: Number of input features
            learning_rate: Step size for gradient descent (alpha)
            random_seed: Optional seed for reproducible initialization
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize weights randomly (small values near zero)
        self.weights = np.random.randn(n_features) * 0.1
        self.bias = 0.0
        self.learning_rate = learning_rate

        # History tracking for visualization
        self.history = {
            "loss": [],
            "weights": [self.weights.copy()],
            "bias": [self.bias],
            "grad_magnitude": [],
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictions: y_hat = X @ W + b

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predictions, shape (n_samples,)
        """
        return X @ self.weights + self.bias

    def fit(self, X: np.ndarray, y: np.ndarray, n_iterations: int = 100) -> None:
        """
        Train the model using gradient descent.

        This method runs the gradient descent optimization loop to find
        weights that minimize the Mean Squared Error loss.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training targets, shape (n_samples,)
            n_iterations: Number of gradient descent steps

        Math:
            Loss (MSE): L = (1/n) * sum((y_hat - y)^2)

            Gradients (derived from chain rule):
                dL/dW = (2/n) * X.T @ (y_hat - y)
                dL/db = (2/n) * sum(y_hat - y)

            Update rule:
                W = W - learning_rate * dL/dW
                b = b - learning_rate * dL/db
        """
        n_samples = X.shape[0]

        # TODO(human): Implement the gradient descent training loop
        #
        # Your task: Write the complete training loop (~10-15 lines)
        #
        # For each iteration, you need to:
        # 1. FORWARD PASS: Compute predictions using self.predict(X)
        #
        # 2. COMPUTE LOSS: MSE = (1/n) * sum((predictions - y)^2)
        #    Hint: Use np.mean() and squaring with **2
        #
        # 3. COMPUTE GRADIENTS: How does loss change with weights?
        #    dW = (2/n) * X.T @ (predictions - y)
        #    db = (2/n) * np.sum(predictions - y)
        #    Intuition: X.T @ error gives you how much each feature
        #               contributed to the error
        #
        # 4. UPDATE WEIGHTS: Step opposite to gradient direction
        #    self.weights = self.weights - self.learning_rate * dW
        #    self.bias = self.bias - self.learning_rate * db
        #
        # 5. TRACK HISTORY: For visualization
        #    self.history["loss"].append(loss)
        #    self.history["weights"].append(self.weights.copy())
        #    self.history["bias"].append(self.bias)
        #    self.history["grad_magnitude"].append(np.linalg.norm(dW))
        #
        # Structure:
        # for i in range(n_iterations):
        #     ... your code here ...
        #
        for _ in range(n_iterations):
            predictions = self.predict(X)
            loss = np.mean((predictions - y) ** 2)

            # Partial Derivitives of MSE
            dW = (2 / n_samples) * X.T @ (predictions - y)
            db = (2 / n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.learning_rate * dW
            self.bias = self.bias - self.learning_rate * db

            self.history["loss"].append(loss)
            self.history["weights"].append(self.weights.copy())
            self.history["bias"].append(self.bias)
            self.history["grad_magnitude"].append(np.linalg.norm(dW))


def generate_linear_data(
    n_samples: int = 100,
    true_weight: float = 3.0,
    true_bias: float = 7.0,
    noise_std: float = 2.0,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Generate synthetic linear data: y = weight * x + bias + noise

    Args:
        n_samples: Number of data points
        true_weight: Ground truth slope
        true_bias: Ground truth intercept
        noise_std: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility

    Returns:
        X: Features, shape (n_samples, 1)
        y: Targets, shape (n_samples,)
        true_weight: The actual weight used
        true_bias: The actual bias used
    """
    np.random.seed(random_seed)

    # Generate x values spread across a range
    X = np.random.uniform(-5, 5, size=(n_samples, 1))

    # Generate y = wx + b + noise
    y = (
        true_weight * X.squeeze()
        + true_bias
        + np.random.normal(0, noise_std, n_samples)
    )

    return X, y, true_weight, true_bias


def compute_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R-squared (coefficient of determination).

    R² = 1 - (SS_res / SS_tot)
    where SS_res = sum((y_true - y_pred)^2)  # residual sum of squares
          SS_tot = sum((y_true - y_mean)^2)  # total sum of squares

    R² = 1.0 means perfect fit, R² = 0 means model is no better than mean.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        R-squared score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def run_regression_demo(
    animate: bool = True,
    delay: float = 0.05,
    n_iterations: int = 200,
    learning_rate: float = 0.05,
) -> None:
    """
    Run animated demonstration of linear regression with gradient descent.

    Visualization shows:
    - Top-left: Data points and regression line (updates as model learns)
    - Top-right: Loss curve over iterations
    - Bottom-left: Gradient magnitude over time
    - Bottom-right: Weight trajectory in parameter space

    Args:
        animate: If True, show step-by-step animation
        delay: Seconds between animation frames
        n_iterations: Number of gradient descent iterations
        learning_rate: Step size for optimization
    """
    print("=" * 60)
    print("LINEAR REGRESSION WITH GRADIENT DESCENT")
    print("=" * 60)
    print(f"Learning rate: {learning_rate}")
    print(f"Iterations: {n_iterations}")
    print()

    # Generate synthetic data
    X, y, true_w, true_b = generate_linear_data(
        n_samples=50, true_weight=3.0, true_bias=7.0, noise_std=2.0
    )
    print(f"True parameters: weight = {true_w:.2f}, bias = {true_b:.2f}")

    # Initialize model
    model = LinearRegression(n_features=1, learning_rate=learning_rate, random_seed=123)
    print(f"Initial weights: {model.weights[0]:.4f}, bias: {model.bias:.4f}")
    print()

    # Set up visualization
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Linear Regression: Gradient Descent in Action", fontsize=14)

    ax_data, ax_loss = axes[0]
    ax_grad, ax_params = axes[1]

    # Data plot setup (top-left)
    ax_data.scatter(X, y, c="blue", alpha=0.6, label="Data points", s=30)
    x_line = np.linspace(X.min() - 0.5, X.max() + 0.5, 100).reshape(-1, 1)
    (line_pred,) = ax_data.plot([], [], "r-", linewidth=2, label="Model prediction")
    (line_true,) = ax_data.plot(
        x_line,
        true_w * x_line + true_b,
        "g--",
        linewidth=2,
        alpha=0.7,
        label="True line",
    )
    ax_data.set_xlabel("x")
    ax_data.set_ylabel("y")
    ax_data.set_title("Fitting a Line to Data")
    ax_data.legend()
    ax_data.grid(True, alpha=0.3)

    # Loss plot setup (top-right)
    (line_loss,) = ax_loss.plot([], [], "b-", linewidth=2)
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("MSE Loss")
    ax_loss.set_title("Loss Over Time (Should Decrease!)")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_xlim(0, n_iterations)

    # Gradient magnitude plot setup (bottom-left)
    (line_grad,) = ax_grad.plot([], [], "purple", linewidth=2)
    ax_grad.set_xlabel("Iteration")
    ax_grad.set_ylabel("Gradient Magnitude")
    ax_grad.set_title("Gradient Magnitude (Should Shrink)")
    ax_grad.grid(True, alpha=0.3)
    ax_grad.set_xlim(0, n_iterations)

    # Parameter trajectory setup (bottom-right)
    ax_params.axhline(
        y=true_w, color="g", linestyle="--", alpha=0.7, label=f"True w={true_w}"
    )
    ax_params.axvline(
        x=true_b, color="g", linestyle="--", alpha=0.7, label=f"True b={true_b}"
    )
    (scatter_params,) = ax_params.plot([], [], "ro-", markersize=3, alpha=0.5)
    ax_params.set_xlabel("Bias")
    ax_params.set_ylabel("Weight")
    ax_params.set_title("Parameter Trajectory")
    ax_params.legend()
    ax_params.grid(True, alpha=0.3)

    plt.tight_layout()

    # Train the model
    print("Training...")
    model.fit(X, y, n_iterations=n_iterations)

    # Check if fit was implemented
    if len(model.history["loss"]) == 0:
        print("\n" + "!" * 60)
        print("NOTE: fit() method not yet implemented!")
        print("Complete the TODO(human) section in the fit() method,")
        print("then run this demo again to see gradient descent in action.")
        print("!" * 60)
        plt.ioff()
        plt.show()
        return

    # Animate the training history
    losses = model.history["loss"]
    grads = model.history["grad_magnitude"]
    weights_hist = np.array(model.history["weights"])
    bias_hist = np.array(model.history["bias"])

    # Set axis limits based on data
    ax_loss.set_ylim(0, max(losses) * 1.1)
    if len(grads) > 0:
        ax_grad.set_ylim(0, max(grads) * 1.1)
    ax_params.set_xlim(min(bias_hist) - 1, max(max(bias_hist), true_b) + 1)
    ax_params.set_ylim(min(weights_hist) - 0.5, max(max(weights_hist), true_w) + 0.5)

    for i in range(n_iterations):
        # Update regression line
        w = weights_hist[i + 1][0]  # +1 because history includes initial
        b = bias_hist[i + 1]
        y_line = w * x_line.squeeze() + b
        line_pred.set_data(x_line.squeeze(), y_line)

        # Update loss curve
        line_loss.set_data(range(i + 1), losses[: i + 1])

        # Update gradient curve
        line_grad.set_data(range(i + 1), grads[: i + 1])

        # Update parameter trajectory
        scatter_params.set_data(bias_hist[: i + 2], weights_hist[: i + 2, 0])

        if animate:
            if i % 10 == 0:
                print(
                    f"Iteration {i:3d}: loss = {losses[i]:.4f}, w = {w:.4f}, b = {b:.4f}"
                )
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(delay)

    # Final results
    final_w = model.weights[0]
    final_b = model.bias
    predictions = model.predict(X)
    r_squared = compute_r_squared(y, predictions)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Learned: weight = {final_w:.4f}, bias = {final_b:.4f}")
    print(f"True:    weight = {true_w:.4f}, bias = {true_b:.4f}")
    print(
        f"Error:   weight = {abs(final_w - true_w):.4f}, bias = {abs(final_b - true_b):.4f}"
    )
    print(f"R² score: {r_squared:.4f}")
    print("=" * 60)

    plt.ioff()
    plt.savefig("output/linear_regression_result.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved to output/linear_regression_result.png")
    plt.show()


if __name__ == "__main__":
    run_regression_demo(animate=True, delay=0.03, n_iterations=200, learning_rate=0.05)
