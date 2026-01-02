"""
Neural Network from Scratch — Multi-Layer Perceptron with Backpropagation

This module implements a neural network from scratch using NumPy, demonstrating
the complete forward and backward passes that power all of deep learning.

Architecture:
    Input (2) → Hidden (4, ReLU) → Output (1, Sigmoid)

Test problem: XOR — the classic non-linearly-separable problem that proves
your hidden layer is learning useful representations.

Why this matters for robotics:
- Policy networks: mapping sensor states to actions
- Perception: neural networks for object recognition
- Control: learned dynamics models

Key concepts:
- Forward pass: compute predictions by propagating through layers
- Backward pass (backpropagation): compute gradients using the chain rule
- Gradient descent: update weights to minimize loss

This is exactly how PyTorch and TensorFlow work under the hood — just with
automatic differentiation instead of manual gradient computation.
"""

import matplotlib

matplotlib.use("TkAgg")  # Must come before pyplot import for Linux

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


class NeuralNetwork:
    """
    A 2-layer neural network (1 hidden layer) for binary classification.

    Architecture:
        Input (n_input) → Hidden (n_hidden, ReLU) → Output (1, Sigmoid)

    Attributes:
        W1 (np.ndarray): Weights for input→hidden, shape (n_input, n_hidden)
        b1 (np.ndarray): Biases for hidden layer, shape (1, n_hidden)
        W2 (np.ndarray): Weights for hidden→output, shape (n_hidden, 1)
        b2 (np.ndarray): Biases for output layer, shape (1, 1)
        learning_rate (float): Step size for gradient descent
        history (dict): Training history for visualization

    The forward pass computes:
        z1 = X @ W1 + b1      (linear transform)
        a1 = relu(z1)         (activation)
        z2 = a1 @ W2 + b2     (linear transform)
        a2 = sigmoid(z2)      (output probability)

    The backward pass computes gradients using the chain rule.
    """

    def __init__(
        self,
        n_input: int = 2,
        n_hidden: int = 4,
        n_output: int = 1,
        learning_rate: float = 0.5,
        random_seed: int | None = None,
    ):
        """
        Initialize network with random weights.

        Args:
            n_input: Number of input features (2 for XOR)
            n_hidden: Number of hidden neurons
            n_output: Number of output neurons (1 for binary classification)
            learning_rate: Step size for gradient descent
            random_seed: Optional seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Xavier/He initialization for better convergence
        # Scale weights by sqrt(2/fan_in) for ReLU
        self.W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2.0 / n_input)
        self.b1 = np.zeros((1, n_hidden))

        self.W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((1, n_output))

        self.learning_rate = learning_rate

        # Store intermediate values for backpropagation
        self.z1 = None  # Pre-activation hidden
        self.a1 = None  # Post-activation hidden
        self.z2 = None  # Pre-activation output
        self.a2 = None  # Post-activation output (prediction)

        # History for visualization
        self.history = {
            "loss": [],
            "accuracy": [],
        }

    def relu(self, z: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, z)"""
        return np.maximum(0, z)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation: 1 / (1 + exp(-z))"""
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute predictions from input.

        Propagates input through hidden layer (ReLU) and output layer (Sigmoid).

        Args:
            X: Input data, shape (n_samples, n_input)

        Returns:
            Predictions (probabilities), shape (n_samples, 1)

        Math:
            z1 = X @ W1 + b1       # Linear transform to hidden
            a1 = relu(z1)          # Hidden activation
            z2 = a1 @ W2 + b2      # Linear transform to output
            a2 = sigmoid(z2)       # Output probability
        """
        # TODO(human): Implement the forward pass
        #
        # Your task: Propagate input through both layers (~8-10 lines)
        #
        # Step 1: Hidden layer
        #   self.z1 = X @ self.W1 + self.b1    # Linear transform
        #   self.a1 = self.relu(self.z1)       # Apply ReLU activation
        #
        # Step 2: Output layer
        #   self.z2 = self.a1 @ self.W2 + self.b2    # Linear transform
        #   self.a2 = self.sigmoid(self.z2)          # Apply Sigmoid activation
        #
        # Step 3: Return the prediction
        #   return self.a2
        #
        # IMPORTANT: Store z1, a1, z2, a2 as self.* because backward() needs them!
        # The intermediate values are used to compute gradients during backprop.
        #

        # Input Layer -> Hidden Layer
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        # Hidden Layer -> Output Layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2

    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Backward pass: compute gradients and update weights.

        Uses the chain rule to compute gradients layer by layer, from output
        back to input. Then updates all weights using gradient descent.

        Args:
            X: Input data, shape (n_samples, n_input)
            y: True labels, shape (n_samples, 1)

        Math (chain rule from output to input):
            # Output layer gradients
            dz2 = a2 - y                    # Derivative of BCE + sigmoid
            dW2 = a1.T @ dz2 / n_samples    # Gradient for W2
            db2 = sum(dz2) / n_samples      # Gradient for b2

            # Hidden layer gradients (backprop through W2, then ReLU)
            da1 = dz2 @ W2.T                # Backprop through W2
            dz1 = da1 * (z1 > 0)            # Backprop through ReLU (derivative is 0 or 1)
            dW1 = X.T @ dz1 / n_samples     # Gradient for W1
            db1 = sum(dz1) / n_samples      # Gradient for b1

            # Update weights (gradient descent)
            W2 = W2 - learning_rate * dW2
            b2 = b2 - learning_rate * db2
            W1 = W1 - learning_rate * dW1
            b1 = b1 - learning_rate * db1
        """
        n_samples = X.shape[0]

        # TODO(human): Implement backpropagation
        #
        # Your task: Compute gradients using chain rule and update weights (~10-12 lines)
        #
        # Part 1: Output layer gradients
        #   dz2 = self.a2 - y                           # Error at output
        #   dW2 = self.a1.T @ dz2 / n_samples           # How W2 affects loss
        #   db2 = np.sum(dz2, axis=0, keepdims=True) / n_samples
        #
        # Part 2: Hidden layer gradients (chain rule!)
        #   da1 = dz2 @ self.W2.T                       # Backprop through W2
        #   dz1 = da1 * (self.z1 > 0)                   # Backprop through ReLU
        #   dW1 = X.T @ dz1 / n_samples                 # How W1 affects loss
        #   db1 = np.sum(dz1, axis=0, keepdims=True) / n_samples
        #
        # Part 3: Update weights (gradient descent)
        #   self.W2 = self.W2 - self.learning_rate * dW2
        #   self.b2 = self.b2 - self.learning_rate * db2
        #   self.W1 = self.W1 - self.learning_rate * dW1
        #   self.b1 = self.b1 - self.learning_rate * db1
        #

        dz2 = self.a2 - y
        dW2 = self.a1.T @ dz2 / n_samples
        db2 = np.sum(dz2, axis=0, keepdims=True) / n_samples

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = X.T @ dz1 / n_samples
        db1 = np.sum(dz1, axis=0, keepdims=True) / n_samples

        self.W2 = self.W2 - self.learning_rate * dW2
        self.b2 = self.b2 - self.learning_rate * db2
        self.W1 = self.W1 - self.learning_rate * dW1
        self.b1 = self.b1 - self.learning_rate * db1

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.

        BCE = -mean(y * log(pred) + (1-y) * log(1-pred))

        Args:
            y_true: True labels, shape (n_samples, 1)
            y_pred: Predicted probabilities, shape (n_samples, 1)

        Returns:
            Scalar loss value
        """
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        verbose: bool = True,
    ) -> None:
        """
        Train the network using gradient descent.

        Args:
            X: Training data, shape (n_samples, n_input)
            y: Labels, shape (n_samples, 1)
            epochs: Number of training iterations
            verbose: Print progress every 100 epochs
        """
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)

            # Check if forward is implemented
            if predictions is None:
                print("!" * 60)
                print("forward() not implemented yet!")
                print("Complete the TODO(human) section in forward() first.")
                print("!" * 60)
                return

            # Compute loss
            loss = self.compute_loss(y, predictions)
            accuracy = np.mean((predictions > 0.5) == y)

            # Store history
            self.history["loss"].append(loss)
            self.history["accuracy"].append(accuracy)

            # Backward pass (compute gradients and update weights)
            self.backward(X, y)

            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d}: loss = {loss:.4f}, accuracy = {accuracy:.2%}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions (returns probabilities)."""
        return self.forward(X)

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Make class predictions (returns 0 or 1)."""
        return (self.forward(X) > 0.5).astype(int)


def generate_xor_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the XOR dataset.

    XOR (exclusive or) is the classic problem that proves neural networks
    need hidden layers. A linear classifier cannot solve XOR.

    Returns:
        X: Input features, shape (4, 2)
        y: Labels, shape (4, 1)
    """
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.float64,
    )

    y = np.array(
        [
            [0],  # 0 XOR 0 = 0
            [1],  # 0 XOR 1 = 1
            [1],  # 1 XOR 0 = 1
            [0],  # 1 XOR 1 = 0
        ],
        dtype=np.float64,
    )

    return X, y


def animate_decision_boundary(
    nn: NeuralNetwork,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 2000,
    animate_every: int = 50,
    delay: float = 0.05,
) -> None:
    """
    Animate the decision boundary as the network learns.

    Shows how the network carves up the 2D input space to classify XOR.

    Args:
        nn: Neural network to train
        X: Training data
        y: Labels
        epochs: Total training epochs
        animate_every: Update visualization every N epochs
        delay: Seconds between frames
    """
    print("=" * 60)
    print("NEURAL NETWORK: LEARNING XOR")
    print("=" * 60)
    print("Watch the decision boundary evolve as the network learns!")
    print("The network must learn a NON-LINEAR boundary to solve XOR.")
    print()

    # Set up visualization
    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Neural Network Learning XOR", fontsize=14)

    ax_boundary, ax_loss, ax_accuracy = axes

    # Create mesh grid for decision boundary
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Decision boundary plot
    ax_boundary.set_xlim(-0.5, 1.5)
    ax_boundary.set_ylim(-0.5, 1.5)
    ax_boundary.set_xlabel("Input 1")
    ax_boundary.set_ylabel("Input 2")
    ax_boundary.set_title("Decision Boundary")

    # Plot XOR points
    colors = ["red" if label == 0 else "blue" for label in y.flatten()]
    ax_boundary.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors="black", zorder=5)

    # Add labels to points
    for i, (xi, yi) in enumerate(X):
        label = int(y[i, 0])
        ax_boundary.annotate(
            f"XOR={label}",
            (xi, yi),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=10,
        )

    # Initialize contour (will be updated)
    contour = None

    # Loss plot
    (line_loss,) = ax_loss.plot([], [], "b-", linewidth=2)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Binary Cross-Entropy Loss")
    ax_loss.set_xlim(0, epochs)
    ax_loss.set_ylim(0, 1)
    ax_loss.grid(True, alpha=0.3)

    # Accuracy plot
    (line_acc,) = ax_accuracy.plot([], [], "g-", linewidth=2)
    ax_accuracy.set_xlabel("Epoch")
    ax_accuracy.set_ylabel("Accuracy")
    ax_accuracy.set_title("Classification Accuracy")
    ax_accuracy.set_xlim(0, epochs)
    ax_accuracy.set_ylim(0, 1.1)
    ax_accuracy.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="100%")
    ax_accuracy.legend()
    ax_accuracy.grid(True, alpha=0.3)

    plt.tight_layout()

    # Training loop with animation
    for epoch in range(epochs):
        # Forward pass
        predictions = nn.forward(X)

        if predictions is None:
            print("!" * 60)
            print("forward() not implemented yet!")
            print("!" * 60)
            plt.ioff()
            plt.show()
            return

        # Compute loss and accuracy
        loss = nn.compute_loss(y, predictions)
        accuracy = np.mean((predictions > 0.5) == y)
        nn.history["loss"].append(loss)
        nn.history["accuracy"].append(accuracy)

        # Backward pass
        nn.backward(X, y)

        # Update visualization
        if epoch % animate_every == 0:
            # Update decision boundary
            if contour is not None:
                # Remove previous contour (matplotlib 3.8+ compatible)
                try:
                    for c in contour.collections:
                        c.remove()
                except AttributeError:
                    contour.remove()  # matplotlib 3.8+

            Z = nn.forward(grid)
            if Z is not None:
                Z = Z.reshape(xx.shape)
                contour = ax_boundary.contourf(
                    xx, yy, Z, levels=20, cmap="RdBu", alpha=0.6
                )

            # Update loss curve
            line_loss.set_data(range(len(nn.history["loss"])), nn.history["loss"])
            ax_loss.set_ylim(0, max(0.8, max(nn.history["loss"]) * 1.1))

            # Update accuracy curve
            line_acc.set_data(
                range(len(nn.history["accuracy"])), nn.history["accuracy"]
            )

            ax_boundary.set_title(f"Decision Boundary (Epoch {epoch})")

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(delay)

            if epoch % 200 == 0:
                print(f"Epoch {epoch:4d}: loss = {loss:.4f}, accuracy = {accuracy:.2%}")

    # Final update
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final loss: {nn.history['loss'][-1]:.4f}")
    print(f"Final accuracy: {nn.history['accuracy'][-1]:.2%}")
    print()

    # Test predictions
    print("Predictions:")
    for i in range(len(X)):
        xi = X[i]  # Shape (2,) - the input point
        pred = nn.predict(xi.reshape(1, -1))[0, 0]
        true = int(y[i, 0])
        print(
            f"  {int(xi[0])} XOR {int(xi[1])} = {true}  |  Network predicts: {pred:.3f} → {int(pred > 0.5)}"
        )

    print("=" * 60)

    plt.ioff()
    plt.savefig("neural_network_xor.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved to neural_network_xor.png")
    plt.show()


def run_neural_network_demo() -> None:
    """Run the neural network demo on XOR problem."""
    # Generate XOR data
    X, y = generate_xor_data()

    print("XOR Dataset:")
    print("  Input    | Output")
    print("  ---------|-------")
    for i in range(len(X)):
        print(f"  {int(X[i, 0])}, {int(X[i, 1])}    |   {int(y[i, 0])}")
    print()

    # Create and train network
    nn = NeuralNetwork(
        n_input=2,
        n_hidden=4,
        n_output=1,
        learning_rate=0.5,
        random_seed=42,
    )

    # Animate training
    animate_decision_boundary(nn, X, y, epochs=2000, animate_every=50)


if __name__ == "__main__":
    run_neural_network_demo()
