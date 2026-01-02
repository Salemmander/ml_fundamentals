"""
CNN Image Classifier with PyTorch — From Scratch to Framework

This module implements a Convolutional Neural Network using PyTorch for
MNIST digit classification. Now that you've built convolution and backprop
from scratch, you'll see how PyTorch abstracts these operations.

Architecture (LeNet-style):
    Input (1, 28, 28)
    → Conv2d(1, 6, 5) → ReLU → MaxPool(2)      # (6, 12, 12)
    → Conv2d(6, 16, 5) → ReLU → MaxPool(2)     # (16, 4, 4)
    → Flatten                                   # (256,)
    → Linear(256, 120) → ReLU
    → Linear(120, 84) → ReLU
    → Linear(84, 10)                           # 10 digit classes

Connection to what you built:
    - nn.Conv2d = your convolve2d() (but with learnable kernels + batches)
    - loss.backward() = your backward() (but automatic via autograd)
    - optimizer.step() = your weight update (W -= lr * dW)

Why MNIST?
    - Classic benchmark: 28x28 grayscale handwritten digits
    - Small enough to train on CPU in minutes
    - 60K training images, 10K test images
"""

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple


# =============================================================================
# CNN MODEL
# =============================================================================


class LeNet(nn.Module):
    """
    LeNet-style CNN for MNIST digit classification.

    Architecture:
        Conv2d(1→6, 5x5) → ReLU → MaxPool(2x2)
        Conv2d(6→16, 5x5) → ReLU → MaxPool(2x2)
        Flatten → Linear(256→120) → ReLU
        Linear(120→84) → ReLU → Linear(84→10)

    Input: (batch, 1, 28, 28) grayscale images
    Output: (batch, 10) logits for each digit class
    """

    def __init__(self):
        """
        Define all layers of the network.

        Layers to define:
            self.conv1: nn.Conv2d(1, 6, kernel_size=5)
            self.conv2: nn.Conv2d(6, 16, kernel_size=5)
            self.pool: nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1: nn.Linear(16 * 4 * 4, 120)  # 16 channels, 4x4 spatial
            self.fc2: nn.Linear(120, 84)
            self.fc3: nn.Linear(84, 10)
        """
        super(LeNet, self).__init__()

        # TODO(human): Define the network layers
        #
        # Your task: Create all layers (~6 lines)
        #
        # Convolutional layers:
        #   self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        #   self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        #
        # Pooling layer (reused for both conv layers):
        #   self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # Fully connected layers:
        #   After conv+pool: 28→24→12→8→4, so 16 channels * 4 * 4 = 256
        #   self.fc1 = nn.Linear(16 * 4 * 4, 120)
        #   self.fc2 = nn.Linear(120, 84)
        #   self.fc3 = nn.Linear(84, 10)
        #

        self.hidden_layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.connected_layers = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input images, shape (batch, 1, 28, 28)

        Returns:
            Logits, shape (batch, 10)

        Flow:
            x (batch, 1, 28, 28)
            → conv1 → (batch, 6, 24, 24)
            → relu → pool → (batch, 6, 12, 12)
            → conv2 → (batch, 16, 8, 8)
            → relu → pool → (batch, 16, 4, 4)
            → flatten → (batch, 256)
            → fc1 → relu → (batch, 120)
            → fc2 → relu → (batch, 84)
            → fc3 → (batch, 10)
        """
        # TODO(human): Implement the forward pass
        #
        # Your task: Pass input through all layers (~8-10 lines)
        #
        # Conv block 1:
        #   x = self.pool(torch.relu(self.conv1(x)))
        #
        # Conv block 2:
        #   x = self.pool(torch.relu(self.conv2(x)))
        #
        # Flatten for fully connected layers:
        #   x = x.view(-1, 16 * 4 * 4)  # or x.view(x.size(0), -1)
        #
        # Fully connected layers:
        #   x = torch.relu(self.fc1(x))
        #   x = torch.relu(self.fc2(x))
        #   x = self.fc3(x)  # No activation - raw logits for CrossEntropyLoss
        #
        # return x
        #

        x = self.hidden_layers(x)
        x = x.flatten(start_dim=1)
        x = self.connected_layers(x)
        return x


# =============================================================================
# DATA LOADING
# =============================================================================


def get_data_loaders(
    batch_size: int = 64,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for MNIST training and test sets.

    Args:
        batch_size: Number of images per batch
        data_dir: Directory to download/store MNIST data

    Returns:
        train_loader: DataLoader for training set (60K images)
        test_loader: DataLoader for test set (10K images)

    Transforms to apply:
        1. ToTensor(): Convert PIL image to tensor, scale to [0, 1]
        2. Normalize((0.1307,), (0.3081,)): MNIST mean and std
    """
    # TODO(human): Implement data loading
    #
    # Your task: Create train and test DataLoaders (~8-10 lines)
    #
    # Step 1: Define transforms
    #   transform = transforms.Compose([
    #       transforms.ToTensor(),
    #       transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean, std
    #   ])
    #
    # Step 2: Load datasets
    #   train_dataset = datasets.MNIST(
    #       root=data_dir, train=True, download=True, transform=transform
    #   )
    #   test_dataset = datasets.MNIST(
    #       root=data_dir, train=False, download=True, transform=transform
    #   )
    #
    # Step 3: Create DataLoaders
    #   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #   test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #
    # return train_loader, test_loader
    #

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,),
                (0.3081,),
            ),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# =============================================================================
# TRAINING
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: The neural network
        train_loader: DataLoader for training data
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Optimizer (SGD or Adam)
        device: Device to run on (cpu or cuda)

    Returns:
        avg_loss: Average loss over the epoch
        accuracy: Training accuracy for the epoch
    """
    model.train()  # Set to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # TODO(human): Implement the training loop
    #
    # Your task: Iterate over batches through training (~10-12 lines)
    #
    # for inputs, labels in train_loader:
    #     # Move data to device
    #     inputs, labels = inputs.to(device), labels.to(device)
    #
    #     # Zero gradients (important! otherwise they accumulate)
    #     optimizer.zero_grad()
    #
    #     # Forward pass
    #     outputs = model(inputs)
    #
    #     # Compute loss
    #     loss = criterion(outputs, labels)
    #
    #     # Backward pass (this is your backward()!)
    #     loss.backward()
    #
    #     # Update weights (this is your W -= lr * dW!)
    #     optimizer.step()
    #
    #     # Track metrics
    #     running_loss += loss.item()
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()
    #
    # avg_loss = running_loss / len(train_loader)
    # accuracy = correct / total
    # return avg_loss, accuracy
    #

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on the test set.

    Args:
        model: The neural network
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run on

    Returns:
        avg_loss: Average loss on test set
        accuracy: Test accuracy
    """
    model.eval()  # Set to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    # TODO(human): Implement evaluation
    #
    # Your task: Evaluate without computing gradients (~8-10 lines)
    #
    # with torch.no_grad():  # No need to track gradients for evaluation
    #     for inputs, labels in test_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #
    #         running_loss += loss.item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    # avg_loss = running_loss / len(test_loader)
    # accuracy = correct / total
    # return avg_loss, accuracy

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy


# =============================================================================
# VISUALIZATION
# =============================================================================


def visualize_filters(model: nn.Module) -> None:
    """Visualize the learned convolutional filters from the first layer."""
    # Get first conv layer weights (handle both Sequential and individual layer approaches)
    if hasattr(model, "conv1"):
        conv1 = model.conv1
    elif hasattr(model, "hidden_layers"):
        conv1 = model.hidden_layers[0]  # First layer in Sequential
    else:
        print("Model doesn't have conv1 layer yet!")
        return

    weights = conv1.weight.data.cpu().numpy()
    n_filters = weights.shape[0]

    fig, axes = plt.subplots(1, n_filters, figsize=(12, 2))
    fig.suptitle("Learned Conv1 Filters (what patterns does the network detect?)")

    for i in range(n_filters):
        ax = axes[i] if n_filters > 1 else axes
        # weights[i, 0] is the 5x5 kernel for filter i
        ax.imshow(weights[i, 0], cmap="gray")
        ax.set_title(f"Filter {i + 1}")
        ax.axis("off")

    plt.tight_layout()


def show_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    n_samples: int = 10,
) -> None:
    """Show sample predictions with confidence scores."""
    model.eval()

    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images, labels = images[:n_samples], labels[:n_samples]

    with torch.no_grad():
        outputs = model(images.to(device))
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)

    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Sample Predictions")

    for i, ax in enumerate(axes.flat):
        if i >= n_samples:
            break

        # Show image
        img = images[i].squeeze().numpy()
        ax.imshow(img, cmap="gray")

        pred = predictions[i].item()
        true = labels[i].item()
        conf = probabilities[i, pred].item()

        color = "green" if pred == true else "red"
        ax.set_title(f"Pred: {pred} ({conf:.1%})\nTrue: {true}", color=color)
        ax.axis("off")

    plt.tight_layout()


def plot_training_history(history: dict) -> None:
    """Plot training and test loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train")
    ax1.plot(epochs, history["test_loss"], "r-", label="Test")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Over Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Train")
    ax2.plot(epochs, history["test_acc"], "r-", label="Test")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Over Training")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()


# =============================================================================
# MAIN DEMO
# =============================================================================


def run_cnn_demo(
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 0.01,
) -> None:
    """
    Run the full CNN training demo.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    print("=" * 60)
    print("CNN IMAGE CLASSIFIER — MNIST")
    print("=" * 60)

    # Device setup (CPU for now)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    # Load data
    print("Loading MNIST dataset...")
    loaders = get_data_loaders(batch_size=batch_size)

    if loaders is None:
        print("!" * 60)
        print("get_data_loaders() not implemented yet!")
        print("Complete the TODO(human) section first.")
        print("!" * 60)
        return

    train_loader, test_loader = loaders
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print()

    # Create model
    print("Creating LeNet model...")
    model = LeNet().to(device)

    # Check if model is implemented
    test_input = torch.randn(1, 1, 28, 28).to(device)
    try:
        test_output = model(test_input)
        if test_output is None:
            raise ValueError("forward() returned None")
    except Exception as e:
        print("!" * 60)
        print(f"LeNet not fully implemented: {e}")
        print("Complete the TODO(human) sections in __init__ and forward.")
        print("!" * 60)
        return

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    # Training loop
    print("Training...")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        # Train
        train_result = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        if train_result is None:
            print("!" * 60)
            print("train_one_epoch() not implemented!")
            print("!" * 60)
            return

        train_loss, train_acc = train_result

        # Evaluate
        test_result = evaluate(model, test_loader, criterion, device)

        if test_result is None:
            print("!" * 60)
            print("evaluate() not implemented!")
            print("!" * 60)
            return

        test_loss, test_acc = test_result

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch:2d}/{epochs}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%}"
        )

    print("-" * 60)
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.2%}")
    print()

    # Visualizations
    plt.ion()

    # Plot training curves
    plot_training_history(history)
    plt.savefig("cnn_training_curves.png", dpi=150, bbox_inches="tight")

    # Visualize learned filters
    visualize_filters(model)
    plt.savefig("cnn_learned_filters.png", dpi=150, bbox_inches="tight")

    # Show predictions
    show_predictions(model, test_loader, device)
    plt.savefig("cnn_predictions.png", dpi=150, bbox_inches="tight")

    print("Figures saved:")
    print("  - cnn_training_curves.png")
    print("  - cnn_learned_filters.png")
    print("  - cnn_predictions.png")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_cnn_demo(epochs=10, batch_size=64, learning_rate=0.01)
