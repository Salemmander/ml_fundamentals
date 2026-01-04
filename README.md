# ML Fundamentals: A Robotics-Focused Learning Journey

Personal learning project for understanding core machine learning algorithms, with emphasis on robotics and computer vision applications.

## Progress Overview

| Algorithm | Status | File | Key Concept |
|-----------|--------|------|-------------|
| 1D Kalman Filter | ✅ Complete | `phase1_fundamentals/kalman_filter.py` | State estimation, predict-update cycle |
| 2D Kalman Filter | ✅ Complete | `phase1_fundamentals/kalman_filter_2d.py` | Matrix formulation, estimating hidden state (velocity) |
| Q-Learning | ✅ Complete | `phase1_fundamentals/q_learning.py` | Reinforcement learning, Bellman equation, ε-greedy |
| Linear Regression | ✅ Complete | `phase1_fundamentals/linear_regression.py` | Gradient descent, optimization fundamentals |
| Image Convolutions | ✅ Complete | `phase2_vision/image_convolution.py` | Filters, edge detection, CNN foundations |
| Neural Network (MLP) | ✅ Complete | `phase2_vision/neural_network.py` | Backpropagation from scratch |
| CNN Image Classifier | ✅ Complete | `phase3_deep_learning/cnn_classifier.py` | PyTorch, LeNet on MNIST |
| Feature Detection | ✅ Complete | `phase2_vision/feature_detection.py` | Harris corners, SIFT/ORB intro |
| Object Detection Concepts | ✅ Complete | `phase3_deep_learning/object_detection_concepts.py` | Anchors, IoU, NMS |
| DQN (Deep Q-Network) | ✅ Complete | `phase4_robotics_ml/dqn_cartpole.py` | Function approximation, replay buffer, target network |

---

## What I Learned

### 1. Kalman Filter (State Estimation)

**Question it answers:** "Where am I, given noisy sensors?"

**Key equations (1D):**

```
Predict:  x = x + v*dt       P = P + Q
Update:   K = P/(P+R)        x = x + K*(z-x)        P = (1-K)*P
```

**Key equations (2D matrix form):**

```
Predict:  x = F @ x          P = F @ P @ F.T + Q
Update:   K = P @ H.T @ inv(S)   x = x + K @ y    P = (I - K@H) @ P
```

**Key insights:**

- Kalman Gain (K) balances trust between prediction and measurement
- Uncertainty grows during prediction, shrinks during update
- 2D filter can estimate velocity from position-only measurements!
- Foundation for SLAM, sensor fusion, robot localization

**What I implemented:** The `update()` step in 1D, the `predict()` step in 2D

### 2. Q-Learning (Reinforcement Learning)

**Question it answers:** "What action should I take?"

**Key equation (Bellman update):**

```
Q(s,a) = Q(s,a) + α * (reward + γ * max(Q(s',:)) - Q(s,a))
```

**Key insights:**

- Q-table maps (state, action) → expected future reward
- ε-greedy balances exploration vs exploitation
- Values propagate backward from goal through repeated experience
- No explicit path planning — optimal behavior emerges from local updates
- Foundation for DQN, policy gradients, modern Deep RL

**What I implemented:** The `choose_action()` method (ε-greedy action selection)

### 3. Linear Regression (Optimization Fundamentals)

**Question it answers:** "How do I find the best parameters to fit my data?"

**Key equations:**

```
Model:    ŷ = X @ W + b
Loss:     L = (1/n) * Σ(ŷ - y)²
Gradient: dW = (2/n) * X.T @ (ŷ - y)    db = (2/n) * Σ(ŷ - y)
Update:   W = W - α * dW                 b = b - α * db
```

**Key insights:**

- MSE loss measures "how wrong" — gradient tells you "which direction is worse"
- `X.T @ error` computes how much each feature contributed to the error
- Learning rate (α) controls step size: too big = overshoot, too small = slow
- This exact loop (forward → loss → gradient → update) is how ALL neural networks train
- Foundation for backpropagation, PyTorch optimizers, deep learning

**What I implemented:** The full `fit()` method — gradient descent training loop

### 4. Image Convolutions (Computer Vision Fundamentals)

**Question it answers:** "How do I detect features in images?"

**Key operation:**

```
output[i,j] = Σ Σ image[i+m, j+n] * kernel[m, n]
              m n
```

**Key insights:**

- Convolution slides a kernel across an image, computing weighted sums
- Sobel X detects vertical edges, Sobel Y detects horizontal edges
- Gradient magnitude `√(Gx² + Gy²)` captures edges at any orientation
- Different kernels = different features (edges, blur, sharpen)
- CNNs learn kernels through backprop instead of hand-designing them
- Foundation for all deep learning in computer vision

**What I implemented:** `convolve2d()` (sliding window operation) + `detect_edges()` (Sobel gradient magnitude)

### 5. Neural Network / MLP (Deep Learning Fundamentals)

**Question it answers:** "How do neural networks learn?"

**Architecture:**

```
Input (2) → Hidden (4, ReLU) → Output (1, Sigmoid)
```

**Key equations:**

```
Forward:  z1 = X @ W1 + b1,  a1 = relu(z1)
          z2 = a1 @ W2 + b2, a2 = sigmoid(z2)

Backward: dz2 = a2 - y
          dW2 = a1.T @ dz2,  da1 = dz2 @ W2.T
          dz1 = da1 * (z1 > 0),  dW1 = X.T @ dz1
          W = W - lr * dW  (for all weights)
```

**Key insights:**

- Forward pass: data flows input → output through matrix multiplies + activations
- Backward pass (backprop): gradients flow output → input via chain rule
- ReLU introduces non-linearity (without it, stacked layers = just one linear layer)
- Sigmoid squashes output to probability (0,1) for classification
- XOR proves the hidden layer learns useful representations (linear classifiers fail)
- This is exactly how PyTorch/TensorFlow work under the hood

**What I implemented:** `forward()` (full forward pass) + `backward()` (backpropagation with chain rule)

### 6. CNN Image Classifier (PyTorch)

**Question it answers:** "How do I classify images with deep learning?"

**Architecture (LeNet-style):**

```
Input (1, 28, 28)
→ Conv2d(1→6, 5x5) → ReLU → MaxPool(2x2)
→ Conv2d(6→16, 5x5) → ReLU → MaxPool(2x2)
→ Flatten → Linear(256→120) → ReLU
→ Linear(120→84) → ReLU → Linear(84→10)
```

**PyTorch training loop:**

```python
optimizer.zero_grad()      # Clear old gradients
outputs = model(inputs)    # Forward pass
loss = criterion(outputs, labels)
loss.backward()            # Backward pass (autograd!)
optimizer.step()           # Update weights
```

**Key insights:**

- `nn.Sequential` chains layers — activations (ReLU) must be included as layers too
- `loss.backward()` replaces your manual backprop — autograd handles the chain rule
- `optimizer.step()` replaces `W -= lr * dW` — with momentum, Adam, etc. built in
- `DataLoader` handles batching and shuffling automatically
- `torch.no_grad()` disables gradient tracking for evaluation (saves memory)
- Always preserve batch dimension when reshaping: `x.flatten(start_dim=1)` not `x.flatten()`

**What I implemented:** Full pipeline — `LeNet` class (with `nn.Sequential`), `get_data_loaders()`, `train_one_epoch()`, `evaluate()`

### 7. Feature Detection (Harris Corners)

**Question it answers:** "Where are the trackable points in this image?"

**Structure Tensor M:**

```
M = [ Σ(Ix²)   Σ(Ix·Iy) ]    (Ix, Iy from Sobel gradients)
    [ Σ(Ix·Iy) Σ(Iy²)   ]
```

**Corner Response:**

```
R = det(M) - k * trace(M)²
R = λ1·λ2 - k*(λ1 + λ2)²
```

- R > 0 (large): Corner — both eigenvalues large
- R < 0 (large): Edge — one eigenvalue dominates
- R ≈ 0: Flat region — both eigenvalues small

**Key insights:**

- Corners are uniquely identifiable in ALL directions (edges are ambiguous along their length)
- Harris reuses Sobel gradients — builds directly on convolution work
- "Valid" convolution shrinks output — must track coordinate offsets when overlaying results
- SIFT/ORB add scale invariance and descriptors for matching between images
- Feature detection → matching → motion estimation → SLAM (Phase 4)

**What I implemented:** `harris_response()` — structure tensor computation and corner response formula

### 8. Object Detection Concepts

**Question it answers:** "How do detectors find and localize objects?"

**Core Components:**

```
Anchor Boxes: Pre-defined boxes at multiple scales/aspect ratios
IoU: Intersection over Union = Area(A ∩ B) / Area(A ∪ B)
NMS: Non-Maximum Suppression — keep best, remove overlapping duplicates
```

**NMS Algorithm:**

```
1. Sort boxes by confidence (highest first)
2. Keep highest-scoring box
3. Remove all boxes with IoU > threshold
4. Repeat until no boxes remain
```

**Key insights:**

- Anchors avoid sliding window — predict offsets from fixed positions instead
- IoU uses min/max for intersection: `max` for start coords, `min` for end coords
- Batch IoU uses broadcasting: reshape `(N,4)` and `(M,4)` to `(N,1,4)` and `(1,M,4)`
- Integer indexing drops dimensions, slice indexing keeps them: `arr[i]` vs `arr[i:i+1]`
- NMS threshold is a precision/recall tradeoff (0.3 aggressive, 0.5 balanced)

**What I implemented:** `generate_anchors()`, `compute_iou()`, `compute_iou_batch()`, `non_maximum_suppression()`

### 9. DQN — Deep Q-Network (Deep Reinforcement Learning)

**Question it answers:** "How do I learn to act in continuous state spaces?"

**Key difference from Q-Learning:**

```
Q-Learning: Q-table[state, action] → value lookup
DQN:        Neural network(state) → Q-values for all actions
```

**Core components:**

```
Replay Buffer: Store (s, a, r, s', done), sample random batches
Target Network: Slowly-updated copy for stable TD targets
TD Loss: MSE(Q(s,a), r + γ * max Q_target(s'))
```

**Key insights:**

- Neural networks enable RL on continuous/high-dimensional states
- Experience replay breaks correlation between consecutive samples
- Target network prevents "moving target" instability
- `gather()` selects Q(s, action_taken), `max()` selects best future action
- Solved CartPole in ~350 episodes on CPU

**What I implemented:** `ReplayBuffer` class, `select_action()`, `compute_td_loss()`, `soft_update_target()`

---

## Running the Code

```bash
# Set up environment (using uv)
cd ml_fundamentals
uv sync  # Install dependencies from pyproject.toml

# Phase 1: Fundamentals
uv run python phase1_fundamentals/kalman_filter.py      # 1D Kalman Filter
uv run python phase1_fundamentals/kalman_filter_2d.py   # 2D velocity estimation
uv run python phase1_fundamentals/q_learning.py         # Q-Learning GridWorld
uv run python phase1_fundamentals/linear_regression.py  # Gradient descent

# Phase 2: Computer Vision
uv run python phase2_vision/image_convolution.py   # Convolution + edge detection
uv run python phase2_vision/neural_network.py      # MLP from scratch (XOR)
uv run python phase2_vision/feature_detection.py   # Harris + SIFT/ORB

# Phase 3: Deep Learning
uv run python phase3_deep_learning/cnn_classifier.py            # LeNet on MNIST
uv run python phase3_deep_learning/object_detection_concepts.py # Anchors, IoU, NMS

# Phase 4: Robotics ML
uv run python phase4_robotics_ml/dqn_cartpole.py  # DQN on CartPole
```

---

## Learning Plan: What's Next

### Phase 2: Computer Vision Fundamentals ✅

1. ~~**Image Convolutions & Filters**~~ — ✅ Complete
2. ~~**Feature Detection**~~ — ✅ Complete (Harris corners, SIFT/ORB)
3. ~~**Neural Network from Scratch**~~ — ✅ Complete

### Phase 3: Deep Learning for Vision ✅

1. ~~**CNN for Image Classification**~~ — ✅ Complete (LeNet on MNIST)
2. ~~**Object Detection**~~ — ✅ Complete (Anchors, IoU, NMS)

### Phase 4: Advanced Robotics ML (in progress)

1. ~~**Deep Q-Network (DQN)**~~ — ✅ Complete (CartPole, experience replay, target network)
2. **Policy Gradients / Actor-Critic** — REINFORCE, A2C, continuous actions
3. **SLAM Concepts** — Visual odometry, feature matching

---

## Key Connections to Robotics

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROBOTICS PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Sensors → [Kalman Filter] → State Estimate → [RL/Planning] → Actions
│             "Where am I?"                     "What do I do?" │
│                                                                 │
│  Camera → [CNN/Vision] → Object Detection → [RL] → Manipulation │
│           "What do I see?"                  "How do I act?"    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dependencies

- `numpy` — Matrix operations, numerical computing
- `matplotlib` — Visualization and animation
- `opencv-python` — SIFT/ORB feature detection and matching
- `torch` — Deep learning framework (Phase 3+)
- `torchvision` — Datasets and transforms for computer vision
- `gymnasium` — RL environments (CartPole, etc.)
- `pygame` — Environment rendering

Managed with `uv`. See `pyproject.toml` for details.

---

## Resources & References

- [Kalman Filter visual explanation](https://www.kalmanfilter.net/)
- [Sutton & Barto RL Book](http://incompleteideas.net/book/the-book.html) (free online)
- [PyTorch tutorials](https://pytorch.org/tutorials/) (for Phase 3)

---

*Started: January 2026*
*Focus: Robotics & Computer Vision*
