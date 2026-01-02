# ML Fundamentals: A Robotics-Focused Learning Journey

Personal learning project for understanding core machine learning algorithms, with emphasis on robotics and computer vision applications.

## Progress Overview

| Algorithm | Status | File | Key Concept |
|-----------|--------|------|-------------|
| 1D Kalman Filter | ✅ Complete | `kalman_filter.py` | State estimation, predict-update cycle |
| 2D Kalman Filter | ✅ Complete | `kalman_filter_2d.py` | Matrix formulation, estimating hidden state (velocity) |
| Q-Learning | ✅ Complete | `q_learning.py` | Reinforcement learning, Bellman equation, ε-greedy |
| Linear Regression | ✅ Complete | `linear_regression.py` | Gradient descent, optimization fundamentals |
| Image Convolutions | ✅ Complete | `image_convolution.py` | Filters, edge detection, CNN foundations |
| Neural Network (MLP) | ⬜ Planned | — | Backpropagation from scratch |

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

---

## Running the Code

```bash
# Set up environment (using uv)
cd ml_fundamentals
uv sync  # Install dependencies from pyproject.toml

# Run Kalman Filter demos
uv run python kalman_filter.py      # 1D: animated position tracking
uv run python kalman_filter_2d.py   # 2D: velocity estimation

# Run Q-Learning demo
uv run python q_learning.py         # Watch robot learn to navigate!

# Run Linear Regression demo
uv run python linear_regression.py  # Watch gradient descent fit a line!

# Run Image Convolution demo
uv run python image_convolution.py  # Animated kernel + filter comparison!
```

---

## Learning Plan: What's Next

### Phase 2: Computer Vision Fundamentals
1. **Image Convolutions & Filters** — Edge detection, blurring from scratch, then OpenCV
2. **Feature Detection** — Harris corners, intro to SIFT/ORB
3. **Neural Network from Scratch** — MLP with backpropagation

### Phase 3: Deep Learning for Vision
4. **CNN for Image Classification** — PyTorch, robotics-relevant dataset
5. **Object Detection** — YOLO/SSD concepts

### Phase 4: Advanced Robotics ML
6. **Deep Reinforcement Learning** — DQN, Policy Gradients, PyBullet/MuJoCo
7. **SLAM Concepts** — Visual odometry, feature matching

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

Managed with `uv`. See `pyproject.toml` for details.

---

## Resources & References

- [Kalman Filter visual explanation](https://www.kalmanfilter.net/)
- [Sutton & Barto RL Book](http://incompleteideas.net/book/the-book.html) (free online)
- [PyTorch tutorials](https://pytorch.org/tutorials/) (for Phase 3)

---

*Started: January 2026*
*Focus: Robotics & Computer Vision*
