# ML Fundamentals: A Robotics-Focused Learning Journey

Personal learning project for understanding core machine learning algorithms, with emphasis on robotics and computer vision applications.

## Progress Overview

| Algorithm | Status | File | Key Concept |
|-----------|--------|------|-------------|
| 1D Kalman Filter | ✅ Complete | `kalman_filter.py` | State estimation, predict-update cycle |
| 2D Kalman Filter | ✅ Complete | `kalman_filter_2d.py` | Matrix formulation, estimating hidden state (velocity) |
| Q-Learning | ✅ Complete | `q_learning.py` | Reinforcement learning, Bellman equation, ε-greedy |
| Linear Regression | ⬜ Planned | — | Gradient descent, optimization fundamentals |
| Image Convolutions | ⬜ Planned | — | Filters, edge detection, CNN foundations |
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
