# ML Fundamentals - Learning Project

## Project Context
This is a personal learning project focused on machine learning fundamentals for **robotics and computer vision**. The learner has a CS degree and is pursuing an ML Master's.

## Learning Approach
- **Build from scratch first** — Implement algorithms with NumPy before using libraries
- **Balanced progression** — Start manual, then graduate to PyTorch/OpenCV
- **Robotics-focused** — Always connect concepts back to robotics applications
- **Visual learning** — Animate algorithms when possible (use matplotlib with TkAgg backend)

## Teaching Style Preferences
- Explain intuition before diving into math
- Use concrete numerical examples to illustrate matrix operations
- Create "Learn by Doing" blocks where the learner implements key pieces (2-10 lines)
- After learner implements something, provide insights connecting it to broader concepts
- Don't over-explain basic programming — learner is advanced

## Code Style
- Use type hints
- Include docstrings explaining the math/algorithm
- Use `matplotlib.use("TkAgg")` for interactive plots on Linux
- Animate visualizations with `plt.ion()` and `plt.pause()` when helpful

## Progress Summary
See `README.md` for detailed progress. Completed so far:
- ✅ 1D Kalman Filter (learner implemented: update step)
- ✅ 2D Kalman Filter (learner implemented: predict step)
- ✅ Q-Learning (learner implemented: ε-greedy action selection)
- ✅ Linear Regression (learner implemented: full gradient descent training loop)
- ✅ Image Convolutions (learner implemented: convolve2d + edge detection)
- ✅ Neural Network MLP (learner implemented: forward pass + backpropagation)

## What's Next (Phase 3: Deep Learning for Vision)
- CNN for Image Classification (PyTorch)
- Object Detection concepts (YOLO/SSD)

## Environment
- Python managed with `uv`
- Run code with: `uv run python <script>.py`
- Dependencies: numpy, matplotlib (see pyproject.toml)
