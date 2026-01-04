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
- ✅ Phase 1: Kalman Filters, Q-Learning, Linear Regression
- ✅ Phase 2: Image Convolutions, Neural Network MLP, Feature Detection
- ✅ Phase 3: CNN Classifier, Object Detection Concepts
- ✅ Phase 4 (in progress): DQN

## What's Next (Phase 4: Advanced Robotics ML)
- Policy Gradients / Actor-Critic
- SLAM Concepts

## Project Structure
```
ml_fundamentals/
├── phase1_fundamentals/    # Kalman, Q-Learning, Linear Regression
├── phase2_vision/          # Convolution, Neural Network, Feature Detection
├── phase3_deep_learning/   # CNN, Object Detection
├── phase4_robotics_ml/     # DQN, Policy Gradients
└── output/                 # Generated visualizations (gitignored)
```

## Environment
- Python managed with `uv`
- Run code with: `uv run python phase1_fundamentals/kalman_filter.py`
- Dependencies: numpy, matplotlib, torch, gymnasium (see pyproject.toml)
