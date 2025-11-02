# 🤚 Gesture Recognition System

A clean, modular, and extensible real-time hand gesture recognition project. This repository provides tools for capturing hand landmarks, building datasets, training lightweight models, and running real-time inference using a webcam. It is designed for reproducibility, easy experimentation, and productionization.

Badges

- Build: ![build](https://img.shields.io/badge/build-passing-brightgreen)
- Python: ![python](https://img.shields.io/badge/python-3.9%2B-blue)
- License: ![license](https://img.shields.io/badge/license-MIT-lightgrey)

Table of contents

- About
- Features
- Requirements
- Quick start
- Typical workflows
  - Capture / Data collection
  - Training
  - Evaluation & Inference
- Development & Tooling
- Hardware & performance notes
- Troubleshooting
- Contributing
- License
- Resources

About
A compact, well-structured pipeline for hand gesture recognition using modern computer-vision and ML tools (OpenCV, MediaPipe, and your chosen ML framework). The project focuses on:

- Real-time detection and classification
- Clean dataset format for reproducible experiments
- Developer-friendly scripts and tooling (formatting, linting, type checks, tests)

Features

- Real-time hand detection and landmark extraction (MediaPipe)
- Scripts for setup verification and camera testing
- Dataset capture utilities and standardized storage
- Training and evaluation utilities (plug-in ML backend: PyTorch/TensorFlow/sklearn)
- CI-friendly commands and developer tooling

Requirements

- macOS / Linux / Windows
- Python 3.9+
- A webcam (for local testing)
- Optional GPU for faster training (CUDA-compatible GPU if using PyTorch/TensorFlow with GPU)
- Dependencies managed with Poetry

Quick start (development)

1. Install dependencies and enter the virtual environment:

```bash
poetry install
poetry shell
```

2. Verify environment and hardware:

```bash
# Sanity checks (Python, OpenCV, MediaPipe)
poetry run python scripts/setup_check.py

# Test camera preview and OpenCV
poetry run python scripts/hello_opencv.py
```

3. Run tests:

```bash
poetry run pytest
```

Typical workflows

Capture / Data collection

- Use the data collection utilities (e.g., scripts/collect.py or src/collect) to record examples.
- Save a standard dataset format: - Per-sample metadata (label, timestamp, source camera) - Landmark arrays (NumPy .npy) or CSV records - Optional annotated video/image copies for auditing
  Example command (adjust script path to your project layout):

```bash
poetry run python scripts/collect.py --label open_palm --out data/open_palm --frames 500
```

Training

- Provide a training script (src/train.py) that accepts:
  - dataset path
  - model architecture / backbone
  - hyperparameters (batch size, LR, epochs)
- Recommended to support checkpoints and TensorBoard / Weights & Biases logging.
  Example:

```bash
poetry run python src/train.py --data data/ --model lightweight_mlp --epochs 60 --batch-size 64
```

Suggestion: start with a small model (MLP on normalized landmarks) for fast iteration, then experiment with temporal models (LSTM/1D-CNN/Transformer) if working with gesture sequences.

Evaluation & Inference

- Provide evaluation script (src/eval.py) for common metrics: accuracy, precision, recall, confusion matrix.
- Provide a real-time demo script for webcam inference:

```bash
poetry run python src/infer.py --camera 0 --model checkpoints/latest.pt
```

Development & Tooling
Formatting, linting, and typing:

```bash
# Format code
poetry run black src/ tests/ scripts/

# Lint / static checks
poetry run ruff check src/

# Type checking
poetry run mypy src/
```

Testing

```bash
poetry run pytest
```

Hardware & performance notes

- CPU-only: use small models and lower input resolution for real-time performance.
- GPU: enable batch training, mixed precision (AMP) for speed and memory gains.
- Camera: 720p is usually sufficient; reduce resolution if inference lags.

Troubleshooting

- Camera not detected: confirm permissions and that no other app is using the camera.
- MediaPipe errors: ensure compatible versions; check virtual environment and reinstall dependencies.
- Poor accuracy: collect more diverse data, balance classes, augment, and review preprocessing.

Contributing

- Follow the repository's branching and PR guidelines.
- Keep changes small and well documented.
- Run tests and linters before opening PRs.

License
This project is licensed under the MIT License. See LICENSE file for details.

Resources & references

- OpenCV: https://opencv.org
- MediaPipe Hands: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
- PyTorch: https://pytorch.org
- TensorFlow: https://www.tensorflow.org
- Weights & Biases (experiment tracking): https://wandb.ai

Acknowledgements

- Inspired by open-source CV and ML tooling and community examples for real-time hand-gesture pipelines.

