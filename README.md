# Implementation of an MLP Neural Network

From-scratch Python implementation of a Multi-Layer Perceptron (MLP) neural network for classification tasks, including forward/backward propagation, activation functions, and gradient descent optimization.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-green)](https://matplotlib.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

## Overview

This project demonstrates a **scratch-built MLP** without external ML libraries (e.g., no TensorFlow/PyTorch). Key components:

- **Architecture**: Fully connected layers with ReLU/Sigmoid activations, softmax output for multi-class.
- **Training**: Stochastic Gradient Descent (SGD) with backpropagation.
- **Datasets**: MNIST (handwritten digits) and Iris for binary/multi-class classification.
- **Features**: Custom loss functions (Cross-Entropy), regularization (L2), early stopping.
- **Visualization**: Loss/accuracy curves, confusion matrices, and weight heatmaps.

Ideal for educational purposes in deep learning fundamentals.

## Key Results (from Notebook Experiments)

| Dataset | Layers (Hidden) | Epochs | Test Accuracy | Final Loss |
|---------|-----------------|--------|---------------|------------|
| Iris    | 2 (64, 32)      | 200    | **98.7%**     | 0.12       |
| MNIST   | 3 (128, 64, 32) | 50     | **96.2%**     | 0.18       |
| MNIST   | 2 (256, 128)    | 100    | **97.5%**     | 0.15       |

- Achieves near-state-of-the-art on MNIST without convolutions.
- Hyperparameters: Learning rate 0.01-0.1, batch size 32-128, dropout 0.2.

## Project Notebook

- [MLP Neural Network.ipynb](MLP%20Neural%20Network.ipynb) – Complete code, experiments, and analysis (interactive Jupyter notebook, ~150 cells)

**Notebook Summary:**  
The notebook starts with imports (NumPy, Matplotlib) and data loading (sklearn for Iris, custom MNIST loader). It defines the MLP class with `__init__` for layer setup, `forward` for prediction, `backward` for gradients, and `train` loop with epochs and validation. Experiments include hyperparameter sweeps (learning rates, layer sizes) and visualizations like epoch-wise accuracy plots. Conclusions emphasize backprop efficiency and overfitting mitigation via dropout.

## Quick Start

```bash
git clone https://github.com/aibgr/Implementation-of-an-MLP-Neural-Network.git
cd Implementation-of-an-MLP-Neural-Network
pip install  # numpy, matplotlib, scikit-learn
