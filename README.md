# Machine Learning Mini-Project: Physics-Informed Neural Networks (PINNs)

This repository contains the source code for solving the **1D Consolidation Problem** (a diffusion equation) using a Physics-Informed Neural Network (PINN), based on the paper *"DATA DRIVEN SOLUTIONS AND DISCOVERIES IN MECHANICS USING PHYSICS INFORMED NEURAL NETWORK"*.

## Problem Statement

The goal is to approximate the solution to the 1D consolidation process, governed by the following dimensionless Partial Differential Equation (PDE):

$$ \frac{\partial \hat{p}}{\partial \hat{t}} - \frac{\partial^2 \hat{p}}{\partial \hat{x}^2} = 0 \quad (0 < \hat{x} < 1, \hat{t} > 0) $$

With boundary and initial conditions:
- $\hat{p}(0, \hat{t}) = 0$ 
- $\frac{\partial \hat{p}}{\partial \hat{x}}(1, \hat{t}) = 0$
- $\hat{p}(\hat{x}, 0) = 1 \quad (0 < \hat{x} \le 1)$

## Project Structure

- `src/pinn.py`: The PyTorch implementation of the PINN.
- `requirements.txt`: Python dependencies.
- `README.md`: Setup and run instructions.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-private-repo-url>
   cd ml_mini_project
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

To train the PINN model, run the following command from the root of the project:

```bash
python src/pinn.py
```

This will run the training loop (using the Adam optimizer) and periodically print the loss. Once finished, it will save the trained model weights to `pinn_model.pth`. You can easily adjust the number of training epochs in `src/pinn.py`.
