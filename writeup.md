# Mini-Project Write-up: Physics-Informed Neural Networks

## Problem Statement
The objective of this project is to employ a Physics-Informed Neural Network (PINN) to solve a fundamental problem in soil mechanics: the one-dimensional consolidation process. Under standard assumptions, this process can be quantified as a 1D diffusion problem. In dimensionless form, the governing partial differential equation (PDE) is:

$$ \frac{\partial \hat{p}}{\partial \hat{t}} - \frac{\partial^2 \hat{p}}{\partial \hat{x}^2} = 0 $$

for $0 < \hat{x} < 1$ and $\hat{t} > 0$. The boundary conditions are $\hat{p}(0, \hat{t}) = 0$ (drainage boundary) and $\frac{\partial \hat{p}}{\partial \hat{x}}(1, \hat{t}) = 0$ (impermeable base). The initial condition is $\hat{p}(\hat{x}, 0) = 1$ for $0 < \hat{x} \le 1$. The goal is to accurately approximate the pore pressure field $\hat{p}(\hat{x}, \hat{t})$ using a deep neural network without relying on traditional mesh-based numerical solvers like Finite Element Methods (FEM).

## Dataset Details
A distinct feature of PINNs is that they are primarily *data-free* for forward problems; they do not require a labeled dataset of exact solutions. Instead, the "dataset" consists of spatial and temporal coordinates (collocation points) sampled from the problem domain.
- **Interior points ($\Gamma_f$)**: 500 points randomly sampled from the domain $\hat{x} \in (0, 1)$ and $\hat{t} \in (0, 0.5)$.
- **Boundary points ($\Gamma_b$)**: 250 points sampled along the boundaries $\hat{x} = 1$ (to enforce the Neumann boundary condition) and $\hat{t} = 0$ (to enforce the initial condition).
The condition at $\hat{x} = 0$ is exactly enforced through the network architecture.

## Approach
We implemented the PINN framework introduced by Raissi et al. (and explored in the provided reference paper). The deep neural network acts as a universal function approximator $\hat{p}(\hat{x}, \hat{t}; \theta)$, parameterized by weights and biases $\theta$. 
Instead of a purely data-driven loss, we used a physics-informed loss function:
$$ \mathcal{L}(\theta) = \mathcal{L}_f(\theta) + \mathcal{L}_b(\theta) + \mathcal{L}_{ic}(\theta) $$
- **$\mathcal{L}_f$**: The mean squared error of the PDE residual evaluated at the interior collocation points. The gradients (derivatives) with respect to inputs $\hat{x}$ and $\hat{t}$ are calculated using automatic differentiation (Autograd).
- **$\mathcal{L}_b$ and $\mathcal{L}_{ic}$**: The mean squared error of the network output compared to the given boundary and initial conditions at the boundary collocation points.

By minimizing this combined loss function using gradient descent, the network learns a continuous approximation of the solution that obeys the underlying conservation laws.

## Implementation Overview
The model was implemented in Python using the **PyTorch** framework. 
- **Architecture**: A Multi-Layer Perceptron (MLP) with 5 hidden layers, each containing 50 neurons. The `Tanh` activation function was chosen due to its smooth derivatives, which are crucial for computing higher-order PDE gradients.
- **Hard Constraints**: To strictly satisfy the Dirichlet boundary condition $\hat{p}(0, \hat{t}) = 0$, the raw network output $N(\hat{x}, \hat{t})$ is multiplied by $\hat{x}$ to yield the final predicted pressure: $\hat{p}_{pred} = \hat{x} \cdot N(\hat{x}, \hat{t})$. 
- **Training**: We utilized the Adam optimizer with a learning rate of 0.001. The loss function uses PyTorch's `autograd.grad` to compute the necessary spatial and temporal derivatives for the PDE residual.

## Conclusions
The implemented Physics-Informed Neural Network successfully learns to approximate the solution to the 1D diffusion equation representing soil consolidation. By embedding the differential equations directly into the loss function, PINNs provide a highly flexible, mesh-free alternative to traditional numerical methods. This approach is not only continuously differentiable across the spatio-temporal domain but can also be naturally extended to handle inverse problems (data discovery) with minimal changes to the underlying code.
