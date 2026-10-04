import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PINN_2D(nn.Module):
    def __init__(self):
        super(PINN_2D, self).__init__()
        # 4 hidden layers, 50 neurons as per Section 4.2
        layers = [2] + [50]*4 + [1]
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layers) - 2):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i+1]))
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        self.activation = nn.Tanh()

    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        out = inputs
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        out = self.output_layer(out)
        # Hard boundary constraint: T=0 on x=+-1, y=+-1
        # The paper suggests a smooth distance function d(x) to boundary
        distance = (1 - x**2) * (1 - y**2)
        return distance * out

def pde_residual(model, x, y):
    x.requires_grad_(True)
    y.requires_grad_(True)
    T = model(x, y)
    
    dT_dx = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    d2T_dx2 = torch.autograd.grad(dT_dx, x, grad_outputs=torch.ones_like(dT_dx), create_graph=True)[0]
    
    dT_dy = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    d2T_dy2 = torch.autograd.grad(dT_dy, y, grad_outputs=torch.ones_like(dT_dy), create_graph=True)[0]
    
    # PDE: d2T/dx2 + d2T/dy2 + 1 = 0
    residual = d2T_dx2 + d2T_dy2 + 1.0
    return residual

def exact_solution_2d(x, y, terms=40):
    """Computes analytical Fourier series solution for the 2D Poisson equation."""
    T = 0.5 * (1 - x**2)
    for n in range(1, terms*2, 2):
        sign = -1 if (n % 4 == 3) else 1
        coef = (16.0 * sign) / ((n**3) * (np.pi**3))
        term = coef * (np.cosh(n * np.pi * y / 2.0) / np.cosh(n * np.pi / 2.0)) * np.cos(n * np.pi * x / 2.0)
        T -= term
    return T

def train_and_evaluate():
    model = PINN_2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 4000
    print("Training 2D Heat Conduction PINN...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Interior points: (-1, 1) x (-1, 1)
        x_f = (torch.rand(1500, 1) * 2 - 1).to(device)
        y_f = (torch.rand(1500, 1) * 2 - 1).to(device)
        
        # Since we use hard constraints, we only need PDE residual loss!
        residual = pde_residual(model, x_f, y_f)
        loss = torch.mean(residual**2)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.5f}')
            
    print("Training finished! Generating 2D Heat graphs...")

    # Evaluate over grid
    x_grid = np.linspace(-1, 1, 100)
    y_grid = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    X_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
    Y_flat = torch.tensor(Y.flatten()[:, None], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        T_pred = model(X_flat, Y_flat).cpu().numpy().reshape(100, 100)
        
    T_exact = exact_solution_2d(X, Y)
    error = np.abs(T_exact - T_pred)
    max_err = np.max(error)
    
    # Plot identical to user's screenshot
    plt.style.use('default')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    im1 = ax1.pcolormesh(X, Y, T_exact, cmap='plasma', shading='auto')
    ax1.set_title('Analytical Fourier / Numerical Reference')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label('Temperature $T$')
    
    im2 = ax2.pcolormesh(X, Y, T_pred, cmap='plasma', shading='auto')
    ax2.set_title('PINN Predicted Temperature')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label('Temperature $T$')
    
    im3 = ax3.pcolormesh(X, Y, error, cmap='hot', shading='auto')
    ax3.set_title(f'Absolute Error (Max: {max_err:.3e})')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    cbar3 = fig.colorbar(im3, ax=ax3)
    cbar3.set_label('Absolute Error $|T_{PINN} - T_{ref}|$')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'heat_2d_comparison.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Graph saved to {ASSETS_DIR}/heat_2d_comparison.png")

if __name__ == "__main__":
    train_and_evaluate()
