import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# Create the assets directory for storing graphs
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # 5 hidden layers with 50 neurons each, as specified in the paper
        layers = [2] + [50]*5 + [1]
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layers) - 2):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i+1]))
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        self.activation = nn.Tanh()

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        out = inputs
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        out = self.output_layer(out)
        # Multiply by x to automatically satisfy the boundary condition p(0, t) = 0
        return x * out

def pde_residual(model, x, t):
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    p = model(x, t)
    
    # Calculate gradients
    dp_dt = torch.autograd.grad(p, t, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    dp_dx = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    d2p_dx2 = torch.autograd.grad(dp_dx, x, grad_outputs=torch.ones_like(dp_dx), create_graph=True)[0]
    
    # PDE: dp/dt - d2p/dx2 = 0
    residual = dp_dt - d2p_dx2
    return residual, dp_dx

def exact_solution(x, t, terms=50):
    """Computes the analytical solution from the paper for comparison."""
    p_exact = np.zeros_like(x)
    for m in range(1, terms*2, 2):
        term = (4.0 / (m * np.pi)) * np.sin(m * np.pi * x / 2.0) * np.exp(-(m**2 * np.pi**2 * t) / 4.0)
        p_exact += term
    return p_exact

def train_and_evaluate():
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 3000
    loss_history = []
    
    print("Training PINN...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Collocation points for PDE (interior domain)
        x_f = torch.rand(500, 1).to(device)
        t_f = (torch.rand(500, 1) * 0.5).to(device)
        residual, _ = pde_residual(model, x_f, t_f)
        loss_f = torch.mean(residual**2)
        
        # 2. Boundary conditions (x = 1)
        x_b1 = torch.ones(250, 1).to(device)
        t_b1 = (torch.rand(250, 1) * 0.5).to(device)
        _, dp_dx_b1 = pde_residual(model, x_b1, t_b1)
        loss_b1 = torch.mean(dp_dx_b1**2)
        
        # 3. Initial condition (t = 0)
        x_ic = torch.rand(250, 1).to(device)
        t_ic = torch.zeros(250, 1).to(device)
        p_ic = model(x_ic, t_ic)
        loss_ic = torch.mean((p_ic - 1.0)**2)
        
        loss = loss_f + loss_b1 + loss_ic
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if epoch % 500 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.5f}')
            
    print("Training finished! Generating graphs...")

    # --- PLOTTING ---
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color='#E94560', linewidth=2, label='Total Loss')
    plt.yscale('log')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.title('PINN Training Convergence', fontsize=14, pad=15)
    plt.legend()
    plt.savefig(os.path.join(ASSETS_DIR, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap Comparison
    model.eval()
    x_grid = np.linspace(0, 1, 100)
    t_grid = np.linspace(0, 0.5, 100)
    X, T = np.meshgrid(x_grid, t_grid)
    
    X_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to(device)
    T_flat = torch.tensor(T.flatten()[:, None], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        P_pred = model(X_flat, T_flat).cpu().numpy().reshape(100, 100)
    P_exact = exact_solution(X, T)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    im1 = ax1.contourf(T, X, P_exact, levels=100, cmap='viridis')
    ax1.set_title('Exact Analytical Solution', fontsize=14)
    ax1.set_xlabel('Time ($\hat{t}$)', fontsize=12)
    ax1.set_ylabel('Space ($\hat{x}$)', fontsize=12)
    fig.colorbar(im1, ax1=ax1)
    
    im2 = ax2.contourf(T, X, P_pred, levels=100, cmap='viridis')
    ax2.set_title('PINN Prediction', fontsize=14)
    ax2.set_xlabel('Time ($\hat{t}$)', fontsize=12)
    fig.colorbar(im2, ax2=ax2)
    
    error = np.abs(P_exact - P_pred)
    im3 = ax3.contourf(T, X, error, levels=100, cmap='magma')
    ax3.set_title('Absolute Error', fontsize=14)
    ax3.set_xlabel('Time ($\hat{t}$)', fontsize=12)
    fig.colorbar(im3, ax3=ax3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'pressure_field_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 1D Cross-Sections (Profiles)
    plt.figure(figsize=(10, 6))
    test_times = [0.05, 0.1, 0.3]
    colors = ['#4361EE', '#3A0CA3', '#F72585']
    
    x_test = np.linspace(0, 1, 100)
    x_test_tensor = torch.tensor(x_test[:, None], dtype=torch.float32).to(device)
    
    for i, t_val in enumerate(test_times):
        t_test_tensor = torch.ones_like(x_test_tensor) * t_val
        with torch.no_grad():
            p_pred_t = model(x_test_tensor, t_test_tensor).cpu().numpy()
        p_exact_t = exact_solution(x_test, t_val)
        
        plt.plot(x_test, p_exact_t, color=colors[i], linestyle='-', linewidth=2, alpha=0.6, label=f'Exact t={t_val}')
        plt.plot(x_test, p_pred_t, color=colors[i], linestyle='--', marker='o', markersize=4, markevery=5, label=f'PINN t={t_val}')
        
    plt.xlabel('Space ($\hat{x}$)', fontsize=12)
    plt.ylabel('Pore Pressure ($\hat{p}$)', fontsize=12)
    plt.title('Pore Pressure Profiles at Specific Times', fontsize=14, pad=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'pressure_profiles.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Graphs successfully saved to {ASSETS_DIR}")

if __name__ == "__main__":
    train_and_evaluate()
