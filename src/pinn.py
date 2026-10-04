import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

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
    # Requires grad for autograd
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    p = model(x, t)
    
    # Calculate gradients
    dp_dt = torch.autograd.grad(
        p, t, 
        grad_outputs=torch.ones_like(p),
        create_graph=True
    )[0]
    
    dp_dx = torch.autograd.grad(
        p, x, 
        grad_outputs=torch.ones_like(p),
        create_graph=True
    )[0]
    
    d2p_dx2 = torch.autograd.grad(
        dp_dx, x, 
        grad_outputs=torch.ones_like(dp_dx),
        create_graph=True
    )[0]
    
    # PDE: dp/dt - d2p/dx2 = 0
    residual = dp_dt - d2p_dx2
    return residual, dp_dx

def train_pinn():
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5000  # Reduced from 50000 for quick testing; increase for full convergence
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Collocation points for PDE (interior domain)
        # Randomly sample from (0, 1) x (0, 0.5)
        x_f = torch.rand(500, 1).to(device)
        t_f = (torch.rand(500, 1) * 0.5).to(device)
        
        residual, _ = pde_residual(model, x_f, t_f)
        loss_f = torch.mean(residual**2)
        
        # 2. Boundary conditions
        # BC: dp/dx = 0 at x = 1
        x_b1 = torch.ones(250, 1).to(device)
        t_b1 = (torch.rand(250, 1) * 0.5).to(device)
        _, dp_dx_b1 = pde_residual(model, x_b1, t_b1)
        loss_b1 = torch.mean(dp_dx_b1**2)
        
        # 3. Initial condition
        # p = 1 at t = 0 for 0 < x <= 1
        x_ic = torch.rand(250, 1).to(device)
        t_ic = torch.zeros(250, 1).to(device)
        p_ic = model(x_ic, t_ic)
        loss_ic = torch.mean((p_ic - 1.0)**2)
        
        # Total loss
        loss = loss_f + loss_b1 + loss_ic
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.5f}')
            
    print("Training finished!")
    return model

if __name__ == "__main__":
    print("Training PINN for 1D consolidation problem...")
    trained_model = train_pinn()
    
    # Save the model
    torch.save(trained_model.state_dict(), "pinn_model.pth")
    print("Model saved to pinn_model.pth")
