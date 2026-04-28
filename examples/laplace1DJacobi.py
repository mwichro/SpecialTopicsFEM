import numpy as np
import matplotlib.pyplot as plt

# --- Setup the Grid ---
N_elements = 32
N_nodes = N_elements + 1
x = np.linspace(0, 1, N_nodes)

# Damping parameter for 1D Laplace
omega = 2.0 / 3.0 

# --- Initialization ---
# Start with a random noise solution (contains all error frequencies)
np.random.seed(42) # Seeded for reproducible lecture plots
u = np.random.rand(N_nodes) - 0.5 # Shift to be centered around 0

# Apply Dirichlet Boundary Conditions: u(0) = 0, u(1) = 0
u[0] = 0.0
u[-1] = 0.0

# --- Plotting Setup ---
plt.figure(figsize=(10, 6))
plt.plot(x, u, label='Initial Guess (k=0)', color='lightgray', marker='.', linestyle='--')

# --- Damped Jacobi Iteration ---
max_iter = 10
plot_steps = [2, 5, 10]

for step in range(1, max_iter + 1):
    u_new = np.copy(u)
    
    # Update internal nodes using Damped Jacobi formula:
    # u^{k+1}_i = (1 - w) * u^k_i + w * (u^k_{i-1} + u^k_{i+1}) / 2
    u_new[1:-1] = (1 - omega) * u[1:-1] + omega * 0.5 * (u[:-2] + u[2:])
    
    u = np.copy(u_new)
    
    # Plot at specified steps
    if step in plot_steps:
        # Gradually darken the line color for later iterations
        alpha_val = step / max_iter
        plt.plot(x, u, label=f'Iteration k={step}', marker='.', linewidth=2)

# --- Formatting the Plot ---
plt.title(f"Smoothing Property of Damped Jacobi ($\\omega$={omega:.2f})")
plt.xlabel("Spatial coordinate $x$")
plt.ylabel("Error / Solution $u(x)$")

# The exact solution to Laplace with 0 BCs and 0 RHS is exactly 0.
plt.axhline(0, color='black', linewidth=1.5, linestyle=':', label='Exact Solution (u=0)')

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()