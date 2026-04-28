# Lecture 8: Investigating Convergence (Why Damped Jacobi Stalls)

**Context for today:** We know Jacobi and Gauss-Seidel are just subspace corrections. But are they *good* solvers? Today we analyze their convergence mathematically. We will map the error to geometric shapes (eigenvectors) and discover a fatal flaw in Jacobi for solving PDEs, which perfectly motivates the need for Multigrid.

## 1. Iterative Solvers as Fixed-Point Iterations (Banach)

Any iterative solver can be written as a mapping:
$$ u^{k+1} = G(u^k) $$

By the **Banach Fixed-Point Theorem**, this iteration converges to a unique solution if $G$ is a *contraction mapping*. For linear iterative methods, the mapping takes the form $G(u) = M u + c$, where $M$ is a matrix.
To guarantee convergence for any initial guess, the matrix norm must be strictly less than 1:
$$ \|M\| < 1 $$

## 2. The Iteration Matrix in Richardson

Let's find the matrix $M$ for our preconditioned Richardson method. The update formula is:
$$ u^{k+1} = u^k + B(f - A u^k) $$
$$ u^{k+1} = (I - BA) u^k + Bf $$

We define the **Iteration Matrix**:
$$ M = I - BA $$

**The Error Equation:**
Let the exact solution be $u$ (so $Au=f$). Let the error at step $k$ be $e^k = u - u^k$. 
Let's see how the error evolves:
$$ u - e^{k+1} = M(u - e^k) + Bf $$
$$ u - e^{k+1} = Mu - M e^k + Bf $$
Since $u = Mu + Bf$ (the exact solution is a fixed point), we can cancel those terms:
$$ e^{k+1} = M e^k $$

By induction, the error at step $k$ is $e^k = M^k e^0$.
To drive the error to zero, we need the **spectral radius** (the maximum absolute eigenvalue) of $M$ to be less than 1:
$$ \rho(M) < 1 $$

## 3. Eigenvalues of the Iteration Matrix

How do the eigenvalues of our preconditioner $BA$ affect $M$?
Let $v$ be an eigenvector such that $(BA) v = \lambda v$. 
Apply the iteration matrix $M$ to $v$:
$$ M v = (I - BA) v = v - \lambda v = (1 - \lambda) v $$

**Key Rule:** If $BA$ has an eigenvalue $\lambda$, the iteration matrix $M$ has an eigenvalue $\mu = 1 - \lambda$. To converge, we need $|\mu| < 1$.

---

## 4. Solid Example: 1D Laplacian and the Anatomy of the Error

Let's evaluate the standard Jacobi method on a 1D string with $N$ internal nodes.

*   The stiffness matrix $A$ is $\text{tridiag}(-1, 2, -1)$.
*   The Jacobi preconditioner is $B = D^{-1} = \frac{1}{2}I$.
*   $BA = \text{tridiag}(-1/2, 1, -1/2)$.

It is a known mathematical property of this matrix that it has exactly $N$ eigenvalues $\lambda_m$ and corresponding eigenvectors $v_m$:
$$ \lambda_m(BA) = 1 - \cos\left(\frac{m \pi}{N+1}\right) $$
$$ (v_m)_j = \sin\left(\frac{m \pi j}{N+1}\right) $$
*(where $m = 1, \dots, N$ is the mode number, and $j = 1, \dots, N$ is the spatial node index).*

Therefore, the eigenvalues of the Jacobi iteration matrix $M_{Jac} = I - BA$ are:
$$ \mu_m = 1 - \left[1 - \cos\left(\frac{m \pi}{N+1}\right)\right] = \cos\left(\frac{m \pi}{N+1}\right) $$

### Vizual inspection

```python
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
plt.title(f"Smoothing Property of Damped Jacobi ($\omega$={omega:.2f})")
plt.xlabel("Spatial coordinate $x$")
plt.ylabel("Error / Solution $u(x)$")

# The exact solution to Laplace with 0 BCs and 0 RHS is exactly 0.
plt.axhline(0, color='black', linewidth=1.5, linestyle=':', label='Exact Solution (u=0)')

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

1. **Initial Guess (Gray):** Highly jagged. This means it contains a ton of high-frequency error.
2. **Iteration 2:** The sharpest spikes are instantly blunted. The curve is already looking continuous.
3. **Iteration 5 & 10:** The jaggedness is completely gone! However, the overall "bulk" of the error (the wide humps) is barely moving toward the zero-line. 
4. **The Conclusion:** Damped Jacobi did exactly what we designed it to do—it smoothed the solution. To get rid of the remaining smooth humps, we will need to transition to a coarser grid (where these wide humps will suddenly look "sharp" and "high-frequency" again!).


### 4.1 Look Where the Error Is!

Let's take our initial guess error $e^0$ and express it as a linear combination of these eigenvectors:
$$ e^0 = \sum_{m=1}^N c_m v_m $$

Because the error equation is $e^k = M^k e^0$, and $M v_m = \mu_m v_m$, the error after $k$ iterations is exactly:
$$ e^k = \sum_{m=1}^N c_m (\mu_m)^k v_m $$

**This is the most important realization of the lecture:** The error does not decay uniformly! It decays mode by mode, and the decay rate is strictly governed by $\mu_m$.

Let's look at the extremes:

*   **Low Frequencies ($m$ is small, e.g., $m=1$):** $\mu_1 = \cos\left(\frac{\pi}{N+1}\right) \approx 1$. 
    Because $\mu_1 \approx 1$, $(\mu_1)^k \approx 1$. The coefficient $c_1$ barely changes. **The solver stalls.**
*   **High Frequencies ($m$ is close to $N$):** $\mu_N = \cos\left(\frac{N \pi}{N+1}\right) \approx -1$. 
    Again, $|-1|^k \approx 1$. The standard Jacobi method also stalls on high frequencies (it just flips their sign every iteration)!

### 4.2 Visualizing the Eigenvectors (The Pattern of the Error)

The stalling is not random; it has a pure geometric pattern. The eigenvectors $v_m$ represent the "shape" of the error modes. 

```python
import numpy as np
import matplotlib.pyplot as plt

N = 32 # Number of internal nodes
x = np.linspace(0, 1, N+2)[1:-1] # Physical grid points

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot the 3 LOWEST frequency eigenvectors
for m in [1, 2, 3]:
    v_m = np.sin(m * np.pi * x)
    ax1.plot(x, v_m, label=f'm={m} (Low Freq)', marker='.')

ax1.set_title("Low Frequency Error Modes (Stalls)")
ax1.legend()

# Plot the 3 HIGHEST frequency eigenvectors
for m in [N-2, N-1, N]:
    v_m = np.sin(m * np.pi * x)
    ax2.plot(x, v_m, label=f'm={m} (High Freq)', marker='.')

ax2.set_title("High Frequency Error Modes")
ax2.legend()
plt.show()
```

**Physical Intuition:**
Look at the plot for $m=1$ (Low Frequency). It forms a massive, smooth bump across the entire domain. 
Jacobi is a *local* averaging process: it updates a node based only on its immediate left and right neighbors. If you sit on the top of the $m=1$ bump, your left and right neighbors have almost the exact same value as you. The local derivative appears to be zero. Therefore, Jacobi "thinks" the solution is locally flat and makes almost zero correction.

**Low frequency is the troublemaker.** Local solvers fundamentally cannot "see" global, low-frequency errors. 

---

## 5. Damping ($\omega$) to the Rescue (Sort of)

We can introduce a damping parameter $\omega$ to shift the eigenvalues to fix the high-frequency oscillation problem.
$$ B = \omega D^{-1} $$
$$ M_\omega = I - \omega D^{-1} A $$
The damped eigenvalues become:
$$ \mu_m(\omega) = 1 - \omega \lambda_m(BA) = 1 - \omega \left[1 - \cos\left(\frac{m \pi}{N+1}\right)\right] $$

**Deriving the Optimal $\omega$ (The Multigrid Perspective):**
We already know we cannot fix the low frequencies with a local solver ($\mu_1 \to 1$ regardless of $\omega$). 
Instead, we pivot our strategy. In Multigrid, we only need Jacobi to be an excellent **smoother**—meaning it must rapidly annihilate the highly oscillating *high-frequency* errors (the zigzag patterns on the right side of our plot).

Let's look at the upper half of the frequency spectrum ($m \ge N/2$). 
For these high frequencies, the values of $\lambda_m(BA)$ range between $1$ and $2$.
We want to find $\omega$ that minimizes the maximum absolute value of $\mu_m$ in this specific range $\lambda \in [1, 2]$.

We solve for the minimax condition: $\max_{\lambda \in [1, 2]} |1 - \omega \lambda|$ is minimized when the boundaries yield equal and opposite values:
$$ 1 - \omega(1) = - (1 - \omega(2)) $$
$$ 1 - \omega = -1 + 2\omega $$
$$ 3\omega = 2 \implies \omega = \frac{2}{3} $$

If we use **Damped Jacobi with $\omega = 2/3$**, any high-frequency zigzag noise ($m \ge N/2$) is multiplied by a factor of at most $1/3$ every single iteration! 
*It violently destroys high-frequency error, leaving ONLY smooth, low-frequency error behind.* This perfectly sets the stage for transferring the problem to a coarser grid.

---

## 6. Iteration Matrices for Preconditioner Combinations

Before closing, recall our subspace corrections (additive vs. multiplicative). What is the overall iteration matrix $M$ when combining multiple preconditioners $B_1, \dots, B_p$?

**1. Additive Combination (Parallel):**
The overall preconditioner is $B_{add} = \sum_{i=1}^p B_i$.
Using our formula $M = I - B_{add} A$:
$$ M_{add} = I - \left( \sum_{i=1}^p B_i \right) A = I - \sum_{i=1}^p B_i A $$

**2. Multiplicative Combination (Successive):**
Here, we apply the preconditioners sequentially. Let $e^0$ be the initial error.

*   After applying $B_1$: $e^1 = (I - B_1 A) e^0$
*   After applying $B_2$: $e^2 = (I - B_2 A) e^1 = (I - B_2 A)(I - B_1 A) e^0$

Continuing this to $B_p$, the final error is:
$$ e^p = (I - B_p A) \dots (I - B_2 A) (I - B_1 A) e^0 $$
Therefore, the multiplicative iteration matrix is the product of the individual iteration matrices:
$$ M_{mult} = \prod_{i=p}^{1} (I - B_i A) $$
*(Note: Order matters! Matrices do not commute. The rightmost matrix $(I-B_1A)$ is applied to the error first).*




## 7. A Handwavy Thought Experiment: The Perfect Pair

To see where we are going next, let's do a handwavy thought experiment using the multiplicative combination formula. 

Suppose we have an arbitrary error $e^0$ that is a mix of all frequencies. Let's conceptually split this error into two parts:
$$ e^0 = e_{low} + e_{high} $$
where $e_{low}$ is composed of the first $k$ eigenvectors, and $e_{high}$ is composed of the remaining eigenvectors.

Now, imagine we possess two idealized, magical preconditioners:

1.  **$B_{low}$**: It perfectly and exactly solves the problem for the first $k$ eigenvalues, but completely ignores the rest. 
    Its iteration matrix $M_{low} = I - B_{low} A$ acts as a perfect filter:
    $$ M_{low}(e_{low} + e_{high}) = 0 + e_{high} $$
2.  **$B_{high}$**: It perfectly and exactly solves the problem for the remaining high-frequency eigenvalues, but ignores the low ones.
    Its iteration matrix $M_{high} = I - B_{high} A$ filters the other way:
    $$ M_{high}(e_{low} + e_{high}) = e_{low} + 0 $$

What happens if we combine these two idealized preconditioners **multiplicatively**? 
The overall iteration matrix is $M_{mult} = M_{high} M_{low}$. Let's track the error through one single global iteration:

*   **Step 1 (Apply $B_{low}$):** 
    $$ e^1 = M_{low} e^0 = M_{low}(e_{low} + e_{high}) = e_{high} $$
    *The low frequencies are completely annihilated! Only zigzag noise remains.*
*   **Step 2 (Apply $B_{high}$):** 
    $$ e^2 = M_{high} e^1 = M_{high}(e_{high}) = 0 $$
    *The remaining high frequencies are annihilated!*

**The Result:** $M_{mult} e^0 = 0$. 
By multiplicatively combining two solvers that act on completely different segments of the eigenvalue spectrum, the exact solution is reached in **exactly one iteration**.

*(Spoiler for the next lecture: We don't have these magical idealized preconditioners. However, **Damped Jacobi** will act as our $B_{high}$ to kill the oscillatory error, and a **Coarse Grid Correction** will act as our $B_{low}$ to kill the smooth error. Together, they form the Multigrid method!)*