


# Homework 1: Evaluating the Local Laplace Operator

**Objective:** In this assignment, you will implement the evaluation of the Laplace operator on a single 3D Cartesian reference cell, $v_c = A_c u_c$. You will implement this using three mathematically equivalent but computationally distinct methods to understand the transition from classical finite elements to high-performance matrix-free tensor contractions.

**Prerequisites:** You will need `numpy` and `jax`. 

---

##  Notes & Theoretical Background

We are solving the Laplace equation $-\Delta u = f$ on a reference cell $[-1, 1]^3$.
Let the 3D basis functions be the tensor product of 1D Lagrange polynomials:
$$ \phi_{ijk}(x,y,z) = \psi_i(z) \psi_j(y) \psi_k(x) $$
*(Note our index ordering: $i \to z$ (slowest), $j \to y$, $k \to x$ (fastest)).*

The local stiffness matrix entry is defined by the weak form:
$$ (A_c)_{(lmn), (ijk)} = \int_{-1}^1 \int_{-1}^1 \int_{-1}^1 \nabla \phi_{ijk} \cdot \nabla \phi_{lmn} \, dx \, dy \, dz $$

### The Three Methods

**Method 1: Classical Assembly**
We explicitly build the dense $N^3 \times N^3$ matrix $A_c$ using Kronecker products and perform a standard matrix-vector multiplication.
$$ A_c = D \otimes M \otimes M + M \otimes D \otimes M + M \otimes M \otimes D $$
Where $M$ is the 1D Mass matrix and $D$ is the 1D Stiffness matrix.

**Method 2: Quadrature-Based Evaluation (Matrix-Free Integration)**
Instead of assembling $A_c$, we evaluate the weak form directly.

1. Evaluate the gradient of the solution at the quadrature points: $\nabla u(\mathbf{x}_q)$
2. Multiply by the quadrature weights $w_q$.
3. Multiply by the gradient of the test functions and sum over quadrature points (pull-back).

**Method 3: Sum Factorization**
We use the tensor-product structure of $A_c$ directly on the 3D grid of Degrees of Freedom (DoFs) $U_{ijk}$, applying the 1D matrices dimension-by-dimension.
$$ V_{ijk} = M_{il} M_{jm} D_{kn} U_{lmn} + M_{il} D_{jm} M_{kn} U_{lmn} + D_{il} M_{jm} M_{kn} U_{lmn} $$

---

### Provided Code & 1D Building Blocks


The provided code to generate the 1D evaluation matrices $\Phi$ and $\nabla\Phi$. Then, use them to compute the 1D Mass matrix ($M$) and 1D Stiffness matrix ($D$).

> Note: that is just simple example so it is written as a script. Please write code that is organized.

```python
import jax
import jax.numpy as jnp
import numpy as np

# --- 1. Quadrature Setup ---
def get_gauss_legendre(degree):
    """Returns quadrature points and weights for exactly integrating degree 2p-1."""
    # We need p+1 points to exactly integrate the mass matrix (degree 2p)
    n_points = degree + 1 
    points, weights = np.polynomial.legendre.leggauss(n_points)
    return jnp.array(points), jnp.array(weights)

# --- 2. Basis Function Setup ---
def evaluate_basis_1d(point, nodes):
    """
    Evaluates all 1D Lagrange basis functions at a single scalar point.
    """
    diffs = point - nodes
    num_nodes = nodes.shape[0]
    
    # Mask to ignore the (point - node[i]) term for the i-th basis function
    mask = jnp.eye(num_nodes, dtype=bool)
    numerators = jnp.where(mask, 1.0, diffs[None, :])
    denominators = jnp.where(mask, 1.0, nodes[:, None] - nodes[None, :])
    
    return jnp.prod(numerators, axis=1) / jnp.prod(denominators, axis=1)

# Derivative of the basis functions using JAX Automatic Differentiation
evaluate_basis_deriv_1d = jax.jacfwd(evaluate_basis_1d, argnums=0)

# --- Generate 1D Operators ---
degree = 2
nodes_1d = jnp.linspace(-1, 1, degree + 1)
q_points, q_weights = get_gauss_legendre(degree)

# Evaluate the functions over all quadrature points using a loop/comprehension 
# instead of jax.vmap. We then stack the list of 1D arrays into a 2D array.
# Phi shape: (N_quad, N_dof)
Phi = jnp.stack([evaluate_basis_1d(q, nodes_1d) for q in q_points])
dPhi = jnp.stack([evaluate_basis_deriv_1d(q, nodes_1d) for q in q_points])

# Compute the 1D Mass matrix (M) and 1D Stiffness matrix (D)
# M_ij = sum_q w_q * Phi_qi * Phi_qj
# TODO: write a formula here!
M = ....

# D_ij = sum_q w_q * dPhi_qi * dPhi_qj
# TODO: write a formula here!
D = .... 
```

---

### Method 1 - Classical Assembly

In this method, we flatten our 3D grid of DoFs into a 1D vector and multiply it by the fully assembled $A_c$ matrix.

---

### Method 2 - Quadrature-Based Evaluation

Here, we evaluate the integrals strictly over the quadrature points using tensor contractions (`jnp.einsum`/`np.einsum`).

1. Evaluate the $x, y, z$ derivatives of $U$ at the 3D quadrature grid. 
2. Integrate back by multiplying by the 3D weights ($w_q w_r w_s$) and the test function derivatives.
3. Sum the $x, y, z$ contributions to form `V_quad`.

---

### Method 2 - Sum Factorization

This method is the sweet spot: it is mathematically identical to Method 1, but computes the action dimension-by-dimension without ever forming the dense matrix.


Evaluate the operator using the 1D matrices `M` and `D` directly on `U` via `jnp.einsum`.


---

## Verification

Verify that all three methods produce the exact same 3D array (up to floating point precision).
Test for various polynomial degrees.

```python
print("Difference Classical vs Quad:    ", jnp.linalg.norm(V_classical - V_quad))
print("Difference Classical vs SumFact: ", jnp.linalg.norm(V_classical - V_sumfact))

# Both should be close to 1e-15.
# Remember to enable fp64 in JAX.
```