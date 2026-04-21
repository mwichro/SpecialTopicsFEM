
# Lecture 5: Fast Diagonalization

**Context for today:** In upcoming lectures on Subspace Correction and Multigrid, we will need to efficiently invert local block matrices (e.g., solving the problem exactly on a single cell). Today, we show how to invert a 3D Laplace operator on a tensor-product cell exactly, not in $\mathcal{O}(N^9)$ (dense inverse), but in $\mathcal{O}(N^4)$ using a technique called *Fast Diagonalization*.

## 1. Tensor Product Evaluation of the Laplace Operator

We want to construct the matrix for the Laplace operator $-\Delta u$ on a single 3D Cartesian cell $\Omega =[x_0, x_1] \times [y_0, y_1] \times[z_0, z_1]$.

**Step 1.1: The Weak Form**
The finite element matrix is generated from the weak form. We want to evaluate the bilinear form:
$$ a(u, v) = \int_{\Omega} \nabla u \cdot \nabla v \, d\Omega = \int_{\Omega} \left( \frac{\partial u}{\partial x}\frac{\partial v}{\partial x} + \frac{\partial u}{\partial y}\frac{\partial v}{\partial y} + \frac{\partial u}{\partial z}\frac{\partial v}{\partial z} \right) dx \, dy \, dz $$

**Step 1.2: Tensor Product Basis**
Let the 3D basis functions be the product of 1D basis functions (e.g., 1D Lagrange polynomials $\psi(x)$):
$$ \phi_{ijk}(x,y,z) = \psi_i(z) \psi_j(y) \psi_k(x) $$
*(Note the index mapping: $i \to z$ (slowest), $j \to y$, $k \to x$ (fastest)).*

We expand our trial function $u$ and test function $v$ as:
$$ u(x,y,z) = \sum_{i,j,k} U_{ijk} \phi_{ijk}(x,y,z) \quad \text{and} \quad v = \phi_{lmn}(x,y,z) = \psi_l(z) \psi_m(y) \psi_n(x) $$

**Step 1.3: Separating the Integrals**
Let's analyze just the $x$-derivative term of the weak form for the matrix entry $(l,m,n), (i,j,k)$:
$$ A^{(x)}_{(lmn), (ijk)} = \int_{\Omega} \frac{\partial \phi_{ijk}}{\partial x} \frac{\partial \phi_{lmn}}{\partial x} \, dx \, dy \, dz $$

Substitute the 1D basis functions. Notice that the $x$-derivative only hits $\psi_k(x)$ and $\psi_n(x)$:
$$ = \int_{\Omega} \big( \psi_i(z) \psi_j(y) \psi_k'(x) \big) \big( \psi_l(z) \psi_m(y) \psi_n'(x) \big) \, dx \, dy \, dz $$

Because the domain is a Cartesian box, the 3D integral separates perfectly into a product of 1D integrals:
$$ A^{(x)}_{(lmn), (ijk)} = \left( \int \psi_i(z) \psi_l(z) dz \right) \left( \int \psi_j(y) \psi_m(y) dy \right) \left( \int \psi_k'(x) \psi_n'(x) dx \right) $$

**Definition of 1D Matrices:**

*   1D Mass Matrix: $M_{rs} = \int \psi_s(\xi) \psi_r(\xi) d\xi$
*   1D Stiffness (Laplace) Matrix: $D_{rs} = \int \psi'_s(\xi) \psi'_r(\xi) d\xi$

Substituting these into our separated integrals:
$$ A^{(x)}_{(lmn), (ijk)} = M_{li} M_{mj} D_{nk} $$
By our multi-index Kronecker product rules (Lecture 2), this is exactly $M \otimes M \otimes D$.

By symmetry, applying the exact same separation to the $y$ and $z$ derivatives yields the total 3D operator $L$:
$$ L = M \otimes M \otimes D + M \otimes D \otimes M + D \otimes M \otimes M $$

---

## 2. Inverting the Operator: The Simplified Case ($M=I$)

Suppose our basis is orthogonal, meaning the mass matrix is the identity ($M=I$). Our operator simplifies to:
$$ L = I \otimes I \otimes D + I \otimes D \otimes I + D \otimes I \otimes I $$

To invert $L$, we will look at its eigenvalues. 

**Step 2.1: Eigenvalues of a Kronecker Product**
**Theorem:** If $D q = \lambda q$, then $(I \otimes I \otimes D)(q_3 \otimes q_2 \otimes q_1) = \lambda_1 (q_3 \otimes q_2 \otimes q_1)$.

**Proof:**
Using the mixed-product property on vectors:
$$ (I \otimes I \otimes D)(q_3 \otimes q_2 \otimes q_1) = (I q_3) \otimes (I q_2) \otimes (D q_1) $$
$$ = q_3 \otimes q_2 \otimes (\lambda_1 q_1) = \lambda_1 (q_3 \otimes q_2 \otimes q_1) \quad \blacksquare $$

This means the eigenvectors of the 3D operator are just tensor products of the 1D eigenvectors! 
If $D q_i = \lambda_i q_i$, then applying the full operator $L$ yields:
$$ L (q_k \otimes q_j \otimes q_i) = (\lambda_k + \lambda_j + \lambda_i) (q_k \otimes q_j \otimes q_i) $$

**Step 2.2: Deriving the Inverse**
Let $D$ be diagonalized by an orthogonal matrix $Q$, such that its columns are the eigenvectors $q$, and $Q^T = Q^{-1}$.
$$ D = Q \Lambda Q^T $$
where $\Lambda$ is the diagonal matrix of 1D eigenvalues. Note that $I = Q Q^T$.

Substitute these into $L$:
$$ L = (QQ^T \otimes QQ^T \otimes Q\Lambda Q^T) + (QQ^T \otimes Q\Lambda Q^T \otimes QQ^T) + (Q\Lambda Q^T \otimes QQ^T \otimes QQ^T) $$

Using the Mixed-Product Property in reverse to pull $Q$ out to the left and $Q^T$ out to the right:
$$ L = (Q \otimes Q \otimes Q) \big[ I \otimes I \otimes \Lambda + I \otimes \Lambda \otimes I + \Lambda \otimes I \otimes I \big] (Q^T \otimes Q^T \otimes Q^T) $$

Let $\mathbf{Q} = Q \otimes Q \otimes Q$. The bracketed term is a purely diagonal matrix, which we will call $\Sigma$, where $\Sigma_{ijk} = \lambda_i + \lambda_j + \lambda_k$. 
$$ L = \mathbf{Q} \Sigma \mathbf{Q}^T $$

Because $\mathbf{Q}$ is orthogonal ($\mathbf{Q}^{-1} = \mathbf{Q}^T$), the inverse of $L$ is trivial!
$$ L^{-1} = \mathbf{Q} \Sigma^{-1} \mathbf{Q}^T $$

---

## 3. Generalizing to $M \neq I$ (Standard FEM)

In standard FEM, $M$ is tridiagonal (or denser for higher $p$), not the identity. 
$$ L = M \otimes M \otimes D + M \otimes D \otimes M + D \otimes M \otimes M $$

To apply the same trick, we must diagonalize $M$ and $D$ *simultaneously*.

**Step 3.1: The Generalized Eigenvalue Problem**
Define the generalized eigenvalue problem for our 1D operators:
$$ D q = \lambda M q $$

Because $D$ and $M$ are symmetric matrices, and $M$ is positive-definite (it's a mass matrix), linear algebra guarantees there exists a matrix of eigenvectors $Q$ and a diagonal matrix of eigenvalues $\Lambda$ such that:

1.  $Q^T M Q = I$  *(The eigenvectors are $M$-orthonormal)*
2.  $Q^T D Q = \Lambda$ *(The eigenvectors diagonalize $D$)*

**Step 3.2: Expressing $M$ and $D$ via $Q$**
From the above properties, we can isolate $M$ and $D$:
$$ Q^T M Q = I \implies M = (Q^T)^{-1} I Q^{-1} = Q^{-T} Q^{-1} $$
$$ Q^T D Q = \Lambda \implies D = (Q^T)^{-1} \Lambda Q^{-1} = Q^{-T} \Lambda Q^{-1} $$

**Step 3.3: Substituting into $L$**
Let's substitute these representations into the first term of $L$ ($M \otimes M \otimes D$):
$$ M \otimes M \otimes D = (Q^{-T} Q^{-1}) \otimes (Q^{-T} Q^{-1}) \otimes (Q^{-T} \Lambda Q^{-1}) $$

Using the Mixed-Product property, factor out $Q^{-T}$ on the left and $Q^{-1}$ on the right:
$$ = (Q^{-T} \otimes Q^{-T} \otimes Q^{-T}) \big( I \otimes I \otimes \Lambda \big) (Q^{-1} \otimes Q^{-1} \otimes Q^{-1}) $$

Let $\mathbf{Q}^{-1} = Q^{-1} \otimes Q^{-1} \otimes Q^{-1}$ and $\mathbf{Q}^{-T} = Q^{-T} \otimes Q^{-T} \otimes Q^{-T}$.
Doing this for all three terms of $L$, we get:
$$ L = \mathbf{Q}^{-T} \big( I \otimes I \otimes \Lambda + I \otimes \Lambda \otimes I + \Lambda \otimes I \otimes I \big) \mathbf{Q}^{-1} $$

Let $\Sigma$ be the diagonal matrix of the sum of eigenvalues ($\Sigma_{ijk} = \lambda_i + \lambda_j + \lambda_k$):
$$ L = \mathbf{Q}^{-T} \Sigma \mathbf{Q}^{-1} $$

**Step 3.4: The Final Inverse**
Inverting this sequence of matrices reverses the order and flips the inverses:

$$ L^{-1} = (\mathbf{Q}^{-1})^{-1} \Sigma^{-1} (\mathbf{Q}^{-T})^{-1} $$
$$ L^{-1} = \mathbf{Q} \Sigma^{-1} \mathbf{Q}^{T} $$
Substituting $\mathbf{Q}$ back in:
$$ L^{-1} = (Q \otimes Q \otimes Q) \Sigma^{-1} (Q^T \otimes Q^T \otimes Q^T) $$

**How to compute $u = L^{-1} f$ in practice:**

We never form the dense 3D matrix. We solve the block exactly via sum-factorization (tensor contractions):

1.  **Transform to Eigen-space:** $\hat{f} = (Q^T \otimes Q^T \otimes Q^T) f$ (Contract $f$ with $Q^T$ along $x,y,z$)
2.  **Scale by 3D Eigenvalues:** $\hat{u}_{ijk} = \frac{\hat{f}_{ijk}}{\lambda_i + \lambda_j + \lambda_k}$ (Simple element-wise division!)
3.  **Transform to Physical space:** $u = (Q \otimes Q \otimes Q) \hat{u}$ (Contract $\hat{u}$ with $Q$ along $x,y,z$)

*Computational cost:* Three 1D contractions + one element-wise array division. Total cost is $\mathcal{O}(N^4)$. We have solved a 3D Laplace problem exactly on a cell at the cost of a single matrix-vector multiplication!