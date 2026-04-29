

# Lecture 10: Tensor Product Structure of Transfer Operators

**Context for today:** In 3D, if our parent polynomial degree is $p=3$, a single cell has $4^3 = 64$ DoFs. Prolonging to its 8 children (each also having 64 DoFs) naively requires a massive $512 \times 64$ matrix. Today, we will use our dimensional splitting (from Lecture 1) to break Prolongation and Restriction into tiny 1D operations. 

For today's lecture, **we will look strictly at a single parent cell in isolation.** We will not worry about neighboring parent cells or global assembly yet.

## 1. The 1D Reference Cell and its Children

Let's start in 1D. Consider a single parent reference cell $\Omega_H = [-1, 1]$.
When we refine this grid, we split the parent into $2^1 = 2$ children:
*   **Child 0 (Left):** $\Omega_0 = [-1, 0]$
*   **Child 1 (Right):** $\Omega_1 = [0, 1]$

Let our finite element space on the parent have $N$ basis functions (e.g., Lagrange polynomials of degree $p$, so $N = p+1$). 
Any function on the parent is represented by a coefficient vector $u_H \in \mathbb{R}^N$:
$$ u_H(x) = \sum_{j=1}^N (u_H)_j \Phi_j(x) $$

When we move to the fine grid, *each* child cell also has $N$ basis functions. 
We need to find the fine coefficients on Child 0, $u_h^{(0)} \in \mathbb{R}^N$, and the fine coefficients on Child 1, $u_h^{(1)} \in \mathbb{R}^N$, such that the function shape remains exactly the same.

## 2. 1D Prolongation and Restriction Matrices

Because the coarse space is a subspace of the fine space, evaluating the parent function at the local nodes of the child gives us the exact fine coefficients.

**Constructing $P^{(0)}$ (Prolongation to Left Child):**
We want a matrix $P^{(0)}$ that maps the parent DoFs to Child 0's DoFs: $u_h^{(0)} = P^{(0)} u_H$.
Let $x_i^{(0)}$ be the coordinates of the nodes inside Child 0.
The $i$-th fine coefficient on Child 0 is simply the parent function evaluated at that specific child node:
$$ (u_h^{(0)})_i = u_H(x_i^{(0)}) = \sum_{j=1}^N \Phi_j(x_i^{(0)}) (u_H)_j $$
This defines the entries of our first prolongation matrix:
$$ P^{(0)}_{ij} = \Phi_j(x_i^{(0)}) $$
*(Row $i$ is the fine node on Child 0; Column $j$ is the coarse parent basis function).*

**Constructing $P^{(1)}$ (Prolongation to Right Child):**
Exactly the same logic applies to the right child. Let $x_i^{(1)}$ be the local nodes of Child 1.
$$ P^{(1)}_{ij} = \Phi_j(x_i^{(1)}) $$

**The 1D Restriction Matrices:**
As we proved in Lecture 9, transferring a *functional* (like a residual) from the fine grid to the coarse grid is the transpose of Prolongation. 
If we have a residual on Child 0 ($r_h^{(0)}$) and a residual on Child 1 ($r_h^{(1)}$), their contributions to the parent cell are computed using the transpose matrices:
*   Restriction from Left Child: $R^{(0)} = (P^{(0)})^T$
*   Restriction from Right Child: $R^{(1)} = (P^{(1)})^T$

To get the total coarse residual on the parent, we just restrict and sum the contributions from both children:
$$ r_H = R^{(0)} r_h^{(0)} + R^{(1)} r_h^{(1)} = (P^{(0)})^T r_h^{(0)} + (P^{(1)})^T r_h^{(1)} $$

---

## 3. Generalizing to $d$-Dimensions (Lexicographical Ordering)

Now let's step up to a $d$-dimensional parent cell (e.g., a 2D quad or 3D hex). 
If we split a cell in half along every axis, a parent has exactly **$2^d$ children**.
*   1D: 2 children.
*   2D: 4 children.
*   3D: 8 children.

How do we index these children systematically? We use a **multi-index** $\mathbf{c}$.
For a 3D cell, let $\mathbf{c} = (c_x, c_y, c_z)$, where each index is either $0$ (left/bottom/back) or $1$ (right/top/front).
*   Child $(0,0,0)$ is the bottom-left-back child.
*   Child $(1,0,0)$ is the bottom-right-back child.
*   Child $(1,1,1)$ is the top-right-front child.

This multi-index is exactly equivalent to binary counting (lexicographical ordering). 
Child $(c_x, c_y, c_z)$ corresponds to the integer index $c_x + 2 c_y + 4 c_z$. 

---

## 4. Tensor Product Structure of $d$-Dimensional Prolongation

Let's derive the 3D Prolongation matrix to a specific child $\mathbf{c} = (c_x, c_y, c_z)$.

**Step 1: The 3D Basis**
Our parent 3D basis functions are tensor products of 1D basis functions. Let $(k, j, i)$ be the $x, y, z$ indices of the parent DoF:
$$ \Phi_{ijk}(x,y,z) = \psi_i(z) \psi_j(y) \psi_k(x) $$

**Step 2: The 3D Child Nodes**
The nodes of child $\mathbf{c}$ are also formed by a tensor product of 1D child nodes. Let $(n, m, l)$ be the $x, y, z$ indices of the fine DoF on the child. 
The physical coordinates of this node are:
$$ (x_n^{(c_x)}, \ y_m^{(c_y)}, \ z_l^{(c_z)}) $$

**Step 3: Evaluating the Matrix Entries**
By our 1D definition, the entry of the 3D Prolongation matrix $\mathbf{P}^{(\mathbf{c})}$ is the parent basis function evaluated at the child node:
$$ \mathbf{P}^{(\mathbf{c})}_{(lmn), (ijk)} = \Phi_{ijk} \left( x_n^{(c_x)}, y_m^{(c_y)}, z_l^{(c_z)} \right) $$

Substitute the separated 3D basis:
$$ = \psi_i(z_l^{(c_z)}) \cdot \psi_j(y_m^{(c_y)}) \cdot \psi_k(x_n^{(c_x)}) $$

**Step 4: Recognizing the 1D Matrices**
Look closely at the three terms. These are exactly the definitions of our 1D Prolongation matrices evaluated for the specific child directions!
*   $\psi_k(x_n^{(c_x)})$ is the entry $P^{(c_x)}_{nk}$
*   $\psi_j(y_m^{(c_y)})$ is the entry $P^{(c_y)}_{mj}$
*   $\psi_i(z_l^{(c_z)})$ is the entry $P^{(c_z)}_{li}$

Therefore, the full 3D Prolongation matrix operator for child $\mathbf{c}$ is:
$$ \mathbf{P}^{(\mathbf{c})}_{(lmn), (ijk)} = P^{(c_z)}_{li} P^{(c_y)}_{mj} P^{(c_x)}_{nk} $$

By the multi-index definition of the Kronecker product (Lecture 2), this is:
$$ \mathbf{P}^{(c_x, c_y, c_z)} = P^{(c_z)} \otimes P^{(c_y)} \otimes P^{(c_x)} $$

---

## 5. Tensor Product Structure of $d$-Dimensional Restriction

Because $R = P^T$, and the transpose of a Kronecker product is the Kronecker product of the transposes ($(A \otimes B)^T = A^T \otimes B^T$), the Restriction matrix from child $\mathbf{c}$ is trivially:

$$ \mathbf{R}^{(c_x, c_y, c_z)} = \left( P^{(c_z)} \otimes P^{(c_y)} \otimes P^{(c_x)} \right)^T $$
$$ \mathbf{R}^{(c_x, c_y, c_z)} = (P^{(c_z)})^T \otimes (P^{(c_y)})^T \otimes (P^{(c_x)})^T $$
$$ \mathbf{R}^{(c_x, c_y, c_z)} = R^{(c_z)} \otimes R^{(c_y)} \otimes R^{(c_x)} $$

If we have 8 local residual tensors $r_h^{(c_x, c_y, c_z)}$ on the 8 children, the total restricted residual on the parent is the sum of the restrictions from all children:
$$ r_H = \sum_{c_x=0}^1 \sum_{c_y=0}^1 \sum_{c_z=0}^1 \left( R^{(c_z)} \otimes R^{(c_y)} \otimes R^{(c_x)} \right) r_h^{(c_x, c_y, c_z)} $$

---

## 6. Matrix-Free Application (Einstein Summation)

Why does this matter? Because we never actually build the massive $512 \times 64$ matrix. 
To prolong a coarse tensor $U_{ijk}$ (parent) to a specific child $(c_x, c_y, c_z)$ to get the fine tensor $V_{lmn}$, we simply perform three 1D tensor contractions.

Using Einstein notation:
$$ V_{lmn} = P^{(c_z)}_{li} P^{(c_y)}_{mj} P^{(c_x)}_{nk} U_{ijk} $$

In JAX/Python, assuming we precomputed the two $N \times N$ 1D matrices `P0` and `P1`:

```python
# Example: Prolonging to the front-top-left child (cx=0, cy=1, cz=1)
# P_cx = P0
# P_cy = P1
# P_cz = P1

# U_parent is our (N, N, N) coarse grid coefficient tensor
V_child = jnp.einsum('li, mj, nk, ijk -> lmn', P1, P1, P0, U_parent)
```

**Complexity Analysis (Why we do this):**
*   **Naive 3D Matrix Vector Multiply:** Size of $\mathbf{P}$ is $N^3 \times N^3$. Cost is $\mathcal{O}(N^6)$ per child.
*   **Sum Factorization (einsum):** Applying three 1D matrices sequentially. Cost is $\mathcal{O}(N^4)$ per child. 
*   For a degree $p=3$ hex ($N=4$), this reduces the FLOPs by roughly a factor of 5 per child, while completely eliminating the memory required to store the assembled interpolation matrices.