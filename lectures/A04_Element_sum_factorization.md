

# Lecture 4: Tensor Product Elements and Sum Factorization

Today, we apply our tensor contraction machinery to the heart of Matrix-Free FEM: evaluating the Laplace operator ($\int \nabla u \cdot \nabla v \, dx$) on a single 3D hexahedral cell.

## 1. Tensor Product Shape Functions

We map a physical hexahedral cell to a reference cube $\hat{K} = [-1, 1]^3$ with local coordinates $(\xi, \eta, \zeta)$.
Let $\phi_i(x)$ be our standard 1D Lagrange polynomial basis functions (where $i = 1, \dots, N$).

On the 3D reference element, the shape function associated with the 3D local node $(i,j,k)$ is the tensor product of the 1D shape functions:
$$ \Phi_{i,j,k}(\xi, \eta, \zeta) = \phi_i(\xi) \phi_j(\eta) \phi_k(\zeta) $$
where $i, j, k \in \{1, \dots, N\}$. 

A discrete field (like our solution) on this element is:
$$ u(\xi, \eta, \zeta) = \Phi_{i,j,k}(\xi, \eta, \zeta) U_{ijk} $$
*(Einstein summation over $i, j, k$ is implied!)*

## 2. Tensor Product Quadrature

To integrate over $\hat{K}$, we use a 1D Gaussian quadrature rule defined by points $\xi_q$ and weights $w_q$ (for $q = 1, \dots, Q$).

The 3D quadrature rule is simply the tensor product of the 1D rule:

* **3D Quadrature Points:** $X_{q,r,s} = (\xi_q, \xi_r, \xi_s)$
* **3D Quadrature Weights:** $W_{q,r,s} = w_q w_r w_s$

## 3. Evaluation of Values (Interpolation to Quadrature)

We want to evaluate our discrete field $U$ at all 3D quadrature points. 
Let $U^{qp}_{qrs}$ be the value of the field at quadrature point $X_{q,r,s}$.

**Define the 1D Basis Matrix ($B$):**
Let $\mathbf{B} \in \mathbb{R}^{Q \times N}$ be the 1D interpolation matrix. It evaluates the $i$-th basis function at the $q$-th quadrature point:
$$ B_{qi} = \phi_i(\xi_q) $$

Let's evaluate $U$ at a specific point $X_{q,r,s}$:
$$ U^{qp}_{qrs} = \phi_i(\xi_q) \phi_j(\xi_r) \phi_k(\xi_s) U_{ijk} $$

Substituting our matrix definition, we get a perfect tensor contraction:
$$ U^{qp}_{qrs} = B_{qi} B_{rj} B_{sk} U_{ijk} $$
> Apply $B$ to the $z$-direction ($k \to s$), then $y$-direction ($j \to r$), then $x$-direction ($i \to q$).

## 4. Integration (The Transpose Trick)

Suppose we have some arbitrary function evaluated at the quadrature points, $F_{qrs}$, and we want to compute its weak integral against all test functions $\Phi_{ijk}$:
$$ I_{ijk} = \int_{\hat{K}} F(\xi, \eta, \zeta) \Phi_{ijk} \, d\hat{K} \approx \sum_{q,r,s} F_{qrs} \Phi_{ijk}(\xi_q, \xi_r, \xi_s) W_{qrs} $$

Let's absorb the weights into our function for simplicity: $\hat{F}_{qrs} = F_{qrs} W_{qrs}$.
Now write the sum using our 1D matrix $B$:
$$ I_{ijk} = \sum_{q,r,s} \hat{F}_{qrs} B_{qi} B_{rj} B_{sk} $$

**The Transpose Observation:**
In Einstein notation, we drop the $\sum$ and write:
$$ I_{ijk} = B_{qi} B_{rj} B_{sk} \hat{F}_{qrs} $$
Notice that we are summing over $q, r, s$ (the quadrature indices, which are the *rows* of $B$). 
In standard matrix algebra, summing over the rows of a matrix $B$ is equivalent to multiplying by its transpose $B^T$:
$$ \sum_q B_{qi} (\dots) = \sum_q (B^T)_{iq} (\dots) $$

**Takeaway:** Interpolation maps DoFs to Quad points using $\mathbf{B}$. Integration maps Quad points back to DoFs using $\mathbf{B^T}$.

## 5. Evaluation of the Gradient

To compute the Laplace operator, we need the gradient of $U$. 

**Define the 1D Derivative Matrix ($D$):**
Let $\mathbf{D} \in \mathbb{R}^{Q \times N}$ be the matrix of 1D basis derivatives evaluated at quadrature points:
$$ D_{qi} = \frac{d\phi_i}{d\xi}\Big|_{\xi_q} $$

The gradient in the **reference element** has three components: $\hat{\nabla} U = [\partial_\xi U, \partial_\eta U, \partial_\zeta U]^T$.
By differentiating the tensor product $\Phi_{i,j,k}$, the derivative hits only one 1D function at a time:

1. $\partial_\xi U_{qrs} = D_{qi} B_{rj} B_{sk} U_{ijk}$
2. $\partial_\eta U_{qrs} = B_{qi} D_{rj} B_{sk} U_{ijk}$
3. $\partial_\zeta U_{qrs} = B_{qi} B_{rj} D_{sk} U_{ijk}$

To get the gradient in **physical space**, we apply the inverse Jacobian of the element mapping, $J^{-1}_{qrs} \in \mathbb{R}^{3 \times 3}$:
$$ \nabla U_{qrs} = J_{qrs}^{-T} \begin{bmatrix} \partial_\xi U_{qrs} \\ \partial_\eta U_{qrs} \\ \partial_\zeta U_{qrs} \end{bmatrix} $$

## 6. Integration of the Laplace Operator

The weak form of the Laplace operator on one element is $\int_K \nabla v \cdot \nabla u \, dx$.
Mapped to the reference element, this becomes:
$$ \int_{\hat{K}} \hat{\nabla} v^T (J^{-1} J^{-T}) \hat{\nabla} u \, |J| \, d\hat{K} $$

Let's compute the action of this matrix-free operator on a vector $U$, yielding an output vector $V_{ijk}$:

**Step A: Interpolate Gradients (Forward)**
Compute the reference gradients: $\hat{\nabla} U_{qrs} = [\partial_\xi U, \partial_\eta U, \partial_\zeta U]^T$ using $B$ and $D$ as shown in Section 5.

**Step B: The Quadrature Loop (Apply Physics/Geometry)**
At every independent quadrature point $(q,r,s)$, we multiply by the geometry and weights. We define the "flux" vector $\hat{G}_{qrs}$:
$$ \hat{G}_{qrs} = W_{qrs} |J_{qrs}| (J_{qrs}^{-1} J_{qrs}^{-T}) \hat{\nabla} U_{qrs} $$
This vector has three components: $[\hat{G}^\xi_{qrs}, \hat{G}^\eta_{qrs}, \hat{G}^\zeta_{qrs}]^T$. 

**Step C: Integrate by Gradient (Backward/Transpose)**
Now we integrate these fluxes against the gradients of the test functions. Because the $\xi$-flux is paired with the $\xi$-derivative of the test function, we integrate it back using $D^T$ in the $\xi$-direction! 

The final output DoFs $V_{ijk}$ are the sum of the three integrated fluxes:
$$ V_{ijk} = \underbrace{D_{qi} B_{rj} B_{sk} \hat{G}^\xi_{qrs}}_{\text{integrate } \xi\text{-flux}} + \underbrace{B_{qi} D_{rj} B_{sk} \hat{G}^\eta_{qrs}}_{\text{integrate } \eta\text{-flux}} + \underbrace{B_{qi} B_{rj} D_{sk} \hat{G}^\zeta_{qrs}}_{\text{integrate } \zeta\text{-flux}} $$
*(Remember: Summing over $q,r,s$ implies we are inherently applying $B^T$ and $D^T$)*

---

## Conclusion: The Universal Matrix-Free Blueprint

This derivation reveals the universal three-step blueprint for Matrix-Free evaluation of *any* PDE on a tensor-product cell:

1. **Interpolation (Forward Contractions):** 
   * Use $B$ and $D$ matrices to interpolate DoF values and reference gradients to quadrature points.
2. **Quadrature Loop (Pointwise Physics):** 
   * A completely local, pointwise operation. Apply weights, Jacobian transformations, material parameters, and non-linear constitutive laws (e.g., hyperelasticity). We output physical "fluxes".
3. **Integration (Backward Contractions):** 
   * Use the transpose matrices $B^T$ and $D^T$ to integrate the fluxes back to the DoFs.
   * **Crucial Rule:** We must separate the parts multiplied by the gradient vs. the parts multiplied by the value. If a flux acts on $\nabla v$, it integrates back via $D^T$. If a source term acts on $v$, it integrates back via $B^T$.