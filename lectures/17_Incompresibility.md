

***

# Lecture 17: Linear Elasticity, Constrained Optimization, and the Stokes Problem

This lecture provides a mathematical pivot: we step back from the extreme non-linearities of finite strain to derive standard Linear Elasticity from pure energy principles (without the heavy 4th-order tensor notation). We then use the physical limit of incompressibility (volumetric locking) to naturally motivate constrained optimization, Lagrange multipliers, and finally derive the **Stokes Equations**.

---

## 1. Linear Elasticity from Energy (The Simple Way)

Instead of the complex finite-strain $\mathbf{F}$ and $\mathbf{C}$ tensors, linear elasticity assumes infinitesimal displacements. 

### Kinematics
*   **Small Strain Tensor:** $\boldsymbol{\varepsilon}(\mathbf{u}) = \frac{1}{2}(\nabla \mathbf{u} + (\nabla \mathbf{u})^T) = \nabla^s \mathbf{u}$
*   **Volumetric Strain (Dilation):** The trace of the strain tensor gives the local volume change. 
    $\text{tr}(\boldsymbol{\varepsilon}) = \varepsilon_{11} + \varepsilon_{22} + \varepsilon_{33} = \nabla \cdot \mathbf{u}$ (the divergence of displacement).

### The Energy Functional
For an isotropic, linear elastic material, the strain energy density $\Psi$ can be written using just two scalars—the **Lamé parameters**, $\mu$ (shear modulus) and $\lambda$ (first Lamé parameter):
$$ \Psi(\mathbf{u}) = \underbrace{\frac{1}{2} \lambda (\nabla \cdot \mathbf{u})^2}_{\text{Volumetric energy}} + \underbrace{\mu \boldsymbol{\varepsilon}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{u})}_{\text{Shear/Distortion energy}} $$

The Total Potential Energy (including external body forces $\mathbf{b}$ and boundary tractions $\mathbf{t}^*$) is:
$$ \Pi(\mathbf{u}) = \int_\Omega \left[ \frac{1}{2} \lambda (\nabla \cdot \mathbf{u})^2 + \mu \boldsymbol{\varepsilon}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{u}) \right] dV - \int_\Omega \mathbf{b} \cdot \mathbf{u} \, dV - \int_{\Gamma_N} \mathbf{t}^* \cdot \mathbf{u} \, dS $$

---

## 2. First Variation: From Energy to the Equation

To find equilibrium, we minimize $\Pi(\mathbf{u})$. We set the Gâteaux derivative in an arbitrary virtual direction $\mathbf{v}$ to zero: $D_{\mathbf{v}}\Pi(\mathbf{u}) = 0$.

### Step 1: The Weak Form (Blackboard Derivation)
Apply the derivative to the volumetric term (using the chain rule):
$$ \frac{d}{d\epsilon} \left[ \frac{1}{2} \lambda (\nabla \cdot (\mathbf{u} + \epsilon \mathbf{v}))^2 \right]_{\epsilon=0} = \lambda (\nabla \cdot \mathbf{u})(\nabla \cdot \mathbf{v}) $$

Apply the derivative to the shear term (note that $\boldsymbol{\varepsilon} : \boldsymbol{\varepsilon} = \varepsilon_{ij}\varepsilon_{ij}$, so the derivative brings down a factor of 2):
$$ \frac{d}{d\epsilon} \left[ \mu \boldsymbol{\varepsilon}(\mathbf{u} + \epsilon \mathbf{v}) : \boldsymbol{\varepsilon}(\mathbf{u} + \epsilon \mathbf{v}) \right]_{\epsilon=0} = 2\mu \boldsymbol{\varepsilon}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{v}) $$

Putting it together gives the **Weak Form**:
$$ \int_\Omega \left[ \lambda (\nabla \cdot \mathbf{u})(\nabla \cdot \mathbf{v}) + 2\mu \boldsymbol{\varepsilon}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{v}) \right] dV = \int_\Omega \mathbf{b} \cdot \mathbf{v} \, dV + \int_{\Gamma_N} \mathbf{t}^* \cdot \mathbf{v} \, dS $$
*(Instructor note: Point out that the bracketed term on the left is exactly the linear Cauchy stress $\boldsymbol{\sigma}(\mathbf{u})$ contracted with $\nabla \mathbf{v}$.)*

### Step 2: Integration by Parts (Extracting the Strong Form)
To find the governing PDE, we must peel the derivatives off the test function $\mathbf{v}$. 
We use the tensor identity $\nabla \cdot (\boldsymbol{\sigma} \mathbf{v}) = (\nabla \cdot \boldsymbol{\sigma}) \cdot \mathbf{v} + \boldsymbol{\sigma} : \nabla \mathbf{v}$, and apply the Divergence Theorem:

$$ \int_\Omega \boldsymbol{\sigma} : \nabla \mathbf{v} \, dV = \int_{\partial \Omega} (\boldsymbol{\sigma} \mathbf{n}) \cdot \mathbf{v} \, dS - \int_\Omega (\nabla \cdot \boldsymbol{\sigma}) \cdot \mathbf{v} \, dV $$

Substitute this back into the weak form. Assume $\mathbf{v} = 0$ on the Dirichlet boundary $\Gamma_D$, so the boundary integral only survives on $\Gamma_N$:
$$ \int_\Omega \left[ - \nabla \cdot \boldsymbol{\sigma} - \mathbf{b} \right] \cdot \mathbf{v} \, dV + \int_{\Gamma_N} \left[ \boldsymbol{\sigma}\mathbf{n} - \mathbf{t}^* \right] \cdot \mathbf{v} \, dS = 0 $$

Since this must hold for *all* $\mathbf{v}$, the bracketed terms must independently be zero.
**The Strong Form (Navier-Cauchy Equation):**
$$ \nabla \cdot \boldsymbol{\sigma} + \mathbf{b} = \mathbf{0} \quad \implies \quad \mu \Delta \mathbf{u} + (\lambda + \mu)\nabla(\nabla \cdot \mathbf{u}) + \mathbf{b} = \mathbf{0} $$

---

## 3. The Crisis of Incompressibility ($\nu \to 0.5$)

Linear elasticity works beautifully until we model rubber or fluid-like materials. 
The Lamé parameter $\lambda$ is related to Poisson's ratio $\nu$ by:
$$ \lambda = \frac{2\mu\nu}{1 - 2\nu} $$
As a material becomes perfectly incompressible (preserves volume exactly), $\nu \to 0.5$, which means **$\lambda \to \infty$**. 

Look back at the energy: $\Pi = \int \frac{1}{2} \lambda (\nabla \cdot \mathbf{u})^2 + \dots$
If $\lambda = \infty$, the energy explodes for *any* vector field where $\nabla \cdot \mathbf{u} \neq 0$. Standard standard finite elements (e.g., Q1/Q1) cannot find a piecewise polynomial field that is *exactly* divergence-free everywhere. The element becomes infinitely stiff, a phenomenon known as **Volumetric Locking**.

---

## 4. Constrained Optimization and Lagrange Multipliers

Instead of treating incompressibility as a penalty term with an infinite coefficient ($\lambda \to \infty$), we reframe the math as a **Constrained Optimization Problem**.

### The General Math Concept
Suppose we want to minimize $f(x)$ subject to the strict constraint $g(x) = 0$.
Instead of substituting, we introduce a new unknown variable $p$ (the Lagrange multiplier) and form the Lagrangian:
$$ \mathcal{L}(x, p) = f(x) - p \cdot g(x) $$
To find the constrained optimum, we find the saddle point of the Lagrangian by taking derivatives with respect to *both* variables:
1.  $\frac{\partial \mathcal{L}}{\partial x} = 0 \implies f'(x) - p \cdot g'(x) = 0$ (Optimality)
2.  $\frac{\partial \mathcal{L}}{\partial p} = 0 \implies g(x) = 0$ (Constraint recovery)

---

## 5. Deriving the Stokes Problem (Saddle Point Formulation)

Let's apply Lagrange multipliers to our continuum mechanics problem. We drop the $\lambda$ penalty term entirely and instead enforce pure incompressibility ($\nabla \cdot \mathbf{u} = 0$).
*(Semantic shift: In solids, $\mathbf{u}$ is displacement. In fluids, $\mathbf{u}$ is velocity and $\mu$ is dynamic viscosity. The math is identical. Let's use the fluid interpretation for Stokes).*

### Step 1: The Lagrangian Functional
*   **Base Energy (Viscous Dissipation):** $J(\mathbf{u}) = \int_\Omega \mu \nabla \mathbf{u} : \nabla \mathbf{u} \, dV - \int_\Omega \mathbf{b} \cdot \mathbf{u} \, dV$ *(Note: we simplify $\boldsymbol{\varepsilon}:\boldsymbol{\varepsilon}$ to $\nabla \mathbf{u} : \nabla \mathbf{u}$ for standard Stokes).*
*   **Constraint:** $\nabla \cdot \mathbf{u} = 0$ everywhere in $\Omega$.
*   **Lagrange Multiplier:** A scalar field $p(\mathbf{x})$, representing the **hydrostatic pressure**.

We construct the Mixed Lagrangian:
$$ \mathcal{L}(\mathbf{u}, p) = \int_\Omega \mu \nabla \mathbf{u} : \nabla \mathbf{u} \, dV - \int_\Omega \mathbf{b} \cdot \mathbf{u} \, dV - \int_\Omega p (\nabla \cdot \mathbf{u}) \, dV $$
*(Sign convention: We use $-p (\nabla \cdot \mathbf{u})$ so that positive pressure corresponds to physical compression).*

### Step 2: The Mixed Weak Form (Stationarity)
We take the Gâteaux derivatives with respect to both $\mathbf{u}$ (direction $\mathbf{v}$) and $p$ (direction $q$).

**1. Variation w.r.t velocity ($\mathbf{u}$):** $D_{\mathbf{v}}\mathcal{L} = 0$
$$ \int_\Omega \mu \nabla \mathbf{u} : \nabla \mathbf{v} \, dV - \int_\Omega p (\nabla \cdot \mathbf{v}) \, dV = \int_\Omega \mathbf{b} \cdot \mathbf{v} \, dV \qquad \forall \mathbf{v} $$

**2. Variation w.r.t pressure ($p$):** $D_{q}\mathcal{L} = 0$
$$ -\int_\Omega q (\nabla \cdot \mathbf{u}) \, dV = 0 \qquad \forall q $$

This system of equations is the **Weak Form of the Stokes Equations**.

### Step 3: Block Matrix Structure
If we discretize this with FEM ($\mathbf{u} \approx \sum U_j \mathbf{\phi}_j$, $p \approx \sum P_k \psi_k$), we get a massive block matrix system:
$$ \begin{bmatrix} A & B^T \\ B & 0 \end{bmatrix} \begin{bmatrix} U \\ P \end{bmatrix} = \begin{bmatrix} F \\ 0 \end{bmatrix} $$
*   $A$: Discrete Laplacian / Viscous block (SPD)
*   $B$: Discrete Divergence operator
*   $B^T$: Discrete Gradient operator
*   $0$: A zero block on the diagonal! This makes the matrix **indefinite** (a saddle point problem, not a simple minimization). 

### Step 4: The Strong Form PDE
To see the PDE we just derived, integrate the mixed weak form by parts. 
Moving the gradient from $\mathbf{v}$ onto $p$ gives $\int_\Omega p(\nabla \cdot \mathbf{v}) dV = - \int_\Omega (\nabla p) \cdot \mathbf{v} dV + \text{boundary terms}$.

This yields the strong form of the **Stokes Equations**:
$$ -\mu \Delta \mathbf{u} + \nabla p = \mathbf{b} \quad \text{(Momentum balance)} $$
$$ \nabla \cdot \mathbf{u} = 0 \quad \text{(Mass conservation / Incompressibility)} $$

---
