# Lecture 14 Notes: Solid Mechanics, Finite Strain Kinematics, and Stress

## 1. Kinematics of Deformation

### Definitions: Configurations and the Deformation Map
*   **Reference Configuration ($\Omega_0$):** The undeformed state of the solid body at time $t=0$. A material point is denoted by the vector $\mathbf{X}$ (coordinates $X_J$).
*   **Deformed (Spatial) Configuration ($\Omega_t$):** The state of the body at current time $t$. The position of the material point is denoted by $\mathbf{x}$ (coordinates $x_i$).
*   **Deformation Map ($\boldsymbol{\varphi}$):** A sufficiently smooth, invertible mapping that tracks a material point over time:
    $$ \mathbf{x} = \boldsymbol{\varphi}(\mathbf{X}, t) $$
*   **Displacement ($\mathbf{u}$):** The difference between current and reference positions:
    $$ \mathbf{u}(\mathbf{X}, t) = \mathbf{x} - \mathbf{X} $$

### Definition: The Deformation Gradient ($\mathbf{F}$)
The deformation gradient is the fundamental two-point tensor (mapping vectors from the reference tangent space to the deformed tangent space):
$$ \mathbf{F} = \nabla_0 \boldsymbol{\varphi} \quad \iff \quad F_{iJ} = \frac{\partial x_i}{\partial X_J} $$
In terms of displacement (as evaluated in FEM):
$$ \mathbf{F} = \mathbf{I} + \nabla_0 \mathbf{u} $$

**Properties of $\mathbf{F}$:**
1.  Maps material line elements to spatial line elements: $d\mathbf{x} = \mathbf{F} \, d\mathbf{X}$.
2.  **Jacobian ($J$):** Measures local volume change. $dv = J \, dV$, where $J = \det(\mathbf{F})$. Physical admissibility strictly requires $J > 0$.

---

## 2. Polar Decomposition, $\mathbf{C}$, and Invariants

### Polar Decomposition
The tensor $\mathbf{F}$ mixes **stretching** (generates stress) and **rigid body rotation** (zero stress). By the Polar Decomposition theorem, any invertible tensor $\mathbf{F}$ with $J > 0$ decomposes uniquely as:
$$ \mathbf{F} = \mathbf{R} \mathbf{U} = \mathbf{V} \mathbf{R} $$
*   $\mathbf{R}$: Orthogonal rotation tensor ($\mathbf{R}^T \mathbf{R} = \mathbf{I}$, $\det \mathbf{R} = 1$).
*   $\mathbf{U}$: Right Stretch Tensor (Symmetric Positive Definite).

### Definition: Right Cauchy-Green Deformation Tensor ($\mathbf{C}$)
Computing $\mathbf{U}$ requires an expensive eigenvalue decomposition. Instead, we use $\mathbf{C}$:
$$ \mathbf{C} = \mathbf{F}^T \mathbf{F} $$
*Proof that $\mathbf{C}$ isolates stretch:*
$$ \mathbf{C} = (\mathbf{R}\mathbf{U})^T (\mathbf{R}\mathbf{U}) = \mathbf{U}^T \mathbf{R}^T \mathbf{R} \mathbf{U} = \mathbf{U}^T \mathbf{I} \mathbf{U} = \mathbf{U}^2 $$
$\mathbf{C}$ operates entirely in the reference configuration ($C_{IJ} = F_{kI} F_{kJ}$). If the body undergoes rigid motion ($\mathbf{F} = \mathbf{R}$), then $\mathbf{C} = \mathbf{I}$.

### Eigenvalues and Principal Invariants of $\mathbf{C}$
Because $\mathbf{C}$ is symmetric and positive definite, it has three real, positive eigenvalues: $\lambda_1^2, \lambda_2^2, \lambda_3^3$. The values $\lambda_i$ are the *principal stretches*.

The characteristic equation for $\mathbf{C}$ is:
$$ \det(\mathbf{C} - \lambda^2 \mathbf{I}) = -(\lambda^2)^3 + I_1 (\lambda^2)^2 - I_2 (\lambda^2) + I_3 = 0 $$

The coefficients $I_1, I_2, I_3$ are the **Principal Invariants** of $\mathbf{C}$. They do not change if the coordinate system is rotated:
1.  **First Invariant (Trace):**
    $$ I_1 = \text{tr}(\mathbf{C}) = C_{II} = \lambda_1^2 + \lambda_2^2 + \lambda_3^2 $$
2.  **Second Invariant:**
    $$ I_2 = \frac{1}{2} \left[ (\text{tr}\mathbf{C})^2 - \text{tr}(\mathbf{C}^2) \right] = \lambda_1^2\lambda_2^2 + \lambda_2^2\lambda_3^2 + \lambda_3^2\lambda_1^2 $$
3.  **Third Invariant (Determinant):**
    $$ I_3 = \det(\mathbf{C}) = J^2 = \lambda_1^2\lambda_2^2\lambda_3^2 $$

**Link to Hyperelasticity:** For isotropic materials, the strain energy density $W$ (energy per unit reference volume) cannot depend on arbitrary coordinate choices. Therefore, it is formulated strictly as a function of the invariants:
$$ W(\mathbf{F}) = \hat{W}(I_1, I_2, I_3) $$
*(Example: Neo-Hookean material uses $W = \frac{\mu}{2}(I_1 - 3) - \mu \ln J + \frac{\lambda}{2}(\ln J)^2$.)*

---

## 3. Cauchy's Stress Theorem

We need to formalize the concept of internal forces. Let $\mathbf{t}(\mathbf{x}, t, \mathbf{n})$ be the traction vector (force per unit area) acting on a surface with outward normal $\mathbf{n}$. 

**Theorem:** There exists a rank-2 spatial tensor field $\boldsymbol{\sigma}$ such that $\mathbf{t} = \boldsymbol{\sigma} \mathbf{n}$.

### The 2D Cauchy Wedge 

https://en.wikipedia.org/wiki/Cauchy_stress_tensor#Cauchy's_stress_theorem%E2%80%94stress_tensor

**1. Draw the geometry:**
*   Draw a 2D Cartesian coordinate system ($x_1, x_2$).
*   Draw an infinitesimal right triangle (a wedge) with its right angle at the origin.
*   Let the hypotenuse have length $\Delta s$ and outward normal vector $\mathbf{n} = (n_1, n_2)$.
*   Let the vertical face have length $\Delta x_2$ and normal $-\mathbf{e}_1 = (-1, 0)$.
*   Let the horizontal face have length $\Delta x_1$ and normal $-\mathbf{e}_2 = (0, -1)$.

**2. Relate the areas (lengths in 2D):**
By basic trigonometry, the lengths of the legs are projections of the hypotenuse:
$$ \Delta x_2 = \Delta s \, n_1 $$
$$ \Delta x_1 = \Delta s \, n_2 $$

**3. Apply Newton's Second Law to the wedge:**
Let $\rho$ be density, $\mathbf{b}$ be body force, and $\mathbf{a}$ be acceleration. Summing the forces on the three faces and the body:
$$ \mathbf{t}(\mathbf{n}) \Delta s + \mathbf{t}(-\mathbf{e}_1) \Delta x_2 + \mathbf{t}(-\mathbf{e}_2) \Delta x_1 + \rho \mathbf{b} \left(\frac{1}{2}\Delta x_1 \Delta x_2\right) = \rho \mathbf{a} \left(\frac{1}{2}\Delta x_1 \Delta x_2\right) $$

**4. The limit as the wedge shrinks:**
Divide the entire equation by the hypotenuse length $\Delta s$:
$$ \mathbf{t}(\mathbf{n}) + \mathbf{t}(-\mathbf{e}_1) n_1 + \mathbf{t}(-\mathbf{e}_2) n_2 + \frac{1}{2}\rho (\mathbf{b} - \mathbf{a}) \frac{\Delta x_1 \Delta x_2}{\Delta s} = 0 $$
Note that $\frac{\Delta x_1 \Delta x_2}{\Delta s} = \Delta s (n_1 n_2)$. As we shrink the wedge to a point ($\Delta s \to 0$), the volume terms (body force and inertia) vanish faster than the surface area terms. We are left with:
$$ \mathbf{t}(\mathbf{n}) = -\mathbf{t}(-\mathbf{e}_1) n_1 - \mathbf{t}(-\mathbf{e}_2) n_2 $$

**5. Newton's Third Law and the Stress Tensor:**
By applying the same limit process to a vanishingly thin rectangle, we can prove $\mathbf{t}(-\mathbf{n}) = -\mathbf{t}(\mathbf{n})$. Therefore:
$$ \mathbf{t}(\mathbf{n}) = \mathbf{t}(\mathbf{e}_1) n_1 + \mathbf{t}(\mathbf{e}_2) n_2 $$
Define the Cauchy stress tensor $\boldsymbol{\sigma}$ such that its columns are the traction vectors on the coordinate planes ($\sigma_{ij}$ is the $i$-th component of traction on the face with normal $\mathbf{e}_j$). In matrix form:
$$ \mathbf{t}(\mathbf{n}) = \begin{bmatrix} \mathbf{t}(\mathbf{e}_1) & \mathbf{t}(\mathbf{e}_2) \end{bmatrix} \begin{bmatrix} n_1 \\ n_2 \end{bmatrix} = \boldsymbol{\sigma} \mathbf{n} $$
*(Index notation: $t_i = \sigma_{ij} n_j$)*.

---

## 4. The Cauchy Momentum Equation

Consider an arbitrary control volume $\Omega_t$ in the deformed configuration, boundary $\partial \Omega_t$.
The balance of linear momentum (Rate of change of momentum = Total forces):
$$ \frac{d}{dt} \int_{\Omega_t} \rho \mathbf{v} \, dv = \int_{\partial \Omega_t} \mathbf{t} \, da + \int_{\Omega_t} \rho \mathbf{b} \, dv $$

**Step 1:** Substitute Cauchy's Stress Theorem ($\mathbf{t} = \boldsymbol{\sigma} \mathbf{n}$):
$$ \int_{\partial \Omega_t} \boldsymbol{\sigma} \mathbf{n} \, da = \int_{\Omega_t} \nabla \cdot \boldsymbol{\sigma} \, dv \quad \text{(by Divergence Theorem)} $$

**Step 2:** Assuming conservation of mass, the material time derivative acts only on velocity:
$$ \int_{\Omega_t} \rho \dot{\mathbf{v}} \, dv = \int_{\Omega_t} \nabla \cdot \boldsymbol{\sigma} \, dv + \int_{\Omega_t} \rho \mathbf{b} \, dv $$

**Step 3:** Collect all terms:
$$ \int_{\Omega_t} \left( \rho \dot{\mathbf{v}} - \nabla \cdot \boldsymbol{\sigma} - \rho \mathbf{b} \right) dv = 0 $$
Because $\Omega_t$ is arbitrary, the integrand must be identically zero everywhere (Localization Theorem). 

**Result (Strong Form):**
$$ \rho \dot{\mathbf{v}} = \nabla \cdot \boldsymbol{\sigma} + \rho \mathbf{b} \quad \text{or in index notation} \quad \rho \dot{v}_i = \partial_j \sigma_{ij} + \rho b_i $$

---

## 5. Symmetry of the Stress Tensor

Why must $\boldsymbol{\sigma} = \boldsymbol{\sigma}^T$? This arises from the **Conservation of Angular Momentum**. The rate of change of angular momentum equals applied torques:
$$ \frac{d}{dt} \int_{\Omega_t} \mathbf{x} \times (\rho \mathbf{v}) \, dv = \int_{\partial \Omega_t} \mathbf{x} \times \mathbf{t} \, da + \int_{\Omega_t} \mathbf{x} \times (\rho \mathbf{b}) \, dv $$

**Step 1: Simplify LHS**
Using conservation of mass, the time derivative goes inside. By the product rule: $\frac{d}{dt}(\mathbf{x} \times \mathbf{v}) = \dot{\mathbf{x}} \times \mathbf{v} + \mathbf{x} \times \dot{\mathbf{v}}$. Since $\dot{\mathbf{x}} = \mathbf{v}$ and $\mathbf{v} \times \mathbf{v} = 0$, the LHS is:
$$ \int_{\Omega_t} \mathbf{x} \times \rho \dot{\mathbf{v}} \, dv $$

**Step 2: Expand Traction Term (using Levi-Civita $\epsilon_{ijk}$)**
Write the $i$-th component of the surface integral:
$$ \int_{\partial \Omega_t} \epsilon_{ijk} x_j t_k \, da = \int_{\partial \Omega_t} \epsilon_{ijk} x_j (\sigma_{kl} n_l) \, da $$
Apply the Divergence Theorem ($n_l da \to \partial_l dv$):
$$ \int_{\Omega_t} \partial_l (\epsilon_{ijk} x_j \sigma_{kl}) \, dv $$
Apply the product rule: $\partial_l (x_j \sigma_{kl}) = (\partial_l x_j) \sigma_{kl} + x_j (\partial_l \sigma_{kl})$. Since $\partial_l x_j = \delta_{jl}$:
$$ = \delta_{jl} \sigma_{kl} + x_j \partial_l \sigma_{kl} = \sigma_{kj} + x_j \partial_l \sigma_{kl} $$
Multiply by $\epsilon_{ijk}$:
$$ \int_{\Omega_t} \left( \epsilon_{ijk} \sigma_{kj} + \epsilon_{ijk} x_j \partial_l \sigma_{kl} \right) dv $$

**Step 3: Assemble and Cancel**
Substitute everything back into the angular momentum equation:
$$ \int_{\Omega_t} \epsilon_{ijk} x_j (\rho \dot{v}_k) \, dv = \int_{\Omega_t} \epsilon_{ijk} \sigma_{kj} \, dv + \int_{\Omega_t} \epsilon_{ijk} x_j (\partial_l \sigma_{kl}) \, dv + \int_{\Omega_t} \epsilon_{ijk} x_j (\rho b_k) \, dv $$
Group terms with $\epsilon_{ijk} x_j$:
$$ \int_{\Omega_t} \epsilon_{ijk} x_j \underbrace{\left[ \rho \dot{v}_k - \partial_l \sigma_{kl} - \rho b_k \right]}_{= 0 \text{ (by Linear Momentum)}} dv = \int_{\Omega_t} \epsilon_{ijk} \sigma_{kj} \, dv $$

**Step 4: Conclusion**
We are left with $\int_{\Omega_t} \epsilon_{ijk} \sigma_{kj} \, dv = 0$. Since the volume is arbitrary:
$$ \epsilon_{ijk} \sigma_{kj} = 0 $$
For $i=1$: $\epsilon_{123}\sigma_{32} + \epsilon_{132}\sigma_{23} = \sigma_{32} - \sigma_{23} = 0 \implies \sigma_{32} = \sigma_{23}$. This holds for all components.
$$ \boldsymbol{\sigma} = \boldsymbol{\sigma}^T $$