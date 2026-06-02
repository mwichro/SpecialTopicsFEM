
# Lecture 16: The Non-Linear Problem in Finite-Strain Elasticity
*(Ref: Section 2.1 of Wichrowski et al., https://arxiv.org/html/2505.15535v1 (publ) )*

**Goal for this class:** We will formulate the continuous weak form and tangent operator for finite-strain elasticity. 

---

## 1. Problem Formulation & Kinematics Recap
Let a hyperelastic body occupy domain $\Omega \subset \mathbb{R}^d$ in the **reference configuration**, with boundaries partitioned into a fixed Dirichlet part $\Gamma_{\rm D}$ and a Neumann part $\Gamma_{\rm N}$ subject to conservative traction $\mathbf{T}^\ast$. 
The deformation is governed by the mapping $\bm{\varphi}: \Omega \to \omega$ to the **current configuration** $\omega$.

*   **Displacement:** $\mathbf{u}(\mathbf{X}) = \bm{\varphi}(\mathbf{X}) - \mathbf{X}$
*   **Deformation Gradient:** $\mathbf{F} := \text{Grad} \bm{\varphi} = \mathbf{I} + \text{Grad} \mathbf{u}$ 
    *(Note: We use $\text{Grad}$ for derivatives with respect to reference coordinates $\mathbf{X}$, and $\text{grad}$ for spatial coordinates $\mathbf{x}$.)*
*   **Jacobian:** $J := \det\mathbf{F} > 0$

The material's physics are encoded entirely in a strain energy density function $\Psi(\mathbf{F})$. The total potential energy is:
$$ \mathcal{E}(\mathbf{u}) := \int_\Omega \Psi(\mathbf{F}) \; dV - \int_{\Gamma_{\rm N}} \mathbf{T}^\ast \cdot \mathbf{u} \; dS $$

---

## 2. The Weak Form (Residual) in the Reference Configuration
To find equilibrium, we require the first variation (Gâteaux derivative) of the energy in the direction of an admissible virtual displacement $\delta\mathbf{u}$ to be zero:
$$ \mathcal{F}(\mathbf{u},\delta\mathbf{u}) := \text{D}_{\delta\mathbf{u}} \mathcal{E} = 0 $$

Applying the chain rule (as we practiced in Lecture 15):
$$ \text{D}_{\delta\mathbf{u}} \Psi(\mathbf{F}) = \frac{\partial\Psi}{\partial \text{Grad}\mathbf{u}} : \text{Grad}\delta\mathbf{u} $$

We define the **First Piola-Kirchhoff (PK1) stress tensor**, $\mathbf{P}$:
$$ \mathbf{P} := \frac{\partial\Psi}{\partial\mathbf{F}} $$
This yields our fundamental reference weak form (Virtual Work):
$$ \mathcal{F}(\mathbf{u},\delta\mathbf{u}) = \int_\Omega \mathbf{P} : \text{Grad}\delta\mathbf{u} \; dV - \int_{\Gamma_{\rm N}} \mathbf{T}^\ast \cdot \delta\mathbf{u} \; dS = 0 \quad \forall \, \delta\mathbf{u} $$

*Point out that $\mathbf{P}$ is generally an asymmetric, two-point tensor. This asymmetry is computationally annoying. We want to expose the hidden physics/symmetry to make our matrix-free solvers faster.*

---

## 3. Push-Forward to the Current Configuration (Blackboard Derivation)

The paper states that the volume integral $\int_\Omega \mathbf{P} : \text{Grad}\delta\mathbf{u} \, dV$ can be equivalently expressed using spatial stress measures. **Let's prove this step-by-step.**

**Step 1: The Spatial Gradient via Chain Rule**
We need to convert the reference gradient $\text{Grad}\delta\mathbf{u}$ (derivatives w.r.t $X_J$) into the spatial gradient $\text{grad}\delta\mathbf{u}$ (derivatives w.r.t $x_k$).
Using index notation:
$$ (\text{Grad}\delta\mathbf{u})_{iJ} = \frac{\partial (\delta u_i)}{\partial X_J} = \frac{\partial (\delta u_i)}{\partial x_k} \frac{\partial x_k}{\partial X_J} = (\text{grad}\delta\mathbf{u})_{ik} F_{kJ} $$
In tensor notation: $\text{Grad}\delta\mathbf{u} = (\text{grad}\delta\mathbf{u}) \mathbf{F}$

**Step 2: Tensor Contraction and Kirchhoff Stress ($\bm{\tau}$)**
Substitute this into the stress power term:
$$ \mathbf{P} : \text{Grad}\delta\mathbf{u} = P_{iJ} (\text{Grad}\delta\mathbf{u})_{iJ} = P_{iJ} (\text{grad}\delta\mathbf{u})_{ik} F_{kJ} $$
Because scalar multiplication commutes, we can re-associate the terms:
$$ = \big( P_{iJ} F_{kJ} \big) (\text{grad}\delta\mathbf{u})_{ik} = (\mathbf{P}\mathbf{F}^{\rm T})_{ik} (\text{grad}\delta\mathbf{u})_{ik} $$
We define the **Kirchhoff stress tensor**:
$$ \bm{\tau} := \mathbf{P}\mathbf{F}^{\rm T} $$
So the integrand becomes $\bm{\tau} : \text{grad}\delta\mathbf{u}$.

**Step 3: Why only the Symmetric Gradient?**
Unlike $\mathbf{P}$, the Kirchhoff stress $\bm{\tau}$ is symmetric ($\tau_{ik} = \tau_{ki}$), which stems from the conservation of angular momentum. 
Any rank-2 tensor (like $\text{grad}\delta\mathbf{u}$) can be additively decomposed into a symmetric part ($\text{grad}^s$) and a skew-symmetric part ($\text{grad}^{skew}$):
$$ \text{grad}\delta\mathbf{u} = \text{grad}^s\delta\mathbf{u} + \text{grad}^{skew}\delta\mathbf{u} $$
When you double-contract a symmetric tensor ($\bm{\tau}$) with a skew-symmetric tensor ($\text{grad}^{skew}$), the result is identically zero!
*Proof to show if asked:* $\tau_{ik} W_{ik} = \tau_{ki} (-W_{ki}) = -\tau_{ki} W_{ki}$. Thus it equals its own negative, so it must be 0.
Therefore:
$$ \bm{\tau} : \text{grad}\delta\mathbf{u} = \bm{\tau} : \text{grad}^s\delta\mathbf{u} $$

**Step 4: Change of Integration Variables (Cauchy Stress $\bm{\sigma}$)**
We now switch the domain of integration from $\Omega$ to $\omega$. 
Recall the Jacobian $J = dv / dV \implies dV = J^{-1} dv$.
$$ \int_\Omega \bm{\tau} : \text{grad}^s\delta\mathbf{u} \; dV = \int_\omega (J^{-1}\bm{\tau}) : \text{grad}^s\delta\mathbf{u} \; dv $$
We define the true **Cauchy stress tensor**: $\bm{\sigma} := J^{-1}\bm{\tau} = J^{-1} \mathbf{P}\mathbf{F}^{\rm T}$.

**Result:**
We have successfully derived the equivalent formulations of the residual:
$$ \int_\Omega \mathbf{P} : \text{Grad}\delta\mathbf{u} \; dV = \int_\Omega \bm{\tau} : \text{grad}^s \delta\mathbf{u} \; dV = \int_\omega \bm{\sigma} : \text{grad}^s \delta\mathbf{u} \; dv $$

---

## 4. The Tangent Operator (Newton's Method)
Because $\mathbf{P}$ is a non-linear function of $\mathbf{u}$, we use Newton's method. Given current guess $\bar{\mathbf{u}}$, we want to find increment $\Delta\mathbf{u}$ by linearizing $\mathcal{F}$:
$$ \mathcal{F}(\bar{\mathbf{u}} + \Delta\mathbf{u}, \delta\mathbf{u}) \approx \mathcal{F}(\bar{\mathbf{u}}, \delta\mathbf{u}) + \mathcal{K}(\bar{\mathbf{u}}; \Delta\mathbf{u}, \delta\mathbf{u}) = 0 $$

The Tangent Operator $\mathcal{K}$ is the Gâteaux derivative of the residual in the direction $\Delta\mathbf{u}$:
$$ \mathcal{K}(\bar{\mathbf{u}}; \Delta\mathbf{u}, \delta\mathbf{u}) := \text{D}_{\Delta\mathbf{u}} \mathcal{F}(\bar{\mathbf{u}}, \delta\mathbf{u}) = \int_\Omega \text{D}_{\Delta\mathbf{u}}\mathbf{P} : \text{Grad}\delta\mathbf{u} \; dV $$

By the chain rule: $\text{D}_{\Delta\mathbf{u}}\mathbf{P} = \frac{\partial\mathbf{P}}{\partial\mathbf{F}} : \text{Grad}\Delta\mathbf{u}$. We define the **First Material Tangent Stiffness Tensor** ($\mathbb{L}$):
$$ \mathbb{L} := \frac{\partial\mathbf{P}}{\partial\mathbf{F}} = \frac{\partial^2\Psi}{\partial\mathbf{F}\otimes\partial\mathbf{F}} $$
This is a 4th-order tensor, giving us our reference bilinear form:
$$ \mathcal{K} = \int_\Omega \text{Grad}\Delta\mathbf{u} : \mathbb{L} : \text{Grad} \delta\mathbf{u} \; dV $$

---

## 5. Spatial Tangent Operator

Just like the stress, $\mathbb{L}$ is computationally ugly. It only has **major symmetry** ($L_{iAjB} = L_{jBiA}$). 
If we transform the entire bilinear form $\mathcal{K}$ to the current configuration $\omega$, the derivative produces **two** distinct terms (derived via rigorous push-forward of Lie derivatives, following Wriggers 2008 / Davydov 2020):

$$ \mathcal{K} = \underbrace{\int_\omega \text{grad}^s \Delta\mathbf{u} : \mathbb{c} : \text{grad}^s \delta\mathbf{u} \, dv}_{\text{Material Part}} + \underbrace{\int_\omega \text{grad} \delta\mathbf{u} : \big( \text{grad}^s \Delta\mathbf{u} \cdot \bm{\sigma} \big) \, dv}_{\text{Geometric Part}} $$

**The Breakdown:**
1.  **Geometric Part:** It arises purely because the geometry is deforming, acting on the already existing stress $\bm{\sigma}$.
2.  **Material Part ($\mathbb{c}$):** The spatial elasticity tensor $\mathbb{c}$ is the push-forward of the material elasticity tensor $\mathbb{C} = 4 \frac{\partial^2 \Psi}{\partial\mathbf{C}\otimes\partial\mathbf{C}}$.
    $$ J \mathbb{c}_{ijkl} = F_{iA}F_{jB}F_{kC}F_{lD}\mathbb{C}_{ABCD} $$

**The Computational Payoff:**
By evaluating our physics through $\mathbb{C}$ and pushing forward to $\mathbb{c}$, the spatial elasticity tensor $\mathbb{c}$ gains **both minor and major symmetries!**
$$ c_{ijkl} = c_{jikl} = c_{klij} $$
*Why do we care?* In 3D, a general 4th-order tensor has $3^4 = 81$ components. With minor and major symmetries, this drops to just **21 independent components**! Mapping symmetric tensors allows us to skip contracting nearly 75% of the data.

---

## 6. FEM Discretization
To map this to our matrix-free solver, we discretize $\Omega$ into a mesh $\mathcal{T}_h$ of hexahedral elements ($\mathbb{Q}_p$). 
We represent our virtual test functions $\delta\mathbf{u}$ by basis functions $\bm{\phi}_i$ and our unknown increment $\Delta\mathbf{u}$ by coefficients $\Delta\mathbf{U}_j$:
$$ \Delta\mathbf{u} = \sum_j \Delta\mathbf{U}_j \bm{\phi}_j $$

At each Newton step, we seek the coefficient vector $\Delta\mathbf{U}$ solving the linear system:
$$ \sum_j \mathcal{K}(\bar{\mathbf{u}}; \bm{\phi}_j, \bm{\phi}_i) \Delta\mathbf{U}_j = -\mathcal{F}(\bar{\mathbf{u}}, \bm{\phi}_i) \qquad \forall i $$




--- 



## 5. Again. Spatial Tangent Operator (The Push-Forward Derivation)

If we blindly evaluate the tangent operator in the reference configuration:
$$ \mathcal{K} = \int_\Omega \text{Grad}\Delta\mathbf{u} : \mathbb{L} : \text{Grad} \delta\mathbf{u} \; dV $$
we face a computational bottleneck. The First Material Tangent Stiffness Tensor $\mathbb{L} = \frac{\partial \mathbf{P}}{\partial \mathbf{F}}$ is a two-point tensor that only possesses "major symmetry" ($L_{iAjB} = L_{jBiA}$). It lacks minor symmetries, making it extremely expensive to evaluate and store. 

To fix this, we need to mathematically transform (push forward) this integral to the current configuration $\omega$, where physical symmetries are restored.

### Step 1: Switch to a fully symmetric reference stress ($\mathbf{S}$)
The root of the asymmetry is $\mathbf{P}$. Instead, we define the physics using the **Second Piola-Kirchhoff stress tensor**, $\mathbf{S}$.
$$ \mathbf{P} = \mathbf{F} \mathbf{S} $$
Unlike $\mathbf{P}$, $\mathbf{S}$ is fully symmetric. It is the derivative of the energy with respect to the symmetric Green-Lagrange strain tensor $\mathbf{E} = \frac{1}{2}(\mathbf{F}^T\mathbf{F} - \mathbf{I})$:
$$ \mathbf{S} = \frac{\partial \Psi}{\partial \mathbf{E}} = 2 \frac{\partial \Psi}{\partial \mathbf{C}} $$

### Step 2: Linearize $\mathbf{P}$ using the Product Rule
To find the tangent operator, we need the Gâteaux derivative of $\mathbf{P}$ in the direction of the increment $\Delta\mathbf{u}$. Applying the product rule to $\mathbf{P} = \mathbf{F}\mathbf{S}$:
$$ \text{D}_{\Delta\mathbf{u}} \mathbf{P} = (\text{D}_{\Delta\mathbf{u}} \mathbf{F}) \mathbf{S} + \mathbf{F} (\text{D}_{\Delta\mathbf{u}} \mathbf{S}) $$
Since $\mathbf{F} = \mathbf{I} + \text{Grad}\mathbf{u}$, its derivative is simply $\text{D}_{\Delta\mathbf{u}} \mathbf{F} = \text{Grad}\Delta\mathbf{u}$. Thus:
$$ \text{D}_{\Delta\mathbf{u}} \mathbf{P} = (\text{Grad}\Delta\mathbf{u}) \mathbf{S} + \mathbf{F} (\text{D}_{\Delta\mathbf{u}} \mathbf{S}) $$

Substitute this back into the weak form $\mathcal{K} = \int_\Omega (\text{D}_{\Delta\mathbf{u}} \mathbf{P}) : \text{Grad}\delta\mathbf{u} \, dV$. The integral naturally splits into two distinct parts:
$$ \mathcal{K} = \underbrace{\int_\Omega (\text{Grad}\Delta\mathbf{u}) \mathbf{S} : \text{Grad}\delta\mathbf{u} \, dV}_{\text{Geometric Part}} + \underbrace{\int_\Omega \mathbf{F} (\text{D}_{\Delta\mathbf{u}} \mathbf{S}) : \text{Grad}\delta\mathbf{u} \, dV}_{\text{Material Part}} $$

---

### Step 3: Pushing Forward the Geometric Part (Blackboard Proof)
Let's transform the first integral from $\Omega$ to $\omega$. 
Using index notation, the integrand is: $(\partial_J \Delta u_i) S_{JK} (\partial_K \delta u_i)$.
We convert the reference gradients to spatial gradients using the chain rule ($\frac{\partial}{\partial X_J} = \frac{\partial}{\partial x_m} F_{mJ}$):
$$ = (\partial_m \Delta u_i F_{mJ}) S_{JK} (\partial_n \delta u_i F_{nK}) $$
Rearrange the scalars to group the tensors together:
$$ = (\partial_n \delta u_i) (\partial_m \Delta u_i) \big( F_{mJ} S_{JK} F_{nK} \big) $$
Recall the definition of the Cauchy stress from earlier: $\bm{\sigma} = \frac{1}{J} \mathbf{F} \mathbf{S} \mathbf{F}^T$. Therefore, the bracketed term $F_{mJ} S_{JK} F_{nK}$ is exactly $J \sigma_{mn}$.
Substituting this back in gives:
$$ \int_\Omega (\partial_n \delta u_i) (\partial_m \Delta u_i) J \sigma_{mn} \, dV $$
Finally, change the integration domain ($dV = J^{-1} dv$). The $J$ cancels out entirely, leaving:
$$ \int_\omega (\text{grad} \delta\mathbf{u}) : (\text{grad} \Delta\mathbf{u} \cdot \bm{\sigma}) \, dv $$
*(Note: Depending on the specific algebraic regrouping of skew-symmetric parts with the material tensor—as done in the Davydov 2020 formulation cited in the paper—this term is often expressed using $\text{grad}^s \Delta\mathbf{u}$.)*

**Physical Meaning:** This is the "Initial Stress Stiffness". It arises purely because the geometry is deforming, acting on the already existing stress state $\bm{\sigma}$, completely independent of any change in material elasticity.

---

### Step 4: Pushing Forward the Material Part 
Now for the second integral: $\int_\Omega \mathbf{F} (\text{D}_{\Delta\mathbf{u}} \mathbf{S}) : \text{Grad}\delta\mathbf{u} \, dV$.
First, use the identity $\mathbf{A}\mathbf{B} : \mathbf{C} = \mathbf{B} : \mathbf{A}^T\mathbf{C}$ to move $\mathbf{F}$:
$$ \int_\Omega (\text{D}_{\Delta\mathbf{u}} \mathbf{S}) : (\mathbf{F}^T \text{Grad}\delta\mathbf{u}) \, dV $$

By definition, the derivative of the Second Piola-Kirchhoff stress is $\text{D}_{\Delta\mathbf{u}} \mathbf{S} = \mathbb{C} : \text{D}_{\Delta\mathbf{u}}\mathbf{E}$, where $\mathbb{C} = 4 \frac{\partial^2 \Psi}{\partial \mathbf{C} \otimes \partial \mathbf{C}}$ is the purely symmetric Material Elasticity Tensor.
Because $\mathbb{C}$ and $\mathbf{S}$ are symmetric, they only extract the symmetric part of $\mathbf{F}^T \text{Grad}\delta\mathbf{u}$, which happens to be exactly the variation of the Green-Lagrange strain, $\text{D}_{\delta\mathbf{u}}\mathbf{E}$. The material part is therefore beautifully symmetric:
$$ \int_\Omega \text{D}_{\delta\mathbf{u}}\mathbf{E} : \mathbb{C} : \text{D}_{\Delta\mathbf{u}}\mathbf{E} \, dV $$

Now, push the strain variations to the spatial configuration. $\text{D}_{\delta\mathbf{u}}\mathbf{E} = \text{sym}(\mathbf{F}^T \text{Grad}\delta\mathbf{u})$. Using the chain rule $\text{Grad} = \text{grad} \mathbf{F}$:
$$ \text{D}_{\delta\mathbf{u}}\mathbf{E} = \text{sym}(\mathbf{F}^T \text{grad}\delta\mathbf{u} \, \mathbf{F}) = \mathbf{F}^T (\text{grad}^s\delta\mathbf{u}) \mathbf{F} $$
Plug this into the integral (using index notation for clarity):
$$ \int_\Omega (F_{kI} \, \text{grad}^s\delta u_{kl} \, F_{lJ}) \, \mathbb{C}_{IJKL} \, (F_{mK} \, \text{grad}^s\Delta u_{mn} \, F_{nL}) \, dV $$
Group all the deformation gradients $\mathbf{F}$ together with the material tensor $\mathbb{C}$ to define the **Spatial Elasticity Tensor ($\mathbb{c}$)**:
$$ J \mathbb{c}_{klmn} = F_{kI} F_{lJ} F_{mK} F_{nL} \mathbb{C}_{IJKL} $$
Switch the domain to $\omega$ ($dV = J^{-1} dv$), and we obtain the final spatial material tangent:
$$ \int_\omega \text{grad}^s \delta\mathbf{u} : \mathbb{c} : \text{grad}^s \Delta\mathbf{u} \, dv $$

---

### Final Result & The Computational Payoff

Combining both parts, we arrive exactly at the formulation shown in the paper:
$$ \mathcal{K} = \underbrace{\int_\omega \text{grad}^s \Delta\mathbf{u} : \mathbb{c} : \text{grad}^s \delta\mathbf{u} \, dv}_{\text{Material Part}} + \underbrace{\int_\omega \text{grad} \delta\mathbf{u} : \big( \text{grad}^s \Delta\mathbf{u} \cdot \bm{\sigma} \big) \, dv}_{\text{Geometric Part}} $$

**Why did we go through all this algebra?**
By evaluating our physics through the symmetric $\mathbb{C}$ and pushing it forward to $\mathbb{c}$, the spatial elasticity tensor $\mathbb{c}$ directly inherits **both minor and major symmetries!**
$$ c_{ijkl} = c_{jikl} = c_{ijlk} = c_{klij} $$
In 3D, a general 4th-order tensor ($\mathbb{L}$) has $3^4 = 81$ independent components. With minor and major symmetries, $\mathbb{c}$ drops to just **21 independent components**! 

When we implement our matrix-free solver in JAX, evaluating 81 terms at millions of quadrature points would destroy our memory bandwidth. Mapping symmetric tensors allows us to skip computing and contracting nearly 75% of the data, which is the entire basis of state-of-the-art hyperelastic FEM codes.