
# Lecture 15: From Energy to the Weak Form (Gateaux Derivatives)

**Goal for this class:** Learn how to systematically transition from an arbitrary energy functional to a non-linear residual (weak form) and its corresponding linear tangent operator (Jacobian matrix) for Newton's method. 

## 1. The Gâteaux Derivative (Directional Derivative for Functionals)
In classical calculus, the directional derivative of $f(x)$ in direction $v$ is $\lim_{\epsilon \to 0} \frac{f(x + \epsilon v) - f(x)}{\epsilon}$.
In the calculus of variations (and FEM), our "variables" are entire functions (e.g., displacement fields $u(\mathbf{x})$). The Gâteaux derivative generalizes this concept. 

**Definition:** Let $\Pi(u)$ be a functional. The Gâteaux derivative of $\Pi$ at $u$ in the direction of an arbitrary test function $v$ is:
$$ D_v \Pi(u) = \left. \frac{d}{d\epsilon} \Pi(u + \epsilon v) \right|_{\epsilon=0} $$
*Intuition to write on board:* $u$ is the current state. $v = \delta u$ is an infinitesimal virtual variation.

## 2. Setting up the Non-Linear Model Problem
Let's consider a generic scalar problem defined by an energy functional $\Pi(u)$ over a domain $\Omega$:
$$ \Pi(u) = \int_{\Omega} \Psi(u, \nabla u) \, dx $$
where $\Psi$ is the energy density. To make things concrete, let's use a highly non-linear toy model. Let $\mathbf{g} = \nabla u$. 
**Model Problem:**
$$ \Psi(u, \mathbf{g}) = (\mathbf{g} \cdot \mathbf{g})^4 + u^6 + u^2(\mathbf{g} \cdot \mathbf{g}) $$
*Note: This contains high-order gradient terms, a high-order scalar term, and a cross-term.*

## 3. The First Variation: The Residual (Weak Form)
To find the equilibrium state, we minimize energy by setting the first variation to zero for all test functions $v$:
$$ \mathcal{F}(u, v) = D_v \Pi(u) = \left. \frac{d}{d\epsilon} \int_{\Omega} \Psi(u + \epsilon v, \mathbf{g} + \epsilon \nabla v) \, dx \right|_{\epsilon=0} = 0 $$

Applying the chain rule inside the integral:
$$ \mathcal{F}(u, v) = \int_{\Omega} \left[ \frac{\partial \Psi}{\partial u} v + \frac{\partial \Psi}{\partial \mathbf{g}} \cdot \nabla v \right] dx $$

**Compute the derivatives for our toy model:**
*   $\frac{\partial \Psi}{\partial u} = 6u^5 + 2u(\mathbf{g} \cdot \mathbf{g})$
*   $\frac{\partial \Psi}{\partial \mathbf{g}} = 4(\mathbf{g} \cdot \mathbf{g})^3 (2\mathbf{g}) + u^2(2\mathbf{g}) = 8(\mathbf{g} \cdot \mathbf{g})^3\mathbf{g} + 2u^2\mathbf{g}$

*(Plug these back into $\mathcal{F}$ to get the explicit non-linear weak form that evaluates the Residual vector).*

## 4. The Second Variation: The Tangent Operator (Jacobian)
To solve $\mathcal{F}(u, v) = 0$ via Newton's method, we need the Jacobian. If $u$ is our current guess and $\Delta u$ is the unknown correction, the linear Newton step is:
$$ \mathcal{F}(u, v) + \mathcal{K}(u; \Delta u, v) = 0 $$
where $\mathcal{K}$ is the Gâteaux derivative of the residual in the direction of the increment $\Delta u$:
$$ \mathcal{K}(u; \Delta u, v) = D_{\Delta u} \mathcal{F}(u, v) $$

Apply the Gâteaux derivative to our residual integral using the multi-variable chain rule. The variations $v$ and $\nabla v$ are fixed directions, so the derivative only hits the $\Psi$ terms:
$$ \mathcal{K}(u; \Delta u, v) = \int_{\Omega} \left[ v \left( \frac{\partial^2 \Psi}{\partial u^2} \Delta u + \frac{\partial^2 \Psi}{\partial u \partial \mathbf{g}} \cdot \nabla \Delta u \right) + \nabla v \cdot \left( \frac{\partial^2 \Psi}{\partial \mathbf{g} \partial u} \Delta u + \frac{\partial^2 \Psi}{\partial \mathbf{g} \partial \mathbf{g}} \nabla \Delta u \right) \right] dx $$

**Block-Matrix Representation for FEM Assembly:**
$$ \mathcal{K} = \int_{\Omega} \begin{bmatrix} v \\ \nabla v \end{bmatrix}^T \begin{bmatrix} \frac{\partial^2 \Psi}{\partial u^2} & \frac{\partial^2 \Psi}{\partial u \partial \mathbf{g}} \\ \frac{\partial^2 \Psi}{\partial \mathbf{g} \partial u} & \frac{\partial^2 \Psi}{\partial \mathbf{g} \partial \mathbf{g}} \end{bmatrix} \begin{bmatrix} \Delta u \\ \nabla \Delta u \end{bmatrix} dx $$

**Compute the 4 Hessian Blocks for our toy model:**
1.  **Scalar block ($u$-$u$):**
    $$ \frac{\partial^2 \Psi}{\partial u^2} = \frac{\partial}{\partial u} \left[ 6u^5 + 2u(\mathbf{g} \cdot \mathbf{g}) \right] = 30u^4 + 2(\mathbf{g} \cdot \mathbf{g}) $$
2.  **Vector block ($u$-$\mathbf{g}$):**
    $$ \frac{\partial^2 \Psi}{\partial u \partial \mathbf{g}} = \frac{\partial}{\partial u} \left[ 8(\mathbf{g} \cdot \mathbf{g})^3\mathbf{g} + 2u^2\mathbf{g} \right] = 4u\mathbf{g} $$
3.  **Vector block ($\mathbf{g}$-$u$):** (Symmetric check)
    $$ \frac{\partial^2 \Psi}{\partial \mathbf{g} \partial u} = \frac{\partial}{\partial \mathbf{g}} \left[ 6u^5 + 2u(\mathbf{g} \cdot \mathbf{g}) \right] = 4u\mathbf{g} $$
4.  **Tensor block ($\mathbf{g}$-$\mathbf{g}$):**
    We need the derivative of a vector w.r.t a vector, yielding a 2nd-order tensor. *Tip: Write it in index notation on the board to avoid confusion!*
    $$ \frac{\partial}{\partial g_j} \left[ 8(g_k g_k)^3 g_i + 2u^2 g_i \right] $$
    Apply product rule to the first term:
    $$ = 8 \left[ 3(g_k g_k)^2 (2g_j) g_i + (g_k g_k)^3 \delta_{ij} \right] + 2u^2 \delta_{ij} $$
    Convert back to tensor/vector notation:
    $$ \frac{\partial^2 \Psi}{\partial \mathbf{g} \partial \mathbf{g}} = 48 (\mathbf{g} \cdot \mathbf{g})^2 (\mathbf{g} \otimes \mathbf{g}) + \left[ 8(\mathbf{g} \cdot \mathbf{g})^3 + 2u^2 \right] \mathbf{I} $$

