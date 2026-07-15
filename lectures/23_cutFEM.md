
---

# Cut Finite Element Methods (CutFEM) 

### Prerequisites:
*   Standard Finite Element Method (FEM)
*   Discontinuous Galerkin (DG) Methods, including jump operators and trace inequalities.
*   Nitsche's method for weakly enforcing Dirichlet boundary conditions.

### Lecture Outline:
1.  **Recap:** The Unstabilized CutFEM Formulation.
2.  **The Breakdown Part I: Ill-Conditioning.** A rigorous look at how small cuts lead to a singular stiffness matrix.
3.  **The Breakdown Part II: Failure of Coercivity.** A proof that the constant in the discrete trace inequality blows up, invalidating the Nitsche method.
4.  **The Solution:** Ghost Penalty Stabilization.
5.  **Proof of Stability:** A step-by-step proof of coercivity, independent of the cut location.

---

## 1. Recap: The Unstabilized CutFEM Formulation

We consider the Poisson problem on a domain $\Omega$ with boundary $\Gamma = \partial\Omega$:
$$ -\Delta u = f \quad \text{in } \Omega, \qquad u = g \quad \text{on } \Gamma $$

The CutFEM approach is to embed $\Omega$ in a background mesh $\mathcal{T}_{bg}$ and define the active mesh $\mathcal{T}_h = \{ T \in \mathcal{T}_{bg} : T \cap \Omega \neq \emptyset \}$. For an element $T \in \mathcal{T}_h$, we denote the physical part as $T_\Omega = T \cap \Omega$. Our space is $V_h = \{ v \in C^0(\cup_{T \in \mathcal{T}_h} T) : v|_T \in P_k(T) \}$.

A naive application of Nitsche's method yields the following formulation: Find $u_h \in V_h$ such that for all $v_h \in V_h$,
$$ a_h(u_h, v_h) = L_h(v_h) $$
where
$$ a_h(u, v) = \int_\Omega \nabla u \cdot \nabla v \, dx - \int_\Gamma (\partial_n u \, v + u \, \partial_n v) \, ds + \frac{\gamma}{h} \int_\Gamma u \, v \, ds $$
$$ L_h(v) = \int_\Omega f \, v \, dx - \int_\Gamma g \, \partial_n v \, ds + \frac{\gamma}{h} \int_\Gamma g \, v \, ds $$

We will now prove that this formulation is fundamentally ill-posed.

---

## 2. The Breakdown Part I: Ill-Conditioning

The problem arises when an element $T$ is cut such that its physical volume $|T_\Omega|$ is arbitrarily small compared to its total volume $|T|$. Let's quantify this.

**Definition:** For a cut element $T$, let $\epsilon_T = |T_\Omega|/|T|$. We are concerned with the case where $\epsilon_T \to 0$.

Consider a basis function $\phi_i$ associated with a node located in the "fictitious" part of a cut element $T$ (i.e., in $T \setminus T_\Omega$). The corresponding diagonal entry of the global stiffness matrix $A$ is $A_{ii} = a_h(\phi_i, \phi_i)$.

Let's analyze the scaling of this term.
$$ A_{ii} = \int_{T_\Omega} |\nabla \phi_i|^2 \, dx - 2 \int_{\Gamma \cap T} (\partial_n \phi_i) \phi_i \, ds + \frac{\gamma}{h} \int_{\Gamma \cap T} \phi_i^2 \, ds $$

Using standard scaling arguments on a reference element, we know that for a shape-regular mesh:
*   $|\nabla \phi_i|^2 \sim \mathcal{O}(h^{-2})$
*   $|\phi_i|^2 \sim \mathcal{O}(1)$ on $T$
*   $|T_\Omega| = \epsilon_T |T| \sim \epsilon_T h^d$ (where $d$ is the spatial dimension)
*   $|\Gamma \cap T| \sim \mathcal{O}(h^{d-1})$

Let's analyze the stiffness integral (the first term):
$$ \int_{T_\Omega} |\nabla \phi_i|^2 \, dx \sim \mathcal{O}(h^{-2}) \cdot |T_\Omega| \sim \mathcal{O}(h^{-2}) \cdot \epsilon_T h^d = \mathcal{O}(\epsilon_T h^{d-2}) $$
The boundary terms scale as $\mathcal{O}(h^{d-2})$.

When $\epsilon_T \to 0$, the dominant stiffness contribution from the volume integral vanishes. The matrix entry $A_{ii}$ becomes entirely dependent on the boundary terms, which may not be sufficient or can themselves be small. Crucially, the volume integral term, which provides the positive-definite part of the standard Laplacian operator, scales with $\epsilon_T$.

**Conclusion:** As $\epsilon_T \to 0$, $A_{ii} \to 0$ (or becomes dominated by potentially negative boundary terms before stabilization). The matrix $A$ has eigenvalues that approach zero. The condition number, $\kappa(A) = \lambda_{max}/\lambda_{min}$, scales badly. Since $\lambda_{max} \sim \mathcal{O}(h^{-2})$ (as for standard FEM) and now $\lambda_{min} \sim \mathcal{O}(\epsilon_T)$, we get a catastrophic scaling:
$$ \kappa(A) \sim \mathcal{O}(\epsilon_T^{-1} h^{-2}) $$
For a practical solver, if $\epsilon_T$ is near machine precision, the matrix is numerically singular.

---

## 3. The Breakdown Part II: Failure of Coercivity

Recall from your Nitsche lecture that the proof of coercivity relies on a **discrete trace inequality** to control the consistency term. Specifically, we need to show that for a sufficiently large penalty parameter $\gamma$,
$$ a_h(v_h, v_h) \geq \alpha \| \nabla v_h \|_{L^2(\Omega)}^2 $$
The key step involves bounding the term $\int_\Gamma (\partial_n v_h) v_h \, ds$. This requires a trace inequality of the form:
$$ \| v_h \|_{L^2(\Gamma \cap T)}^2 \leq C_{tr,1} h \| \nabla v_h \|_{L^2(T_\Omega)}^2 \quad \text{and} \quad h^2 \| \partial_n v_h \|_{L^2(\Gamma \cap T)}^2 \leq C_{tr,2} \| \nabla v_h \|_{L^2(T_\Omega)}^2 $$
Let's focus on the second one (often called an inverse inequality). The coercivity of Nitsche's method is only guaranteed if the penalty parameter $\gamma > C_{tr,2}$. We will now prove that $C_{tr,2}$ depends on $\epsilon_T^{-1}$.

**Theorem (Degenerate Trace Inequality):** Let $T$ be a cut element. The constant $C_{tr}$ in the inequality
$$ \| \partial_n v_h \|_{L^2(\Gamma \cap T)}^2 \leq C_{tr} h^{-1} \| \nabla v_h \|_{L^2(T_\Omega)}^2 $$
scales as $C_{tr} \sim \mathcal{O}(\epsilon_T^{-1})$.

**Proof:**
We prove this by constructing a specific function $v_h$ for which the inequality degenerates.
Consider a 2D case. Let $T = [0,h] \times [0,h]$ be a square element. Suppose the boundary $\Gamma$ is the line $x = \delta$, where $\delta \ll h$. The physical domain is $T_\Omega = [0,\delta] \times [0,h]$.
In this case, $|T_\Omega| = \delta h$ and $|T|=h^2$, so $\epsilon_T = \delta/h$.

Let's choose a linear function $v_h(x,y) = x/h \in P_1(T)$.
*   The gradient is $\nabla v_h = (1/h, 0)$.
*   The normal on $\Gamma \cap T$ is $\mathbf{n} = (1,0)$.
*   The normal derivative is $\partial_n v_h = \nabla v_h \cdot \mathbf{n} = 1/h$.

Now we compute the terms in the inequality:
1.  **Left-Hand Side (LHS):** The boundary segment is $\{ \delta \} \times [0,h]$.
    $$ \| \partial_n v_h \|_{L^2(\Gamma \cap T)}^2 = \int_0^h (1/h)^2 \, dy = \frac{1}{h^2} \cdot h = \frac{1}{h} $$
2.  **Right-Hand Side (RHS):** We integrate over the sliver domain $T_\Omega$.
    $$ \| \nabla v_h \|_{L^2(T_\Omega)}^2 = \int_0^h \int_0^\delta (1/h)^2 \, dx \, dy = \frac{1}{h^2} (\text{Area of } T_\Omega) = \frac{1}{h^2} (\delta h) = \frac{\delta}{h} $$

Now, let's find the constant $C_{tr}$:
$$ C_{tr} \geq h \frac{\| \partial_n v_h \|_{L^2(\Gamma \cap T)}^2}{\| \nabla v_h \|_{L^2(T_\Omega)}^2} = h \frac{1/h}{\delta/h} = \frac{h}{\delta} = \frac{1}{\epsilon_T} $$
We have shown that $C_{tr} \sim \mathcal{O}(\epsilon_T^{-1})$.

**Consequence:** To ensure coercivity, we need to choose the Nitsche penalty $\gamma > C_{tr}$. But as $\epsilon_T \to 0$, $C_{tr} \to \infty$. This means we would need an infinitely large penalty parameter, which is computationally unworkable and leads to a different kind of ill-conditioning known as "locking." The unstabilized method is broken.

---

## 4. The Solution: Ghost Penalty Stabilization

The root of the problem is that the degrees of freedom in the fictitious part $T \setminus T_\Omega$ are not controlled by the PDE. The Ghost Penalty (Burman, 2010) fixes this by weakly enforcing continuity of the gradient across faces of cut elements.

**Definition (Ghost Faces):** Let $\mathcal{F}_{ghost}$ be the set of all interior faces $F$ of the active mesh $\mathcal{T}_h$ such that $F$ is a face of at least one cut element $T$ (where $|T_\Omega|/|T| < 1$).

**Definition (Ghost Penalty Stabilizer):** The ghost penalty bilinear form $j(\cdot, \cdot)$ is defined as:
$$ j(u, v) = \gamma_g \sum_{F \in \mathcal{F}_{ghost}} h_F \int_F \llbracket \partial_n u \rrbracket \llbracket \partial_n v \rrbracket \, ds $$
where $\llbracket \cdot \rrbracket$ is the standard DG jump operator, $\partial_n$ is the normal derivative, and $\gamma_g > 0$ is a user-defined stabilization parameter. For $P_k$ elements, this is often extended to penalize jumps in all derivatives up to order $k$.

The **stabilized CutFEM formulation** is: Find $u_h \in V_h$ such that
$$ A_h(u_h, v_h) := a_h(u_h, v_h) + j(u_h, v_h) = L_h(v_h) \quad \forall v_h \in V_h $$

---

## 5. Proof of Stability (Independent of the Cut)

We will now prove that the bilinear form $A_h(\cdot, \cdot)$ is coercive with a constant that is independent of how the boundary cuts the mesh.

**Key Lemma 1 (Gradient Extension):** There exists a constant $C_{ext}$, independent of the cut geometry $\epsilon_T$, such that for any $v_h \in V_h$ and any cut element $T$,
$$ \| \nabla v_h \|_{L^2(T)}^2 \leq C_{ext} \left( \| \nabla v_h \|_{L^2(T_\Omega)}^2 + \sum_{F \in \mathcal{F}_{ghost}, F \subset \partial T} h_F \int_F \llbracket \partial_n v_h \rrbracket^2 \, ds \right) $$

*Proof Sketch:* This is a Poincare-type inequality. If the gradient norm on $T_\Omega$ is zero and the jumps of the normal derivative across all faces are zero, it implies that the polynomial gradient $\nabla v_h$ is constant on $T$ and matches the gradient of its neighbors. By extending this argument through a patch of elements connected to the stable interior of $\Omega$, one can show that $\nabla v_h$ must be zero on the entire element $T$. A full proof uses a scaling argument and contradiction on a reference element.

**Key Lemma 2 (Stabilized Trace Inequality):** There exists a constant $C_{tr}^*$, independent of the cut geometry, such that
$$ \| \partial_n v_h \|_{L^2(\Gamma \cap T)}^2 \leq C_{tr}^* h^{-1} \left( \| \nabla v_h \|_{L^2(T_\Omega)}^2 + j_T(v_h, v_h) \right) $$
where $j_T$ is the contribution to the ghost penalty from faces of $T$.

*Proof:*
1. Start with the standard inverse inequality on the **full, uncut, shape-regular element $T$**:
   $$ \| \partial_n v_h \|_{L^2(\Gamma \cap T)}^2 \leq C_{std} h^{-1} \| \nabla v_h \|_{L^2(T)}^2 $$
   Here, $C_{std}$ depends only on the polynomial degree and the element shape regularity, NOT on the cut.

2. Apply **Key Lemma 1** to the RHS:
   $$ \| \partial_n v_h \|_{L^2(\Gamma \cap T)}^2 \leq C_{std} h^{-1} \left[ C_{ext} \left( \| \nabla v_h \|_{L^2(T_\Omega)}^2 + j_T(v_h, v_h) \right) \right] $$
3. Set $C_{tr}^* = C_{std} C_{ext}$. This constant is independent of the cut. This completes the proof of the lemma.

**Theorem (Coercivity of the Stabilized Form):** The bilinear form $A_h(v,v)$ is coercive on $V_h$ with a constant $\alpha > 0$ independent of the cut geometry.

*Proof:*
We start with the full bilinear form:
$$ A_h(v,v) = \int_\Omega |\nabla v|^2 dx - 2 \int_\Gamma (\partial_n v) v \, ds + \frac{\gamma}{h} \int_\Gamma v^2 ds + j(v,v) $$
Apply Cauchy-Schwarz and Young's inequality ($2ab \leq \delta a^2 + \delta^{-1} b^2$) to the consistency term:
$$ \left| 2 \int_\Gamma (\partial_n v) v \, ds \right| \leq 2 \| h^{1/2} \partial_n v \|_\Gamma \| h^{-1/2} v \|_\Gamma \leq \delta h \| \partial_n v \|_\Gamma^2 + \frac{1}{\delta} h^{-1} \| v \|_\Gamma^2 $$
for any $\delta > 0$. Substituting this back gives:
$$ A_h(v,v) \geq \| \nabla v \|_\Omega^2 - \delta h \| \partial_n v \|_\Gamma^2 + \left( \gamma - \frac{1}{\delta} \right) h^{-1} \| v \|_\Gamma^2 + j(v,v) $$
Now, we use our **Key Lemma 2 (Stabilized Trace Inequality)** to bound the negative term. Summing over all boundary segments:
$$ h \| \partial_n v \|_\Gamma^2 = \sum_T h \| \partial_n v \|_{\Gamma \cap T}^2 \leq C_{tr}^* \left( \| \nabla v \|_\Omega^2 + j(v,v) \right) $$
Substitute this into the coercivity estimate:
$$ A_h(v,v) \geq \| \nabla v \|_\Omega^2 - \delta C_{tr}^* \left( \| \nabla v \|_\Omega^2 + j(v,v) \right) + \left( \gamma - \frac{1}{\delta} \right) h^{-1} \| v \|_\Gamma^2 + j(v,v) $$
Group the terms:
$$ A_h(v,v) \geq (1 - \delta C_{tr}^*) \| \nabla v \|_\Omega^2 + \left( \gamma - \frac{1}{\delta} \right) h^{-1} \| v \|_\Gamma^2 + (1 - \delta C_{tr}^*) j(v,v) $$
Now, we choose the parameters. Since $C_{tr}^*$ is a constant independent of the cut:
1.  Choose $\delta = \frac{1}{2 C_{tr}^*}$. This makes the coefficient $(1 - \delta C_{tr}^*) = 1/2$.
2.  Choose the Nitsche penalty $\gamma$ to make the second term positive. For example, choose $\gamma = \frac{1}{\delta} = 2 C_{tr}^*$.

With these choices, both of which are independent of the cut geometry, we get:
$$ A_h(v,v) \geq \frac{1}{2} \| \nabla v \|_\Omega^2 + \frac{1}{2} j(v,v) $$
This shows coercivity with respect to the gradient and stabilization norms. A slightly more careful argument including a Poincaré inequality establishes coercivity with respect to a full $H^1$-like norm, proving well-posedness. $\blacksquare$