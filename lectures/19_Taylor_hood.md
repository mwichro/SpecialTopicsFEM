
The strategy for this lecture is highly effective for the blackboard: instead of proving the discrete LBB condition directly with brutal algebra, we will use **Fortin's Trick**, which reduces the whole problem to answering one question: *"Does our velocity element have enough 'bubble' nodes to correct interpolation errors?"*

***

# Lecture: Proving the Taylor-Hood ($Q_2-Q_1$) Element Satisfies LBB

**Goal:** Prove that the Taylor-Hood element on quadrilateral/hexahedral meshes (biquadratic continuous velocity, bilinear continuous pressure) is stable by satisfying the discrete LBB condition.

## 1. The Challenge of Discrete LBB
Recall the continuous LBB condition we proved last time:
$$ \inf_{q \in Q} \sup_{v \in V} \frac{b(v, q)}{\|v\|_V \|q\|_Q} \ge \beta > 0 $$

When we discretize to $V_h \subset V$ and $Q_h \subset Q$, we must ensure the *discrete* spaces still satisfy this:
$$ \inf_{q_h \in Q_h} \sup_{v_h \in V_h} \frac{b(v_h, q_h)}{\|v_h\|_V \|q_h\|_Q} \ge \beta_h > 0 $$
If we just guess spaces (like $Q_1-Q_1$), $\beta_h \to 0$ as the mesh refines, leading to unstable checkerboard pressures. We need a systematic way to prove our chosen spaces are stable.

---

## 2. The Tool: Fortin's Lemma

**Theorem (Fortin, 1977):** 
Assume the continuous spaces $V, Q$ satisfy the continuous LBB condition with constant $\beta$. 
If we can construct a projection operator $\Pi_h : V \to V_h$ such that for a constant $C > 0$:
1.  **Divergence Preservation:** $b(\Pi_h v, q_h) = b(v, q_h) \quad \forall q_h \in Q_h$
2.  **Stability:** $\|\Pi_h v\|_{V} \le C \|v\|_{V}$
Then the discrete LBB condition is automatically satisfied with constant $\beta_h = \frac{\beta}{C}$.

**Proof of Fortin's Trick (Write this on the board, it's elegant!):**
Let $q_h \in Q_h$. Because $Q_h \subset Q$, we know the *continuous* LBB holds for $q_h$. There exists some continuous velocity $v \in V$ such that:
$$ b(v, q_h) \ge \beta \|v\|_V \|q_h\|_Q $$

Now, substitute $v$ with its discrete projection $\Pi_h v \in V_h$:
$$ \sup_{v_h \in V_h} \frac{b(v_h, q_h)}{\|v_h\|_V} \ge \frac{b(\Pi_h v, q_h)}{\|\Pi_h v\|_V} $$
Use Property 1 (Divergence Preservation):
$$ = \frac{b(v, q_h)}{\|\Pi_h v\|_V} $$
Use Property 2 (Stability, $\|\Pi_h v\|_V \le C\|v\|_V$):
$$ \ge \frac{b(v, q_h)}{C \|v\|_V} $$
Substitute the continuous LBB bound:
$$ \ge \frac{\beta \|v\|_V \|q_h\|_Q}{C \|v\|_V} = \frac{\beta}{C} \|q_h\|_Q \quad \blacksquare $$

*Takeaway:* We don't have to prove discrete inf-sup directly! We just have to build the operator $\Pi_h$.

---

## 3. Applying Fortin to the Taylor-Hood Element

Let's define our spaces on a quadrilateral mesh:
*   $V_h$: $Q_2$ (Continuous Biquadratic). Nodes at 4 vertices, 4 edge midpoints, 1 cell center.
*   $Q_h$: $Q_1$ (Continuous Bilinear). Nodes at 4 vertices only.

Let's rewrite Fortin's Property 1: $b(\Pi_h v, q_h) = b(v, q_h)$.
$$ b(\Pi_h v - v, q_h) = 0 $$
$$ -\int_\Omega (\nabla \cdot (\Pi_h v - v)) q_h \, dV = 0 $$
Because $q_h \in Q_1$ is **globally continuous**, we can safely integrate by parts without any edge-jump terms appearing:
$$ \int_\Omega (\Pi_h v - v) \cdot \nabla q_h \, dV = 0 \quad \forall q_h \in Q_h $$

---

## 4. Constructing the Fortin Operator $\Pi_h$

We split the construction of $\Pi_h v$ into two parts: 
$$ \Pi_h v = I_h v + \mathbf{w}_h $$
1.  **$I_h v$ (The Coarse Linear Part):** This is a basic bilinear ($Q_1$) approximation of the continuous velocity, built using **only the corner vertices** of the elements. 
    *   *How do we build it?* Since our true velocity $v$ lives in the abstract $H^1$ space, it might have undefined exact values at specific points. So, instead of just evaluating $v$ exactly at the corners, we take a **local spatial average** of the velocity in the elements surrounding each vertex. 
    *   *Why do we do this?* It captures the large-scale "bulk" flow of the fluid. Because it is based on averages rather than exact point-values, it never "blows up," making it mathematically stable: $\|\nabla (I_h v)\| \le C\|\nabla v\|$.
2.  **$\mathbf{w}_h$ (The Correction):** A function built strictly from the remaining nodes in $Q_2$ (the edge midpoints and cell centers). These are called "Bubble Functions" because they are zero at all element vertices.

Substitute this split into our requirement:
$$ \int_\Omega (I_h v + \mathbf{w}_h - v) \cdot \nabla q_h \, dV = 0 $$
$$ \int_\Omega \mathbf{w}_h \cdot \nabla q_h \, dV = \int_\Omega (v - I_h v) \cdot \nabla q_h \, dV $$

**The Physical Meaning:** The standard interpolation $I_h v$ creates an error in the divergence field. We must ask: *Does our velocity space have enough bubble degrees of freedom ($\mathbf{w}_h$) to absorb and correct this error?*

---

## 5. Dimension Counting (Why Taylor-Hood Works)

Let's look at the constraints locally on a single element $K$. 
The pressure $q_h \in Q_1$, so it has the form $q_h = c_0 + c_1 x + c_2 y + c_3 xy$.
The gradient is:
$$ \nabla q_h = \begin{pmatrix} c_1 + c_3 y \\ c_2 + c_3 x \end{pmatrix} $$
Notice that $\nabla q_h$ contains constants ($c_1, c_2$) and linear terms ($c_3$). This means we have exactly **4 independent constraints** per element to satisfy.

What tools (Degrees of Freedom) do we have in our bubble correction $\mathbf{w}_h$?
1.  **Cell Bubble (Center Node):** This function is zero on all boundaries of $K$. It has 2 DOFs ($x$ and $y$ velocity). Testing with the cell bubble controls the **constant** parts ($c_1, c_2$) of the pressure gradient perfectly.
2.  **Edge Bubbles (Edge Midpoint Nodes):** But the cell bubble cannot cancel out the linear $c_3$ terms (due to symmetric integration, the integrals vanish). This is where the 4 edge bubbles come in! They provide $4 \text{ edges} \times 2 \text{ components} = 8$ DOFs per element.

**Global Solvability:**
*   Total Constraints: $4 \times \text{Number of Elements}$ (from $\nabla q_h$).
*   Total Bubble DOFs: $2 \times \text{Number of Elements}$ (centers) + $2 \times \text{Number of Edges}$ (edges).
*   Because $\text{Number of Edges} \approx 2 \times \text{Number of Elements}$, we have roughly **$6E$ DOFs to satisfy $4E$ equations**. 
*   The system is massively under-determined, meaning we can *always* find a local bubble correction $\mathbf{w}_h$ to satisfy the integral. 

*(Therefore, $\Pi_h$ exists, stability holds, and Fortin's Trick proves LBB!)*

---

## Why $Q_1-Q_1$ Fails

To cement the students' understanding, ask them to apply this exact proof to the Equal-Order $Q_1-Q_1$ element.

*   If velocity is $Q_1$, there are **zero** edge midpoints and **zero** cell centers.
*   Therefore, the bubble space is empty: $\mathbf{w}_h = 0$.
*   We are left forcing the standard interpolant to satisfy $\int (I_h v - v) \cdot \nabla q_h = 0$, which is mathematically impossible for arbitrary functions. 
*   The Fortin operator cannot be constructed.

**Conclusion:** The LBB condition mandates that the velocity space must be strictly richer (higher degree) than the pressure space. The "extra" nodes (edge and cell bubbles) in the Taylor-Hood $Q_2$ element act as the mathematical "shock absorbers" that absorb interpolation errors, preventing spurious checkerboard pressure fields!