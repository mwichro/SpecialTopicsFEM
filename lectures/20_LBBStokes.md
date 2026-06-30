
***

# Lecture: Proving the Continuous LBB Condition for Stokes

Proving the continuous LBB condition directly from functional analysis is abstract (it typically invokes the Closed Range Theorem or Nečas's inequality). There is a more constructive, PDE-based path that is better suited to a lecture. We will show that the continuous LBB condition is equivalent to solving a specific boundary-value problem: finding a velocity field whose divergence reproduces any given pressure field.


**Goal:** We have relied on the continuous LBB condition for the Stokes equations to establish everything else (Brezzi's conditions, Fortin's trick). Here we prove it. 

## 1. The Formal Statement and the Zero-Mean Pressure Space

The Stokes weak form uses the bilinear form $b(v, q) = -\int_\Omega (\nabla \cdot v) q \, dV$. 
The continuous LBB condition states there exists a constant $\beta > 0$ such that:
$$ \inf_{q \in Q} \sup_{v \in V} \frac{-\int_\Omega (\nabla \cdot v) q \, dV}{\|v\|_V \|q\|_Q} \ge \beta $$

**Defining the Spaces:**
*   **Velocity ($V$):** $H^1_0(\Omega)^d$. These are vector fields with bounded gradients that are exactly zero on the boundary $\partial \Omega$. Norm: $\|v\|_V = \|\nabla v\|_{L^2}$.
*   **Pressure ($Q$):** $L^2_0(\Omega)$. This is the space of square-integrable functions with **strictly zero mean**. 
    $$ L^2_0(\Omega) = \left\{ q \in L^2(\Omega) : \int_\Omega q \, dV = 0 \right\} $$

**Why Zero-Mean Pressure?**
What if we allowed a constant pressure $q = C \neq 0$ everywhere? 
Evaluate the numerator of the LBB condition:
$$ b(v, C) = -\int_\Omega (\nabla \cdot v) C \, dV = -C \int_\Omega \nabla \cdot v \, dV $$
By the Divergence Theorem (Gauss's Theorem):
$$ b(v,C) = -C \int_\Omega \nabla\cdot v\,dV = -C \int_{\partial \Omega} v \cdot n \, dS . $$
But because $v \in H^1_0$, $v = 0$ on the boundary, so $b(v, C) = 0$ for *every* velocity. 
If $b(v, q) = 0$ for all $v$, the supremum in the numerator is $0$ while $\|q\|_Q\neq 0$, so the quotient is $0$, the infimum over $q$ is $0$, and LBB fails with $\beta=0$. **We must factor out the constant pressure mode** — equivalently, we measure pressure only up to a constant, which is exactly what the physics says (only $\nabla p$ enters the momentum equation). Quotienting $L^2$ by constants and choosing the zero-mean representative gives precisely $L^2_0(\Omega)$.

---

## 2. The Surjectivity Equivalence Theorem

We can replace the abstract $\inf$-$\sup$ with a concrete PDE problem. The whole point is to translate the analyst's question — *"is the inf-sup constant positive?"* — into the PDE-maker's question — *"can I always find a flow whose divergence equals my prescribed source?"*

**Theorem.** Let $B : V \to Q'$ be the operator associated with $b$, i.e. $\langle Bv, q\rangle = b(v,q) = -\int_\Omega (\nabla\cdot v)\, q\, dV$. The following are equivalent:

*   **(LBB)** There is $\beta>0$ with $\displaystyle \inf_{q\in Q}\sup_{v\in V}\frac{b(v,q)}{\|v\|_V\|q\|_Q}\ge\beta$.
*   **(SURJ)** The divergence is *boundedly surjective*: for every $q \in L^2_0(\Omega)$ there exists $v_q \in H^1_0(\Omega)^d$ with
    1.  **It matches the pressure:** $-\nabla \cdot v_q = q$;
    2.  **It is bounded:** $\|v_q\|_V \le C \|q\|_Q$, with $C$ independent of $q$.

The two constants are reciprocal: one may take $C = 1/\beta$.

> **Direction we actually need: (SURJ) $\implies$ (LBB).**
> This is the useful half — it lets us *certify* LBB by *building* a velocity. (The reverse implication is the abstract one; see the note at the end of the section.)

**Proof that (SURJ) $\implies$ (LBB).**
Fix an arbitrary $q \in L^2_0(\Omega)$ and let $v_q$ be the field promised by (SURJ). To bound the supremum from below we are free to *test with this one special $v_q$* instead of searching over all of $V$:
$$ \sup_{v \in V} \frac{-\int_\Omega (\nabla \cdot v) q \, dV}{\|v\|_V} \ge \frac{-\int_\Omega (\nabla \cdot v_q) q \, dV}{\|v_q\|_V} . $$
Substitute the matching property $-\nabla \cdot v_q = q$ in the numerator:
$$ = \frac{\int_\Omega q^2 \, dV}{\|v_q\|_V} = \frac{\|q\|^2_{L^2}}{\|v_q\|_V} . $$
Now use the boundedness property $\|v_q\|_V \le C \|q\|_{L^2}$ in the denominator (a larger denominator only makes the fraction smaller, so the inequality is preserved):
$$ \ge \frac{\|q\|^2_{L^2}}{C \|q\|_{L^2}} = \frac{1}{C} \|q\|_{L^2} . $$
This holds for the chosen $q$, so dividing the bound by $\|q\|_{L^2}=\|q\|_Q$ gives
$$ \sup_{v\in V}\frac{b(v,q)}{\|v\|_V\|q\|_Q}\ge \frac1C \qquad\text{for every } q\in Q, $$
and taking the infimum over $q$ yields $\beta = \tfrac1C > 0$. $\blacksquare$

*(Note what was used: the matching property turned the bilinear form into $\|q\|^2$, and the bound turned it into a single power of $\|q\|$. So LBB is the statement that the divergence operator is surjective onto $L^2_0$ with a uniform bound.)*

> **The other direction, (LBB) $\implies$ (SURJ), for completeness.**
> This is the genuinely abstract half. LBB says $B^\top$ (the gradient, acting on pressures) is bounded below: $\|B^\top q\|_{V'}\ge\beta\|q\|_Q$. By the **Closed Range Theorem**, $B^\top$ bounded below is equivalent to $B$ having closed range *and* being surjective onto $(\ker B^\top)^\circ$. Since LBB also forces $\ker B^\top=\{0\}$ on $Q=L^2_0$, $B$ is onto all of $Q$, and the open mapping theorem supplies the uniform bound $C=1/\beta$. We do **not** need this direction below — we only build velocities — but it explains why the equivalence is exact rather than one-sided.

So the rest of the lecture is devoted to **(SURJ)**: given a zero-mean $q$, *construct* the field $v_q$.

---

## 3. Constructive Proof: Building the Velocity Field

Now the entire proof of LBB reduces to: *Given a zero-mean function $q$, how do we mathematically construct a bounded vector field $v \in H^1_0$ whose divergence is exactly $-q$?*

We do this in two steps using a "Poisson Trick".

### Step 1: The Bulk Field (Solving Poisson)
Consider the auxiliary Neumann problem for a scalar potential $\phi$:
$$ \Delta \phi = -q \quad \text{in } \Omega $$
$$ \nabla \phi \cdot n = 0 \quad \text{on } \partial \Omega $$
Does a solution to this exist? By the Fredholm alternative, a purely Neumann Poisson problem only has a solution if the right-hand side integrates to zero. 
We check: $\int_\Omega -q \, dV = 0$, which holds because we restricted the pressure space to $L^2_0(\Omega)$. 

So, $\phi$ exists. By standard elliptic regularity, $\|\phi\|_{H^2} \le C \|q\|_{L^2}$.
Now define our first velocity attempt:
$$ v_1 = \nabla \phi $$
*Check the properties:* 
*   Divergence: $-\nabla \cdot v_1 = -\Delta \phi = q$, as required.
*   Boundary: $v_1 \cdot n = \nabla \phi \cdot n = 0$, so the normal velocity vanishes.
*   *The catch:* the tangential velocity on the boundary is **not** necessarily zero, so $v_1 \notin H^1_0$.

### Step 2: The Boundary Corrector
The catch above is the only thing standing between us and $H^1_0$: $v_1$ has the right divergence but the wrong (tangential) trace. So we subtract a second field $v_2$ that carries away that trace *without* injecting any divergence. We require:
1.  $\nabla \cdot v_2 = 0 \quad \text{in } \Omega$ (so subtracting it leaves the divergence of $v_1$ untouched);
2.  $v_2 = v_1 \quad \text{on } \partial \Omega$ (so the traces cancel).

Why can such a $v_2$ exist? A divergence-free field has zero net flux through any closed boundary, $\int_{\partial\Omega} v_2\cdot n\,dS = \int_\Omega \nabla\cdot v_2\,dV = 0$. So we may only prescribe a boundary datum that *itself* has zero net flux. Here the datum is the trace of $v_1$, and we computed in Step 1 that $v_1\cdot n = \nabla\phi\cdot n = 0$ on $\partial\Omega$ — flux zero, pointwise even. **The Neumann condition we imposed on $\phi$ was chosen precisely to make this compatibility hold.**

With the compatibility condition met, the existence of a bounded, divergence-free extension of the boundary data is a standard (but nontrivial) result, realized e.g. by **Bogovskii's operator** or by writing $v_2 = \nabla\times\psi$ for a suitable vector potential $\psi$. It comes with a bound, via the trace theorem ($\|v_1|_{\partial\Omega}\|_{H^{1/2}}\le C\|v_1\|_{H^1}$) and the stability of the extension:
$$\|v_2\|_{H^1} \le C_2 \|v_1\|_{H^1}.$$

### Step 3: The Final Combination
Define our final velocity: 
$$ v_q = v_1 - v_2 $$
1.  **Boundary condition:** On $\partial \Omega$, $v_q = v_1 - v_1 = 0$. Thus, $v_q \in H^1_0(\Omega)^d$.
2.  **Divergence:** $-\nabla \cdot v_q = -\nabla \cdot v_1 + \nabla \cdot v_2 = q + 0 = q$.
3.  **Boundedness:** By the triangle inequality and our elliptic/trace bounds:
    $$ \|v_q\|_{H^1} \le \|v_1\|_{H^1} + \|v_2\|_{H^1} \le (1 + C_2)\|v_1\|_{H^1} $$
    $$ \le (1 + C_2) C \|\phi\|_{H^2} \le C_{final} \|q\|_{L^2} $$

## 4. Summary and Conclusion

We have proven the continuous LBB condition.

1. We reduced LBB to the surjectivity statement: for any pressure $q$, there is a velocity $v$ with $-\nabla \cdot v = q$ and a uniform bound.
2. We used the zero-mean property of the pressure to guarantee a solution to an auxiliary Neumann–Poisson equation.
3. We used a boundary corrector to recover the no-slip condition.
4. Elliptic regularity provided the bound $C$, giving $\beta = 1/C > 0$.

This connects three threads from the course: why we need $L^2_0$, how the divergence operator acts, and how a PDE construction can establish an abstract functional-analytic estimate.

---

## 5. Practical Application: Chorin's Projection Method

The proof above is not only an existence argument: its central step — *split a field into a gradient part and a divergence-free part, using a Poisson equation as the bridge* — is a standard tool in computational fluid dynamics. The same Helmholtz decomposition that produced our velocity $v_q$ reappears as a time-stepping scheme for the **incompressible Navier–Stokes equations**:
$$ \partial_t u + (u\cdot\nabla)u - \nu\,\Delta u + \nabla p = f, \qquad \nabla\cdot u = 0, \qquad u|_{\partial\Omega}=0. $$

The difficulty is the same one LBB is about: the pressure $p$ is **not** a prognostic variable with its own evolution equation. It is a **Lagrange multiplier** whose only job is to enforce $\nabla\cdot u = 0$. Chorin's method (1968) sidesteps the coupled saddle-point solve by *decoupling* velocity and pressure at each time step — and the tool that lets it do so is exactly our decomposition.

### The Helmholtz–Hodge decomposition (the engine)
**Theorem.** Any vector field $w \in L^2(\Omega)^d$ splits *uniquely and orthogonally* as
$$ w = u + \nabla\phi, \qquad \nabla\cdot u = 0,\quad u\cdot n|_{\partial\Omega}=0. $$
The two pieces are $L^2$-orthogonal, $\int_\Omega u\cdot\nabla\phi\,dV = 0$ (integrate by parts and use $\nabla\cdot u=0$, $u\cdot n=0$). To compute the split, take the divergence:
$$ \nabla\cdot w = \Delta\phi, \qquad \nabla\phi\cdot n = w\cdot n \text{ on }\partial\Omega, $$
a Neumann–Poisson problem — *the very same one* we solved in Step 1 of the proof. Then $u = w - \nabla\phi$ is the divergence-free projection $\mathbb{P}\,w$.

### The algorithm (one time step $u^n \to u^{n+1}$, step size $\Delta t$)
1.  **Predictor — ignore the constraint.** Advance momentum *without* the pressure term, producing an intermediate velocity $u^\star$ that is generally **not** divergence-free:
    $$ \frac{u^\star - u^n}{\Delta t} + (u^n\cdot\nabla)u^n - \nu\,\Delta u^\star = f^{n+1}, \qquad u^\star|_{\partial\Omega}=0. $$
    This is a (vector) advection–diffusion solve — symmetric positive definite, with no saddle point.

2.  **Projection — restore incompressibility.** We *want* $u^{n+1}$ to be the divergence-free part of $u^\star$. Writing the (skipped) pressure term back in,
    $$ \frac{u^{n+1}-u^\star}{\Delta t} = -\nabla p^{n+1}, \qquad \nabla\cdot u^{n+1}=0. $$
    This is a Helmholtz decomposition of $u^\star$: $u^\star = u^{n+1} + \Delta t\,\nabla p^{n+1}$. Taking the divergence gives the **pressure Poisson equation**
    $$ \Delta p^{n+1} = \frac{1}{\Delta t}\,\nabla\cdot u^\star, \qquad \nabla p^{n+1}\cdot n = 0 \text{ on }\partial\Omega. $$

3.  **Correct.** Subtract the gradient to land on the divergence-free manifold:
    $$ u^{n+1} = u^\star - \Delta t\,\nabla p^{n+1}. $$

### Payoff and caveats
*   **Why it works:** Steps 2–3 are the projection $u^{n+1}=\mathbb{P}\,u^\star$ onto divergence-free fields. The indefinite Stokes saddle-point system is replaced by **one symmetric vector solve and one scalar Poisson solve** per step, both of which are standard and parallelizable.
*   **The LBB connection:** solvability of the pressure Poisson equation requires the right-hand side $\tfrac1{\Delta t}\nabla\cdot u^\star$ to be compatible (zero mean against the Neumann data) — the same Fredholm/zero-mean condition that forced us into $L^2_0$ above. At the *discrete* level, choosing velocity/pressure spaces so that this projection is stable is the discrete LBB condition.
*   **The price of decoupling:** the artificial Neumann condition $\nabla p\cdot n = 0$ is not satisfied by the true pressure, which produces a numerical boundary layer and limits the basic scheme to first order in $\Delta t$ for the pressure. Incremental and rotational pressure-correction variants (Goda; Timmermans; Guermond–Shen) modify the boundary condition and recover higher order, but the structure is still predict → Poisson solve → project.

The "Poisson trick plus boundary corrector" used to *prove* LBB is the same construction that, applied once per time step, *solves* Navier–Stokes in practice.