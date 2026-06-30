
***

# Lecture: Proving the Continuous LBB Condition for Stokes

Proving the continuous LBB condition directly from functional analysis is famously abstract (often invoking the Closed Range Theorem or Nečas's Inequality). However, for a blackboard lecture, there is a much more intuitive, PDE-based path. We will prove that the continuous LBB condition is exactly equivalent to solving a specific boundary-value problem: finding a velocity field that "absorbs" any given pressure field.


**Goal:** We have relied on the fact that the continuous LBB condition holds for the Stokes equations to prove everything else (Brezzi's conditions, Fortin's trick). Today, we actually prove it. 

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
But because $v \in H^1_0$, $v = 0$ on the boundary! Thus, $b(v, C) = 0$ for *every possible velocity*. 
If $b(v, q) = 0$ for all $v$, the supremum in the numerator is $0$ while $\|q\|_Q\neq 0$, so the quotient is $0$, the infimum over $q$ is $0$, and LBB fails instantly with $\beta=0$. **We must factor out the constant pressure mode** — equivalently, we measure pressure only up to a constant, which is exactly what the physics says (only $\nabla p$ enters the momentum equation). Quotienting $L^2$ by constants and choosing the zero-mean representative gives precisely $L^2_0(\Omega)$.

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

*(Teacher note: read off what was used. The matching property turned the bilinear form into $\|q\|^2$; the bound turned it into a single power of $\|q\|$. So LBB is *exactly* the statement that the divergence operator is surjective onto $L^2_0$ with a uniform bound — "can it generate any zero-mean scalar field, cheaply?")*

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
We check: $\int_\Omega -q \, dV = 0$. Yes! Because we specifically restricted our pressure space to $L^2_0(\Omega)$. 

So, $\phi$ exists. By standard elliptic regularity, $\|\phi\|_{H^2} \le C \|q\|_{L^2}$.
Now define our first velocity attempt:
$$ v_1 = \nabla \phi $$
*Check the properties:* 
*   Divergence: $-\nabla \cdot v_1 = -\Delta \phi = q$. (Perfect!)
*   Boundary: $v_1 \cdot n = \nabla \phi \cdot n = 0$. (Normal velocity is zero).
*   *The Catch:* The tangential velocity on the boundary is **not** necessarily zero. So $v_1 \notin H^1_0$.

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

We have successfully proven the continuous LBB condition!

1. We showed that proving LBB is identical to proving that for any pressure $q$, we can find a velocity $v$ that satisfies $-\nabla \cdot v = q$.
2. We used the zero-mean property of the pressure to guarantee a solution to an auxiliary Poisson equation.
3. We used a boundary corrector to satisfy the no-slip condition.
4. Elliptic regularity provided the necessary bound $C$, proving that $\beta = 1/C > 0$.

*(This brilliantly ties together everything the students have learned: why we need $L^2_0$, how divergence works, and how PDEs are used as tools to prove abstract functional analysis theorems!)*