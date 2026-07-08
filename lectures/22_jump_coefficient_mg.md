#  Elliptic Problems with Highly Discontinuous Coefficients

## 1. Introduction and Problem Setup

In many physical applications (groundwater flow, electromagnetics, composite materials) we encounter diffusion problems where the material properties change abruptly and by many orders of magnitude. We want solvers whose performance does **not** degrade as the contrast grows.

We consider the second-order elliptic boundary value problem:
$$
\begin{cases}
-\nabla \cdot (\omega \nabla u) = f & \text{in } \Omega, \\
u = g_D & \text{on } \Gamma_D, \\
-\omega \dfrac{\partial u}{\partial n} = g_N & \text{on } \Gamma_N,
\end{cases}
$$
where $\omega(x)$ is a highly discontinuous, piecewise constant coefficient. We assume $\Omega$ is partitioned into $M$ disjoint subdomains $\Omega_m$ with
$$ \omega|_{\Omega_m} = \omega_m > 0, \qquad m = 1, \dots, M. $$

Discretizing with standard linear finite elements on a quasi-uniform mesh of size $h$ gives the linear system
$$ \mathcal{A}\mu = b, \qquad a_{ij} = \int_\Omega \omega\, \nabla \phi_i \cdot \nabla \phi_j, $$
where $\mathcal{A}$ is the (symmetric positive definite) stiffness matrix.

Throughout we use the **weighted norms**
$$
\|v\|_{0,\omega}^2 = \int_\Omega \omega\, v^2, \qquad
|v|_{1,\omega}^2 = \int_\Omega \omega\, |\nabla v|^2 = \nu^t \mathcal{A}\nu,
$$
so that the energy inner product is exactly the bilinear form of the problem, and $\nu$ denotes the coefficient vector of the finite element function $v$.

### The Challenge

The condition number of the unpreconditioned stiffness matrix degrades with **two** independent factors — the mesh size and the coefficient contrast:
$$ \kappa(\mathcal{A}) \simeq h^{-2}\, \mathcal{J}(\omega), \qquad
\mathcal{J}(\omega) = \frac{\max_m \omega_m}{\min_m \omega_m}. $$
If the contrast is large (say $\mathcal{J}(\omega) = 10^6$), standard iterative solvers stall almost completely. Our goal is a preconditioner $\mathcal{B}$ whose effect on convergence is (essentially) independent of $\mathcal{J}(\omega)$.

---

## 2. PCG and the "Effective" Condition Number

When we precondition, we solve $\mathcal{B}\mathcal{A}\mu = \mathcal{B}b$, and the classical CG bound is governed by $\kappa(\mathcal{B}\mathcal{A})$. But CG is a *polynomial* method, and it is much smarter than the worst-case bound suggests: if the condition number is large **only** because of a *few* isolated small eigenvalues, CG rapidly annihilates the corresponding error components and thereafter converges as if those eigenvalues were absent.

**Definition (Effective condition number).**
Let $\lambda_1 \le \lambda_2 \le \dots \le \lambda_n$ be the eigenvalues of $\mathcal{B}\mathcal{A}$. The $(m{+}1)$-th *effective condition number* is
$$ \kappa_{m+1}(\mathcal{B}\mathcal{A}) = \frac{\lambda_{\max}(\mathcal{B}\mathcal{A})}{\lambda_{m+1}(\mathcal{B}\mathcal{A})}. $$

**Why it controls convergence.** The CG error after $k$ steps satisfies
$$
\frac{\|e_k\|_{\mathcal A}}{\|e_0\|_{\mathcal A}}
\le \min_{\substack{p\in \mathbb P_k \\ p(0)=1}} \max_{i} |p(\lambda_i)|.
$$
Choose the polynomial $p = q\cdot r$, where $q$ is the degree-$m$ polynomial that vanishes at the $m$ small eigenvalues $\lambda_1,\dots,\lambda_m$ (and $q(0)=1$), and $r$ is the standard Chebyshev polynomial on $[\lambda_{m+1},\lambda_{\max}]$. Then $p$ kills the bad modes exactly, and on the remaining spectrum it decays at the Chebyshev rate governed by $\kappa_{m+1}$. Hence, after an initial *latency* of about $m$ iterations, CG converges at the rate
$$
\frac{\|e_k\|_{\mathcal A}}{\|e_0\|_{\mathcal A}}
\lesssim 2\left(\frac{\sqrt{\kappa_{m+1}}-1}{\sqrt{\kappa_{m+1}}+1}\right)^{k-m}.
$$
So if $m$ is small (a handful of outliers) and $\kappa_{m+1}$ is moderate, CG is fast — even though $\kappa$ itself is huge.

**Strategy.** We will *not* try to make $\kappa(\mathcal{B}\mathcal{A})$ small. Instead we allow a fixed, small number $m_0$ of jump-dependent outliers and control $\kappa_{m_0+1}$.

---

## 3. Analysis of Simple Preconditioners (Jacobi / Gauss–Seidel)

Take the diagonal (Jacobi) preconditioner $\mathcal{B} = \mathcal{D}^{-1}$, $\mathcal{D} = \operatorname{diag}(\mathcal{A})$.

> **Claim.** Jacobi preconditioning confines the coefficient-jump dependence to exactly $m_0$ small eigenvalues, where $m_0$ is the number of subdomains not touching $\Gamma_D$. The effective condition number $\kappa_{m_0+1}$ is bounded *independently of the jump* $\mathcal J(\omega)$; only the usual $h^{-2}$ mesh dependence remains.

### 3.1 Where the bad eigenvectors come from

Let
$$ I = \{\, m : \operatorname{meas}(\partial \Omega_m \cap \Gamma_D) = 0 \,\}, \qquad m_0 = \#I $$
be the set of **floating** subdomains — those insulated from the Dirichlet boundary.

*Physical picture.* Add a constant $c$ to $u$ inside one floating subdomain $\Omega_m$. Inside $\Omega_m$ the gradient is unchanged, so the energy contribution of $\Omega_m$ itself is unchanged; energy only changes in a thin layer of neighboring elements where the jump is bridged. If the neighbors have very small $\omega$, that extra energy $\sim \omega_{\text{neigh}}\,c^2$ is tiny compared to the $L^2$-mass $\sim \omega_m c^2$ of the mode. The Rayleigh quotient $\nu^t\mathcal A\nu / \nu^t\mathcal D\nu$ is therefore $\mathcal O(\mathcal J^{-1})$ — a near-null mode of $\mathcal D^{-1}\mathcal A$. There is one such (nearly) constant-patch mode per floating subdomain, i.e. $m_0$ of them.

To remove these modes from the analysis, define the subspace with zero mean on every floating subdomain:
$$ \widetilde{\mathcal{V}} = \Big\{ v \in \mathcal{V}_h : \int_{\Omega_m} v\, dx = 0 \ \ \forall m \in I \Big\}. $$
This is $m_0$ linear constraints, so $\operatorname{codim}\widetilde{\mathcal V} = m_0$ in $\mathcal V_h$.

**Poincaré–Friedrichs, uniformly in the jump.** On each floating subdomain the constant is now pinned, and on non-floating subdomains it is pinned by the Dirichlet data; hence a Poincaré/Friedrichs inequality holds *subdomain by subdomain* with constants depending only on the shapes $\{\Omega_m\}$, not on $\omega$. Summing the subdomain inequalities weighted by the (constant) $\omega_m$ preserves the weights identically on both sides:
$$
\|v\|_{0,\omega}^2 = \sum_m \omega_m \|v\|_{0,\Omega_m}^2
\le \sum_m \omega_m\, C_m\, |v|_{1,\Omega_m}^2
\le C\, |v|_{1,\omega}^2 ,
$$
so
$$ c_0\, \|v\|_{0,\omega}^2 \le |v|_{1,\omega}^2, \qquad \forall v \in \widetilde{\mathcal{V}}, \tag{PF}$$
with $c_0$ **independent of $\mathcal J(\omega)$**. This is the crucial place where the weighted norm and the mean-zero constraints cooperate: the weights cancel because $\omega$ is constant on each $\Omega_m$.

### 3.2 Bounding the effective condition number

**Lemma (Diagonal equivalence).** For a quasi-uniform mesh and linear elements,
$$ \nu^t \mathcal{D} \nu \simeq h^{-2}\, \|v\|_{0,\omega}^2 . \tag{D}$$

*Proof.* A diagonal entry is $a_{ii} = \int_\Omega \omega |\nabla\phi_i|^2$. On the support of $\phi_i$ (a patch of diameter $\sim h$) one has $|\nabla\phi_i|\sim h^{-1}$ and $\int \phi_i^2 \sim h^d$, while $\int|\nabla\phi_i|^2 \sim h^{-2}\int\phi_i^2$. Because $\omega$ is (elementwise) constant, the same weight multiplies both integrals, so
$$
a_{ii} = \int_\Omega \omega|\nabla\phi_i|^2 \simeq h^{-2}\int_\Omega \omega\,\phi_i^2 .
$$
Summing $\nu_i^2 a_{ii}$ and using the local $L^2$ equivalence $\sum_i \nu_i^2 \int\omega\phi_i^2 \simeq \int \omega v^2$ (mass-matrix / diagonal-lumping equivalence, again weight-preserving) gives (D). $\qquad\square$

**Theorem (Jacobi spectrum).** The Jacobi-preconditioned operator $\mathcal{D}^{-1}\mathcal{A}$ has at most $m_0$ small eigenvalues, and
$$ \boxed{\ \kappa_{m_0+1}(\mathcal{D}^{-1}\mathcal{A}) \lesssim h^{-2}\ } \qquad \text{independently of } \mathcal J(\omega). $$

*Proof.*

**1. Upper bound (no restriction).** By the standard inverse inequality $|v|_{1,\Omega_m}^2 \lesssim h^{-2}\|v\|_{0,\Omega_m}^2$ applied elementwise and weighted by $\omega_m$,
$$
\nu^t\mathcal A\nu = |v|_{1,\omega}^2 \lesssim h^{-2}\|v\|_{0,\omega}^2 \simeq \nu^t\mathcal D\nu ,
$$
using (D). Hence $\lambda_{\max}(\mathcal D^{-1}\mathcal A)\lesssim 1$.

**2. The full-space lower bound is jump-polluted.** Over *all* of $\mathcal V_h$ one can only guarantee
$$ \nu^t \mathcal{A} \nu \gtrsim h^{2}\, \mathcal{J}(\omega)^{-1}\, \nu^t \mathcal{D} \nu, $$
because a constant-patch mode on a floating subdomain realizes this ratio. This is exactly the $m_0$-dimensional bad space.

**3. Lower bound on the good subspace.** For $v\in\widetilde{\mathcal V}$, combine (D), (PF), and the energy identity $|v|_{1,\omega}^2 = \nu^t\mathcal A\nu$:
$$
\nu^t \mathcal{D} \nu \;\simeq\; h^{-2}\|v\|_{0,\omega}^2
\;\overset{(PF)}{\lesssim}\; h^{-2}\,|v|_{1,\omega}^2
\;=\; h^{-2}\,\nu^t\mathcal A\nu .
$$
Rearranging,
$$ \frac{\nu^t \mathcal{A}\nu}{\nu^t \mathcal{D}\nu} \gtrsim h^{2} \qquad \forall\, 0\ne v \in \widetilde{\mathcal{V}}, \tag{$\star$}$$
with a constant that does **not** involve $\mathcal J(\omega)$.

**4. Minimax counting.** The Courant–Fischer characterization gives, for any subspace $W$ of codimension $m_0$,
$$
\lambda_{m_0+1}(\mathcal D^{-1}\mathcal A)
= \max_{\operatorname{codim} S = m_0}\ \min_{0\ne v\in S} \frac{\nu^t\mathcal A\nu}{\nu^t\mathcal D\nu}
\;\ge\; \min_{0\ne v\in W} \frac{\nu^t\mathcal A\nu}{\nu^t\mathcal D\nu}.
$$
Take $W = \widetilde{\mathcal V}$ (codimension exactly $m_0$) and apply $(\star)$:
$$ \lambda_{m_0+1}(\mathcal D^{-1}\mathcal A) \gtrsim h^{2}. $$

Dividing the upper bound by this lower bound,
$$ \kappa_{m_0+1}(\mathcal D^{-1}\mathcal A) = \frac{\lambda_{\max}}{\lambda_{m_0+1}} \lesssim \frac{1}{h^{2}} = h^{-2}. \qquad\square$$

**Remark.** The identical argument, with $\mathcal D$ replaced by the symmetric Gauss–Seidel preconditioner $\mathcal B_{SGS}^{-1}$, gives the same conclusion, because $\nu^t\mathcal B_{SGS}\nu \simeq \nu^t\mathcal D\nu$ for these matrices. Thus **stationary smoothers remove the jump dependence up to $m_0$ CG-cheap outliers.**

---

## 4. Upgrading to Multigrid

Section 3 removed the *jump* dependence (modulo $m_0$ outliers) but left the standard $h^{-2}$ *mesh* dependence in $\kappa_{m_0+1}$. On fine meshes CG still slows down. Multigrid fixes precisely this.

**Idea.** In classical MG theory for the Poisson problem, the smoother (Jacobi/Gauss–Seidel) damps high-frequency error while coarse-grid corrections handle the low-frequency error, and the two together give an $h$-independent contraction. The question is whether this survives huge coefficient jumps.

**It does.** Wrap the (Gauss–Seidel) smoother in a **multigrid V-cycle**, or equivalently use a **BPX** additive preconditioner, and use the result as $\mathcal B_{MG}$ in PCG. The subspace decomposition underlying the analysis uses **weighted $L^2$-projections** onto the mesh hierarchy, which — exactly as in the Poincaré step of §3.1 — preserve the piecewise-constant weights and therefore stay uniform in $\mathcal J(\omega)$. The coarse spaces absorb the $h^{-2}$ factor attached to the *good* eigenvalues, while the $m_0$ jump-dependent modes remain isolated and CG-cheap. The result:
$$ \boxed{\ \kappa_{m_0+1}(\mathcal{B}_{MG}\mathcal{A}) \lesssim |\log h|^{2}\quad(\text{3D}),\qquad \mathcal O(1)\quad(\text{1D, 2D}).\ } $$

So the dependence on the mesh drops to at worst a mild polylogarithm, and the dependence on the contrast is confined to $m_0$ outliers that CG steps over in its first $m_0$ iterations. The resulting **Multigrid Preconditioned CG (MGCG)** converges almost uniformly, no matter how extreme the material contrast.

---

## Summary / Takeaways

1. **Two sources of ill-conditioning.** $\kappa(\mathcal A)\simeq h^{-2}\mathcal J(\omega)$: mesh *and* jump.
2. **Bad modes are localized and few.** Discontinuous coefficients produce $m_0$ near-null modes — essentially constant patches on the $m_0$ subdomains cut off from $\Gamma_D$.
3. **CG only sees the effective spectrum.** A degree-$(k)$ CG polynomial can annihilate the $m_0$ outliers exactly, so only $\kappa_{m_0+1}$ governs the asymptotic rate.
4. **Jacobi / Gauss–Seidel kill the jump dependence.** Restricting to the mean-zero subspace $\widetilde{\mathcal V}$ (codimension $m_0$) gives a jump-independent Poincaré–Friedrichs inequality; with the diagonal equivalence and minimax counting this yields $\kappa_{m_0+1}(\mathcal D^{-1}\mathcal A)\lesssim h^{-2}$.
5. **Multigrid kills the mesh dependence.** Weighted-$L^2$ coarse corrections reduce $h^{-2}$ to $|\log h|^2$ (3D) or $\mathcal O(1)$ (1D/2D).
6. **MGCG** combines both and is robust for extreme multi-material simulations.
