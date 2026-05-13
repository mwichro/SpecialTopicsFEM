
# Lecture 11: Convergence Proofs of Multigrid

**Context for today:** In Lectures 6 and 7, we did a "hand-wavy" derivation of Subspace Correction. We assumed the fine and coarse spaces were the *same size*, which allowed us to use the inverse matrix $P^{-1}$. 

In true Multigrid, $P$ is a tall, rectangular matrix ($N_h \times N_H$)—it has no inverse! We will prove the Two-Grid convergence theorem focusing on the Laplace problem.

## 1. Setting the Stage and "Reasonable Assumptions"

Recall our split of the Two-Grid iteration matrix from Lecture 9:
$$ M_{TG} = \underbrace{(A_h^{-1} - P A_H^{-1} P^T)}_{\text{Approximation Property}} \underbrace{(A_h S^\nu)}_{\text{Smoothing Property}} $$

We want to prove that $\| M_{TG} \| \le \frac{C_A C_S}{\nu} < 1$. We will use the standard Euclidean matrix norm (spectral norm).

**Assumptions on the Grid and Problem:**
We are solving the Poisson problem $-\Delta u = f$. 

The proofs hold for unstructured meshes under two assumptions:

1.  **Quasi-uniformity:** The elements don't suddenly jump in size or become extremely skewed. (This ensures standard matrix eigenvalue scaling).
2.  **Convex Domain:** The physical boundary has no "re-entrant corners" (like a Pac-Man shape). In PDE theory, a convex domain guarantees *Full Elliptic Regularity* (specifically $H^2$-regularity), meaning the exact solution is sufficiently smooth if the RHS is smooth. We need this so our Finite Element approximation bounds hold.

---


## 2. Proof of the Approximation Property

**Goal:** Prove that for any vector $f_h$, $\| (A_h^{-1} - P A_H^{-1} P^T) f_h \| \le \frac{C_A}{\|A_h\|} \|f_h\|$.

This proof connects pure linear algebra to Finite Element error theory. We do not use matrix inverses like $P^{-1}$. We only use the Galerkin property: $A_H = P^T A_h P$.

**Step 1: Identify the components.**
Let's apply the operator to a generic Right-Hand Side vector $f_h$.
*   Let $u_h = A_h^{-1} f_h$. This is the **exact solution** on the fine grid.
*   Let $u_H = A_H^{-1} P^T f_h$. This is the **coarse grid correction** (a vector in $\mathbb{R}^{N_H}$).
*   The fine-grid representation of the coarse correction is $P u_H$.

Therefore, the approximation operator measures exactly the difference between the fine solution and the coarse correction:
$$ (A_h^{-1} - P A_H^{-1} P^T) f_h = u_h - P u_H $$

**Step 2: The A-Orthogonality (Galerkin) Condition.**
Let's look at the residual of our coarse correction evaluated on the fine grid:
$$ r_{new} = f_h - A_h (P u_H) $$
Let's restrict this residual to the coarse grid (multiply by $P^T$):
$$ P^T r_{new} = P^T f_h - P^T A_h P u_H $$
Substitute our definition of $A_H$:
$$ P^T r_{new} = P^T f_h - A_H u_H $$
Since $u_H = A_H^{-1} P^T f_h$, we know $A_H u_H = P^T f_h$. Therefore:
$$ P^T r_{new} = 0 $$
This means $P^T A_h (u_h - P u_H) = 0$. The error $u_h - P u_H$ is $A$-orthogonal to the coarse space. This confirms that $P u_H$ is mathematically the "best possible" fit for $u_h$ in the coarse space (an orthogonal projection).

**Step 3: Applying Finite Element Approximation Theory.**
Because $P u_H$ is the best fit, bounding $u_h - P u_H$ is a standard Finite Element error analysis problem. We don't need to reinvent the wheel; we borrow the classic $L_2$ error bound (Céa's Lemma + Aubin-Nitsche duality trick).

For the Laplace problem with linear elements, the $L_2$ error between a fine grid solution ($h$) and a coarse grid solution ($H$) scales with the square of the coarse element size:
$$ \| u_h - P u_H \|_{\text{discrete } L_2} \le C_{fem} H^2 \| f_h \|_{\text{discrete } L_2} $$
> The above relies on our assumption of a convex domain.

**Step 4: Connecting Physics ($H^2$) to Linear Algebra ($\|A_h\|$).**
How do we get from $H^2$ to the matrix norm $\|A_h\|$?
Recall the standard Finite Element stiffness matrix for the Laplacian. Because it approximates a 2nd spatial derivative, its largest eigenvalues scale inversely with the square of the grid spacing $h$:
$$ \lambda_{\max}(A_h) = \|A_h\| \approx \frac{c}{h^2} \implies h^2 \approx \frac{c}{\|A_h\|} $$

Because our grids are nested (standard refinement), the coarse grid spacing is exactly twice the fine grid spacing: $H = 2h$.
$$ H^2 = 4h^2 \approx \frac{4c}{\|A_h\|} $$

**Step 5: The Final Bound.**
Substitute the $H^2$ scaling back into our FEM error bound:
$$ \| u_h - P u_H \| \le C_{fem} \left( \frac{4c}{\|A_h\|} \right) \|f_h\| $$

Let $C_A = 4 c C_{fem}$. We obtain:
$$ \| (A_h^{-1} - P A_H^{-1} P^T) f_h \| \le \frac{C_A}{\|A_h\|} \|f_h\| $$

**Conclusion:** The coarse grid captures the physical PDE behavior up to an error bounded by $C_A / \|A_h\|$.  $\blacksquare$

---

## 3. Proof of the Smoothing Property

**Goal:** Prove that $\| A_h S^\nu \| \le \frac{C_S}{\nu} \|A_h\|$.


Let us use the preconditioned Richardson method as our smoother, where the preconditioner is simply a scaled identity matrix. The iteration updates the guess by taking a step $\omega$ in the direction of the residual: 
$$u^{k+1} = u^k + \omega (f_h - A_h u^k).$$
 This yields the smoother iteration matrix $S = I - \omega A_h$. 

To ensure that this method strictly reduces error and acts as a proper smoother, we must choose $\omega$ based on the maximum eigenvalue of the fine grid matrix. Let $L = \|A_h\| = \lambda_{\max}(A_h)$. By choosing the damping parameter $\omega = 1/L$, our smoother iteration matrix is explicitly defined as:
$$ S = I - \frac{1}{L} A_h $$

Because $A_h$ and $S$ share the same eigenvectors, we can analyze this purely through their eigenvalues. 
If $\lambda_i$ is an eigenvalue of $A_h$, the corresponding eigenvalue of the operator $A_h S^\nu$ is:
$$ \mu_i = \lambda_i \left( 1 - \frac{\lambda_i}{L} \right)^\nu $$

The norm $\|A_h S^\nu\|$ is simply the maximum absolute value of $\mu_i$ across all eigenvalues $0 < \lambda_i \le L$.
Let's define a continuous function for $x \in [0, L]$ and find its maximum using standard calculus:
$$ g(x) = x \left( 1 - \frac{x}{L} \right)^\nu $$

**Step 1: Take the derivative and set to zero.**
$$ g'(x) = 1 \cdot \left( 1 - \frac{x}{L} \right)^\nu + x \cdot \nu \left( 1 - \frac{x}{L} \right)^{\nu - 1} \left( -\frac{1}{L} \right) = 0 $$

Factor out the common term $(1 - x/L)^{\nu-1}$:
$$ \left( 1 - \frac{x}{L} \right) - \frac{x \nu}{L} = 0 $$
$$ 1 = \frac{x}{L} + \frac{x \nu}{L} = \frac{x}{L}(1 + \nu) $$

Solving for the critical point $x^*$:
$$ x^* = \frac{L}{\nu + 1} $$

**Step 2: Evaluate the maximum.**
Substitute $x^*$ back into $g(x)$:
$$ g(x^*) = \frac{L}{\nu + 1} \left( 1 - \frac{L / (\nu+1)}{L} \right)^\nu $$
$$ g(x^*) = \frac{L}{\nu + 1} \left( 1 - \frac{1}{\nu + 1} \right)^\nu $$

**Step 3: Bound the maximum.**
Look at the term in the parenthesis. For any integer $\nu \ge 1$, the term $(1 - \frac{1}{\nu+1})^\nu$ is strictly less than 1. (In fact, as $\nu \to \infty$, it approaches $1/e$).
Therefore:
$$ g(x^*) \le \frac{L}{\nu + 1} < \frac{L}{\nu} $$

Substitute $L = \|A_h\|$ back in:
$$ \| A_h S^\nu \| \le \frac{1}{\nu} \|A_h\| $$

**Conclusion:** The constant is exactly $C_S = 1$. The smoother removes the residual inversely proportional to the number of smoothing steps $\blacksquare$

---

## Convergence of multigrid

When we multiply the Approximation factor by the Smoothing factor ($\|A_h\| / \nu$), the grid dependencies completely cancel out, giving us unconditional, mesh-independent convergence $\blacksquare$




## 4. Polynomial Smoothers (Chebyshev)

**Context:** In the previous proof, we applied the exact same Richardson step $\nu$ times. We can view this from the perspective of polynomials. 

If we apply $S = I - \frac{1}{L}A_h$ for $\nu$ iterations, the eigenvalues of the initial error are multiplied by the polynomial:
$$ P_\nu(\lambda) = \left( 1 - \frac{\lambda}{L} \right)^\nu $$
Notice two things about this polynomial:
1.  $P_\nu(0) = 1$. (The solver does not touch the zero-eigenvalue/constant mode).
2.  In our proof, we essentially bounded the function $g(\lambda) = \lambda P_\nu(\lambda)$ over the interval $\lambda \in [0, L]$.

### 4.1 The Optimization Problem

Can we choose a *better* polynomial? 
What if, instead of using a constant damping parameter $\omega = 1/L$, we use a different parameter $\omega_k$ at every single micro-step $k = 1, \dots, \nu$?
The iteration matrix for the whole pre-smoothing phase becomes a product:
$$ S_{poly} = \prod_{k=1}^\nu (I - \omega_k A_h) $$
And the eigenvalues are now scaled by a general polynomial of degree $\nu$:
$$ P_\nu(\lambda) = \prod_{k=1}^\nu (1 - \omega_k \lambda) $$

**The Multigrid Objective:** We don't need the smoother to kill *all* error. The coarse grid correction completely handles the low frequencies (near $\lambda \approx 0$). We only need the smoother to aggressively annihilate the high frequencies. 

Let's define a high-frequency cutoff, $\lambda_{cut}$. (For standard coarsening, high frequencies typically live in the upper portion of the spectrum, e.g., $\lambda \in [\lambda_{cut}, L]$). 

We want to find the optimal polynomial $P_\nu(\lambda)$ that satisfies:
1.  $P_\nu(0) = 1$ (Required so the formula corresponds to valid iteration steps $I - \omega A_h$).
2.  **Minimizes the maximum absolute value** over the high-frequency range $[\lambda_{cut}, L]$.

### 4.2 The Chebyshev Solution

This specific "minimax" optimization problem is a famous one in approximation theory. The mathematical, provably optimal solution is given by a shifted and scaled **Chebyshev Polynomial of the first kind**.

Standard Chebyshev polynomials $T_\nu(x)$ oscillate between $[-1, 1]$ on the interval $[-1, 1]$. 
By shifting the interval to $[\lambda_{cut}, L]$ and scaling so that the value at $\lambda = 0$ is exactly $1$, we get our optimal smoothing polynomial:
$$ P_\nu^{Cheb}(\lambda) = \frac{T_\nu \left( \frac{L + \lambda_{cut} - 2\lambda}{L - \lambda_{cut}} \right)}{T_\nu \left( \frac{L + \lambda_{cut}}{L - \lambda_{cut}} \right)} $$

**Why?**
*   The Richardson polynomial $(1 - \lambda/L)^\nu$ gently slopes downward. 
*   The Chebyshev polynomial $P_\nu^{Cheb}(\lambda)$ drops rapidly from $1$ at $\lambda=0$, and then *oscillates* extremely close to $0$ across the entire high-frequency interval $[\lambda_{cut}, L]$. 
*   Because it optimizes the worst-case high-frequency mode, the maximum value in that interval drops **exponentially** with the degree $\nu$. For the same number of FLOPs, a Chebyshev smoother can easily provide an order of magnitude more error reduction than standard damped Jacobi.

### 4.3 Implementation via 3-Term Recurrence

You might think we need to compute the roots of the Chebyshev polynomial to find the individual parameters $\omega_k$. While mathematically valid, doing this sequentially (e.g., $v^k = v^{k-1} + \omega_k r^{k-1}$) is known to be highly numerically unstable. If the roots are applied in the wrong order, intermediate vectors can explode to infinity.

Instead, because Chebyshev polynomials satisfy a recursive relation ($T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)$), we can implement the Chebyshev smoother iteratively using a **3-term recurrence**. 

To compute the next step $u^{k+1}$, we don't just use the current step $u^k$ and the residual $r^k$; we also incorporate the *previous* step $u^{k-1}$:
$$ u^{k+1} = \alpha_k u^k + (1 - \alpha_k) u^{k-1} + \beta_k r^k $$

*(Where $\alpha_k$ and $\beta_k$ are cheap scalar coefficients derived directly from the bounds $L$ and $\lambda_{cut}$).*

This requires storing one extra vector in memory, but it guarantees completely stable execution. We have upgraded our standard smoother into an optimal, high-performance filter for our Multigrid cycle!



### 4.4 Accelerating Other Smoothers (Preconditioned Chebyshev)

**The Conceptual Shift:** 
Chebyshev smoothing does not have to act on the raw stiffness matrix $A_h$. It acts as an algebraic "accelerator" for *any* baseline smoother, provided that smoother can be represented by a Symmetric Positive Definite (SPD) preconditioner $B$. (For example, $B$ could be the diagonal Jacobi matrix $D^{-1}$, or the exact block-solver derived in our Fast Diagonalization lecture). 

Instead of applying the polynomial to $A_h$, we apply it to the **preconditioned operator** $\hat{A} = B A_h$. 

**Strict Formulation:**
To construct the polynomial, we no longer look at the eigenvalues of $A_h$. We must estimate the spectrum of the preconditioned system $\hat{A}$.
Let the eigenvalues of $B A_h$ be bounded by:
$$ 0 < \lambda_{\min} \le \lambda_i \le \lambda_{\max} $$
We define our high-frequency target interval as $[\lambda_{cut}, \lambda_{\max}]$. (Often in multigrid, $\lambda_{cut} \approx \lambda_{\max}/4$ depending on the coarsening factor).

We want to solve the preconditioned system $B A_h u_h = B f_h$. 
Let $z^k = B(f_h - A_h u^k) = B r^k$ be the **preconditioned residual**. This requires one evaluation of your base smoother.

The formal 3-term Chebyshev recurrence for the preconditioned system is defined as follows. First, we define the interval center $d$ and radius $c$:
$$ d = \frac{\lambda_{\max} + \lambda_{cut}}{2}, \quad c = \frac{\lambda_{\max} - \lambda_{cut}}{2} $$

**The Preconditioned Algorithm:**
*   **Step 1:** $u^1 = u^0 + \frac{1}{d} z^0$
*   **For $k \ge 1$:**
    1. Compute raw residual: $r^k = f_h - A_h u^k$
    2. Apply base smoother: $z^k = B r^k$
    3. Update extrapolation factors (derived from Chebyshev roots):
       $$ \alpha_k = \frac{2d}{2d^2 - c^2} \quad \text{(for } k=1) $$
       $$ \alpha_k = \left( d - \frac{c^2}{4} \alpha_{k-1} \right)^{-1} \quad \text{(for } k > 1) $$
       $$ \beta_k = \alpha_k d - 1 $$
    4. Apply the strictly bounded step:
       $$ u^{k+1} = u^k + \alpha_k z^k + \beta_k (u^k - u^{k-1}) $$



<!-- We have strictly decoupled the physics from the algebra. The matrix $B$ handles the localized physical coupling (e.g., solving an entire cell block exactly to handle element anisotropy). The Chebyshev recurrence handles the global algebraic spectrum, actively shifting the search direction $\alpha_k z^k$ using the history term $\beta_k (u^k - u^{k-1})$ to guarantee the mathematically optimal attenuation of high-frequency eigenvectors of $B A_h$. -->