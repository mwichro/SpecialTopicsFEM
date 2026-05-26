
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
This means $P^T A_h (u_h - P u_H) = 0$. The error $u_h - P u_H$ is $A$-orthogonal to the coarse space. This confirms that $P u_H$ is the "best possible" fit for $u_h$ in the coarse space (an orthogonal projection).

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

This specific "minimax" optimization problem is a famous one in approximation theory. The provably optimal solution is given by a shifted and scaled **Chebyshev Polynomial of the first kind**.

Standard Chebyshev polynomials $T_\nu(x)$ oscillate between $[-1, 1]$ on the interval $[-1, 1]$. 
By shifting the interval to $[\lambda_{cut}, L]$ and scaling so that the value at $\lambda = 0$ is exactly $1$, we get our optimal smoothing polynomial:
$$ P_\nu^{Cheb}(\lambda) = \frac{T_\nu \left( \frac{L + \lambda_{cut} - 2\lambda}{L - \lambda_{cut}} \right)}{T_\nu \left( \frac{L + \lambda_{cut}}{L - \lambda_{cut}} \right)} $$

**Why?**
*   The Richardson polynomial $(1 - \lambda/L)^\nu$ gently slopes downward. 
*   The Chebyshev polynomial $P_\nu^{Cheb}(\lambda)$ drops rapidly from $1$ at $\lambda=0$, and then *oscillates* extremely close to $0$ across the entire high-frequency interval $[\lambda_{cut}, L]$. 
*   Because it optimizes the worst-case high-frequency mode, the maximum value in that interval drops **exponentially** with the degree $\nu$. For the same number of FLOPs, a Chebyshev smoother can easily provide an order of magnitude more error reduction than standard damped Jacobi.

### 4.3 Implementation via 3-Term Recurrence

You might think we need to compute the roots of the Chebyshev polynomial to find the individual parameters $\omega_k$. While valid, doing this sequentially (e.g., $v^k = v^{k-1} + \omega_k r^{k-1}$) is known to be highly numerically unstable. If the roots are applied in the wrong order, intermediate vectors can explode to infinity.

Instead, because Chebyshev polynomials satisfy a recursive relation ($T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)$), we can implement the Chebyshev smoother iteratively using a **3-term recurrence**. 

To compute the next step $u^{k+1}$, we don't just use the current step $u^k$ and the residual $r^k$; we also incorporate the *previous* step $u^{k-1}$:
$$ u^{k+1} = \alpha_k u^k + (1 - \alpha_k) u^{k-1} + \beta_k r^k $$

*(Where $\alpha_k$ and $\beta_k$ are cheap scalar coefficients derived directly from the bounds $L$ and $\lambda_{cut}$).*

This requires storing one extra vector in memory, but it guarantees completely stable execution. We have upgraded our standard smoother into an optimal, high-performance filter for our Multigrid cycle!


<!-- DERIVE IP PROPPERLY!! -->
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


Here is the detailed, blackboard-ready derivation of the Chebyshev 3-term recurrence. You can use this to replace or significantly expand Section 4 of Lecture 11.

***

# Deep Dive: Deriving the Chebyshev 3-Term Recurrence

**Context:** We know that applying a smoother $k$ times multiplies the error by a polynomial $P_k(A)$. We want this polynomial to be a Chebyshev polynomial to optimally squash high-frequency errors. However, calculating the roots of a high-degree polynomial and applying them sequentially is numerically unstable. Today, we will derive a stable, iterative algorithm directly from the definition of Chebyshev polynomials.

For notation simplicity, let $A$ be our operator (this could be the raw matrix $A_h$, or the preconditioned matrix $B A_h$). Let its high-frequency eigenvalues be bounded by $[\lambda_{\min}, \lambda_{\max}]$.

## 1. The Chebyshev Mapping

Standard Chebyshev polynomials of the first kind, denoted $T_k(x)$, are defined on the interval $x \in [-1, 1]$.
They satisfy the 3-term recurrence:
$$ T_0(x) = 1 $$
$$ T_1(x) = x $$
$$ T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x) $$

Our matrix eigenvalues live in $\lambda \in [\lambda_{\min}, \lambda_{\max}]$. We must linearly map this interval to $[-1, 1]$. 
Let's define the center $d$ and radius $c$ of our eigenvalue interval:
$$ d = \frac{\lambda_{\max} + \lambda_{\min}}{2}, \quad c = \frac{\lambda_{\max} - \lambda_{\min}}{2} $$
The function that maps $\lambda_{\min} \to 1$ and $\lambda_{\max} \to -1$ is:
$$ x(\lambda) = \frac{d - \lambda}{c} $$


## 2. The Iteration Polynomial $P_k(\lambda)$ and the Consistency Condition

We know that applying our smoother $k$ times multiplies the initial error by some matrix polynomial: $e^k = P_k(A) e^0$. But we cannot just pick *any* polynomial to squash the eigenvalues. The polynomial must represent a valid solver. 

To see what makes a polynomial "valid," let's look at how iterative solvers are actually built.

### 2.1 The Residual is the Only Guide

Every standard iterative solver (Richardson, Jacobi, Conjugate Gradient, Chebyshev) operates on a single fundamental principle: **we only update the solution based on the residual.** 
If we start with $u^0$, any subsequent update is formed by multiplying the residual $r^0 = f - Au^0$ by various matrices (like our preconditioner $B$) and adding them up. 

Mathematically, after $k$ steps, the total accumulated correction to our initial guess is just some matrix polynomial, let's call it $Q_{k-1}(A)$, applied to the initial residual:
$$ u^k = u^0 + Q_{k-1}(A) r^0 $$

### 2.2 Translating the Update to the Error Polynomial

Let's convert this update equation into an error equation. 
Substitute the definitions of the exact solution $u$ and the error $e^k = u - u^k$:
$$ (u - e^k) = (u - e^0) + Q_{k-1}(A) r^0 $$

Cancel the exact solution $u$ from both sides, and multiply by $-1$:
$$ e^k = e^0 - Q_{k-1}(A) r^0 $$

Now, substitute the definition of the residual in terms of the error ($r^0 = A e^0$):
$$ e^k = e^0 - Q_{k-1}(A) A e^0 $$
$$ e^k = \big( I - A \, Q_{k-1}(A) \big) e^0 $$

We have just derived the exact form of our error polynomial matrix:
$$ P_k(A) = I - A \, Q_{k-1}(A) $$

### 2.3 The Strict Consistency Condition at $\lambda = 0$

Now, let's look at what this matrix polynomial does to the scalar eigenvalues $\lambda$ of the matrix $A$:
$$ P_k(\lambda) = 1 - \lambda \, Q_{k-1}(\lambda) $$

What happens if we evaluate this polynomial at exactly $\lambda = 0$?
$$ P_k(0) = 1 - 0 \cdot Q_{k-1}(0) = 1 $$

This is the **Consistency Condition**.
*   **Physical Meaning:** If our initial guess is already the exact solution, the error is zero, and the residual is zero. The solver must not do anything. It must leave the solution untouched. If $P_k(0)$ were, say, $0.5$, it would imply the solver is actively shrinking the exact solution toward zero, which is mathematically invalid!

### 2.4 Scaling the Chebyshev Polynomial

We want to use the Chebyshev polynomial $T_k(x)$ because it has the optimal oscillating properties to kill high frequencies. But there is a problem: if we blindly set $P_k(\lambda) = T_k(x(\lambda))$, it will almost certainly violate the consistency condition ($T_k(x(0)) \neq 1$).

To fix this, we simply scale the entire polynomial by a constant so that its value at $\lambda = 0$ is exactly 1. 

Let the mapped coordinate at the origin ($\lambda=0$) be $x(0) = \frac{d}{c}$.
We define the scaling factor $\sigma_k$ as the raw Chebyshev value at this origin point:
$$ \sigma_k = T_k\left(\frac{d}{c}\right) $$

Our valid, properly normalized smoothing polynomial is therefore:
$$ P_k(\lambda) = \frac{T_k(x(\lambda))}{\sigma_k} $$

By construction, $P_k(0) = \frac{\sigma_k}{\sigma_k} = 1$. We now have an optimal, high-frequency-destroying polynomial that perfectly obeys the algebraic laws of iterative solvers.
## 3. Deriving the Recurrence for the Error

Let's plug our mapped coordinate $x(\lambda)$ into the standard Chebyshev recurrence:
$$ T_{k+1}(x(\lambda)) = 2 \left( \frac{d - \lambda}{c} \right) T_k(x(\lambda)) - T_{k-1}(x(\lambda)) $$

Now, substitute $T_k = \sigma_k P_k$ for all terms to convert this into a recurrence for our smoothing polynomials:
$$ \sigma_{k+1} P_{k+1}(\lambda) = 2 \left( \frac{d - \lambda}{c} \right) \sigma_k P_k(\lambda) - \sigma_{k-1} P_{k-1}(\lambda) $$

Divide everything by $\sigma_{k+1}$ to isolate $P_{k+1}(\lambda)$:
$$ P_{k+1}(\lambda) = \frac{2 \sigma_k}{c \sigma_{k+1}} (d - \lambda) P_k(\lambda) - \frac{\sigma_{k-1}}{\sigma_{k+1}} P_{k-1}(\lambda) $$

Since the error at step $k$ is $e^k = P_k(A) e^0$, we can replace the scalar $\lambda$ with the matrix $A$ and apply it to the initial error:
$$ e^{k+1} = \frac{2 d \sigma_k}{c \sigma_{k+1}} e^k - \frac{2 \sigma_k}{c \sigma_{k+1}} A e^k - \frac{\sigma_{k-1}}{\sigma_{k+1}} e^{k-1} $$

## 4. The Magic Cancellation: From Error to Solution

We cannot compute the error $e^k$ directly in code (because we don't know the exact solution $u$). We must rewrite this equation in terms of the iterate $u^k$.
Substitute $e^k = u - u^k$ into the recurrence:
$$ u - u^{k+1} = \frac{2 d \sigma_k}{c \sigma_{k+1}} (u - u^k) - \frac{2 \sigma_k}{c \sigma_{k+1}} A (u - u^k) - \frac{\sigma_{k-1}}{\sigma_{k+1}} (u - u^{k-1}) $$

Notice the term $A(u - u^k)$. Since $Au = f$, this is exactly the residual $r^k$! Let's substitute $r^k$ and group all the exact solution $u$ terms together:
$$ -u^{k+1} = u \left[ -1 + \frac{2 d \sigma_k}{c \sigma_{k+1}} - \frac{\sigma_{k-1}}{\sigma_{k+1}} \right] - \frac{2 d \sigma_k}{c \sigma_{k+1}} u^k - \frac{2 \sigma_k}{c \sigma_{k+1}} r^k + \frac{\sigma_{k-1}}{\sigma_{k+1}} u^{k-1} $$

**The Magic Step:** Look at the bracketed term multiplying $u$. 
Remember how we defined $\sigma_k$? It comes from the Chebyshev recurrence at the point $d/c$:
$$ \sigma_{k+1} = 2 \left(\frac{d}{c}\right) \sigma_k - \sigma_{k-1} $$
If we divide this equation by $\sigma_{k+1}$, we get exactly $1 = \frac{2 d \sigma_k}{c \sigma_{k+1}} - \frac{\sigma_{k-1}}{\sigma_{k+1}}$.
Therefore, the bracket evaluates to exactly zero! **The unknown exact solution $u$ completely cancels out.**

Multiply the remaining terms by $-1$:
$$ u^{k+1} = \frac{2 d \sigma_k}{c \sigma_{k+1}} u^k + \frac{2 \sigma_k}{c \sigma_{k+1}} r^k - \frac{\sigma_{k-1}}{\sigma_{k+1}} u^{k-1} $$

## 5. Cleaning up the Coefficients ($\alpha_k$ and $\beta_k$)

Let's define a clean step-size coefficient $\alpha_k$ for the residual term:
$$ \alpha_k = \frac{2 \sigma_k}{c \sigma_{k+1}} $$

What is the coefficient for the $u^{k-1}$ term? From our "magic step" identity above, we know that:
$$ \frac{\sigma_{k-1}}{\sigma_{k+1}} = \frac{2 d \sigma_k}{c \sigma_{k+1}} - 1 $$
Substitute $\alpha_k$ into this:
$$ \frac{\sigma_{k-1}}{\sigma_{k+1}} = d \alpha_k - 1 $$
Let's call this term $\beta_k = d \alpha_k - 1$. 

Now rewrite the main $u^{k+1}$ equation using $\alpha_k$ and $\beta_k$:
$$ u^{k+1} = (d \alpha_k) u^k + \alpha_k r^k - \beta_k u^{k-1} $$
Since $d \alpha_k = \beta_k + 1$, we can expand this:
$$ u^{k+1} = (\beta_k + 1) u^k + \alpha_k r^k - \beta_k u^{k-1} $$

Rearrange to get the beautiful, standard algorithm format:
$$ u^{k+1} = u^k + \alpha_k r^k + \beta_k (u^k - u^{k-1}) $$

*(Note for the board: This looks exactly like gradient descent with a momentum term!)*

## 6. The Sequence of $\alpha_k$

Finally, we need a way to calculate $\alpha_k$ in code without actually computing the massive scalar values of $\sigma_k$, which might overflow.

Start with the base definition: $\sigma_{k+1} = 2 \left(\frac{d}{c}\right) \sigma_k - \sigma_{k-1}$.
Divide by $\sigma_k$:
$$ \frac{\sigma_{k+1}}{\sigma_k} = \frac{2d}{c} - \frac{\sigma_{k-1}}{\sigma_k} $$

Using our definition $\alpha_k = \frac{2}{c} \frac{\sigma_k}{\sigma_{k+1}}$, we can invert it to find $\frac{\sigma_{k+1}}{\sigma_k} = \frac{2}{c \alpha_k}$.
Similarly, shifting the index gives $\frac{\sigma_{k-1}}{\sigma_k} = \frac{c \alpha_{k-1}}{2}$.

Substitute these into the divided recurrence:
$$ \frac{2}{c \alpha_k} = \frac{2d}{c} - \frac{c \alpha_{k-1}}{2} $$
Multiply the whole equation by $c/2$:
$$ \frac{1}{\alpha_k} = d - \frac{c^2}{4} \alpha_{k-1} $$
$$ \alpha_k = \left( d - \frac{c^2}{4} \alpha_{k-1} \right)^{-1} $$

**What about step 1 ($k=1$)?** 
At $k=1$, we don't have a $u^{-1}$ to use the momentum term. We just do a standard Richardson step: $u^1 = u^0 + \alpha_1 r^0$.
From definition, $\sigma_0 = T_0(d/c) = 1$, and $\sigma_1 = T_1(d/c) = d/c$.
$$ \alpha_1 = \frac{2 \sigma_1}{c \sigma_2} = \frac{2 (d/c)}{c (2(d/c)(d/c) - 1)} = \frac{2d}{2d^2 - c^2} $$

**Conclusion:** We have derived the 3-term recurrence algorithm. We compute $\alpha_k$ iteratively using only scalars $d$ and $c$, and update the vectors completely safely without ever explicitly knowing the polynomial roots!