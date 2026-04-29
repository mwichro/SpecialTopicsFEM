

# Lecture 9: Introduction to Multigrid

**Context for today:** We have established two critical facts. First, local solvers like Damped Jacobi perfectly annihilate high-frequency error but stall on smooth, low-frequency error. Second, subspace correction allows us to solve problems on different bases. Today, we combine these to cure the stalling problem once and for all.

## 1. Nested Grids and Nested Spaces

Instead of a single grid, imagine we have two grids:
*   A **fine grid** $\Omega_h$ with spacing $h$. 
*   A **coarse grid** $\Omega_H$ created by removing every other node (spacing $H = 2h$).

These grids define two Finite Element spaces: the fine space $V_h$ and the coarse space $V_H$. 
Crucially, these spaces are **nested**: 
$$ V_H \subset V_h $$
This means *any* function that can be drawn using the coarse basis functions can be perfectly recreated using a linear combination of the fine basis functions. 

## 2. Prolongation and Restriction

Because $V_H \subset V_h$, we have a natural, exact mapping from the coarse space to the fine space. 
Let $\Phi_i^H$ be a coarse basis function, and $\phi_j^h$ be the fine basis functions. 
$$ \Phi_i^H = \sum_{j} P_{ji} \phi_j^h $$

This defines our **Prolongation Matrix**, $P$. 
*   $P : V_H \to V_h$
*   It takes a coarse solution vector $u_H$ and interpolates it to the fine grid: $u_h = P u_H$.

Following our Subspace Correction derivation (Lecture 6), the **Restriction Matrix** is simply the transpose:
$$ R = P^T $$

### 2.1 Why $R = P^T$? (Primal vs. Dual Spaces)

*A common mistake:* Why can't we just take a fine vector and geometrically interpolate/sample it on the coarse grid to build $R$? 

Because there is a fundamental mathematical difference between a **Function** and a **Functional**.
*   **Primal Space ($V_h$):** Contains solutions (functions). Geometrically interpolating a function makes sense. $P$ maps Primal $\to$ Primal.
*   **Dual Space ($V_h'$):** Contains Right-Hand Sides and Residuals ($f_h$, $r_h$). These are *functionals*, representing integrals against test functions: $(r_h)_j = \int r \phi_j^h$. 

If we have a residual $r_h$ on the fine grid, we want to know what it looks like tested against the *coarse* basis $\Phi_i^H$. 
Substitute the expansion of the coarse basis:
$$ (r_H)_i = \int r \Phi_i^H = \int r \left( \sum_{j} P_{ji} \phi_j^h \right) = \sum_{j} P_{ji} \left( \int r \phi_j^h \right) $$
$$ (r_H)_i = \sum_{j} (P^T)_{ij} (r_h)_j $$
$$ r_H = P^T r_h $$
**Conclusion:** We use $P^T$ because we are transferring a *functional* from the fine Dual space $V_h'$ to the coarse Dual space $V_H'$. It is not a geometric sampling; it is a change-of-basis for integrals!

## 3. The Coarse Grid Correction (CGC) Preconditioner

Using our Subspace Correction formula from Lecture 7, the subspace $V_H$ defines a preconditioner:
$$ B_{CGC} = P A_H^{-1} P^T $$
Where $A_H = P^T A_h P$ is the Galerkin projection of the operator onto the coarse grid.

**What does $B_{CGC}$ do?**
If we apply this preconditioner $u^{k+1} = u^k + B_{CGC} r^k$, it solves the problem *exactly* for any error that can be represented on the coarse grid. 
Since the coarse grid can only represent smooth, low-frequency shapes, $B_{CGC}$ completely annihilates low-frequency error! However, it is utterly blind to high-frequency zigzags that fall between the coarse nodes. 

## 4. The Two-Grid Method

We now have two complementary "preconditioners":
1.  $B_{smooth}$ (e.g., Damped Jacobi): Destroys high frequencies. Stalls on low frequencies.
2.  $B_{CGC}$ (Coarse Grid Correction): Destroys low frequencies. Blind to high frequencies.

Let's combine them **multiplicatively** (successively applying Richardson steps updating the residual). We will use a symmetric 3-step combination:
1.  **Pre-smooth:** $v^1 = v^0 + B_{smooth} (f - A_h v^0)$
2.  **Coarse Grid Correction:** $v^2 = v^1 + B_{CGC} (f - A_h v^1)$
3.  **Post-smooth:** $v^3 = v^2 + B_{smooth} (f - A_h v^2)$

This is the **Two-Grid Method**. 
Its overall iteration matrix is exactly the product of the individual iteration matrices:
$$ M_{TG} = (I - B_{smooth}A_h) (I - P A_H^{-1} P^T A_h) (I - B_{smooth}A_h) $$
Because the smoothers and the coarse grid solver perfectly cover the entire frequency spectrum (recall our thought experiment from Lecture 8), this method converges incredibly fast.

## 5. From Two-Grid to Multigrid

There is one major computational flaw in the Two-Grid method: evaluating $B_{CGC}$ requires computing $A_H^{-1}$. 
If our fine grid has 10 million DOFs, our coarse grid still has 1.25 million DOFs (in 3D). We cannot exactly invert a matrix of that size!

**The solution is recursion.** 
How do we apply $A_H^{-1}$ to the coarse residual? We treat it as a brand-new PDE problem: $A_H e_H = r_H$.
Instead of solving it exactly, we approximate $A_H^{-1}$ by calling the Two-Grid method again! We smooth on grid $H$, restrict the residual to an even coarser grid $2H$, and so on. We only compute an exact inverse at the very bottom, on a grid with just a few elements.

## 6. The V-Cycle Algorithm

We can clean up this recursive logic into the standard **V-Cycle Algorithm**. 

**Function:** $u_h = \text{VCycle}(A_h, u_h, f_h)$
1. **Pre-smooth:** 
   Apply $\nu_1$ iterations of smoother (e.g., $u_h \leftarrow u_h + \omega D^{-1} (f_h - A_h u_h)$).
2. **Compute Residual:** 
   $r_h = f_h - A_h u_h$
3. **Transfer Down (Restriction):** 
   $r_H = P^T r_h$
4. **Recursion (Coarse Grid Solve):** 
   *    *If*: on the coarsest grid: $e_H = A_H^{-1} r_H$ (Exact solve)
   *    *Else*: $e_H = \text{VCycle}(A_H, \mathbf{0}, r_H)$  *(Note: initial guess for error is 0)*
5. **Transfer Up & Add (Prolongation):** 
   $u_h \leftarrow u_h + P e_H$
6. **Post-smooth:** 
   Apply $\nu_2$ iterations of smoother.
   **Return** $u_h$.

## 7. A Note on Problem Dependency

A Multigrid solver is assembled from distinct parts, some universal and some highly problem-dependent.

*   **Transfers ($P$ and $P^T$):** These are largely independent of the PDE. They depend entirely on geometry and the chosen Finite Element spaces (e.g., evaluating linear basis functions). 
*   **The Smoother:** This is *highly* problem-dependent. The smoother must be tailored to the physics of $A_h$.
    *   *Damped Jacobi:* Simple, but requires tuning the relaxation parameter $\omega$ based on the eigenvalues of the specific PDE.
    *   *Gauss-Seidel:* Can be highly effective, but its asymmetry means the direction of the sweep matters (especially for advection-dominated PDEs).
    *   *Advanced Smoothers:* If we are solving complex, highly coupled systems (like block systems on elements), pointwise Jacobi fails. We can use advanced local solvers—like the **Fast Diagonalization** method from Lecture 5! By doing an exact block-solve locally, we create an incredibly powerful, matrix-free smoother for Multigrid.




## 8. The Convergence Theorem of the Two-Grid Method

To prove that the Two-Grid method converges, we must derive its iteration matrix $M_{TG}$ and show that its norm is strictly less than 1. For simplicity, we will analyze a V-cycle with $\nu$ **pre-smoothing** steps and **no post-smoothing**.

### 8.1 Deriving the Iteration Matrix

Let $e^0$ be the initial error. We apply our two preconditioners multiplicatively:


**1. Pre-smoothing ($\nu$ steps):**
Recall from Lecture 8 that a single iteration of a preconditioned solver ($u^{k+1} = u^k + B_{smooth} r^k$) modifies the error according to its iteration matrix. Let's denote the iteration matrix for our specific smoother as $S$:
$$ S = I - B_{smooth}A_h $$

If we apply exactly *one* step of this smoother to our initial guess, the new error $e^1$ is:
$$ e^1 = S e^0 $$

However, in Multigrid, we usually apply the smoother multiple times in succession to aggressively damp out the high frequencies. If we apply it a second time, we multiply the error by $S$ again:
$$ e^2 = S e^1 = S (S e^0) = S^2 e^0 $$

By induction, if we apply the smoother $\nu$ times consecutively, we are simply compounding the iteration matrix $\nu$ times. Therefore, the error immediately *after* the pre-smoothing phase (which we will call $e^{pre}$) is:
$$ e^{pre} = S^\nu e^0 $$
**2. Coarse Grid Correction (CGC):**
The CGC preconditioner is $B_{CGC} = P A_H^{-1} P^T$. Its iteration matrix is $(I - B_{CGC}A_h)$. Applying this to our smoothed error gives the final error:
$$ e^1 = (I - P A_H^{-1} P^T A_h) e^{pre} $$

Substituting $e^{pre}$, the total Two-Grid iteration matrix is:
$$ M_{TG} = (I - P A_H^{-1} P^T A_h) S^\nu $$

### 8.2 The Magical Algebraic Split

At first glance, bounding $\| M_{TG} \|$ seems difficult because $M_{TG}$ mixes fine matrices, coarse matrices, and smoothers. 
We can cleverly separate these effects by inserting the identity matrix $I = A_h^{-1} A_h$ exactly in the middle of our formula:

$$ M_{TG} = (I - P A_H^{-1} P^T A_h) \mathbf{A_h^{-1} A_h} S^\nu $$

Distribute the $A_h^{-1}$ into the left parentheses:
$$ M_{TG} = (A_h^{-1} - P A_H^{-1} P^T A_h A_h^{-1}) A_h S^\nu $$
$$ M_{TG} = \underbrace{(A_h^{-1} - P A_H^{-1} P^T)}_{\text{Factor 1}} \underbrace{(A_h S^\nu)}_{\text{Factor 2}} $$

This split is the heart of Multigrid theory. It perfectly isolates the two mechanisms of the solver.

### 8.3 The Two Ingredients

Let's define the mathematical properties of our two factors. We use the standard matrix norm $\|\cdot\|$, where $\|A_h\|$ roughly represents the highest eigenvalue (the highest frequency on the fine grid).

**Ingredient 1: The Approximation Property**
The first factor measures how well the coarse grid inverse approximates the fine grid inverse. Because the coarse grid captures low frequencies perfectly, the only difference between the exact inverse and the coarse inverse lies in the highly oscillatory high frequencies. 
Mathematically, the error of this approximation scales inversely with the highest frequency of the grid ($\|A_h\|$):
$$ \exists C_A > 0 \quad \text{such that} \quad \| (A_h^{-1} - P A_H^{-1} P^T) v \| \le \frac{C_A}{\|A_h\|} \| v \| $$
*(Where $C_A$ is a constant depending purely on the finite element interpolation, not the grid size $h$.)*

**Ingredient 2: The Smoothing Property**
The second factor, $A_h S^\nu$, measures the effect of the smoother. Remember that $A_h e^{pre} = r^{pre}$. This factor essentially bounds the residual after smoothing.
Because a good smoother acts specifically to aggressively damp high eigenvalues, taking more steps ($\nu$) rapidly reduces this norm:
$$ \exists C_S > 0 \quad \text{such that} \quad \| A_h S^\nu v \| \le \frac{C_S}{\nu} \|A_h\| \| v \| $$
*(Where $C_S$ depends on the chosen smoother).*

### 8.4 The "Self-Proving" Convergence Theorem

Now that we have algebraically isolated these properties, the proof of Multigrid convergence writes itself.

**Theorem:** For a sufficiently large number of smoothing steps $\nu$, the Two-Grid method unconditionally converges ($\| M_{TG} \| < 1$) independent of the grid size $h$.

**Proof:**
We want to bound the norm of the overall iteration matrix acting on an arbitrary error vector $v$.
$$ \| M_{TG} v \| = \left\| (A_h^{-1} - P A_H^{-1} P^T) (A_h S^\nu v) \right\| $$

Let the smoothed vector be $w = A_h S^\nu v$. Apply the **Approximation Property** to $w$:
$$ \| M_{TG} v \| \le \frac{C_A}{\|A_h\|} \| w \| = \frac{C_A}{\|A_h\|} \| A_h S^\nu v \| $$

Now, apply the **Smoothing Property** to the remaining term:
$$ \| M_{TG} v \| \le \frac{C_A}{\|A_h\|} \left( \frac{C_S}{\nu} \|A_h\| \| v \| \right) $$

Notice how the highly grid-dependent $\|A_h\|$ term perfectly cancels out!
$$ \| M_{TG} v \| \le \frac{C_A C_S}{\nu} \| v \| $$

Therefore, the spectral norm of the Two-Grid iteration matrix is bounded by:
$$ \| M_{TG} \| \le \frac{C_A C_S}{\nu} $$

To guarantee convergence, we just need $\| M_{TG} \| < 1$. Because $C_A$ and $C_S$ are constants independent of the mesh size $h$, we simply choose the number of pre-smoothing steps $\nu$ such that:
$$ \nu > C_A C_S $$
The method converges, and importantly, the convergence rate does not degrade as we refine the mesh. $\blacksquare$




## 9. The $\gamma$-Cycle (V-Cycles vs. W-Cycles)

In our standard recursive algorithm (Section 6), we handled the coarse grid correction by calling the recursive function exactly *once* per level. What if the coarse grid problem is particularly difficult, and one recursive pass doesn't solve $A_H e_H = r_H$ accurately enough? 

We can generalize our algorithm by introducing a parameter **$\gamma$**, which dictates exactly how many times we apply the coarse grid correction at each level.

### 9.1 Modifying the Algorithm

Let's rewrite the recursive step (Step 4) of our Multigrid algorithm to include $\gamma$. 

**Function:** $u_h = \text{MGCycle}(A_h, u_h, f_h, \gamma)$
1. **Pre-smooth:** Apply $\nu_1$ iterations of smoother.
2. **Compute Residual:** $r_h = f_h - A_h u_h$
3. **Restrict:** $r_H = P^T r_h$
4. **Recursion ($\gamma$ times):** 
   If on the coarsest grid: 
       $e_H = A_H^{-1} r_H$ (Exact solve)
   Else:
       Set initial guess $e_H = \mathbf{0}$
       **For $i = 1$ to $\gamma$:**
           $e_H = \text{MGCycle}(A_H, e_H, r_H, \gamma)$
5. **Prolong & Add:** $u_h \leftarrow u_h + P e_H$
6. **Post-smooth:** Apply $\nu_2$ iterations of smoother.
   **Return** $u_h$.

*(Crucial detail for the board: Notice that in the loop, the first iteration uses an initial guess of $\mathbf{0}$, but the subsequent iterations use the updated error $e_H$ from the previous pass!)*

> Draw the $v$-cycle ($\gamma =1$ ) and $w$-cycle ($\gamma=2$ ) on the board. There should be vertical axis: refinement level, finest grid on top, horizontal axis: step of execution. Mark smoothing with full circles, residual evaluation with open circles, and transfers with arrows.  Take picture from https://en.wikipedia.org/wiki/Multigrid_method#Computational_cost

### 9.2 $\gamma = 1$: The V-Cycle

If $\gamma = 1$, we recover our standard algorithm. We go straight down to the coarsest grid, and straight back up. 
If we trace the execution of the grids over time, it draws the letter **V**.

*   **Pros:** The absolute cheapest cycle per iteration. Very fast.
*   **Cons:** The approximation of the exact Two-Grid inverse $A_H^{-1}$ might be quite loose, leading to a poorer convergence rate per cycle for tough PDEs.

 

### 9.3 $\gamma = 2$: The W-Cycle

If $\gamma = 2$, every time we drop to a coarse grid, we solve it twice before returning to the fine grid. 
If we trace the grid execution over time for 3 levels, it looks like this:
1. Down to Grid 2
2. Down to Grid 3 (Solve exactly)
3. Up to Grid 2
4. **Down to Grid 3 again!** (Solve exactly)
5. Up to Grid 2
6. Up to Grid 1

Tracing this path visually draws the letter **W**.

*   **Pros:** By solving the coarse problem twice, the W-cycle drastically improves the accuracy of the Coarse Grid Correction. It behaves much closer to the idealized Two-Grid mathematical proof. It is highly robust for difficult, ill-conditioned PDEs.
*   **Cons:** It requires more computational work per global iteration. 

### 9.4 Computational Complexity (Is the W-Cycle too expensive?)

You might worry that branching twice at every level causes an explosion in computational cost. Let's look at the asymptotic complexity.

Let $W_h$ be the computational work (FLOPs) required to do smoothing and transfers on a grid with $N$ elements.
Because we split cells in every dimension, a coarse grid in $d$-dimensions has roughly $1/2^d$ the number of elements.
So, the work on the coarse grid is $W_H \approx \frac{1}{2^d} W_h$.

The total work for a $\gamma$-cycle is bounded by the geometric series:
$$ \text{Total Work} = W_h + \gamma W_H + \gamma^2 W_{HH} + \dots = W_h \sum_{k=0}^{\text{levels}} \left( \frac{\gamma}{2^d} \right)^k $$

For this series to converge to a small constant bounded by $\mathcal{O}(N)$, we strictly need:
$$ \frac{\gamma}{2^d} < 1 \implies \gamma < 2^d $$

**The Takeaway:**
*   In **1D ($d=1$)**: $2^1 = 2$. So $\gamma=2$ makes the ratio $2/2 = 1$. The W-cycle in 1D is technically $\mathcal{O}(N \log N)$, losing strict linear scaling!
*   In **2D ($d=2$)**: $2^2 = 4$. So $\gamma=2$ gives a ratio of $2/4 = 1/2$. The W-cycle retains perfect $\mathcal{O}(N)$ complexity.
*   In **3D ($d=3$)**: $2^3 = 8$. The ratio is $2/8 = 1/4$. The W-cycle is cheap relative to the fine grid work! 


> In practice your milagage may vary: it all depends on the smoother. It is all about cost, $w$-cycle generates more work on coarser level and there migh just not be enough work to fully utilize the comutatitonal resources. 

> **Personal opinion**: I have never seen any benefit from using $w$-cycle.