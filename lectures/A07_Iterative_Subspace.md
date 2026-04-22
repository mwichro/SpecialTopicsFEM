
# Lecture 7: Iterative Methods as Subspace Corrections

**Context for today:** We will connect the algebraic world of iterative solvers (like Jacobi and Gauss-Seidel) to the geometric world of Finite Elements (subspace corrections). We will see that standard solvers are actually just applying the subspace correction formula $P A_c^{-1} P^T r$ to very specific, simple subspaces.

## 1. The Richardson Method

Suppose we want to solve $A u = f$. 
If we guess a solution $u^k$, we can define the **residual** as the error in the equation:
$$ r^k = f - A u^k $$
*(If $r^k = 0$, then $u^k$ is the exact solution.)*

The **Standard Richardson Method** updates the guess by taking a small step $\omega$ in the direction of the residual:
$$ u^{k+1} = u^k + \omega r^k $$

**Preconditioned Richardson:**
Instead of a scalar $\omega$, we use a matrix $B$ that approximates the inverse of $A$ ($B \approx A^{-1}$). This "steers" and scales the residual to give a much better update.
$$ u^{k+1} = u^k + B r^k $$
$$ u^{k+1} = u^k + B(f - A u^k) $$

## 2. Combining Preconditioners

What if we have a set of several different preconditioners, $B_1, B_2, \dots, B_m$, and we want to combine them? There are two primary ways:

**1. Additive Combination (Parallel)**
We compute the correction from each preconditioner independently based on the *same* initial residual $r^k$, and then add all the corrections together.
$$ B_{add} = \sum_{i=1}^m B_i $$
$$ u^{k+1} = u^k + \left( \sum_{i=1}^m B_i \right) r^k = u^k + \sum_{i=1}^m B_i r^k $$

**2. Multiplicative Combination (Successive)**
Instead of doing everything at once, we apply them one by one, **updating the residual after every single step**. 

*Algorithm for Multiplicative Combination:*
Let $v^0 = u^k$ (start of the iteration).
For $i = 1, 2, \dots, m$:
$$ v^i = v^{i-1} + B_i (f - A v^{i-1}) $$
Let $u^{k+1} = v^m$ (end of the iteration).

> *Interpretation:* Multiplicative combination is simply running $m$ consecutive Richardson iterations, but changing the preconditioner $B_i$ at each micro-step!

---

## 3. Subspace Correction as a Preconditioner

Recall from the last lecture, if we have a subspace $V_i$, the exact correction from that subspace applied to a right-hand side $r$ is:
$$ B_i r = P_i A_i^{-1} P_i^T r $$
We can use this $B_i$ directly as a preconditioner!

**The Setup (1D Subspaces):**
Let $V$ be our full finite element space of size $N$, with standard basis functions $\phi_1, \dots, \phi_N$.
Let's define $N$ different subspaces, $V_1, V_2, \dots, V_N$.
Let each subspace $V_i$ be **1-Dimensional**, spanned by a single basis function: $V_i = \text{span}\{\phi_i\}$.

Let's carefully evaluate the terms of $B_i = P_i A_i^{-1} P_i^T$:

1.  **The Prolongation $P_i$:** Maps a scalar (from the 1D space) to the full space. Since $V_i$ represents only the $i$-th basis function, $P_i$ is simply the standard basis column vector $\mathbf{e}_i =[0, \dots, 0, 1, 0, \dots, 0]^T$ (with a 1 at index $i$).
2.  **The Restriction $P_i^T$:** Maps a full RHS vector to the 1D space. $P_i^T r = \mathbf{e}_i^T r = r_i$ (it simply extracts the $i$-th scalar component of the residual).
3.  **The Subspace Matrix $A_i$:** Using Galerkin projection: $A_i = P_i^T A P_i = \mathbf{e}_i^T A \mathbf{e}_i = A_{ii}$. This is just a $1 \times 1$ matrix (a scalar), representing the $i$-th diagonal element of $A$.
4.  **The Subspace Inverse $A_i^{-1}$:** The inverse of a scalar is just $1 / A_{ii}$.

**The 1D Subspace Preconditioner:**
Put it all together:
$$ B_i r = \mathbf{e}_i \left( \frac{1}{A_{ii}} \right) \mathbf{e}_i^T r = \mathbf{e}_i \frac{r_i}{A_{ii}} $$

> *Physical meaning:* This preconditioner $B_i$ corrects the $i$-th degree of freedom exactly, completely ignoring the rest of the system.

---

## 4. Additive Subspace Correction $\implies$ Jacobi Iteration

What happens if we take all $N$ of our 1D subspace preconditioners and combine them **Additively**?

$$ u^{k+1} = u^k + \sum_{i=1}^N B_i r^k $$
Substitute our formula for $B_i$:
$$ u^{k+1} = u^k + \sum_{i=1}^N \mathbf{e}_i \frac{r_i^k}{A_{ii}} $$

Look closely at the sum: $\sum_{i=1}^N \mathbf{e}_i \frac{r_i^k}{A_{ii}}$.
This builds a vector where the 1st component is $r_1^k/A_{11}$, the 2nd is $r_2^k/A_{22}$, etc. 
In pure matrix notation, this is exactly multiplying the residual by the inverse of the diagonal of $A$. 
Let $D$ be the diagonal matrix containing the diagonal entries of $A$.
$$ \sum_{i=1}^N B_i = D^{-1} $$

Substitute this back:
$$ u^{k+1} = u^k + D^{-1} r^k $$
$$ u^{k+1} = u^k + D^{-1} (f - A u^k) $$

**Conclusion:** Additive subspace correction on 1D basis functions is **exactly the Jacobi Iteration**.

---

## 5. Multiplicative Subspace Correction $\implies$ Gauss-Seidel Iteration

Now, let's take the exact same 1D subspaces, but combine them **Multiplicatively**. We update the residual immediately after correcting each degree of freedom.

Let's look at the $i$-th micro-step of the algorithm:
$$ v^i = v^{i-1} + B_i (f - A v^{i-1}) $$

Let $\hat{r} = f - A v^{i-1}$ be the current residual at this micro-step.
We know that $B_i \hat{r} = \mathbf{e}_i \frac{\hat{r}_i}{A_{ii}}$.
Because $B_i \hat{r}$ is zero everywhere except at index $i$, the vector $v^i$ is identical to $v^{i-1}$ everywhere except at component $i$. Let's write the formula strictly for the $i$-th scalar component:

$$ v^i_i = v^{i-1}_i + \frac{1}{A_{ii}} \left( f_i - \sum_{j=1}^N A_{ij} v^{i-1}_j \right) $$

**Analyzing the sum $\sum_{j=1}^N A_{ij} v^{i-1}_j$:**
Because we are doing this sequentially from $1$ to $N$, at step $i$:

*   For indices $j < i$: These variables have *already* been updated during this global iteration. So, $v^{i-1}_j = u^{k+1}_j$.
*   For indices $j \ge i$: These variables have *not yet* been updated. So, $v^{i-1}_j = u^k_j$.

Let's split the sum into three parts: $j < i$, $j = i$, and $j > i$:
$$ \sum_{j=1}^N A_{ij} v^{i-1}_j = \left( \sum_{j < i} A_{ij} u^{k+1}_j \right) + A_{ii} u^k_i + \left( \sum_{j > i} A_{ij} u^k_j \right) $$

Substitute this expanded sum back into the update equation for $v^i_i$:
$$ v^i_i = u^k_i + \frac{1}{A_{ii}} \left[ f_i - \sum_{j < i} A_{ij} u^{k+1}_j - A_{ii} u^k_i - \sum_{j > i} A_{ij} u^k_j \right] $$

Notice the $- A_{ii} u^k_i$ inside the brackets. When divided by the $1/A_{ii}$ outside, it becomes $-u^k_i$. 
$$ v^i_i = u^k_i - u^k_i + \frac{1}{A_{ii}} \left[ f_i - \sum_{j < i} A_{ij} u^{k+1}_j - \sum_{j > i} A_{ij} u^k_j \right] $$

The $u^k_i$ terms perfectly cancel out! Since $v^i_i$ is the final value for the $i$-th component in this sweep, we can call it $u^{k+1}_i$:
$$ u^{k+1}_i = \frac{1}{A_{ii}} \left( f_i - \sum_{j < i} A_{ij} u^{k+1}_j - \sum_{j > i} A_{ij} u^k_j \right) $$

**Conclusion:** Multiplicative subspace correction on 1D basis functions yields the exact component-wise formula for the **Gauss-Seidel Iteration**. 

>This shows that changing our solvers is nothing more than changing how we combine geometric subspaces!