



# Lecture 6: Introduction to Subspace Correction 
 We will build the mathematical machinery to transition between different bases. Today, we assume the two bases span the exact same space. In the next lecture, we will drop this assumption to solve problems on smaller, cheaper subspaces.

## 1. Re-basing the Operator

Let us define two spaces equipped with different bases. For today, assume they span the exact same underlying mathematical space:
*   **Space $V_1$** equipped with base $\mathcal{B}_1 = \{\phi_1, \phi_2, \dots, \phi_N\}$
*   **Space $V_2$** equipped with base $\mathcal{B}_2 = \{\psi_1, \psi_2, \dots, \psi_N\}$

Because both bases span the same space, any function in $V_2$ can be written as a linear combination of basis functions from $V_1$:
$$ \psi_n = \sum_{j=1}^N P_{jn} \phi_j $$
**Definition:** The matrix $P$ is the **Interpolation Matrix** (or Change of Basis matrix). Its columns contain the coefficients of the $\mathcal{B}_2$ basis functions expressed in the $\mathcal{B}_1$ basis.

### 1.1 Deriving the Operator in the New Base

Suppose we have already assembled the Laplace matrix $A_1$ using $\mathcal{B}_1$. Its entries are defined by the weak form:
$$ (A_1)_{ij} = a(\phi_j, \phi_i) = \int_{\Omega} \nabla \phi_j \cdot \nabla \phi_i \, d\Omega $$
*(Note: $j$ is the column/trial function, $i$ is the row/test function).*

We want to find the matrix $A_2$ assembled in $\mathcal{B}_2$, without re-integrating:
$$ (A_2)_{mn} = a(\psi_n, \psi_m) $$

Substitute the change-of-basis formula $\psi_n = \sum_j P_{jn} \phi_j$ and $\psi_m = \sum_i P_{im} \phi_i$:
$$ (A_2)_{mn} = a\left( \sum_{j} P_{jn} \phi_j, \sum_{i} P_{im} \phi_i \right) $$

Because the integral (bilinear form) is linear with respect to scalars, we can pull the sums and coefficients $P$ outside:
$$ (A_2)_{mn} = \sum_{i} \sum_{j} P_{im} P_{jn} \, a(\phi_j, \phi_i) $$

Substitute $(A_1)_{ij} = a(\phi_j, \phi_i)$:
$$ (A_2)_{mn} = \sum_{i} \sum_{j} P_{im} (A_1)_{ij} P_{jn} $$

To write this in pure matrix notation, recognize that $P_{im}$ is the $(i,m)$ entry of $P$, which means it is the $(m,i)$ entry of the transpose $P^T$:
$$ (A_2)_{mn} = \sum_{i} \sum_{j} (P^T)_{mi} (A_1)_{ij} P_{jn} $$
$$ A_2 = P^T A_1 P $$

### 1.2 Generalization to any Bilinear Form
This relation $A_2 = P^T A_1 P$ relies purely on the algebraic properties of bilinear forms, not on specific integrals or derivatives. Here is the formal algebraic derivation.

Let $a(u,v)$ be *any* bilinear form. By definition, evaluating this continuous form on finite element functions reduces to a vector-matrix-vector product using their discrete coefficient vectors.

In Space $V_1$ (base $\mathcal{B}_1$), let functions $u$ and $v$ be represented by the discrete coefficient vectors $\mathbf{u}_1$ and $\mathbf{v}_1$. The operator matrix $A_1$ is defined precisely such that:
$$ a(u,v) = \mathbf{v}_1^T A_1 \mathbf{u}_1 $$

Now, consider the exact same functions evaluated in Space $V_2$ (base $\mathcal{B}_2$) with coefficient vectors $\mathbf{u}_2$ and $\mathbf{v}_2$. 
By the definition of our interpolation matrix $P$ (which maps coefficients from $\mathcal{B}_2$ to $\mathcal{B}_1$), we have:
$$ \mathbf{u}_1 = P \mathbf{u}_2 \quad \text{and} \quad \mathbf{v}_1 = P \mathbf{v}_2 $$

Substitute these transformations into our bilinear form equation:
$$ a(u,v) = (P \mathbf{v}_2)^T A_1 (P \mathbf{u}_2) $$
Using the matrix transpose rule $(XY)^T = Y^T X^T$:
$$ a(u,v) = \mathbf{v}_2^T (P^T A_1 P) \mathbf{u}_2 $$

However, by definition, the matrix $A_2$ assembled in Space $V_2$ must satisfy:
$$ a(u,v) = \mathbf{v}_2^T A_2 \mathbf{u}_2 $$

Because $\mathbf{v}_2^T (P^T A_1 P) \mathbf{u}_2 = \mathbf{v}_2^T A_2 \mathbf{u}_2$ must hold true for *all* possible arbitrary vectors $\mathbf{u}_2$ and $\mathbf{v}_2$, the core matrices themselves must be identical:
$$ A_2 = P^T A_1 P \quad \blacksquare $$

## 2. Interpretation: The Four Spaces

To rigorously understand what $P$, $P^T$, $A_1$, and $A_2$ are doing, we must map out our algebraic spaces. Finite element methods naturally map functions from a **Primal** space to a **Dual** space. 

*   The **Primal space** contains the coefficients of our solution vectors.
*   The **Dual space** contains continuous linear *functionals* (e.g., $v \mapsto \int f v$). When we test these functionals against our basis functions, we get the algebraic Right-Hand Side (RHS) vectors.

Because we have two bases, we operate between **four** distinct algebraic spaces:
1.  **Primal space $V_1$** (Solutions in $\mathcal{B}_1$)
2.  **Primal space $V_2$** (Solutions in $\mathcal{B}_2$)
3.  **Dual space $V_1'$** (Functionals evaluated with $\mathcal{B}_1$ test functions / RHS vectors)
4.  **Dual space $V_2'$** (Functionals evaluated with $\mathcal{B}_2$ test functions / RHS vectors)

**Mapping the Operators:**
*   $P : V_2 \to V_1$ (Takes a solution in $\mathcal{B}_2$ and expresses it in $\mathcal{B}_1$. Often called **Prolongation**).
*   $A_1 : V_1 \to V_1'$ (Maps a solution to its dual functional representation within $V_1$).
*   $A_2 : V_2 \to V_2'$ (Maps a solution to its dual functional representation within $V_2$).
*   $P^T : V_1' \to V_2'$ (Takes a functional tested against $\mathcal{B}_1$ and projects it to be tested against $\mathcal{B}_2$. Often called **Restriction**).

**The Galerkin Projection:**
The formula $A_2 = P^T A_1 P$ is formally known as the **Galerkin Projection** of operator $A_1$ onto base $\mathcal{B}_2$.
Read from right to left: $P$ moves the state from $V_2$ to $V_1$, $A_1$ calculates the physical action (landing in $V_1'$), and $P^T$ restricts that functional back into $V_2'$.

## 3. "Full Space" Correction

Let's apply this mapping to a practical algebraic solve. 
**The Problem:** We want to solve the system $A_1 u_1 = f_1$. 
**The Catch:** We only have the RHS functional $f_1 \in V_1'$. We do *not* have the inverse of $A_1$. However, we magically have the exact inverse of $A_2$ (perhaps $\mathcal{B}_2$ diagonalizes the matrix easily, like in our previous lecture!).

Because $V_1$ and $V_2$ span the identical full space, the matrix $P$ is square and strictly invertible. 

We know the Galerkin projection:
$$ A_2 = P^T A_1 P $$

Let's isolate $A_1$:
$$ A_1 = (P^T)^{-1} A_2 P^{-1} = P^{-T} A_2 P^{-1} $$

Now, invert $A_1$. Remember that $(XYZ)^{-1} = Z^{-1} Y^{-1} X^{-1}$:
$$ A_1^{-1} = (P^{-1})^{-1} A_2^{-1} (P^{-T})^{-1} $$
$$ A_1^{-1} = P A_2^{-1} P^T $$

Substitute this into our original solve, $u_1 = A_1^{-1} f_1$:
$$ u_1 = P A_2^{-1} P^T f_1 $$

### 3.1 Interpretation of the Solve Algorithm

The equation $u_1 = P A_2^{-1} P^T f_1$ perfectly describes a 3-step physical algorithm mapping through our 4 spaces:

1.  **Transfer RHS / Restrict ($V_1' \to V_2'$):** 
    $$ f_2 = P^T f_1 $$
    We take the load functional $f_1$ and "restrict" it to the test functions of $V_2$.
2.  **Solve ($V_2' \to V_2$):** 
    $$ u_2 = A_2^{-1} f_2 $$
    We solve the dual-to-primal system exactly in Space 2.
3.  **Transfer Solution / Prolong ($V_2 \to V_1$):** 
    $$ u_1 = P u_2 $$
    We "prolong" or interpolate the solution coefficients back into Space 1.

**Crucial Note for Next Lecture:**
Everything today relied on the fact that $V_1$ and $V_2$ are the same size, making $P$ a square, invertible matrix ($N_1 = N_2$). 
*What if $V_2$ is smaller?* What if it is a strict subspace ($V_2 \subset V_1$)?
$P$ becomes a tall, rectangular matrix. $P^{-1}$ no longer exists, and $A_1^{-1} = P A_2^{-1} P^T$ is strictly false! 
However, the algorithmic operation **$P A_2^{-1} P^T f_1$** is still mathematically valid. It won't yield the exact full solution $u_1$, but it will yield an optimal **subspace correction**. This simple geometric idea forms the entire foundation of Multigrid methods!