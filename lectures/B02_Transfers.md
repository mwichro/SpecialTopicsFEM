
# Lecture 12: Implementation of Transfer Operators (Discontinuous Storage)

**Context for today:** We have derived the math for Multigrid and the tensor-product structure of our transfer operators. Today, we map this math to actual data structures. We are using continuous Finite Elements, but we store them **discontinuously** (element-by-element). We will prove a beautiful property: by cleverly ordering our operations, the Restriction operator requires absolutely *zero* inter-cell communication on the fine grid.

## 1. Data Structures: The 6D Tensor

Because we are on a Cartesian grid, we don't need a complex unstructured mesh adjacency graph. We can store our entire field in a single, dense 6D tensor.

Let the fine grid have $E_x, E_y, E_z$ elements in each direction.
Let the polynomial degree be $p$, so there are $N = p+1$ 1D DoFs per element.

Our data is a tensor of shape `(Ex, Ey, Ez, N, N, N)`.
*   Indices `(ex, ey, ez)` identify the macroscopic Cartesian cell.
*   Indices `(i, j, k)` identify the microscopic local DoF within that cell.

**The Parent-Child Mapping:**
On a Cartesian grid, identifying which coarse cell (parent) owns a fine cell (child) is trivial integer division.
For a fine cell at `(ex, ey, ez)`, its parent coarse cell is at:
$$ (px, py, pz) = (ex \, // \, 2, \ ey \, // \, 2, \ ez \, // \, 2) $$

Its local "child index" $\mathbf{c} = (c_x, c_y, c_z) \in \{0,1\}^3$ within that parent is given by the modulo:
$$ (cx, cy, cz) = (ex \ \% \ 2, \ ey \ \% \ 2, \ ez \ \% \ 2) $$

## 2. Continuous Elements in Discontinuous Storage

Because we store elements independently, DoFs on the boundaries between cells are physically duplicated in memory. 

To formalize this, we define the **Extract** (or Gather) operator $E$ and its transpose, the **Assemble** (or Scatter-Add) operator $E^T$.
*   Let $V_{glob}$ be the true, continuous global space (unique DoFs).
*   Let $V_{loc}$ be our 6D tensor space (duplicated DoFs).
*   **Extract ($E : V_{glob} \to V_{loc}$):** Copies continuous global values into the duplicated local element arrays.
*   **Assemble ($E^T : V_{loc} \to V_{glob}$):** Takes local element arrays and sums them at the shared boundaries to produce the continuous global vector.

When computing a standard PDE residual $A_{glob} u_{glob} = f_{glob}$, we evaluate the integrals locally and then assemble:
$$ r_{glob} = f_{glob} - E^T A_{loc} E u_{glob} $$
This $E^T$ represents our **inter-cell exchange** (e.g., MPI communication or shared-memory atomic adds).

## 3. The Commutator Proof: Element-Wise Restriction

To restrict the residual to the coarse grid, we mathematically require $r_H^{glob} = P_{glob}^T r_h^{glob}$. 
This implies we must first fully assemble the fine residual ($E_h^T$), and then apply the global restriction ($P_{glob}^T$). 

Instead, we will prove we can use the **unassembled** local fine residual $r_h^{loc}$.

**Theorem:** Restricting an assembled residual is mathematically equivalent to restricting the unassembled residual purely element-by-element, and then assembling on the coarse grid.
$$ P_{glob}^T E_h^T r_h^{loc} = E_H^T P_{loc}^T r_h^{loc} $$

**Proof:**
1.  Consider the Prolongation of a continuous coarse vector: $u_h^{glob} = P_{glob} u_H^{glob}$.
2.  Let's look at this prolonged vector strictly inside a single fine element (apply $E_h$):
    $$ u_h^{loc} = E_h (P_{glob} u_H^{glob}) $$
3.  Because basis functions are strictly zero outside their elements, evaluating the prolonged function inside a fine element only depends on the coarse parent element it belongs to. 
    Therefore, we get the exact same result if we first extract the coarse parent element ($E_H$), and then apply the dense local prolongation matrix ($P_{loc}$) derived in Lecture 10:
    $$ u_h^{loc} = P_{loc} (E_H u_H^{glob}) $$
4.  Since these two paths are identical for *any* vector $u_H^{glob}$, the linear operators must commute:
    $$ E_h P_{glob} = P_{loc} E_H $$
5.  Take the transpose of both sides (remembering $(XY)^T = Y^T X^T$):
    $$ P_{glob}^T E_h^T = E_H^T P_{loc}^T \quad \blacksquare $$


Look at the right side of the equation: $E_H^T (P_{loc}^T r_h^{loc})$.
*   $P_{loc}^T r_h^{loc}$: This is an entirely local, element-by-element operation. No neighbor communication. We simply take the local 6D fine tensor, apply our tensor-contraction restriction, and write the result directly into a local coarse 6D tensor.
*   $E_H^T$: The inter-cell assembly is pushed to the *coarse* grid. The coarse grid has $8\times$ fewer elements and $8\times$ fewer boundary nodes to communicate!

---

## 4. Implementation Algorithm for Restriction

Based on our proof, the algorithm to compute the coarse right-hand side (residual) from a fine grid state is as follows:

**Input:** Continuous fine solution guess $u_h^{glob}$.

1.  **Extract:** Populate the fine 6D tensor.
    `u_h_loc = extract(u_h_glob)`
2.  **Local Action:** Compute the unassembled local residual tensor.
    `r_h_loc = f_h_loc - local_laplace(u_h_loc)`
3.  **Local Restriction (Element-Wise):** Loop over all fine elements. Determine their parent and child-index $(c_x, c_y, c_z)$. Apply the corresponding 1D restriction matrices (Lecture 10) to `r_h_loc` and accumulate the result directly into the parent's unassambled local tensor `r_H_loc`.
    *(No communication happens here!)*
4.  **Coarse Assembly:** Perform the inter-cell exchange on the *coarse* tensor to get the continuous coarse right-hand side.
    `r_H_glob = assemble(r_H_loc)`


## 6. SOTA Matrix-Free Solvers (e.g., deal.II) and the "Valence" Story

In state-of-the-art (SOTA) matrix-free frameworks like **deal.II**, **MFEM**, we never build global matrices.  

### 6.1 The Overcounting Problem

Recall our two mapping operators:
*   **Extract ($E$):** Copies global continuous values to local elements.
*   **Assemble ($E^T$):** Sums local element values into the global continuous vector.

Notice that $E^T E$ is *not* the Identity matrix.
If you take a global vector of 1s, Extract it to the elements, and Assemble it back ($E^T E \mathbf{1}$), the interior nodes will still be 1, a node on a face shared by 2 elements will become 2. A corner node in a 3D Cartesian mesh shared by 8 elements will become 8.

This integer representing how many elements share a specific Degree of Freedom is called its **Valence** (or Multiplicity). Let $V$ be the diagonal matrix of these valences.
$$ E^T E = V $$

### 6.2 Prolongation in SOTA Codes

Let's look at Prolongation: $u_h = P u_H$. 
We start with a continuous coarse solution $u_H^{glob}$.
1.  Extract to coarse elements: $u_H^{loc} = E_H u_H^{glob}$
2.  Apply local 1D tensor prolongations: $u_h^{loc} = P_{loc} u_H^{loc}$

Now we have the correct fine values inside every element. But we need to build the continuous global vector $u_h^{glob}$. If we just use the Assemble operator $E_h^T u_h^{loc}$, the shared boundary nodes will be added together and artificially multiplied by their valence!

To fix this, SOTA codes explicitly store a precomputed **Inverse Valence** vector (a diagonal matrix $W_h = V_h^{-1}$). 
To get the correct continuous Primal solution, we Assemble and then weight by the inverse valence:
$$ u_h^{glob} = W_h E_h^T u_h^{loc} $$

Combining the steps, the true global Prolongation operator is:
$$ P_{glob} = W_h E_h^T P_{loc} E_H $$

### 6.3 Restriction and the "Partition of Unity"

Because Multigrid relies strictly on the Galerkin condition, the Restriction operator must be the exact algebraic transpose of Prolongation:
$$ R_{glob} = P_{glob}^T = E_H^T P_{loc}^T E_h W_h $$

Read this operator from right to left to see exactly how a solver like `deal.II` restricts an assembled fine residual $r_h^{glob}$:
1.  **Weight by Inverse Valence ($W_h r_h^{glob}$):** We take the global residual and divide it by the number of sharing elements. For example, if a face node has a residual of 10 and is shared by 2 elements, we assign 5 to each.
2.  **Extract ($E_h$):** We copy these divided values into the local element storage.
3.  **Local Restriction ($P_{loc}^T$):** We apply our fast 1D tensor contractions on each element independently.
4.  **Coarse Assemble ($E_H^T$):** We sum the results across the coarse cell boundaries.

*Physical Interpretation:* When restricting a global residual, we conceptually "split" the fine-grid functional among the elements that share it (creating a partition of unity). Each element restricts its own fraction of the residual to the coarse grid, and the coarse assembly naturally pieces it all perfectly back together.

### 6.4 Valence in the Smoother (Matrix-Free Jacobi)

This inverse-valence vector $W$ is reused constantly in SOTA codes, most notably for the smoother. 
If we want to run Damped Jacobi ($u^{k+1} = u^k + \omega D^{-1} r^k$), we need the global diagonal $D$ of the stiffness matrix.

How to compute the diagonal matrix-free?
1.  We loop over all cell. For every loop over all local DoFs, we 
2.  To get the global diagonal, we simply assemble it! $D_{glob} = E^T D_{loc}$.

If we want to apply the Jacobi preconditioner to a residual, we don't divide the residual by valence—we just divide by this correctly assembled global diagonal $D_{glob}$. 

**Summary for Implementation:**
In a modern matrix-free code, the only extra memory overhead required for the entire Multigrid hierarchy is the storage of the 1D transfer matrices (tiny) and one global `inverse_multiplicity` vector ($W$) per grid level. Everything else is pure, on-the-fly tensor math.