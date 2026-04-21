
# Lecture: High-Performance GPU Implementation & The Memory Layout Secret

**Context for today:** We have the mathematics to evaluate the action of a tensor-product operator in $\mathcal{O}(N^4)$ operations. Now, we want to run this on a GPU. GPUs thrive on structured, predictable memory access. Today, we restructure our data into a Discontinuous Galerkin (DG) style storage to achieve peak memory bandwidth, and we will reveal a non-intuitive memory layout trick that yields a 5x speedup on modern hardware.

---

## 1. The Bottleneck: Indirect Memory Access

In standard Continuous Galerkin (CG) FEM, the state vector $U$ is a flat array of global Degrees of Freedom (DoFs). 

To evaluate the local action of an operator on element $E$, we must fetch the local DoFs from the global vector using an element-to-node mapping array:
$$ u_{local} = U_{global}[\text{map}[E]] $$

**Why is this terrible for GPUs?**

*   **Scatter/Gather:** This is an indirect memory access pattern (`A[B[i]]`).
*   **Non-Coalesced Memory:** The GPU wants to load contiguous blocks of memory (e.g., 128 bytes at a time). Indirect arrays force the GPU to fetch scattered cache lines, wasting up to 80-90% of memory bandwidth.
*   **Bandwidth is King:** Matrix-free methods are almost entirely memory-bandwidth bound. If we starve the memory units, our fast $\mathcal{O}(N^4)$ math doesn't matter.

---

## 2. The Solution: Cell-Wise Storage (The Plot Twist)

To fix the memory access, we abandon the global DoF vector. We store the state exactly as it will be consumed by the elements. If a vertex is shared by 8 elements, we store that vertex's value 8 times—once for each element. 

**The Intuitive (but slow) Layout:**
Naturally, you might create a 4D array sized `[E, Z, Y, X]`, grouping all local data for Element 0 together, then Element 1, etc. 

**The Optimized Layout:** `[Z, Y, X, E]`
We put the element index **last**.

*   `Z, Y, X`: Local DoF indices within the element.
*   `E`: Element (Cell) index.

**Why? The Coalesced Access Secret:**
In C/C++/Python, the *last* dimension is contiguous in memory. 

*   If we use `[E, Z, Y, X]`, extracting a specific local node (e.g., the top-right corner) across all elements requires the GPU to jump non-contiguously through memory.
*   If we use `[Z, Y, X, E]`, extracting local node `(z, y, x)` across *all* elements is a single, perfectly contiguous memory block! Adjacent threads reading the same local node for adjacent elements hit adjacent memory addresses. 

> While we trade capacity for bandwidth by storing duplicate nodes, we must arrange those duplicates to respect the GPU's cache lines.

---

## 3. The `Volume Kernel`

We define a `volume_kernel` that applies our local Laplace operator to every cell simultaneously.
$$ L_e = M \otimes M \otimes D + M \otimes D \otimes M + D \otimes M \otimes M $$

*   **Input:** The `[Z, Y, X, E]` array.
*   **Operation:** A batched sum-factorization over the local $Z, Y, X$ dimensions.
*   **Output:** A new `[Z, Y, X, E]` array representing the local residual $v_e = L_e u_e$.

**Performance Trade-off:** 
Because `E` is last, the sum-factorization along $X, Y, Z$ happens on strided memory. This makes the `volume_kernel` *slightly* slower than if we had used `[E, Z, Y, X]`. However, as we will see next, this slight penalty buys us a massive speedup in the communication phase. 

---

## 4. Direct Stiffness Summation (DSS)

Inside the `volume_kernel`, there is zero communication between neighboring elements. The outputs on the element boundaries (faces, edges, corners) are incomplete. We must sum the shared boundaries to get the true physical values. This is called **Direct Stiffness Summation (DSS)**.

Because we are on a structured grid (e.g., a unit cube), we perform DSS using highly efficient face-swaps, dimension by dimension.

### The DSS Algorithm (Dimensional Splitting)

**Strategy: Extract Faces $\to$ Shift to Neighbors $\to$ Add Back**

**Pass 1: X-Axis Communication**
1.  **Extract:** Grab the right face `u[:, :, -1, :]` and left face `u[:, :, 0, :]` for all elements.
    *   *Here is where `[Z, Y, X, E]` shines:* Notice the `E` slice (`:`) is at the end. These faces are extracted using perfectly coalesced, contiguous memory loads!
2.  **Shift:** Shift the face data to align with neighbors (e.g., Right Face goes to Right Neighbor's Left Face).
3.  **Add:** Add the received faces directly to the current cell's boundaries.

**Pass 2: Y-Axis Communication**
1.  **Extract:** Grab the top `u[:, -1, :, :]` and bottom `u[:, 0, :, :]` faces.
    *   *The trick:* These $Y$-faces **already contain** the summed $X$-data from Pass 1 at their corners!
2.  **Shift:** Shift data up and down to the $Y$-neighbors.
3.  **Add:** Add to the current cell's $Y$-boundaries.

**Pass 3: Z-Axis Communication**
1.  **Extract:** Grab the front `u[-1, :, :, :]` and back `u[0, :, :, :]` faces. 
    *   These faces now inherently carry the accumulated $X$ and $Y$ sums at their edges and corners.
2.  **Shift:** Shift to the $Z$-neighbors.
3.  **Add:** Add to the current cell's $Z$-boundaries.

### The A100 Benchmark Reality Check

Why do we accept a slightly slower `volume_kernel`? 
*   If we use `[E, Z, Y, X]`, the face extraction in DSS forces uncoalesced memory access. DSS becomes the dominant bottleneck, starving the GPU.
*   By switching to `[Z, Y, X, E]`, DSS memory access becomes perfectly coalesced. 
*   **Hardware Result:** On an NVIDIA A100 GPU, the `[Z, Y, X, E]` layout makes the DSS algorithm **3x faster**. This completely offsets the minor volume kernel penalty, resulting in a much faster total operator evaluation.

By doing the DSS dimension-by-dimension, we naturally propagate edge and corner values across diagonal neighbors without ever having to explicitly extract 1D edges or 0D corners. Everything is dense, vectorized tensor math.


---

# Simplication: 6D Tensor Layout

Instead of a flat element index `E`, we recognize that the elements themselves form a 3D Cartesian grid. 
We expand `E` into three spatial dimensions: `EZ, EY, EX`.

**The Data Structure:** `[Z, Y, X, EZ, EY, EX]`
*   `Z, Y, X`: Local DoF indices within a single element. (Size $N \times N \times N$)
*   `EZ, EY, EX`: Global element indices in the Cartesian mesh.


1.  **Implicit Topology:** We no longer need an integer array to tell us who Element 5's neighbors are. The right neighbor of element `(ez, ey, ex)` is strictly and mathematically guaranteed to be `(ez, ey, ex + 1)`.
2.  **Coalesced Memory Maintained:** The element indices are still trailing. Extracting local node `(z, y, x)` across all elements simultaneously is still a perfectly contiguous memory read!

---

## 2. The Volume Kernel (Unchanged)

Because tensor contractions (sum-factorization) only operate on the local dimensions `Z, Y, X`, expanding the element dimension changes almost nothing mathematically.

In Einstein summation, the element dimensions simply become passive batch dimensions. 
$$ V_{z y x, \mathbf{e}} = L_{(z, y, x), (k, j, i)} U_{k j i, \mathbf{e}} $$
Where $\mathbf{e}$ is the multi-index `(ez, ey, ex)`. 

In JAX, `einsum` handles this transparently. We just use ellipses (`...`) to broadcast over the spatial grid of elements.

---

## 3. Direct Stiffness Summation (DSS) on a Cartesian Grid

Because the topology is implicit, DSS becomes incredibly elegant. We communicate by extracting the local boundary slices and shifting them along the global element axes.

**The shift logic (e.g., X-Axis):**
If I want to update my local Left Face `(x = 0)`, I need the local Right Face `(x = -1)` from my Left Neighbor.
In tensor terms, I take the `X=-1` slice of the whole mesh, shift the entire array $+1$ along the `EX` axis, and add it to the `X=0` slice.

### The Algorithm:
**Pass 1: X-Axis Communication (Dim `X` and Dim `EX`)**
*   Extract Right Faces (`x = -1`). Shift them $+1$ along `EX`. Add to Left Faces (`x = 0`).
*   Extract Left Faces (`x = 0`). Shift them $-1$ along `EX`. Add to Right Faces (`x = -1`).

**Pass 2: Y-Axis Communication (Dim `Y` and Dim `EY`)**
*   *(Remember: Because we do this sequentially, Y-faces now implicitly carry the X-corners!)*
*   Extract Top Faces (`y = -1`). Shift $+1$ along `EY`. Add to Bottom Faces (`y = 0`).
*   Extract Bottom Faces (`y = 0`). Shift $-1$ along `EY`. Add to Top Faces (`y = -1`).

**Pass 3: Z-Axis Communication (Dim `Z` and Dim `EZ`)**
*   Extract Back Faces (`z = -1`). Shift $+1$ along `EZ`. Add to Front Faces (`z = 0`).
*   Extract Front Faces (`z = 0`). Shift $-1$ along `EZ`. Add to Back Faces (`z = -1`).

> This shifting naturally enforces periodic boundary conditions! If solving a Dirichlet problem, we simply apply a binary mask to the global boundaries afterward to zero them out.

***

### Python / JAX Implementation (For Reference / GitHub)

Here is how the theoretical 6D tensor translates cleanly into JAX code. Notice the total absence of `for` loops or adjacency lists.

```python
import jax.numpy as jnp

def volume_kernel_cartesian(u, K, M):
    """
    u shape: [Z, Y, X, EZ, EY, EX]
    K, M shape: [N, N] (1D stiffness and mass matrices)
    """
    # The '...' cleanly abstracts away the EZ, EY, EX dimensions.
    # Apply X-derivative
    u_Kx = jnp.einsum('ix, zyxi... -> zyii...', K, u)
    u_Mx = jnp.einsum('ix, zyxi... -> zyii...', M, u)
    
    # Apply Y-derivative
    term_Kx_My = jnp.einsum('jy, zyi... -> zji...', M, u_Kx)
    term_Mx_Ky = jnp.einsum('jy, zyi... -> zji...', K, u_Mx)
    term_Mx_My = jnp.einsum('jy, zyi... -> zji...', M, u_Mx)
    
    # Apply Z-derivative
    out_1 = jnp.einsum('kz, zji... -> kji...', M, term_Kx_My)
    out_2 = jnp.einsum('kz, zji... -> kji...', M, term_Mx_Ky)
    out_3 = jnp.einsum('kz, zji... -> kji...', K, term_Mx_My)
    
    return out_1 + out_2 + out_3


def dss_cartesian(u):
    """
    Direct Stiffness Summation for a 6D Cartesian layout.
    u shape:[Z, Y, X, EZ, EY, EX]
    """
    
    # --- PASS 1: X-AXIS ---
    # Local X is axis 2. Global EX is axis -1.
    
    # Roll the Right Faces (-1) to the Right (+1 along EX) to hit the Left Neighbors
    inc_from_left_neighbor = jnp.roll(u[:, :, -1, ...], shift=1, axis=-1)
    # Roll the Left Faces (0) to the Left (-1 along EX) to hit the Right Neighbors
    inc_from_right_neighbor = jnp.roll(u[:, :, 0, ...], shift=-1, axis=-1)
    
    # Add them to the boundaries
    u = u.at[:, :, 0, ...].add(inc_from_left_neighbor)
    u = u.at[:, :, -1, ...].add(inc_from_right_neighbor)

    # --- PASS 2: Y-AXIS ---
    # Local Y is axis 1. Global EY is axis -2.
    
    inc_from_bottom_neighbor = jnp.roll(u[:, -1, :, ...], shift=1, axis=-2)
    inc_from_top_neighbor    = jnp.roll(u[:, 0, :, ...], shift=-1, axis=-2)
    
    u = u.at[:, 0, :, ...].add(inc_from_bottom_neighbor)
    u = u.at[:, -1, :, ...].add(inc_from_top_neighbor)

    # --- PASS 3: Z-AXIS ---
    # Local Z is axis 0. Global EZ is axis -3.
    
    inc_from_back_neighbor  = jnp.roll(u[-1, :, :, ...], shift=1, axis=-3)
    inc_from_front_neighbor = jnp.roll(u[0, :, :, ...], shift=-1, axis=-3)
    
    u = u.at[0, :, :, ...].add(inc_from_back_neighbor)
    u = u.at[-1, :, :, ...].add(inc_from_front_neighbor)

    return u
```