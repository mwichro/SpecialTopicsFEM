
# Lecture 3: JAX, GPU Architecture, and the Memory Wall



## 1. What is JAX (Really)?

At first glance, JAX looks exactly like "NumPy on the GPU." But under the hood, it is fundamentally different. Standard Python is an interpreted language, meaning it reads and executes code line-by-line. This is notoriously slow. For massive numerical problems like Matrix-Free FEM, we need to use as much of available compute power as possible. 

JAX solves this by acting as a **trace-based compiler for numerical computing**. You write Python, but JAX translates it into highly optimized machine code.

### The Core Concepts of JAX

JAX requires functions to be **pure** (meaning they don't change global variables or mutate input arrays; the same inputs always yield the same outputs). Because the code is pure, JAX can safely apply powerful transformations:

1.  **`vmap` (Vectorization):** 
    *   *Vectorization* means executing a single operation on a large block of data simultaneously, rather than iterating through it one by one. 
    *   *How `vmap` works:* `vmap` takes a function you wrote to process just **one single cell**, and automatically transforms it into a function that processes an entire **array of cells** all at once on the hardware level. You never write a spatial `for` loop; you write the math for one element, and `vmap` scales it to the whole mesh.
2.  **`jit` (Just-In-Time Compilation):** This is the compilation engine of JAX. 
3.  **`grad` (Automatic Differentiation):** Automatically computes the exact derivative of your functions. 

### How `jit` and XLA Work

When you decorate a function with `@jax.jit`, JAX executes your Python code *once*. But instead of passing it real numbers, it passes "tracers"—abstract objects that just say "I am an array of size 100x100."  JAX records every mathematical operation applied to these tracers and builds an intermediate computational graph. It then hands this graph to a compiler called **XLA (Accelerated Linear Algebra)**. 

To understand why XLA is so crucial, we first must define the biggest bottleneck in scientific computing: **Memory Bandwidth**.

### Interlude: What is Memory Bandwidth?

In modern computers (especially GPUs), performing mathematical operations (like addition or multiplication) is practically instantaneous. The hardware is so fast that the compute cores spend most of their time sitting idle. 

Why are they idle? Because they are waiting for the numbers to arrive from the main memory (RAM, or in case of GPU: VRAM). 

**Memory Bandwidth** is the rate at which data can be transferred from the main memory to the compute cores. 
*   Think of the compute core as a massive, high-speed factory.
*   Think of main memory as a warehouse far away.
*   Think of Memory Bandwidth as the width of the highway connecting them. 

No matter how fast the factory works, total performance is strictly limited by how many trucks can drive down the highway at once. In Matrix-Free solvers, we are almost always **Memory Bound**, meaning our speed is entirely dictated by memory bandwidth, not computational limits.



### Interlude 2: Arithmetic Intensity (FLOPs per Byte)

To strictly quantify whether a program is memory-bound or compute-bound, we use a crucial metric called **Arithmetic Intensity**. 

**Definition:** Arithmetic Intensity is the ratio of mathematical operations performed to the amount of memory moved.
$$ \text{Arithmetic Intensity} = \frac{\text{Total FLOPs (Floating Point Operations)}}{\text{Total Memory Bytes Transferred}} $$

**The Hardware Reality (Machine Balance):**
Modern GPUs are severely imbalanced. Their ability to do math has scaled much faster than their ability to load memory. 
* A modern GPU (like an NVIDIA A100) can perform roughly **10 to 20 FLOPs for every 1 Byte** of data it reads from main memory (this ratio is even higher for lower precision math).
* If your algorithm's Arithmetic Intensity is $< 10$, the compute cores sit idle waiting for data. You are **Memory Bound**.
* If your algorithm's Arithmetic Intensity is $> 10$, the memory keeps up with the cores. You are **Compute Bound**.

### Why Sparse Matrices on Modern Hardware are bad

In traditional FEM, we assemble a massive, global, sparse stiffness matrix $A$. To solve the system, the most common operation is the Sparse Matrix-Vector product (SpMV): $y = A x$.

Let's calculate the Arithmetic Intensity of evaluating a single non-zero entry in this sparse matrix.

**1. The Compute (FLOPs):**
For a single non-zero entry $A_{ij}$, the operation is $y_i = y_i + A_{ij} x_j$.
This requires 1 multiplication ($A_{ij} \times x_j$) and 1 addition.
* **Total Compute: 2 FLOPs.**

**2. The Memory (Bytes):**
To do this one operation, the hardware must read the data from HBM. How much data?
* We must read the matrix value $A_{ij}$ (Standard double precision float = 8 bytes).
* Because the matrix is sparse, the computer doesn't know where $A_{ij}$ belongs. We must also read the column index `j` from an array (Standard integer = 4 bytes).
* *(We are ignoring the reads/writes for $x$ and $y$, which make this even worse).*
* **Total Memory Read: 12 Bytes.**

**3. The Arithmetic Intensity:**
$$ \text{Intensity of SpMV} = \frac{2 \text{ FLOPs}}{12 \text{ Bytes}} \approx \mathbf{0.16 \text{ FLOPs/Byte}} $$

**The Conclusion:**
A hardware GPU needs an intensity of ~15 to run efficiently, but Sparse Matrix-Vector multiplication yields **0.16**. 
This means when doing traditional assembled FEM on a GPU, **the hardware operates at roughly 1% to 2% of its peak computational capacity.** The insanely fast, expensive compute cores do absolutely nothing while waiting for the 12 bytes of sparse matrix data to inch its way down the memory highway.

### The Matrix-Free Solution

This hardware reality is the entire motivation for this course. 

Matrix-Free FEM completely abandons the assembled matrix $A$. Instead of loading $A_{ij}$ and its indices from memory, we only load the vector $x$ and the raw geometry of the Cartesian cell. We then use our tensor contractions (sum factorization) to **recompute** the action of the stiffness matrix on the fly.

* **We do vastly more math:** We are re-evaluating basis functions, Jacobians, and Kronecker products every single iteration. (High FLOPs).
* **We load almost no data:** We never store or read the massive assembled matrix. (Low Bytes).

By deliberately throwing away stored data and forcing the computer to recalculate the physics from scratch every time, we artificially boost our Arithmetic Intensity from 0.16 up into the 10+ range. **Matrix-Free methods trade cheap, abundant FLOPs to save precious, expensive memory bandwidth.**

### The magic of XLA: Operation Fusion

Now, let's look at why XLA is critical for overcoming the memory bandwidth bottleneck.

Imagine evaluating a standard mathematical expression like: `D = (A * B) + C`.

**Without XLA (Standard Python/NumPy execution):**
The code executes step-by-step.
1.  Load arrays `A` and `B` from the slow main memory over the highway.
2.  Multiply them on the chip to create a temporary array, `T1`.
3.  **Write `T1` back out to the slow main memory.**
4.  Load `T1` and `C` from the slow main memory.
5.  Add them on the chip to create `D`.
6.  **Write `D` back to the slow main memory.**

This requires 4 round-trips over our slow highway. The computer is suffocating on data movement.

**With XLA (Fused Compilation):**
XLA analyzes the entire computational graph beforehand. It realizes it doesn't need to save the intermediate steps to main memory. It generates a single, fused piece of machine code:
1.  Load `A`, `B`, and `C` into the ultra-fast registers directly on the chip.
2.  Multiply `A * B` and immediately add `C` in one fluid hardware step.
3.  Write `D` back to main memory.

**The result:** We eliminated the useless trips to main memory. 
In Matrix-Free FEM, our Einstein summations (`jnp.einsum`) generate massive, complex graphs. XLA compiles these into monolithic GPU kernels that keep data in fast on-chip memory as long as possible, completely bypassing the memory wall. This is why our Python code can run at the absolute physical limits of the GPU.

---


## 2. GPU Architecture 101: Hardware, Memory, and Tensor Cores

To understand why XLA's fusion is critical, and why we structure our math as tensor contractions, we need to look at the physical GPU. A GPU is not just a fast CPU; it is a specialized throughput machine built for massive parallelism.

### 2.1 The Compute Engine: Cores and Warps
* **Streaming Multiprocessors (SMs):** A modern GPU consists of many SMs (e.g., an NVIDIA A100 has 108 SMs). Each SM contains multiple execution cores.
* **Warps:** Threads don't execute totally independently. They are bundled into groups of 32, called a **Warp**. All 32 threads in a warp execute the *exact same instruction* at the *exact same time*, just on different data (SIMT: Single Instruction, Multiple Threads).

### 2.2 Tensor Cores and MMA (Matrix-Multiply-Accumulate)
Historically, standard GPU cores performed **FMA (Fused Multiply-Add)** on scalars: `d = a * b + c`. One operation per thread, per clock cycle. 

Modern GPUs (starting significantly with Volta, and perfected in the A100/H100) dedicate vast amounts of physical silicon to **Tensor Cores**.
* **What they do:** Instead of scalar math, a Tensor Core executes a hardware-level **MMA (Matrix-Multiply-Accumulate)** instruction. It takes small block matrices (e.g., 4x4 or 8x8) and computes $D = A \times B + C$ in a *single clock cycle*.
* **Why it matters for us:** Our Matrix-Free sum factorization (e.g., applying 1D operators to slices of a 2D grid) translates perfectly into small block matrix multiplications. When XLA compiles our `jnp.einsum` code, it maps these dense local contractions directly onto the Tensor Cores. This yields a massive, order-of-magnitude leap in FLOPs compared to traditional sparse-matrix solvers which cannot use Tensor Cores.

### 2.3 Hardware Support for Transposition
In tensor contractions (like switching from processing horizontal slices to vertical slices), we are constantly permuting indices. Mathematically, this is **transposition**.
* **The historical problem:** Transposing a matrix requires reading rows and writing columns. This destroys memory coalescing, resulting in catastrophic memory slowdowns.
* **The hardware solution:** Modern GPUs feature specialized instructions (like `ldmatrix`) and shared memory routing that allow them to load a tile of data from main memory and **transpose it on-the-fly** as it moves into the cores, with near-zero performance penalty. When we write `einsum('ij, kj -> ik', ...)`, the hardware handles the index swapping natively.

### 2.4 The Memory Hierarchy
As discussed, computing is fast, but feeding the cores is hard. 
1. **HBM (High Bandwidth Memory) / Global Memory:** The main GPU memory. Very large, but relatively slow and far from the cores.
2. **Shared Memory (SRAM):** Tiny (e.g., ~164 KB per SM), but located right next to the cores. It is roughly 100x faster than HBM and acts as a user-managed cache. All threads in a block can read/write to the same Shared Memory.
3. **Registers:** The absolute fastest memory. Private to each individual thread.

> Note 1: In comparison to CPU RAM, HBM is very close to the GPU chip, it is connected directly via interposer. Those units are soldered toghether 

> Note 2: HBM is a bunch of chips stacked vertically and soldered toghether. That is why it is so expesive.

### 2.5 The Reality Check: A100 GPU vs. High-End CPU
To put these architectural differences into perspective, let's look at the raw physical limits of a top-tier server CPU (e.g., 64-core AMD EPYC) versus an NVIDIA A100 GPU.


| Metric | NVIDIA A100 (80GB) | NVIDIA Grace | AMD EPYC 9754 (Bergamo) | Intel Xeon 8592+ (Emerald Rapids) |
| :--- | :--- | :--- | :--- | :--- |
| **Architecture** | GPU (Ampere) | CPU (ARM Neoverse V2) | CPU (x86 Zen 4c) | CPU (x86 Raptor Cove) |
| **FP64 TFLOPs** | **9.7** | **7.1** | ~4.5–5.0 | ~3.7–4.0 |
| **FP32 TFLOPs** | **19.5** | **14.2** | ~9.0–10.0 | **15.5 – 23.7** |
| **AI/Tensor notes.** | **312** (FP16)/**156** (TF32)  
| **Memory Bandwidth**| **2,039 GB/s** | **1,000 GB/s** | **460.8 GB/s** | **358.4 GB/s** |
| **Memory Type** | HBM2e | LPDDR5X (On-module) | DDR5-4800 (12-ch) | DDR5-5600 (8-ch) |

> The CPU is a generalist; it handles messy, branching code brilliantly. The GPU is a specialist. **If**—and ONLY if—we structure our FEM solver as dense, coalesced tensor contractions, we gain access to 10 times the memory bandwidth and nearly 20 TeraFLOPs of compute power.

> Intel Advanced Matrix Extensions (AMX) are the CPU-level equivalent of NVIDIA’s MMA (Matrix Multiply-Accumulate) units (commonly known as Tensor Cores). While traditional CPU instructions (like AVX-512) handle data in 1D vectors, AMX is designed to operate on 2D matrices (called "tiles") in a single operation.
---

## 3. Memory Coalescing: The Key to GPU Performance

Because HBM is slow, the GPU hardware tries to be highly efficient when pulling data from it. Memory from HBM is not fetched byte-by-byte; it is fetched in massive chunks (Memory Transactions), typically 32-byte or 128-byte segments.

### Scenario A: Organized Access (Coalesced Memory)
Imagine a warp of 32 threads. They execute an instruction to load a value from a flat array `U`.
* Thread 0 requests `U[offset + 0]`
* Thread 1 requests `U[offset + 1]`
* ...
* Thread 31 requests `U[offset + 31]`

**Hardware Resolution:** The memory controller looks at these 32 requested addresses. It sees they are perfectly contiguous in physical RAM. It issues **ONE single memory transaction** to fetch a 128-byte chunk from HBM. The data arrives immediately for the whole warp. This achieves near 100% bandwidth utilization.

### Scenario B: Chaotic Access (Uncoalesced / Scattered)
Now imagine we are doing unstructured FEM assembly using an indirection array, so threads request `U[map[ThreadID]]`:
* Thread 0 requests `U[5012]`
* Thread 1 requests `U[12]`
* Thread 2 requests `U[99991]`
* ...

**Hardware Resolution:** The memory controller looks at the 32 addresses. They are scattered everywhere. A single 128-byte fetch can only satisfy one thread. The controller is forced to issue **32 separate memory transactions** for a single warp instruction. 
* **The Penalty:** Our memory bandwidth effectively drops by a factor of up to 32. The cores sit entirely idle, starved of data, waiting for the memory controller to finish 32 slow trips to HBM. We become severely **Memory Bound**.

> Matrix-free methods on regular Cartesian grids allow for perfect coalesced memory access, which is why we can push the hardware to its absolute limits. Unstructured meshes suffer heavily from scattered access).

---

## 4. When XLA Isn't Enough: Triton and Pallas

Most of the time, we write our tensor contractions in JAX (`jnp.einsum`), decorate it with `@jax.jit`, and XLA acts as our "safety net". It perfectly schedules the threads and automatically coalesces memory accesses. 

However, sometimes XLA's compiler heuristics fail, or we have a highly specialized algorithm (like incomplete LU factorizations or custom multigrid smoothers) where we *need* explicit control over that fast Shared Memory.

### Triton (The Engine)
Historically, writing custom GPU kernels meant writing raw CUDA in C++. It required manual, agonizing management of threads, warp synchronizations, and memory coalescing rules. 

**Triton** (developed by OpenAI) is a Python-based language and compiler for GPU programming. 
* Instead of programming *individual threads* (SIMT), Triton allows you to program *blocks of threads*.
* You define operations on smaller "blocks" of data (e.g., 64x64 tiles). 
* **The Magic of Triton:** Because you are operating on contiguous blocks, Triton's compiler *automatically handles the memory coalescing and Shared Memory allocation for you*. It provides a high-level, Pythonic interface that achieves ~90-95% of the performance of ninja-level raw C++ CUDA.
* There exist `jax-triton` package that allows execution of `triton` kernels on JAX arrays.

### Pallas (The JAX Interface)
JAX cannot natively execute Triton code out of the box because JAX needs to know how to trace and differentiate everything. 

**Pallas** is JAX's extension for writing custom kernels. 
* It allows you to write kernel logic using Triton-like block semantics directly inside JAX.
* Under the hood, if you are on a GPU, Pallas compiles your code down to Triton. If you are on a TPU, it compiles it down to Mosaic. 
* It acts as an "escape hatch". If your standard JAX matrix-free operator is running too slowly because XLA didn't fuse it correctly, you can write a custom Pallas kernel, explicitly tell it how to tile the mesh cells into Shared Memory, and slot it seamlessly back into your larger JAX pipeline. Hopefully in this couse we won't need that.

**Takeaway for the Course:** 
We will rely on `jax.jit` and XLA for 95% of our work. But understanding HBM, Shared Memory, and Coalescing is mandatory, because it dictates *how* we structure our arrays and `einsum` operations in Python to help XLA generate the fastest possible code.