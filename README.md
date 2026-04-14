Special Topics in Finite Element Methods (JAX-based)
===============================================

This repository contains course materials for an advanced finite-element
methods course with an emphasis on matrix-free techniques, sum-factorization,
subspace correction, and multigrid — implemented and demonstrated using
Python and JAX.

Repository layout
-----------------
- `lectures/` — lecture notes, exercises, and solution files.
- `examples/` — runnable Python examples using JAX / NumPy.

Course plan 
------------------------
1. Einstein summation, Kronecker products. Split Kronecker product by dimension.
2. Sum factorization: evaluation of the Laplace operator on a single Cartesian cell.
3. Fast diagonalization (exercise).
4. Introduction to subspace correction (rebase, prolongation/restriction, Richardson, Jacobi, Gauss–Seidel).
5. Motivation for multigrid and convergence properties of simple smoothers.
6. Two-level method as subspace correction.
7. Multilevel methods and practical implementation.
8. To be planned:
    - Elasticity: Automatic Differentiation,
    - Stokes: LBB, 
    - Stokes: Block solvers
    - Stokes: multigrid, constrained smoothers, patch smoothers.

Getting started (Python + JAX)
------------------------------
We recommend creating an isolated Python environment for the course.

Example (POSIX shells):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install JAX:

- CPU-only (simple):

```bash
pip install jax jaxlib
```



- GPU support: choose the release matching your CUDA/cuDNN version. See the
  JAX CUDA releases page for the correct wheel selector. Example (choose
  the exact tag appropriate for your GPU/CUDA):

```bash
pip install --upgrade pip
pip install "jax[cuda12]"
```  
Or 
```bash
# Example only — check https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade pip
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Note: replace `cuda12` with the CUDA/cuDNN variant that matches your system. JAX also works on AMD hardware.
If unsure, use the CPU install above. 

Running examples
----------------
After activating the virtualenv and installing JAX, try a small example:

```bash
python examples/A01_einsum.py
```

Lecture and example pointers
---------------------------
- Lecture 1: [lectures/A01_Kronecker_and_Einstein.md](lectures/A01_Kronecker_and_Einstein.md)
<!-- - Lecture 1 solutions: [lectures/A01_solutions.md](lectures/A01_solutions.md) -->
- Examples: [examples/](examples/)