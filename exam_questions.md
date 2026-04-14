# Selected Topics in Finite Elements:  Exam Questions

### 1. Damped Jacobi Iteration

1. Jacobi method as a variant of Richardson
2. Derive iteration matrix.
3. Discuss what happens with the error at each iteration for Laplace problem.
4. How is that connected to the eigenvalues of the iteration matrix?
5. Which parts of the error are removed by Jacobi iteration?

### 2. Subspace Correction

1. Describe what is subspace correction method.
2. What do we need to define the subspace correction method?
3. Show that Jacobi is a parallel subspace correction method.
4. Show that Gauss-Seidel is a successive subspace correction method.

### 3. Two Grid Method

1. Show the connection between computing product with a given preconditioner $P$ and Richardson iteration with the same precondition.
2. Show that an iterative method that does two precondition Richardson iterations is also a Richardson method.
3. How can we combine two preconditioners? Derive formulas for multiplicative and additive combinations.
4. Show that a two-grid method is a Richardson iteration with a combination of preconditioners.
5. Derive the iteration matrix of the two-grid method (you can skip pre- or post-smoothing). What are the conditions that guarantee convergence of the two-grid method?
6. Why do the conditions from the point above hold for FE and the Laplace problem?

### 4. Prolongation, Restriction, Hanging Nodes

1. How do we obtain the prolongation operator?
2. How the prolongation operator is connected to the restriction operator?
3. How we can eliminate hanging nodes from the system using ideas behind the prolongation operator?

### 5. Multigrid

1. Describe (draw) a multigrid $\gamma$-cycle. Discuss w- and v- cycles.
2. What assumptions are required for multigrid to converge independently of $h$?
3. What is the computational complexity of a multigrid cycle? How much memory is required?
4. How would you construct a $p$-multigrid method? Will it converge, why? (sketch the reasoning)
5. What do we need to define a multigrid method? Which parts are problem-dependent, and which parts are coming from the finite element method itself?

### 6. Matrix-Free: Preliminaries

1. What factor limits the execution of the program?
2. What does it mean that the algorithm is compute-bound or memory-bound?
3. What is the limiting factor for a typical FE program involving a sparse matrix?
4. What is CPU cache? What impact does it have on the performance?
5. Types of parallelism. Concepts behind distributed computing.
6. Race conditions, why do we care about thread safety?
7. What is SIMD?
8. Sketch naive implementation of matrix-free computing on a cartesian mesh. How can we use SIMD instructions to speed it up?
9. What do we need to solve a problem involving an operator evaluated in a matrix-free manner using multigrid?
10. Polynomial smoother: how does `PreconditionChebyshev` in `deal.ii` work?

### 7. Matrix-Free: Tensor Products Structure of Finite Element

1. How many operations are required to perform cell matrix-vector multiplication?
2. Describe how the cell matrix-vector product is computed in matrix-free computing. Discuss the complexity of each step.
3. Tensor product elements: shape functions, quadrature.
4. How do we exploit the tensor product structure of the finite element? Sum Factorization.
5. What is the complexity of the optimized steps from 2.?

### 8. Matrix-Free: Separability of Certain Operators. Fast Diagonalization

1. Generalized eigenvalues, simultaneous diagonalization of two quadratic forms (sketch the proof of existence)
2. Show tensor product structure of Laplace problem.
3. How to obtain a Laplace operator in higher dimensions using 1D operators? Discuss what kind of boundary conditions are OK.
4. Fast Diagonalization: the idea behind `TensorProductSymmetricSum` in `deal.II`.

### 9. Solid Mechanics

1. Explain how we describe elasticity: reference and deformed configurations, deformation, deformation gradient.
2. Polar decomposition of the deformation gradient tensor. Right Cauchy–Green deformation tensor. Describe the idea.
3. How we can use the right Cauchy–Green tensor to obtain deformation energy? Why invariants/eigenvalues of the right Cauchy–Green tensor are important?
4. What assumption leads to linear elasticity?
5. Stress-strain relation for linear elasticity, Hook's law
6. Derive Cauchy momentum equation. Why the stress tensor has to be symmetric?
7. Derive weak form of linear elasticity equation. Show that
   $$\text{div}(\sigma v) = \sigma:\nabla v = \sigma : \epsilon(v)$$

### 10. Nonlinear Problems

1. Explain how can we solve a non-linear problem.
2. Given a simple energy functional, for example
   $$\int_\Omega u^2 \vert \nabla u \vert ^2 \text{d}x$$
   derive formulas for the residual and tangent.
3. How can we separate gradient and value contributions in the above so that matrix-free framework can be used?
4. Storing vs computing: what can we store to save some computations? How does the storage scales with polynomial degree?
5. Two ways of applying Dirichlet boundary conditions in Newton procedure. Which one is better in context of finite-strain solid mechanics?

### 11. Stokes and Saddle-Point Problems

1. $\inf$-$\sup$ condition and LBB theorem. Statement of the theorem. The idea of proof: splitting into "divergence-free". Bonus point: names in LBB.
2. Convergence of FEM for saddle-point problem.
3. Why $Q_1$-$Q_1$ pair is not suitable for solving the Stokes problem? What can be done instead?
4. Schur complement.
5. Decomposition of block system into lower-triangular, diagonal, and upper triangular block matrix.
6. How do we obtain blocks for preconditioners? Explain using Stokes problem as an example (what are $\tilde{A}$ and $\tilde{S}$)

### 12. Additional Topics

1. Well-posedness of Taylor-Hood element for Stokes. Sketch the idea of the proof.
2. Multigrid for Stokes: Explain the idea of constrained smoother.
3. Jump coefficient problems:
   - Why they might be challenging?
   - What does the spectrum of MG-preconditioned jump coefficient Laplace operator look like?
   - How do we deal with outlying eigenvalues?
   - Can you describe the subspace connected to outlying eigenvalues?
