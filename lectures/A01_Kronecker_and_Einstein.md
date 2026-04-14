
# Lecture 1: Kronecker Products, Sum Factorization, and Tensor Contractions

## 1. Definition of the Kronecker Product

**Definition:** Let $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{p \times q}$. The Kronecker product $A \otimes B \in \mathbb{R}^{mp \times nq}$ is defined as the block matrix:

$$
A \otimes B = \begin{bmatrix}
a_{11}B & a_{12}B & \dots & a_{1n}B \\
a_{21}B & a_{22}B & \dots & a_{2n}B \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1}B & a_{m2}B & \dots & a_{mn}B
\end{bmatrix}
$$

**Key Properties (Mixed-Product Property):**
For matrices of compatible dimensions,
$$ (A \otimes B)(C \otimes D) = (AC) \otimes (BD) $$
$$ (A \otimes B)^T = A^T \otimes B^T $$
$$ (A \otimes B)^{-1} = A^{-1} \otimes B^{-1} \quad \text{(if } A, B \text{ are invertible)} $$


### Proof of the Mixed-Product Property: $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$

#### 1. Explicit Block Setup
Recall the definition: $Q \otimes W$ means multiplying every scalar entry of $Q$ by the entire matrix $W$. 

Let $A$ be an $m \times n$ matrix and $C$ be an $n \times p$ matrix. Writing out the Kronecker products $A \otimes B$ and $C \otimes D$ explicitly as block matrices, we have:

$$
A \otimes B = \begin{bmatrix}
a_{11}B & a_{12}B & \cdots & a_{1n}B \\
a_{21}B & a_{22}B & \cdots & a_{2n}B \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1}B & a_{m2}B & \cdots & a_{mn}B
\end{bmatrix}
$$

$$
C \otimes D = \begin{bmatrix}
c_{11}D & c_{12}D & \cdots & c_{1p}D \\
c_{21}D & c_{22}D & \cdots & c_{2p}D \\
\vdots & \vdots & \ddots & \vdots \\
c_{n1}D & c_{n2}D & \cdots & c_{np}D
\end{bmatrix}
$$

#### 2. Block Matrix Multiplication
Now, multiply these two large block matrices together. Just like standard matrix multiplication, we multiply a "block row" of the first matrix by a "block column" of the second matrix. 

Let's compute just the very first block (top-left, position 1,1) of the resulting matrix:

$$
\text{Block}_{11} = \big[a_{11}B \quad a_{12}B \quad \cdots \quad a_{1n}B \big] 
\begin{bmatrix}
c_{11}D \\
c_{21}D \\
\vdots \\
c_{n1}D
\end{bmatrix}
$$

Expanding this dot product:
$$ \text{Block}_{11} = (a_{11}B)(c_{11}D) + (a_{12}B)(c_{21}D) + \dots + (a_{1n}B)(c_{n1}D) $$

##### 3. Commuting Scalars and Factoring
Because the $a$'s and $c$'s are just regular scalar numbers, we can move them around freely and group the matrices $B$ and $D$ together:

$$ \text{Block}_{11} = (a_{11}c_{11})BD + (a_{12}c_{21})BD + \dots + (a_{1n}c_{n1})BD $$

Since every term is multiplied by the same matrix $(BD)$, we can factor $(BD)$ out:

$$ \text{Block}_{11} = \underbrace{(a_{11}c_{11} + a_{12}c_{21} + \dots + a_{1n}c_{n1})}_{\text{This is exactly the dot product of A's 1st row and C's 1st col!}} BD $$

Therefore:
$$ \text{Block}_{11} = (AC)_{11} BD $$

#### 4. Reassembling the Matrix
If we repeat this exact same row-by-column block multiplication for *every* position $(i, j)$ in the new matrix, the $(i,j)$-th block will always be the dot product of $A$'s $i$-th row and $C$'s $j$-th column, multiplied by $BD$:

$$
(A \otimes B)(C \otimes D) = \begin{bmatrix}
(AC)_{11}BD & (AC)_{12}BD & \cdots & (AC)_{1p}BD \\
(AC)_{21}BD & (AC)_{22}BD & \cdots & (AC)_{2p}BD \\
\vdots & \vdots & \ddots & \vdots \\
(AC)_{m1}BD & (AC)_{m2}BD & \cdots & (AC)_{mp}BD
\end{bmatrix}
$$

Looking at this final matrix, we see every entry of the matrix $(AC)$ is multiplying the entire matrix $(BD)$. By our very first definition, this is exactly $(AC) \otimes (BD)$.

$$ \therefore (A \otimes B)(C \otimes D) = (AC) \otimes (BD) \quad \blacksquare $$

## 2. Dimensional Splitting

In FEM on tensor-product elements (e.g., hexahedra), operations like the mass matrix or evaluation of basis functions decompose along spatial dimensions. 

Let an operator on a 3D Cartesian cell be defined as $M = A \otimes B \otimes C$, where $A, B, C \in \mathbb{R}^{N \times N}$ are 1D operators (e.g., 1D mass matrices). $N$ is the number of 1D Degrees of Freedom ($N = p+1$, where $p$ is the polynomial degree). 

The full operator $M \in \mathbb{R}^{N^3 \times N^3}$. We can split the application of $M$ using the mixed-product property.

**Proof of the split:**
Note that the identity matrix of size $N$ is $I$. We can rewrite $A, B, C$ as:
$$ A = A \cdot I \cdot I $$
$$ B = I \cdot B \cdot I $$
$$ C = I \cdot I \cdot C $$

Using the mixed-product property:
$$ A \otimes B \otimes C = (A I I) \otimes (I B I) \otimes (I I C) $$
$$ A \otimes B \otimes C = (A \otimes I \otimes I) (I \otimes B \otimes I) (I \otimes I \otimes C) $$

This shows that applying the 3D operator $M$ to a vector $u \in \mathbb{R}^{N^3}$ can be done sequentially:
1. $v^{(1)} = (I \otimes I \otimes C) u$  *(operate on the $x$-direction)*
2. $v^{(2)} = (I \otimes B \otimes I) v^{(1)}$ *(operate on the $y$-direction)*
3. $v^{(3)} = (A \otimes I \otimes I) v^{(2)}$ *(operate on the $z$-direction)*


## 3. Exercise: Computational Complexity Analysis

The computational-complexity derivation for the naive assembled approach vs.
sum-factorization is an exercise. Work out the FLOP counts and the crossover
points for various polynomial degrees $p$ (with $N=p+1$).


## 4. Einstein Summation and Tensor Representation

We can map the 1D flat vector $u \in \mathbb{R}^{N^3}$ to a 3D tensor $U \in \mathbb{R}^{N \times N \times N}$.
Let $U_{ijk}$ represent the degree of freedom at node $(i, j, k)$.

**Einstein Summation Convention:** Repeated indices in a single term imply summation over that index.

Let's rewrite the sequential sum factorization using tensor index notation.
Let's rewrite the sequential sum factorization using tensor index notation.

1. Apply $C$ to the fastest changing index $k$ ($x$-direction):
   $$ V^{(1)}_{i j m} = \sum_{k=1}^N C_{m k} U_{i j k} \quad \xrightarrow{\text{Einstein}} \quad C_{m k} U_{i j k} $$
2. Apply $B$ to index $j$ ($y$-direction):
   $$ V^{(2)}_{i n m} = B_{n j} V^{(1)}_{i j m} = B_{n j} C_{m k} U_{i j k} $$
3. Apply $A$ to index $i$ ($z$-direction):
   $$ V^{(3)}_{l n m} = A_{l i} V^{(2)}_{i n m} = A_{l i} B_{n j} C_{m k} U_{i j k} $$

The full evaluation of the Laplace operator (or mass matrix) on a single Cartesian cell is compactly written as
$$ V_{lnm} = A_{li} B_{nj} C_{mk} U_{ijk}. $$

The detailed index-level proof of the 2D/3D Kronecker identities is given as
an exercise; see [A01_solutions.md](A01_solutions.md) for worked solutions.
Additionally, a numeric demonstration of the 2D "transpose trick" is
available as an example script: [examples/A01_transpose_trick.py](examples/A01_transpose_trick.py).
1. Apply $C$ to the fastest changing index $k$ ($x$-direction):
   $$ V^{(1)}_{i j m} = \sum_{k=1}^N C_{m k} U_{i j k} \quad \xrightarrow{\text{Einstein}} \quad C_{m k} U_{i j k} $$
2. Apply $B$ to index $j$ ($y$-direction):
   $$ V^{(2)}_{i n m} = B_{n j} V^{(1)}_{i j m} = B_{n j} C_{m k} U_{i j k} $$
3. Apply $A$ to index $i$ ($z$-direction):
   $$ V^{(3)}_{l n m} = A_{l i} V^{(2)}_{i n m} = A_{l i} B_{n j} C_{m k} U_{i j k} $$

The full evaluation of the Laplace operator (or mass matrix) on a single Cartesian cell is beautifully compressed to:
$$ V_{lnm} = A_{li} B_{nj} C_{mk} U_{ijk} $$

## 5. Implementation Overview with JAX / NumPy

Because the operation is purely multidimensional tensor contraction, it maps perfectly to `einsum`.

See the runnable example in [examples/A01_einsum.py](examples/A01_einsum.py).

## 6. Exercise: 2D Kronecker Product and the Transpose Trick

**Problem:** 
Let $u \in \mathbb{R}^{N^2}$ be a vector, and let $A, B \in \mathbb{R}^{N \times N}$. 
Show that computing $v = (A \otimes B)u$ is mathematically equivalent to computing:
$$ V_{out} = B U A^T $$
where $U \in \mathbb{R}^{N \times N}$ is the tensor (matrix) representation of $u$, and $V_{out}$ is the tensor representation of $v$.

**Proof Outline:**
 Proof of the above is left as an excercise