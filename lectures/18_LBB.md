

# Lecture: Functional Spaces and Brezzi's Conditions for Stokes

When dealing with mixed problems like the Stokes equations, standard tools like the Lax-Milgram theorem fail because the bilinear form is not coercive over the whole space. Instead, we must understand how the operators map between spaces, kernels, and images. 

To do this, we need to generalize the concept of an "orthogonal complement" to infinite-dimensional function spaces and their duals.

## 1. Perpendicular Space ($Z^\perp$)

**Definition 6.4:** For a subspace $Z \subset Q$ of a Hilbert space $Q$, the perpendicular space $Z^\perp$ of $Z$ in $Q$ is:
$$ Z^\perp = \{q \in Q: \langle q, p \rangle_Q = 0, \forall p \in Z\} $$

**Extended Explanation:**
In finite-dimensional Euclidean space, $Z^\perp$ is simply the orthogonal complement (e.g., if $Z$ is a 2D plane in 3D space, $Z^\perp$ is the 1D line normal to that plane). A fundamental theorem in Hilbert spaces is the Orthogonal Decomposition Theorem: $Q = Z \oplus Z^\perp$. 

Why is this useful for mixed problems? We have an operator $B^*: Q \to V'$. The kernel of this operator, $\text{Ker}(B^*)$, consists of all functions $q \in Q$ that map to zero. 
If we split $Q = \text{Ker}(B^*) \oplus (\text{Ker}(B^*))^\perp$, we can completely ignore the kernel part when looking at the image of the operator. Consequently, $B^*$ forms a perfect, one-to-one (bijective) mapping from the perpendicular space $(\text{Ker}(B^*))^\perp$ strictly onto its image $\text{Im}(B^*)$.

## 2. Polar Space ($Z^0$)

**Definition 6.5:** For a subspace $Z$ of a Hilbert space $V$, the polar space (or annihilator) $Z^0$ is the subspace of the *dual space* $V'$ consisting of continuous linear functionals that vanish on $Z$:
$$ Z^0 = \{F \in V': F[v] = 0 \quad \forall v \in Z\} $$

**Extended Explanation:**
In standard linear algebra (like $\mathbb{R}^n$), vectors and co-vectors (row vs. column vectors) are often treated as the same thing via the dot product. Therefore, the Rank-Nullity Theorem says that the image of a transposed matrix is the orthogonal complement of the original matrix's kernel: $\text{Im}(B^*) = (\text{Ker}(B))^\perp$.

In infinite dimensions (e.g., FEM function spaces), $V$ and its dual $V'$ (the space of functionals acting on $V$) are strictly distinct. 
* $B$ maps $V \to Q'$
* $B^*$ maps $Q \to V'$

Because $\text{Im}(B^*) \subset V'$ but $\text{Ker}(B) \subset V$, we **cannot** use a standard inner product to find the perpendicular space; they live in different universes! Instead, we use the polar space. The dual-space equivalent of the Rank-Nullity Theorem (formally derived from the Closed Range Theorem) is:
$$ \text{Im}(B^*) = (\text{Ker}(B))^0 $$
*Meaning:* The image of $B^*$ consists of exactly those force functionals in $V'$ that do zero virtual work on any velocity field $v$ that is in the kernel of $B$ (i.e., divergence-free fields).

---

## 3. Theorem 6.6 (Brezzi’s Conditions)

Let $a(u,v)$ be a continuous bilinear form on $V \times V$, and $b(v,q)$ be a continuous bilinear form on $V \times Q$. We seek $(u,p) \in V \times Q$ such that:
$$ a(u,v) + b(v,p) = F[v], \quad \forall v \in V $$
$$ b(u,q) = G[q], \quad \forall q \in Q $$

Define the kernel of $B$ as $Z = \{u \in V: b(u,q) = 0 \quad \forall q \in Q\}$.

**Assume two conditions:**
1. **Coercivity on the Kernel:** $a(u,v)$ is coercive on $Z$ with constant $\alpha$. (Note: it does not need to be coercive everywhere, only on $Z$).
2. **Inf-Sup (LBB) Condition:** There exists $\beta > 0$ such that:
   $$ \inf_{q \in Q} \sup_{v \in V} \frac{b(v,q)}{\|v\|_V \|q\|_Q} \ge \beta $$

**Conclusion:** There exists a unique solution $(u,p)$ and it satisfies stability bounds dependent on $\alpha, \beta,$ and $M$ (the continuity constant of $a$).

### Proof of Brezzi's Theorem

#### Part 1: Why LBB implies $B$ is surjective
*This is the most critical deduction from the LBB condition.*

We work entirely in **finite dimensions**: $V$ and $Q$ are finite-dimensional Hilbert spaces (as they are after discretization by finite elements). This lets us avoid the heavy machinery of the Closed Range Theorem and argue using nothing more than the rank–nullity theorem and a single inequality.

**Step 0: Translate LBB into an operator inequality.**
Recall that $b(v,q) = \langle Bv, q\rangle = \langle v, B^*q\rangle$, where $B: V \to Q'$ and its adjoint (transpose) $B^*: Q \to V'$. The supremum in the LBB condition is exactly the dual norm of $B^*q$:
$$ \sup_{v \in V} \frac{b(v,q)}{\|v\|_V} = \sup_{v \in V} \frac{\langle v, B^*q\rangle}{\|v\|_V} = \|B^*q\|_{V'}. $$
So the inf–sup condition is precisely the statement that $B^*$ is **bounded below**:
$$ \|B^* q\|_{V'} \ge \beta \|q\|_Q \quad \forall q \in Q. \tag{$\star$} $$

**Step 1: $B^*$ is injective.**
Suppose $B^* q = 0$. Then $(\star)$ gives $0 = \|B^*q\|_{V'} \ge \beta \|q\|_Q$, and since $\beta > 0$ this forces $\|q\|_Q \le 0$, i.e. $q = 0$. Hence
$$ \text{Ker}(B^*) = \{0\}. $$

**Step 2: From injectivity of $B^*$ to surjectivity of $B$ — by dimension counting.**
This is where finite-dimensionality does the work, replacing the Closed Range Theorem.

First, a basic fact: a linear map and its adjoint have the **same rank**, $\dim \text{Im}(B) = \dim \text{Im}(B^*)$. (In coordinates, $B^*$ is represented by the transpose of the matrix of $B$, and a matrix and its transpose have equal rank — the row rank equals the column rank.)

Now apply rank–nullity to $B^*: Q \to V'$:
$$ \dim Q = \dim \text{Ker}(B^*) + \dim \text{Im}(B^*) = 0 + \dim \text{Im}(B^*), $$
using Step 1. Therefore $\dim \text{Im}(B^*) = \dim Q$. Combining with equal ranks,
$$ \dim \text{Im}(B) = \dim \text{Im}(B^*) = \dim Q = \dim Q', $$
where the last equality holds because a finite-dimensional space and its dual have the same dimension.

But $\text{Im}(B)$ is a subspace of $Q'$ whose dimension equals $\dim Q'$. A subspace of a finite-dimensional space that has the full dimension must be the whole space:
$$ \text{Im}(B) = Q'. $$
**Hence $B$ is surjective.**

**Why this works without the Closed Range Theorem.**
In infinite dimensions, "injective adjoint" only gives a *dense* image for $B$, and one needs the range to be *closed* (the Closed Range Theorem, which the LBB inequality is exactly designed to supply) before concluding surjectivity. In finite dimensions every subspace is automatically closed and dimensions are finite, so the equality of ranks plus a counting argument immediately upgrades "$B^*$ injective" to "$B$ surjective." The only role of the constant $\beta > 0$ here is to guarantee $\text{Ker}(B^*) = \{0\}$; in Part 4 the *quantitative* value of $\beta$ resurfaces to control the size of the solution.

#### Part 2: Existence
Because $B$ is surjective, for any data $G \in Q'$, we can definitively find a "particular" velocity $u_g \in V$ such that $B u_g = G$ (which means $b(u_g, q) = G[q]$).
We split our unknown velocity: $u = u_g + u_Z$. Substituting this into our system:
$$ a(u_Z, v) + b(v,p) = F[v] - a(u_g, v) \quad \forall v \in V $$
$$ b(u_Z, q) = 0 \quad \forall q \in Q \implies u_Z \in Z $$

Because $u_Z \in Z$, we can restrict our test functions to $v \in Z$. For these test functions, $b(v,p) = 0$ by definition. The first equation collapses to:
$$ a(u_Z, v) = F[v] - a(u_g, v) \quad \forall v \in Z $$
Because $a(\cdot,\cdot)$ is assumed coercive on $Z$, the Lax-Milgram theorem guarantees a unique $u_Z \in Z$.

Now we must find the pressure $p$. Define a new residual functional $L \in V'$:
$$ L[v] = F[v] - a(u_g + u_Z, v) $$
Notice that for any $v \in Z$, $L[v] = 0$. Therefore, by definition, $L$ resides in the polar space $Z^0 = (\text{Ker}(B))^0$.
As discussed earlier, $(\text{Ker}(B))^0 = \text{Im}(B^*)$. Since $L \in \text{Im}(B^*)$, there **must** exist a $p \in Q$ such that $B^* p = L$, which is exactly $b(v,p) = L[v]$. 
Substitute $L[v]$ back, and we have satisfied the full mixed equation! Existence is proven.

#### Part 3: Uniqueness
Assume two solutions $(u_1, p_1)$ and $(u_2, p_2)$. Let $u = u_1 - u_2$ and $p = p_1 - p_2$. The differences satisfy the homogeneous equations:
1) $a(u,v) + b(v,p) = 0$
2) $b(u,q) = 0$

Equation 2 implies $u \in Z$. Therefore, we can choose $v = u \in Z$ as a test function in Equation 1. Since $b(u,p) = 0$, we get:
$$ a(u,u) = 0 $$
By coercivity on $Z$, $\alpha \|u\|_V^2 \le a(u,u) = 0 \implies u = 0$.

Substitute $u=0$ into Equation 1:
$$ b(v,p) = 0 \quad \forall v \in V \implies B^* p = 0 $$
Because LBB proved $B^*$ is injective, $p$ must be $0$. Uniqueness is proven.

#### Part 4: Stability Bounds
**For $u$:**
Using the surjectivity of $B$, we choose $u_g$ to be the minimal-norm element such that $B u_g = G$. By LBB, it satisfies $\|G\|_{Q'} \ge \beta \|u_g\|_V$, so $\|u_g\|_V \le \frac{1}{\beta}\|G\|_{Q'}$.

Apply Lax-Milgram to the $Z$-space equation:
$$ \alpha \|u_Z\|_V^2 \le a(u_Z, u_Z) = F[u_Z] - a(u_g, u_Z) \le \|F\|_{V'} \|u_Z\|_V + M \|u_g\|_V \|u_Z\|_V $$
Divide by $\alpha \|u_Z\|_V$:
$$ \|u_Z\|_V \le \frac{1}{\alpha}\|F\|_{V'} + \frac{M}{\alpha}\|u_g\|_V $$
Using the triangle inequality $u = u_Z + u_g$:
$$ \|u\|_V \le \|u_Z\|_V + \|u_g\|_V \le \frac{1}{\alpha}\|F\|_{V'} + \left( \frac{M}{\alpha} + 1 \right) \|u_g\|_V $$
Because coercivity implies $M > \alpha$, we have $1 < \frac{M}{\alpha}$, so $\frac{M}{\alpha} + 1 < \frac{2M}{\alpha}$. Substituting our bound for $u_g$:
$$ \|u\|_V \le \frac{1}{\alpha}\|F\|_{V'} + \frac{2M}{\alpha \beta} \|G\|_{Q'} $$

**For $p$:**
Rearrange the momentum equation: $b(v,p) = F[v] - a(u,v)$.
The LBB condition states $\beta \|p\|_Q \le \|B^* p\|_{V'}$. Furthermore:
$$ \|B^* p\|_{V'} = \sup_{v} \frac{F[v] - a(u,v)}{\|v\|_V} \le \|F\|_{V'} + M\|u\|_V $$
Thus:
$$ \beta \|p\|_Q \le \|F\|_{V'} + M \left( \frac{1}{\alpha}\|F\|_{V'} + \frac{2M}{\alpha \beta}\|G\|_{Q'} \right) $$
Divide by $\beta$ and use $1 < \frac{M}{\alpha}$ again to simplify the first term:
$$ \|p\|_Q \le \frac{2M}{\alpha \beta}\|F\|_{V'} + \frac{2M^2}{\alpha \beta^2} \|G\|_{Q'} \quad \blacksquare $$