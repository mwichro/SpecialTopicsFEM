Here are the revised, slower-paced blackboard notes for Lecture 2. We will demystify the indices by explicitly connecting them to physical slices of a 2D grid before proving anything.

***

# Lecture 2: Demystifying Indices and Slices (From Grid to Einstein)

## 1. The 2D Grid of Degrees of Freedom (DoFs)

Forget flat 1D vectors for a moment. In a 2D Matrix-Free method, our DoFs live on a grid. Let's represent the DoFs on a single Cartesian element as a 2D matrix $U$.

*   Let $i$ be the index for the **y-axis** (vertical coordinate).
*   Let $j$ be the index for the **x-axis** (horizontal coordinate).
*   $U_{ij}$ is the value at row $i$, column $j$.

A 3×3 grid of DoFs:
```
      j=1      j=2      j=3
    ┌─────────┬─────────┬─────────┐
i=1 │ U₁₁     │ U₁₂     │ U₁₃     │
    ├─────────┼─────────┼─────────┤
i=2 │ U₂₁     │ U₂₂     │ U₂₃     │
    ├─────────┼─────────┼─────────┤
i=3 │ U₃₁     │ U₃₂     │ U₃₃     │
    └─────────┴─────────┴─────────┘
```

When we write a Kronecker product operator like $(B \otimes A)$ acting on $U$, the convention is:
*   The **first** matrix ($B$) acts on the **first** index ($i$, the y-axis).
*   The **second** matrix ($A$) acts on the **second** index ($j$, the x-axis).

Let's break this down one dimension at a time.

---

## 2. Operating on Horizontal Slices: $(I \otimes A)$

Consider the operator $(I \otimes A)$ acting on our grid $U$ to produce a new grid $V$.
*   $I$ (Identity) acts on the y-axis (index $i$). It does nothing.
*   $A$ acts on the x-axis (index $j$).

**What does this mean physically?**
Take a horizontal slice of our grid. A horizontal slice is a single row, where $i$ is fixed, and $j$ varies. Let's look at row $i$:
$$ \text{Row } i = \begin{bmatrix} U_{i1} & U_{i2} & U_{i3} \end{bmatrix} $$

Applying $(I \otimes A)$ simply means taking the 1D matrix $A$ and multiplying it by this row:
$$ V_{i,:} = A \begin{bmatrix} U_{i1} \\ U_{i2} \\ U_{i3} \end{bmatrix} $$

**Writing this with Indices:**
To find the value of $V$ at position $(i, j)$, we take the dot product of $A$'s $j$-th row with the $i$-th row of $U$:
$$ V_{ij} = \sum_{l} A_{jl} U_{il} $$

> To get the new $j$ position, we sum over the old $x$-coordinates $l$. The $y$-coordinate $i$ just comes along for the ride.

---

## 3. Operating on Vertical Slices: $(B \otimes I)$

Now consider $(B \otimes I)$ acting on $U$.
*   $B$ acts on the y-axis (index $i$).
*   $I$ (Identity) acts on the x-axis (index $j$). It does nothing.

**What does this mean physically?**
Take a vertical slice. A vertical slice is a single column, where $j$ is fixed, and $i$ varies. Let's look at column $j$:
$$ \text{Column } j = \begin{bmatrix} U_{1j} \\ U_{2j} \\ U_{3j} \end{bmatrix} $$

Applying $(B \otimes I)$ means taking the 1D matrix $B$ and multiplying it by this column:
$$ V_{:,j} = B \begin{bmatrix} U_{1j} \\ U_{2j} \\ U_{3j} \end{bmatrix} $$

**Writing this with Indices:**
To find the value of $V$ at $(i, j)$, we take the dot product of $B$'s $i$-th row with the $j$-th column of $U$:
$$ V_{ij} = \sum_{k} B_{ik} U_{kj} $$
> To get the new $i$ position, we sum over the old $y$-coordinates $k$. The $x$-coordinate $j$ comes along for the ride.

---

## 4. Combining Them: Einstein Summation

Now, let's apply the full operator $(B \otimes A)$ to $U$. 
Because $(B \otimes A) = (B \otimes I)(I \otimes A)$, we are simply doing both operations: mixing the vertical slices, and mixing the horizontal slices.

Let's combine our index formulas. We want $V = (B \otimes A) U$.
1.  $B$ hits the first index: $B_{ik}$
2.  $A$ hits the second index: $A_{jl}$
3.  They both multiply $U_{kl}$

$$ V_{ij} = \sum_{k} \sum_{l} B_{ik} A_{jl} U_{kl} $$

**Enter Einstein Summation:**
Writing $\sum_k \sum_l$ every time is tedious. The Einstein convention states: **If an index appears twice in a term, we automatically sum over it.**

So, we simply write:
$$ V_{ij} = B_{ik} A_{jl} U_{kl} $$

Checklist:
*   Are $k$ and $l$ repeated? Yes. They are "dummy indices" (summed over).
*   Are $i$ and $j$ free (appear once)? Yes. They dictate the size of the output $V_{ij}$.

---

## 5. Re-proving the Mixed-Product Property

Now that we know exactly what the indices mean, let's prove the Mixed-Product Property:
$$ (A \otimes B)(C \otimes D) = (AC) \otimes (BD) $$

*Note: For the proof, let's use a standard 4-index tensor definition for the operator itself, rather than its action on $U$.*

Let Operator 1 be $K = (A \otimes B)$. 
How does it map an input grid $(k,l)$ to an output grid $(i,j)$? 
$A$ hits the first dimension, $B$ hits the second:
$$ K_{i j k l} = A_{ik} B_{jl} $$
*(Meaning: Input at $k$ goes to $i$ via $A$. Input at $l$ goes to $j$ via $B$.)*

Let Operator 2 be $L = (C \otimes D)$.
It maps an input $(m,n)$ to an output $(k,l)$:
$$ L_{k l m n} = C_{km} D_{ln} $$

**Step 1: Multiply the Operators**
Matrix multiplication means applying $L$, then $K$. In index notation, we sum over the intermediate grid indices $(k,l)$:
$$ \text{Product}_{i j m n} = K_{i j k l} L_{k l m n} $$

*(Einstein notation at work: $k$ and $l$ appear twice, so they are summed. $i, j, m, n$ appear once, so they are the output dimensions.)*

**Step 2: Substitute our Definitions**
Substitute the slice-by-slice definitions of $K$ and $L$:
$$ \text{Product}_{i j m n} = (A_{ik} B_{jl}) (C_{km} D_{ln}) $$

**Step 3: Commute the Scalars**
Remember, $A_{ik}, B_{jl}$, etc., are just individual numbers at this point. We can shuffle them around freely. Let's group the $y$-axis operators together, and the $x$-axis operators together:
$$ \text{Product}_{i j m n} = (A_{ik} C_{km}) (B_{jl} D_{ln}) $$

**Step 4: Recognize 1D Matrix Multiplication**
Look at the grouped terms. 
*   $A_{ik} C_{km}$ is exactly the row-by-column dot product definition for the matrix $(AC)$ at entry $im$.
*   $B_{jl} D_{ln}$ is exactly the row-by-column dot product definition for the matrix $(BD)$ at entry $jn$.

$$ \text{Product}_{i j m n} = (AC)_{im} (BD)_{jn} $$

**Step 5: Translate Back to Kronecker**
Read the final line out loud: "The operator $(AC)$ acts on the first index $(i,m)$, and the operator $(BD)$ acts on the second index $(j,n)$." 

By our very first rule today, this is the exact definition of the Kronecker product:
$$ (AC) \otimes (BD) $$

Because the indices match perfectly, the operators are identical.
$$ (A \otimes B)(C \otimes D) = (AC) \otimes (BD) \quad \blacksquare $$