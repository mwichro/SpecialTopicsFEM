

# Lecture Notes: Emulating FP64 Matrix Multiplication on TF32/FP16 Hardware

## 1. Introduction & The Challenge
Modern AI hardware (like NVIDIA Tensor Cores) performs matrix multiplications at blazing speeds using low-precision formats like **FP16** or **TF32** (Tensor Float 32). However, many scientific computing tasks require **FP64** (Double Precision). 

**The Challenge:**
* **FP64** has 53 bits of mantissa precision and a massive exponent range.
* **TF32/FP16** have only 11 bits of mantissa precision (10 explicit + 1 implicit).
* **FP32** (usually the accumulation target on Tensor Cores) has 24 bits of mantissa precision.

To achieve exact FP64 results on this hardware without overflow or rounding errors, we use an **Error-Free Transformation** algorithm based on the **Ozaki Scheme**. 

## 2. The Mathematical Foundation

To perfectly emulate FP64 using lower precision, we must reduce the floating-point inputs into **exact integers**. If the hardware computes with integers, it cannot suffer from floating-point rounding errors—provided we respect the hardware's maximum integer bounds.

### Why exactly 11 bits per split?
* Both **TF32** and **FP16** have 11 bits of mantissa. This means they can perfectly represent any integer up to $2^{11} = 2048$.
* If we slice our input matrices $A$ and $B$ into chunks of integers strictly in the range $[-2048, 2048]$, the hardware multipliers will process them with 0% error.
* The maximum product of two such integers is $2048 \times 2048 = 4,194,304$ (which is $2^{22}$).

### Why is this perfect for a 4x4 Matrix?
Matrix multiplication is a dot product. For a $4 \times 4$ matrix, each element in the output is the sum of **4** multiplications.
* Maximum possible sum: $4 \times 4,194,304 = 16,777,216$.
* Conveniently, $16,777,216$ is exactly $2^{24}$. 
* The Tensor Core's **FP32 Accumulator** has exactly 24 bits of mantissa precision. It can perfectly represent integers up to $2^{24}$ without dropping a single bit. 

Thus, for $N=4$, splitting the matrices into 11-bit chunks results in a **mathematically perfect** multiplication in the hardware, with zero rounding error.

## 3. The Algorithm Breakdown

Because FP64 has 53 bits of precision, and we can only process 11 bits safely at a time, we must split our matrices $\lceil 53 / 11 \rceil = \mathbf{5 \text{ times}}$.

### Step 1: Compute Global Exponents
Find the absolute maximum value in matrices $A$ and $B$, and find their base-2 exponents:
$$E_A = \lceil \log_2(\max(|A|)) \rceil$$
$$E_B = \lceil \log_2(\max(|B|)) \rceil$$

### Step 2: Slice into 11-bit Integer Matrices
For $k \in [1, 5]$, we extract the $k$-th slice of Matrix $A$ from a remainder matrix $R_A$ (initially $R_A = A$):
1. **Target LSB:** Calculate the bit-position we are extracting down to: $L_{A,k} = E_A - (11 \times k)$.
2. **Shift:** Scale the remainder so the target bit becomes the "ones" place: $S = R_A \times 2^{-L_{A,k}}$.
3. **Extract:** Round to the nearest integer to create our slice: $A_k = \text{Round}(S)$.
4. **Update Remainder:** $R_A = R_A - (A_k \times 2^{L_{A,k}})$.

Repeat this for Matrix $B$ to get $B_1 \dots B_5$.

### Step 3: Multiply, Prune, and Accumulate
Instead of 1 matrix multiplication, we now have $5 \times 5 = 25$ sub-multiplications. 
We perform $A_k \times B_l$ by casting them to TF32/FP16, doing the math, scaling the answer by $2^{L_{A,k} + L_{B,l}}$, and adding it to an FP64 accumulator.

**Optimization (Pruning):** Not all 25 operations are necessary. If the scale of a sub-product ($L_{A,k} + L_{B,l}$) falls too far below the maximum exponent ($E_A + E_B - 53$), it physically cannot impact the FP64 result. We can safely skip these, reducing the required operations from 25 down to roughly **19**.

---

## 4. Exemplary Python Implementation

This script simulates the hardware boundaries precisely, utilizing 11-bit FP16 matrices for the multiplication, FP32 for the accumulation, and ultimately reconstructing the FP64 result.

```python
import numpy as np

def split_matrix_ozaki(A, num_splits, bits_per_split):
    """Splits an FP64 matrix into slices of perfectly representable integers."""
    max_val = np.max(np.abs(A))
    if max_val == 0:
        return [(np.zeros_like(A, dtype=np.float16), 0) for _ in range(num_splits)]
    
    max_exp = np.ceil(np.log2(max_val))
    splits = []
    rem = A.copy() 
    
    for k in range(1, num_splits + 1):
        # 1. Calculate the exponent of the Least Significant Bit for this slice
        lsb_exp = max_exp - (k * bits_per_split)
        
        # 2. Scale the remainder so the LSB is shifted to the 2^0 (ones) position
        scale = 2.0 ** (-lsb_exp)
        
        # 3. Extract the integer part (Guaranteed to be <= 2^11, perfectly fitting in TF32/FP16)
        A_k_int = np.round(rem * scale)
        splits.append((A_k_int.astype(np.float16), lsb_exp))
        
        # 4. Subtract the extracted portion from the remainder
        rem = rem - (A_k_int / scale)
        
    return splits

def emulate_fp64_matmul_on_tf32(A, B):
    # 1. Split matrices into 5 slices of 11 bits each
    A_splits = split_matrix_ozaki(A, num_splits=5, bits_per_split=11)
    B_splits = split_matrix_ozaki(B, num_splits=5, bits_per_split=11)

    # Accumulator in true FP64
    C_fp64 = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)

    # Exponents for pruning calculations
    E_A = np.ceil(np.log2(np.max(np.abs(A))))
    E_B = np.ceil(np.log2(np.max(np.abs(B))))
    ops_performed = 0

    for A_val, L_A in A_splits:
        for B_val, L_B in B_splits:
            
            # Pruning Logic:
            # Max magnitude of this chunk's values is 2^(L_A + L_B + 24)
            # If this is smaller than the 53rd bit of the highest possible product, 
            # it will be rounded away in FP64 anyway. Skip it!
            if (L_A + L_B + 24) < (E_A + E_B - 53):
                continue
                
            ops_performed += 1
            
            # 2. Hardware Matrix Multiplication mapping exactly to TF32/FP16 cores
            # (Inputs are float16 <= 2048, Accumulation is strictly float32)
            P_fp32 = np.matmul(
                A_val.astype(np.float32), 
                B_val.astype(np.float32), 
                dtype=np.float32
            )
            
            # 3. Scale back up and accumulate in CPU FP64
            scale = 2.0 ** (L_A + L_B)
            C_fp64 += P_fp32.astype(np.float64) * scale

    return C_fp64, ops_performed

# --- Execution & Validation ---
if __name__ == "__main__":
    np.random.seed(42)
    # Generate random 4x4 matrices in FP64
    A = np.random.randn(4, 4).astype(np.float64)
    B = np.random.randn(4, 4).astype(np.float64)

    # Compute using our TF32 emulated algorithm
    C_emulated, ops = emulate_fp64_matmul_on_tf32(A, B)
    
    # Compute using standard FP64 matrix multiplication for baseline
    C_exact = np.matmul(A, B)
    
    # Compare
    max_diff = np.max(np.abs(C_emulated - C_exact))
    
    print(f"Required Matrix Multiplications: {ops} / 25")
    print(f"Maximum difference from native FP64: {max_diff:.3e}")
```

## 5. Conclusion
By strategically scaling floating-point numbers into the strict integer bounds supported by the mantissas of low-precision formats ($[-2048, 2048]$ for TF32/FP16), we can completely subvert precision loss during multiplication. Furthermore, bounding the sum of the inner dimension $N=4$ strictly within the limits of the FP32 accumulator ($16,777,216$) guarantees **zero rounding errors** on the Tensor Cores. 

By accumulating the scaled fragments in FP64 software-side, we achieve double-precision exactness using low-precision ALUs.