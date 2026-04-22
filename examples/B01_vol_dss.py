"""
Benchmark: Continuous Galerkin Operator with Direct Stiffness Summation.
"""
import jax
import jax.numpy as jnp
import time


jax.config.update("jax_enable_x64", False)
DTYPE = jnp.float32

def get_cg_operators(order):
    N = order + 1
    key = jax.random.PRNGKey(42)
    K = jax.random.normal(key, (N, N), dtype=DTYPE)
    M = jax.random.normal(key, (N, N), dtype=DTYPE)
    K = 0.5 * (K + K.T)
    M = 0.5 * (M + M.T)
    return K, M

def volume_kernel_cartesian(u, K, M):
    """
    u shape: [Z, Y, X, EZ, EY, EX]
    K, M shape: [N, N] (1D stiffness and mass matrices)
    """
    # The '...' cleanly abstracts away the EZ, EY, EX dimensions.
    # Apply X-derivative (last of the local Z,Y,X indices)
    u_Kx = jnp.einsum('ix, zyxa... -> zyia...', K, u)
    u_Mx = jnp.einsum('ix, zyxa... -> zyia...', M, u)
    
    # Apply Y-derivative
    term_Kx_My = jnp.einsum('jy, zyia... -> zjia...', M, u_Kx)
    term_Mx_Ky = jnp.einsum('jy, zyia... -> zjia...', K, u_Mx)
    term_Mx_My = jnp.einsum('jy, zyia... -> zjia...', M, u_Mx)
    
    # Apply Z-derivative
    out_1 = jnp.einsum('kz, zjia... -> kjia...', M, term_Kx_My)
    out_2 = jnp.einsum('kz, zjia... -> kjia...', M, term_Mx_Ky)
    out_3 = jnp.einsum('kz, zjia... -> kjia...', K, term_Mx_My)
    
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

def laplace_cartesian(u, K, M, scales):
    # 1. Volume Kernel (Compute Intensive)
    r = volume_kernel_cartesian(u, K, M)
    
    # 2. DSS (Bandwidth Intensive - but now optimized)
    # We apply DSS to the Residual vector
    r_summed = dss_cartesian(r)
    
    return r_summed

def main():
    print(f"--- Optimized CG Solver (Slice-Roll-Add) ---")
    P = 3
    N = P + 1
    # 6D Cartesian layout: [Z, Y, X, EZ, EY, EX]
    # Let's say we have grid of 100x100x100 cells
    grid_size = 30 # roughly 27k cells, manageable for debugging and tests
    NUM_CELLS = grid_size**3
    
    print(f"Order: {P}, Cells: {NUM_CELLS} ({grid_size}^3)")
    
    key = jax.random.PRNGKey(0)
    # Shape matching expectation: [Z, Y, X, EZ, EY, EX]
    u_global = jax.random.normal(key, (N, N, N, grid_size, grid_size, grid_size), dtype=DTYPE)
    
    K, M = get_cg_operators(P)
    scales = (1.0, 1.0, 1.0)
    
    print("JIT Compiling...")
    op_jit = jax.jit(laplace_cartesian)
    _ = op_jit(u_global, K, M, scales).block_until_ready()
    print("Compilation Done.")
    
    print("Running Benchmark...")
    start = time.time()
    iterations = 20
    for _ in range(iterations):
        res = op_jit(u_global, K, M, scales)
        res.block_until_ready()
    end = time.time()
    
    avg_time = (end - start) / iterations
    
    flops_vol = 16 * (N**4)
    # DSS flops are negligible (adds), but we count volume
    total_flops = iterations * NUM_CELLS * flops_vol
    
    gflops = (total_flops / 1e9) / avg_time
    gdofs = (NUM_CELLS * N**3 * iterations / 1e9) / (end - start)
    
    print(f"Time:       {avg_time:.4f} s")
    print(f"Est TFLOPS: {gflops/1000:.2f}")
    print(f"GDoF/s:     {gdofs:.2f}")

if __name__ == "__main__":
    main()