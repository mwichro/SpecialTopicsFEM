"""Example for Lecture 1: Kronecker / Einstein summation with JAX.

This script demonstrates sequential sum-factorization vs. einsum.
"""
import jax.numpy as jnp
import jax


def demo(N=3):
    # Define simple 1D operators and a tensor of DoFs
    A = jnp.arange(N * N).reshape(N, N).astype(jnp.float32) + 1.0
    B = (jnp.arange(N * N).reshape(N, N).astype(jnp.float32) + 1.0) * 0.5
    C = (jnp.arange(N * N).reshape(N, N).astype(jnp.float32) + 1.0) * 0.25
    U = jnp.arange(N * N * N).reshape(N, N, N).astype(jnp.float32) + 1.0

    # Sequential sum-factorization
    V1 = jnp.einsum('mk, ijk -> ijm', C, U)
    V2 = jnp.einsum('nj, ijm -> inm', B, V1)
    V3 = jnp.einsum('li, inm -> lnm', A, V2)

    # One-shot einsum (same result)
    V_out = jnp.einsum('li, nj, mk, ijk -> lnm', A, B, C, U)

    # Optimized contraction path
    V_opt = jnp.einsum('li, nj, mk, ijk -> lnm', A, B, C, U, optimize=True)

    return V3, V_out, V_opt


def main():
    v3, v_out, v_opt = demo(3)
    print('Sequential center:', v3[1, 1, 1])
    print('Einsum center:    ', v_out[1, 1, 1])
    print('Einsum(opt) center:', v_opt[1, 1, 1])


if __name__ == '__main__':
    main()
