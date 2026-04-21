"""Numeric demonstration of the 2D Kronecker <-> matrix reshape "transpose trick".

Compares v = (A ⊗ B) vec(U) with vec(B U A^T) using column-major (Fortran)
vectorization to match the conventional Kronecker identity used in the notes.
"""
import numpy as np


def demo(N=4):
    # deterministic small matrices
    A = np.arange(1, N * N + 1).reshape(N, N)
    B = (np.arange(1, N * N + 1).reshape(N, N)) * 0.5
    U = np.arange(1, N * N + 1).reshape(N, N)

    # vectorize U in column-major (Fortran) order
    u_vec = U.flatten(order='F')

    # Kronecker application
    K = np.kron(A, B)
    v_kron = K.dot(u_vec)

    # Transpose-trick: V_out = B U A^T, then vectorize column-major
    V_out = B.dot(U).dot(A.T)
    v_trick = V_out.flatten(order='F')

    # Compare
    diff = np.linalg.norm(v_kron - v_trick)
    print('N=', N)
    print('|| (A⊗B) vec(U) - vec(B U A^T) || =', diff)
    if diff < 1e-12:
        print('Match: transpose trick verified numerically.')
    else:
        print('Mismatch: check vectorization ordering/computation.')


if __name__ == '__main__':
    demo(4)
