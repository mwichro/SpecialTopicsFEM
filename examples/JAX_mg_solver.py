"""
Standalone JAX example: Continuous-Galerkin Laplace solver on a Cartesian grid.

Layout
------
Every field lives in a 6-D tensor `u[Z, Y, X, EZ, EY, EX]`:
  * (Z, Y, X)     – the (p+1)^3 local Gauss-Lobatto-Legendre (GLL) nodes
                    inside a single hexahedral element.
  * (EZ, EY, EX)  – the structured element index in the global grid.

Because the grid is Cartesian and conformal, every face / line / vertex of
an element has exactly one matching counterpart in the neighbouring element,
so direct stiffness summation (DSS) is implemented with `jnp.roll` plus
`.at[...].add(...)`.

What this file demonstrates
---------------------------
1. Proper spectral-element matrices (GLL nodes/weights, 1-D mass & stiffness).
2. The Laplace operator as a tensor-product volume kernel + DSS.
3. A node-wise Jacobi smoother that needs the assembled operator diagonal.
4. Geometric h-multigrid: prolongation/restriction between two Cartesian
   grids of the same polynomial order, with a V-cycle on top of Jacobi.

x64 must be enabled.
"""
import jax
import jax.numpy as jnp
import numpy as np
import time

jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64


# =============================================================================
# 1.  Reference-element matrices
# =============================================================================

def gauss_lobatto_legendre(p: int):
    """Return (nodes, weights) of the (p+1)-point GLL quadrature on [-1, 1].

    Uses the eigenvalues of the symmetric Jacobi matrix for the
    Legendre polynomial P_p'.  Good enough up to fairly high p; for
    p <= ~30 this is accurate to machine precision.
    """
    if p < 1:
        raise ValueError("Need p >= 1 for GLL.")
    N = p + 1
    # Use Newton iteration on Legendre polynomial derivative.
    # Initial guess: Chebyshev-Gauss-Lobatto nodes.
    x = np.cos(np.pi * np.arange(N) / p)
    # Legendre-Vandermonde matrix.
    P = np.zeros((N, N))
    x_old = 2.0
    while np.max(np.abs(x - x_old)) > 1e-15:
        x_old = x
        P[:, 0] = 1.0
        P[:, 1] = x
        for k in range(2, N):
            P[:, k] = ((2 * k - 1) * x * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k
        x = x_old - (x * P[:, p] - P[:, p - 1]) / (N * P[:, p])
    # Weights
    w = 2.0 / (p * N * P[:, p] ** 2)
    # Sort ascending.
    order = np.argsort(x)
    return x[order], w[order]


def lagrange_derivative_matrix(nodes: np.ndarray) -> np.ndarray:
    """`D[i, j] = phi_j'(nodes[i])` for the Lagrange basis through `nodes`.

    Classical barycentric formula.
    """
    N = nodes.size
    D = np.zeros((N, N))
    # Barycentric weights
    w = np.ones(N)
    for j in range(N):
        for k in range(N):
            if k != j:
                w[j] /= (nodes[j] - nodes[k])
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = (w[j] / w[i]) / (nodes[i] - nodes[j])
        D[i, i] = -np.sum(np.delete(D[i, :], i))
    return D


def reference_element_matrices(p: int):
    """Build the 1-D reference mass `M_ref` and stiffness `K_ref` on [-1, 1].

    Using GLL collocation: mass is diagonal (lumped exactly by GLL quadrature
    of polynomials up to degree 2p-1), stiffness is `D^T diag(w) D`.
    """
    nodes, w = gauss_lobatto_legendre(p)
    D = lagrange_derivative_matrix(nodes)
    M_ref = np.diag(w)            # lumped mass on GLL
    K_ref = D.T @ np.diag(w) @ D  # consistent stiffness
    return (jnp.asarray(nodes, DTYPE),
            jnp.asarray(w,     DTYPE),
            jnp.asarray(M_ref, DTYPE),
            jnp.asarray(K_ref, DTYPE))


def physical_element_matrices(p: int, h: float):
    """Scale the reference matrices to a cube element of edge length `h`.

    On a [-1,1] -> [0,h] affine map, the Jacobian is h/2, so
        M_phys = (h/2)   * M_ref
        K_phys = (2/h)   * K_ref
    """
    _, _, M_ref, K_ref = reference_element_matrices(p)
    J = h / 2.0
    M = J * M_ref
    K = (1.0 / J) * K_ref
    return K, M


# =============================================================================
# 2.  Volume kernel and DSS  (Cartesian, structured)
# =============================================================================

def volume_kernel_cartesian(u, K, M):
    """Apply (K_x M_y M_z + M_x K_y M_z + M_x M_y K_z) tensor-product."""
    # X axis
    u_Kx = jnp.einsum('ix, zyxa... -> zyia...', K, u)
    u_Mx = jnp.einsum('ix, zyxa... -> zyia...', M, u)
    # Y axis
    t_KxMy = jnp.einsum('jy, zyia... -> zjia...', M, u_Kx)
    t_MxKy = jnp.einsum('jy, zyia... -> zjia...', K, u_Mx)
    t_MxMy = jnp.einsum('jy, zyia... -> zjia...', M, u_Mx)
    # Z axis
    o1 = jnp.einsum('kz, zjia... -> kjia...', M, t_KxMy)
    o2 = jnp.einsum('kz, zjia... -> kjia...', M, t_MxKy)
    o3 = jnp.einsum('kz, zjia... -> kjia...', K, t_MxMy)
    return o1 + o2 + o3


def dss_cartesian(u):
    """Direct Stiffness Summation along the EX, EY, EZ element axes.

    Non-periodic: missing neighbours at the domain boundary contribute 0.

    IMPORTANT: per axis, *capture both increments from the same `u`*, then
    apply them.  Reading `u` after a partial update double-counts.
    """
    def shift_right(slab, axis):  # value at e arrives from e-1; e=0 -> 0
        z = jnp.zeros_like(jnp.take(slab, jnp.array([0]), axis=axis))
        return jnp.concatenate([z, jnp.take(slab, jnp.arange(slab.shape[axis] - 1), axis=axis)], axis=axis)

    def shift_left(slab, axis):   # value at e arrives from e+1; e=last -> 0
        z = jnp.zeros_like(jnp.take(slab, jnp.array([0]), axis=axis))
        return jnp.concatenate([jnp.take(slab, jnp.arange(1, slab.shape[axis]), axis=axis), z], axis=axis)

    # X-axis
    right_face_of_left_neighbour = shift_right(u[:, :, -1, ...], axis=-1)
    left_face_of_right_neighbour = shift_left (u[:, :,  0, ...], axis=-1)
    u = u.at[:, :,  0, ...].add(right_face_of_left_neighbour)
    u = u.at[:, :, -1, ...].add(left_face_of_right_neighbour)
    # Y-axis
    top_face_of_bot_neighbour = shift_right(u[:, -1, :, ...], axis=-2)
    bot_face_of_top_neighbour = shift_left (u[:,  0, :, ...], axis=-2)
    u = u.at[:,  0, :, ...].add(top_face_of_bot_neighbour)
    u = u.at[:, -1, :, ...].add(bot_face_of_top_neighbour)
    # Z-axis
    front_face_of_back_neighbour = shift_right(u[-1, :, :, ...], axis=-3)
    back_face_of_front_neighbour = shift_left (u[ 0, :, :, ...], axis=-3)
    u = u.at[ 0, :, :, ...].add(front_face_of_back_neighbour)
    u = u.at[-1, :, :, ...].add(back_face_of_front_neighbour)
    return u


def dirichlet_mask(shape, dtype=DTYPE):
    """Mask that is 1 everywhere except on the global outer faces.

    Layout: u[Z, Y, X, EZ, EY, EX].  The outer faces are
        EX==0   & local X==0
        EX==-1  & local X==-1
    and similarly for Y and Z.  We build the mask as a product of
    per-axis 1-D masks via broadcasting.
    """
    _, _, _, EZ, EY, EX = shape
    def axis_mask(N_loc, N_elem):
        # 2-D (N_loc, N_elem) mask: 0 at (0,0) and (-1,-1), 1 elsewhere.
        outer = jnp.ones((N_loc, N_elem), dtype=dtype)
        outer = outer.at[0,  0 ].set(0.0)
        outer = outer.at[-1, -1].set(0.0)
        return outer
    mZ = axis_mask(shape[0], EZ)
    mY = axis_mask(shape[1], EY)
    mX = axis_mask(shape[2], EX)
    # Broadcast to [Z, Y, X, EZ, EY, EX] by outer-product per axis pair.
    mZ_b = mZ[:, None, None, :, None, None]
    mY_b = mY[None, :, None, None, :, None]
    mX_b = mX[None, None, :, None, None, :]
    return mZ_b * mY_b * mX_b


def apply_laplace(u, K, M, mask=None):
    """Assembled Laplacian: volume kernel + DSS, then optional Dirichlet mask."""
    out = dss_cartesian(volume_kernel_cartesian(u, K, M))
    if mask is not None:
        out = out * mask
    return out


# =============================================================================
# 3.  Operator diagonal & Jacobi smoother
# =============================================================================
#
# For the tensor-product operator A = K_x M_y M_z + M_x K_y M_z + M_x M_y K_z
# the element-local diagonal at local node (i, j, k) is
#
#     diag_e[i,j,k] = K[i,i]*M[j,j]*M[k,k]
#                   + M[i,i]*K[j,j]*M[k,k]
#                   + M[i,i]*M[j,j]*K[k,k]
#
# (M is diagonal because we use GLL lumping.)  The assembled diagonal is
# obtained by broadcasting this to every element and running DSS, exactly
# the same scatter pattern as for the operator itself.

def assembled_diagonal(K, M, grid_shape):
    """Assembled diagonal of the global Laplace operator.

    Local diagonal at node (i,j,k):
        K[i,i] M[j,j] M[k,k] + M[i,i] K[j,j] M[k,k] + M[i,i] M[j,j] K[k,k].
    DSS-sum across elements so shared DoFs carry the assembled diagonal.
    """
    EZ, EY, EX = grid_shape
    kd = jnp.diag(K)
    md = jnp.diag(M)
    d_local = (kd[:, None, None] * md[None, :, None] * md[None, None, :]
             + md[:, None, None] * kd[None, :, None] * md[None, None, :]
             + md[:, None, None] * md[None, :, None] * kd[None, None, :])
    d6 = jnp.broadcast_to(d_local[:, :, :, None, None, None],
                          d_local.shape + (EZ, EY, EX))
    return dss_cartesian(d6)


def jacobi_smoother(u, b, K, M, diag, mask, n_iter=2):
    """Damped Jacobi on the ASSEMBLED residual.

        r_loc = b_local - A_local u        (per-element, before DSS)
        r     = DSS(r_loc)                 (assemble residual)
        u    <- u + 0.5 * D^{-1} * r

    `b` is the element-local (unassembled) RHS; Jacobi DSS's the residual
    internally.  `mask` zeros the update at Dirichlet DoFs.
    """
    inv_d = (1.0 / diag) * mask
    for _ in range(n_iter):
        r_loc = b - volume_kernel_cartesian(u, K, M)
        r = dss_cartesian(r_loc) * mask
        u = u + 0.5 * inv_d * r
    return u


# =============================================================================
# 4.  Multigrid transfers
# =============================================================================
#
# Two grids of the same polynomial order p, the fine grid having 2x the
# number of elements per axis as the coarse grid.  One coarse element maps
# to 2x2x2 fine elements.  Inside the coarse element we evaluate its
# Lagrange basis at the GLL nodes of the two fine sub-intervals along each
# axis; this gives the 1-D prolongation matrix P_left and P_right (both of
# shape (N, N)).  We then stack them into a (2, N, N) operator and apply
# tensor-product style.  DSS finally averages the duplicated boundary
# values; for nodal Lagrange interpolation on shared GLL nodes the
# duplicated entries are already equal, so we instead "DSS-add then divide
# by multiplicity" or, equivalently, place the contribution only once.
# Here we do the "place once, no DSS" route on the fine grid for
# prolongation, and use the transpose for restriction (which *does* need a
# DSS-style sum on the coarse side).

def lagrange_eval_matrix(src_nodes: np.ndarray, dst_nodes: np.ndarray):
    """Evaluate Lagrange basis through `src_nodes` at points `dst_nodes`.

    Returns L[i, j] = phi_j(dst_nodes[i]).
    """
    Ns = src_nodes.size
    Nd = dst_nodes.size
    L = np.ones((Nd, Ns))
    for j in range(Ns):
        for k in range(Ns):
            if k == j:
                continue
            L[:, j] *= (dst_nodes - src_nodes[k]) / (src_nodes[j] - src_nodes[k])
    return L


def build_1d_prolongation(p: int):
    """Return (P_left, P_right), each (N, N), mapping the coarse 1-D
    element [-1,1] to the two fine sub-elements [-1,0] and [0,1] mapped
    back to their own [-1,1] reference."""
    nodes, _ = gauss_lobatto_legendre(p)
    # Fine sub-elements live on [-1,0] and [0,1]; their GLL nodes mapped
    # back to the *coarse* reference are:
    fine_left_in_coarse  = 0.5 * (nodes - 1.0)   # maps [-1,1] -> [-1, 0]
    fine_right_in_coarse = 0.5 * (nodes + 1.0)   # maps [-1,1] -> [ 0, 1]
    P_L = lagrange_eval_matrix(nodes, fine_left_in_coarse)
    P_R = lagrange_eval_matrix(nodes, fine_right_in_coarse)
    return jnp.asarray(P_L, DTYPE), jnp.asarray(P_R, DTYPE)


def _interleave_last(a, b):
    """Interleave two equal-shape arrays along their last axis: out[..., 2k]=a, 2k+1=b."""
    stacked = jnp.stack([a, b], axis=-1)              # (..., E, 2)
    return stacked.reshape(a.shape[:-1] + (2 * a.shape[-1],))


def _split_pairs_last(a):
    """Inverse of _interleave_last along the last axis: returns (a_even, a_odd)."""
    new_shape = a.shape[:-1] + (a.shape[-1] // 2, 2)
    s = a.reshape(new_shape)
    return s[..., 0], s[..., 1]


def prolong(uc, P_L, P_R):
    """Coarse -> fine.

    uc shape:  [N, N, N, ECZ, ECY, ECX]
    returns:   [N, N, N, 2*ECZ, 2*ECY, 2*ECX]
    """
    # ----- X axis (last) -----
    fL = jnp.einsum('ix, zyxa... -> zyia...', P_L, uc)
    fR = jnp.einsum('ix, zyxa... -> zyia...', P_R, uc)
    u  = _interleave_last(fL, fR)                    # doubles EX
    # ----- Y axis (axis -2) -----
    # Move EY to the end, double it, move it back.
    u = jnp.moveaxis(u, -2, -1)
    fL = jnp.einsum('jy, zyia... -> zjia...', P_L, u)
    fR = jnp.einsum('jy, zyia... -> zjia...', P_R, u)
    u = _interleave_last(fL, fR)
    u = jnp.moveaxis(u, -1, -2)
    # ----- Z axis (axis -3) -----
    u = jnp.moveaxis(u, -3, -1)
    fL = jnp.einsum('kz, zjia... -> kjia...', P_L, u)
    fR = jnp.einsum('kz, zjia... -> kjia...', P_R, u)
    u = _interleave_last(fL, fR)
    u = jnp.moveaxis(u, -1, -3)
    return u


def restrict(uf, P_L, P_R):
    """Fine -> coarse (transpose of `prolong`).

    Split each element axis into (left, right) pairs, apply P_L^T to the
    left fine element and P_R^T to the right fine element, sum the two
    contributions.
    """
    PLT = P_L.T
    PRT = P_R.T
    # ----- X axis -----
    uL, uR = _split_pairs_last(uf)                   # along EX
    cL = jnp.einsum('xi, zyia... -> zyxa...', PLT, uL)
    cR = jnp.einsum('xi, zyia... -> zyxa...', PRT, uR)
    u = cL + cR
    # ----- Y axis -----
    u = jnp.moveaxis(u, -2, -1)
    uL, uR = _split_pairs_last(u)
    cL = jnp.einsum('yj, zjia... -> zyia...', PLT, uL)
    cR = jnp.einsum('yj, zjia... -> zyia...', PRT, uR)
    u = cL + cR
    u = jnp.moveaxis(u, -1, -2)
    # ----- Z axis -----
    u = jnp.moveaxis(u, -3, -1)
    uL, uR = _split_pairs_last(u)
    cL = jnp.einsum('zk, kjia... -> zjia...', PLT, uL)
    cR = jnp.einsum('zk, kjia... -> zjia...', PRT, uR)
    u = cL + cR
    u = jnp.moveaxis(u, -1, -3)
    return u


# =============================================================================
# 5.  Multigrid V-cycle
# =============================================================================

def _shape_for(p: int, grid_size: int):
    N = p + 1
    return (N, N, N, grid_size, grid_size, grid_size)


def make_level(p: int, grid_size: int, h_domain: float):
    """Per-level operators (rediscretized FE Laplacian) + Jacobi diagonal + Dirichlet mask."""
    h_elem = h_domain / grid_size
    K, M = physical_element_matrices(p, h_elem)
    diag = assembled_diagonal(K, M, (grid_size, grid_size, grid_size))
    mask = dirichlet_mask(_shape_for(p, grid_size))
    diag_safe = jnp.where(mask > 0, diag, 1.0)   # avoid div-by-zero at boundary
    return {
        "K": K, "M": M, "diag": diag_safe, "mask": mask,
        "grid_size": grid_size, "h": h_elem,
    }


def v_cycle(b, levels, P_L, P_R, n_pre=4, n_post=4):
    """Recursive V-cycle.

    Convention: `b` at every level is the **element-local** (unassembled)
    RHS.  Jacobi handles the DSS step internally.  When restricting the
    residual, we restrict the element-local residual `vol_kernel(u) - b`
    and DSS only at the very end as the smoother on the coarse level
    starts to consume it (Jacobi DSS's it).
    """
    def smooth(u, b, lvl, n):
        L = levels[lvl]
        return jacobi_smoother(u, b, L["K"], L["M"], L["diag"], L["mask"], n_iter=n)

    def cycle(lvl, b):
        L = levels[lvl]
        if lvl == len(levels) - 1:
            return smooth(jnp.zeros_like(b), b, lvl, n=200)   # coarsest "solve"
        u  = smooth(jnp.zeros_like(b), b, lvl, n=n_pre)
        # element-local residual  r_loc = b - A_local u  (Jacobi convention)
        r_loc = b - volume_kernel_cartesian(u, L["K"], L["M"])
        bc = restrict(r_loc, P_L, P_R)
        ec = cycle(lvl + 1, bc)
        u  = u + prolong(ec, P_L, P_R) * L["mask"]
        u  = smooth(u, b, lvl, n=n_post)
        return u

    return cycle(0, b)


# =============================================================================
# 6.  Demo / benchmark
# =============================================================================

def main():
    print("--- CG Laplace + Jacobi + MG demo (x64, JAX) ---")
    P = 1
    N = P + 1
    finest_grid = 32          # 32^3 elements on the finest level
    n_levels    = 4           # 32 -> 16 -> 8 -> 4
    h_domain    = 1.0

    print(f"Order p={P}, finest grid {finest_grid}^3, levels={n_levels}")

    # Build levels (each is a coarsening by 2 of its parent).
    levels = []
    g = finest_grid
    for _ in range(n_levels):
        levels.append(make_level(P, g, h_domain))
        g //= 2
        if g < 1:
            break
    P_L, P_R = build_1d_prolongation(P)

    K_fine    = levels[0]["K"]
    M_fine    = levels[0]["M"]
    diag_fine = levels[0]["diag"]
    mask_fine = levels[0]["mask"]

    # Manufactured solution: u_exact(x,y,z) = sin(pi x) sin(pi y) sin(pi z),
    # which vanishes on the cube's outer faces -> compatible with Dirichlet BCs.
    nodes_1d, _ = gauss_lobatto_legendre(P)
    nodes_1d = np.asarray(nodes_1d)
    h_f = h_domain / finest_grid
    # Per-element node coordinates along one axis:
    #   x[i, e] = h_f * (e + 0.5 * (nodes_1d[i] + 1))
    coords_1d = h_f * (np.arange(finest_grid)[None, :]
                       + 0.5 * (nodes_1d[:, None] + 1.0))   # (N, finest_grid)
    xZ = jnp.asarray(coords_1d, DTYPE)[:, None, None, :, None, None]
    xY = jnp.asarray(coords_1d, DTYPE)[None, :, None, None, :, None]
    xX = jnp.asarray(coords_1d, DTYPE)[None, None, :, None, None, :]
    u_exact = jnp.sin(jnp.pi * xZ) * jnp.sin(jnp.pi * xY) * jnp.sin(jnp.pi * xX)
    u_exact = u_exact * mask_fine                       # zero on Dirichlet
    # Element-local RHS (unassembled).  Jacobi will DSS it as part of the residual.
    b = volume_kernel_cartesian(u_exact, K_fine, M_fine)

    # ---- Jacobi-only baseline ----
    def assembled_residual(u, b_local):
        return dss_cartesian(b_local - volume_kernel_cartesian(u, K_fine, M_fine)) * mask_fine
    resid_jit = jax.jit(assembled_residual)
    smooth_jit = jax.jit(lambda u, b: jacobi_smoother(
        u, b, K_fine, M_fine, diag_fine, mask_fine, n_iter=100))

    _ = smooth_jit(jnp.zeros_like(u_exact), b).block_until_ready()

    print("\nJacobi-only convergence (100 sweeps / report, 1000 total):")
    u = jnp.zeros_like(u_exact)
    r0 = float(jnp.linalg.norm(resid_jit(u, b)))
    print(f"  iter    0   ||r||={r0:.3e}")
    for it in range(10):
        u = smooth_jit(u, b)
        rnorm = float(jnp.linalg.norm(resid_jit(u, b)))
        print(f"  iter {100*(it+1):4d}   ||r||={rnorm:.3e}")

    # ---- MG V-cycle ----
    # Residual on the fine level is element-local (vol_kernel(u) - b); the
    # V-cycle's coarse-solve produces an element-local correction.  We just
    # invoke the V-cycle as a smoother for `u` directly.
    def one_vcycle(u, b):
        # element-local residual  r_loc = b - A_local u
        r_loc = b - volume_kernel_cartesian(u, K_fine, M_fine)
        e = v_cycle(r_loc, levels, P_L, P_R, n_pre=4, n_post=4)
        return u + e
    vcyc_jit = jax.jit(one_vcycle)
    _ = vcyc_jit(jnp.zeros_like(u_exact), b).block_until_ready()

    print("\nMG V-cycle convergence:")
    u = jnp.zeros_like(u_exact)
    for it in range(8):
        u = vcyc_jit(u, b)
        rnorm = float(jnp.linalg.norm(resid_jit(u, b)))
        print(f"  vcycle {it+1:2d}   ||r||={rnorm:.3e}")

    # Crude timing of one V-cycle.
    start = time.time()
    iters = 20
    for _ in range(iters):
        u = vcyc_jit(u, b)
    u.block_until_ready()
    avg = (time.time() - start) / iters
    print(f"\nAvg time / V-cycle: {avg*1e3:.2f} ms  "
          f"(grid {finest_grid}^3, p={P}, x64)")


if __name__ == "__main__":
    main()
