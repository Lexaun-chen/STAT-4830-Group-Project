"""
Fixed Point Continuation
“Fixed point and Bregman iterative methods for matrix rank minimization” (Ma, Goldfarb, Chen, October 2008)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
import time


def norm_est(Y, tol=1e-2):
    """
    Estimate the spectral norm (largest singular value) of a sparse matrix Y.
    Attempts partial SVD; falls back to dense computation if it fails.
    """
    try:
        Y_csr = Y.tocsr()
        u, s, vt = svds(Y_csr, k=1, which='LM')
        return s[0]
    except Exception:
        return np.linalg.norm(Y.toarray(), 2)


def update_sparse(Y, new_vals, i_arr, j_arr):
    """
    Fast update of values in a dok_matrix Y at positions (i_arr, j_arr) using new_vals.
    Utilizes dictionary-style assignment supported by dok_matrix.
    """
    for ii, jj, val in zip(i_arr, j_arr, new_vals):
        Y[ii, jj] = val


def update_sparse_slow(Y, new_vals, i_arr, j_arr):
    """
    Slow fallback version of sparse matrix update function (same as update_sparse),
    used when the fast version is unavailable or problematic.
    """
    for ii, jj, val in zip(i_arr, j_arr, new_vals):
        Y[ii, jj] = val


def X_on_omega(U, sigma, V, i_arr, j_arr):
    """
    Evaluate the low-rank matrix X = U * diag(sigma) * V^T only at observed positions.

    Mathematically:
        X[i,j] = sum_{l=1}^r U[i,l] * sigma[l] * V[j,l]
    """
    if U.shape[1] == 0:
        return np.zeros(len(i_arr))
    US = U * sigma[np.newaxis, :]  # shape (n1, r)
    return np.sum(US[i_arr, :] * V[j_arr, :], axis=1)


def FPC(n, Omega, b, mu_final, maxiter=500, tol=1e-4):
    """
    Fixed Point Continuation (FPC) algorithm to solve:
        min_X mu * ||X||_* + 1/2 ||P_Omega(X) - b||_2^2

    This uses a continuation strategy to gradually reduce mu and recover a low-rank matrix.

    Args:
        n        : Matrix shape. If scalar, assumes square matrix (n x n).
        Omega    : Linear indices of observed entries (column-major order, like MATLAB).
        b        : Observed values corresponding to Omega (i.e., P_Omega(M)).
        mu_final : Target mu value at the end of continuation.
        maxiter  : Maximum iterations per mu level.
        tol      : Relative residual tolerance for convergence.

    Returns:
        U, sigma, V : SVD factors of the completed matrix X = U * diag(sigma) * V^T
        numiter     : Total number of outer and inner iterations.
    """
    tau = 1.99         # Step size parameter (recommended in [1, 2])
    eta_mu = 1 / 4     # Mu decay factor

    # Parse matrix dimensions
    if np.isscalar(n):
        n1, n2 = n, n
    else:
        n1, n2 = n

    SMALLSCALE = (n1 * n2 < 100 * 100)
    m = len(Omega)

    # Sort Omega and convert to (i, j) subscripts in column-major (Fortran) order
    Omega = np.array(Omega)
    sorted_idx = np.argsort(Omega)
    Omega_sorted = Omega[sorted_idx]
    i_indices, j_indices = np.unravel_index(Omega_sorted, (n1, n2), order='F')
    b = np.array(b)
    normb = np.linalg.norm(b)

    # Initialize residual G: since X = 0 initially, G = P_Omega(0) - b = -b
    G = sparse.dok_matrix((n1, n2), dtype=np.float64)
    for ii, jj, val in zip(i_indices, j_indices, b):
        G[ii, jj] = val

    # Estimate initial mu
    mu = norm_est(G, tol=1e-2)

    # Initialize SVD components
    U = np.zeros((n1, 1))
    V = np.zeros((n2, 1))
    sigma_vals = np.array([0])
    S = np.diag(sigma_vals)
    if SMALLSCALE:
        X = np.zeros((n1, n2))

    numiter = 0
    r = 1
    s = r + 1

    # Outer continuation loop: reduce mu gradually
    while mu > mu_final:
        mu = max(mu * eta_mu, mu_final)
        print(f"FPC, mu = {mu:.3e}")
        s = 2 * r + 1  # Estimate the next rank

        # Inner loop: fixed-point iteration at current mu
        for k in range(maxiter):
            numiter += 1
            X_curr = U @ S @ V.T if U.shape[1] > 0 else np.zeros((n1, n2))
            G_dense = G.toarray()
            Y_mat = X_curr - tau * G_dense

            # Perform SVD
            if SMALLSCALE:
                U_temp, s_vals, Vh_temp = np.linalg.svd(Y_mat, full_matrices=False)
                V_temp = Vh_temp.T
            else:
                try:
                    U_temp, s_vals, Vh_temp = svds(Y_mat, k=s, which='LM')
                    idx_sort = np.argsort(s_vals)[::-1]
                    s_vals = s_vals[idx_sort]
                    U_temp = U_temp[:, idx_sort]
                    V_temp = Vh_temp[idx_sort, :].T
                except Exception:
                    U_temp, s_vals, Vh_temp = np.linalg.svd(Y_mat, full_matrices=False)
                    V_temp = Vh_temp.T

            # Soft-thresholding on singular values
            r = np.sum(s_vals > tau * mu)
            if r > 0:
                U = U_temp[:, :r]
                V = V_temp[:, :r]
                sigma_vals = s_vals[:r] - tau * mu
            else:
                U = np.zeros((n1, 0))
                V = np.zeros((n2, 0))
                sigma_vals = np.array([])

            S = np.diag(sigma_vals)
            s = r + 1  # Update rank estimate

            # Compute residual on observed entries
            x_vals = X_on_omega(U, sigma_vals, V, i_indices, j_indices)
            resid = x_vals - b
            relResid = np.linalg.norm(resid) / normb
            print(f"Iteration {numiter}, rank: {r}, relative residual: {relResid:.1e}")

            # Update sparse residual matrix G
            update_sparse(G, resid, i_indices, j_indices)

            # Convergence check
            if relResid < tol:
                break

        print(f"Final relative residual for mu={mu:.3e}: {relResid:.3e}, rank: {r}")

    final_relative_error = relResid
    final_residual_norm = np.linalg.norm(resid)
    print("========================================")
    print(f"Final relative error: {final_relative_error:.3e}")
    print(f"Final residual norm: {final_residual_norm:.3e}")
    print("========================================")

    return U, sigma_vals, V, numiter


#
# n = (n1, n2)
# Omega = np.array([...])  # Observed  positions
# b = np.array([...])      # Observed  data
# mu_final = 0.1           # Final mu value
#
# U, sigma, V, num_iter = FPC(n, Omega, b, mu_final, maxiter=500, tol=1e-4)
#
# X = U @ np.diag(sigma) @ V.T 