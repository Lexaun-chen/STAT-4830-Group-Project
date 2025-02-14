"""
SVT Algorithm for Matrix Completion
----------------------------------------
This implementation solves the matrix completion problem using the Singular Value Thresholding (SVT) method:
    minimize    tau * ||X||_* + 0.5 * ||X||_F^2
    subject to  P_Omega(X) = P_Omega(M)

The main steps are:
1. Initialization:
   - Convert the observation indices (Omega) to (i, j) coordinates.
   - Initialize a sparse matrix Y (and dual variable) using the observed data.
2. Iterative Process:
   - At each iteration, compute the SVD of Y (using full SVD for small-scale or partial SVD for large-scale).
   - Apply a soft-thresholding operation on the singular values (subtract tau).
   - Reconstruct X on the observed positions and compute the relative residual.
   - Update the dual variable (and Y) accordingly.
3. Termination:
   - The iteration stops when the relative residual falls below a tolerance threshold.
   - Final information (iteration count, residual error, etc.) is printed.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
import time


def norm_est(Y, tol=1e-2):
    try:
        Y_csr = Y.tocsr()
        _, s, _ = svds(Y_csr, k=1, which='LM')
        return s[0]
    except Exception:
        return np.linalg.norm(Y.toarray(), 2)

# Same operations since can't get meg file from Matlab used in 'update_sparse'
def update_sparse(Y, new_vals, i_arr, j_arr):
    for ii, jj, val in zip(i_arr, j_arr, new_vals):
        Y[ii, jj] = val


def update_sparse_slow(Y, new_vals, i_arr, j_arr):
    for ii, jj, val in zip(i_arr, j_arr, new_vals):
        Y[ii, jj] = val


def X_on_omega(U, sigma, V, i_arr, j_arr):
    if U.shape[1] == 0:
        return np.zeros(len(i_arr))
    US = U * sigma[np.newaxis, :]
    return np.sum(US[i_arr, :] * V[j_arr, :], axis=1)


def SVT(n, Omega, b, tau, delta, maxiter=500, tol=1e-4, EPS=0):
    verbose = 2
    start_time = time.process_time()
    if np.isscalar(n):
        n1, n2 = n, n
    else:
        n1, n2 = n

    SMALLSCALE = (n1 * n2 < 100 * 100)
    Omega = np.array(Omega)
    b = np.array(b)
    m = len(Omega)
    normb = np.linalg.norm(b)
    sorted_idx = np.argsort(Omega)
    Omega_sorted = Omega[sorted_idx]
    i_indices, j_indices = np.unravel_index(Omega_sorted, (n1, n2), order='F')

    USE_SLOW_UPDATE = False
    if EPS:
        delta /= np.sqrt(2)
        y1 = np.maximum(b, 0)
        y2 = np.maximum(-b, 0)
        Y = sparse.dok_matrix((n1, n2), dtype=np.float64)
        for ii, jj, val in zip(i_indices, j_indices, y1 - y2):
            Y[ii, jj] = val
        normProjM = norm_est(Y, tol=1e-2)
        k0 = int(np.ceil(tau / (delta * normProjM)))
        y1 = k0 * delta * y1
        y2 = k0 * delta * y2
        try:
            update_sparse(Y, y1 - y2, i_indices, j_indices)
        except Exception:
            USE_SLOW_UPDATE = True
            update_sparse_slow(Y, y1 - y2, i_indices, j_indices)
    else:
        Y = sparse.dok_matrix((n1, n2), dtype=np.float64)
        for ii, jj, val in zip(i_indices, j_indices, b):
            Y[ii, jj] = val
        normProjM = norm_est(Y, tol=1e-2)
        k0 = int(np.ceil(tau / (delta * normProjM)))
        y = k0 * delta * b
        try:
            update_sparse(Y, y, i_indices, j_indices)
        except Exception:
            USE_SLOW_UPDATE = True
            update_sparse_slow(Y, y, i_indices, j_indices)

    r = 0
    out = {'residual': np.zeros(maxiter),
           'rank': np.zeros(maxiter, dtype=int),
           'time': np.zeros(maxiter),
           'nuclearNorm': np.zeros(maxiter)}
    incre = 4

    for k in range(maxiter):
        s_val = min(r + 4, n1, n2)
        if SMALLSCALE:
            Y_full = Y.toarray()
            U, s_vals, Vh = np.linalg.svd(Y_full, full_matrices=False)
            V = Vh.T
        else:
            Y_csr = Y.tocsr()
            s_current = s_val
            OK = False
            while not OK:
                try:
                    U, s_vals, Vh = svds(Y_csr, k=s_current, which='LM')
                    idx_sort = np.argsort(s_vals)[::-1]
                    s_vals = s_vals[idx_sort]
                    U = U[:, idx_sort]
                    V = Vh[idx_sort, :].T
                    if s_vals[s_current - 1] <= tau or s_current == min(n1, n2):
                        OK = True
                    else:
                        s_current = min(s_current + incre, min(n1, n2))
                except Exception:
                    Y_dense = Y.toarray()
                    U, s_vals, Vh = np.linalg.svd(Y_dense, full_matrices=False)
                    V = Vh.T
                    OK = True
                    break

        r = np.sum(s_vals > tau)
        if r > 0:
            U_r = U[:, :r]
            V_r = V[:, :r]
            sigma = s_vals[:r] - tau
        else:
            U_r = np.zeros((n1, 0))
            V_r = np.zeros((n2, 0))
            sigma = np.array([])

        x = X_on_omega(U_r, sigma, V_r, i_indices, j_indices)
        eTime = time.process_time() - start_time
        if verbose == 2 and (k + 1) % 20 == 0:
            rel_err = np.linalg.norm(x - b) / normb
            print(f"iteration {k + 1:4d}, rank {r:2d}, rel. residual {rel_err:.1e}")
        relRes = np.linalg.norm(x - b) / normb
        out['residual'][k] = relRes
        out['time'][k] = eTime
        out['rank'][k] = r
        out['nuclearNorm'][k] = np.sum(sigma)
        start_time = time.process_time()

        if relRes < tol:
            print(f"\nFinal iteration: {k + 1:4d}, rank {r:2d}, rel. residual {rel_err:.1e}")
            break
        if EPS and np.linalg.norm(x - b, np.inf) < 2 * EPS:
            print(f"\nFinal iteration: {k + 1:4d}, rank {r:2d}, rel. residual {rel_err:.1e}")
            break
        if np.linalg.norm(x - b) / normb > 1e5:
            print("Divergence!")
            break

        if EPS:
            y1 = np.maximum(y1 + delta * (-(x - b) - EPS), 0)
            y2 = np.maximum(y2 + delta * ((x - b) - EPS), 0)
            new_vals = y1 - y2
            if USE_SLOW_UPDATE:
                update_sparse_slow(Y, new_vals, i_indices, j_indices)
            else:
                update_sparse(Y, new_vals, i_indices, j_indices)
        else:
            y = y + delta * (b - x)
            if USE_SLOW_UPDATE:
                update_sparse_slow(Y, y, i_indices, j_indices)
            else:
                update_sparse(Y, y, i_indices, j_indices)

    num_iter = k + 1
    out['residual'] = out['residual'][:num_iter]
    out['time'] = out['time'][:num_iter]
    out['rank'] = out['rank'][:num_iter]
    out['nuclearNorm'] = out['nuclearNorm'][:num_iter]

    final_relative_error = relRes
    final_residual_norm = np.linalg.norm(x - b)

    print("")
    print("\n========================================")
    print(f"Final relative error: {final_relative_error:.3e}")
    print(f"Final residual norm: {final_residual_norm:.3e}")
    print("========================================\n")

    return U_r, sigma, V_r, num_iter, out


"""
Test Block for the SVT Algorithm

1. Generates a random low-rank matrix M of size (n1 x n2) with rank r.
2. Computes the number of observed entries (m) based on the degrees of freedom.
3. Randomly selects m indices (Omega) to simulate observed data.
4. Optionally adds noise to the observed data.
5. Sets algorithm parameters (tau, delta, maxiter, tol).
6. Runs the SVT algorithm for matrix completion and reports the computation time.
"""

if __name__ == "__main__":
    np.random.seed(40)
    n1, n2, r = 300, 150, 10
    M = np.random.randn(n1, r) @ np.random.randn(r, n2)
    df = r * (n1 + n2 - r) #degree of freedom
    m = min(7 * df, round(0.9 * n1 * n2))
    p = m / (n1 * n2)
    Omega = np.random.choice(n1 * n2, m, replace=False)
    data = M.flatten()[Omega]

    sigma_noise = 0
    data += sigma_noise * np.random.randn(*data.shape)

    print(f"Matrix completion: {n1} x {n2} matrix, rank {r}, {100 * p:.1f}% observations")

    tau = 15 * np.sqrt(n1 * n2)
    delta = min(1.2 / p, 2)
    maxiter = 1000
    tol = 5e-4

    print("\nSolving by SVT...")
    start_time = time.time()
    U, S, V, numiter, out = SVT((n1, n2), Omega, data, tau, delta, maxiter, tol)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")