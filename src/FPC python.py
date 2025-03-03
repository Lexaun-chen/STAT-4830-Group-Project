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
    估计稀疏矩阵 Y 的谱范数（最大奇异值）。
    尝试使用部分 SVD，如出错则退化为 dense 计算。
    """
    try:
        Y_csr = Y.tocsr()
        u, s, vt = svds(Y_csr, k=1, which='LM')
        return s[0]
    except Exception:
        return np.linalg.norm(Y.toarray(), 2)


def update_sparse(Y, new_vals, i_arr, j_arr):
    """
    快速更新 dok_matrix Y 中位于 (i_arr, j_arr) 处的值为 new_vals。
    dok_matrix 支持字典式赋值，这里逐个更新。
    """
    for ii, jj, val in zip(i_arr, j_arr, new_vals):
        Y[ii, jj] = val


def update_sparse_slow(Y, new_vals, i_arr, j_arr):
    """
    慢速版本更新函数（与 update_sparse 相同），在 fast 版本不可用时调用。
    """
    for ii, jj, val in zip(i_arr, j_arr, new_vals):
        Y[ii, jj] = val


def X_on_omega(U, sigma, V, i_arr, j_arr):
    """
    根据低秩分解 X = U * diag(sigma) * V^T，
    仅在观测位置 (i_arr, j_arr) 上计算 X 的取值。

    数学上：
        X[i,j] = sum_{l=1}^r U[i,l] * sigma[l] * V[j,l]
    """
    if U.shape[1] == 0:
        return np.zeros(len(i_arr))
    # 每一列乘以相应的奇异值
    US = U * sigma[np.newaxis, :]  # 形状 (n1, r)
    # 对每个观测点计算对应行和列的内积
    return np.sum(US[i_arr, :] * V[j_arr, :], axis=1)


def FPC(n, Omega, b, mu_final, maxiter=500, tol=1e-4):
    """
    Fixed Point Continuation (FPC) 算法求解问题：
        min_X mu * ||X||_* + 1/2 ||P_Omega(X)-b||_2^2
    利用连续化策略逐步降低 mu，从而逼近最终低秩矩阵解。

    参数：
        n       : 矩阵尺寸。若为标量则为正方形，否则 n = (n1, n2)。
        Omega   : 观测位置的线性索引（应按照 Matlab 的列主序排序）。
        b       : 观测数据向量，对应 M(Omega)。
        mu_final: 最终目标的 mu 值。
        maxiter : 每个 mu 下内层循环的最大迭代次数。
        tol     : 收敛容忍度。

    返回：
        U, sigma, V : 低秩矩阵 X 的奇异值分解（X = U * diag(sigma) * V^T）。
        numiter     : 内外层迭代的总次数。
    """
    tau = 1.99  # tau 建议取值在 1~2 之间
    eta_mu = 1 / 4  # 每次 mu 缩减因子

    # 处理矩阵尺寸
    if np.isscalar(n):
        n1, n2 = n, n
    else:
        n1, n2 = n

    SMALLSCALE = (n1 * n2 < 100 * 100)
    m = len(Omega)

    # 对 Omega 排序并转换为 (i,j) 下标（保持列主序与 Matlab 一致）
    Omega = np.array(Omega)
    sorted_idx = np.argsort(Omega)
    Omega_sorted = Omega[sorted_idx]
    i_indices, j_indices = np.unravel_index(Omega_sorted, (n1, n2), order='F')
    b = np.array(b)
    normb = np.linalg.norm(b)

    # 初始化 G：起始时 X=0，因此 G = P_Omega(0)-b = -b，
    # 这里构造稀疏矩阵 G，其非零位置的值为 b
    G = sparse.dok_matrix((n1, n2), dtype=np.float64)
    for ii, jj, val in zip(i_indices, j_indices, b):
        G[ii, jj] = val

    # 初始 mu 的估计
    mu = norm_est(G, tol=1e-2)

    # 初始化低秩分解变量（初始 X=0）
    U = np.zeros((n1, 1))
    V = np.zeros((n2, 1))
    sigma_vals = np.array([0])
    S = np.diag(sigma_vals)
    if SMALLSCALE:
        X = np.zeros((n1, n2))

    numiter = 0
    r = 1
    s = r + 1

    # 外层循环：逐步降低 mu 到 mu_final
    while mu > mu_final:
        mu = max(mu * eta_mu, mu_final)
        print(f"FPC, mu = {mu:.3e}")
        s = 2 * r + 1  # 为下一次内层迭代预估新的秩

        # 内层循环：固定 mu 下的固定点迭代
        for k in range(maxiter):
            numiter += 1
            # 当前 X = U * S * V^T；初始为空则 X 为 0
            if U.shape[1] > 0:
                X_curr = U @ S @ V.T
            else:
                X_curr = np.zeros((n1, n2))

            # 计算 Y = X - tau * G
            G_dense = G.toarray()
            Y_mat = X_curr - tau * G_dense

            # 计算 Y 的 SVD 分解
            if SMALLSCALE:
                U_temp, s_vals, Vh_temp = np.linalg.svd(Y_mat, full_matrices=False)
                V_temp = Vh_temp.T
            else:
                try:
                    U_temp, s_vals, Vh_temp = svds(Y_mat, k=s, which='LM')
                    idx_sort = np.argsort(s_vals)[::-1]  # 按奇异值降序排列
                    s_vals = s_vals[idx_sort]
                    U_temp = U_temp[:, idx_sort]
                    V_temp = Vh_temp[idx_sort, :].T
                except Exception:
                    U_temp, s_vals, Vh_temp = np.linalg.svd(Y_mat, full_matrices=False)
                    V_temp = Vh_temp.T

            # 对奇异值做阈值处理：只保留大于 tau*mu 的部分
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
            s = r + 1  # 为下一次迭代更新预估秩

            # 在观测位置上重构 X，计算残差
            x_vals = X_on_omega(U, sigma_vals, V, i_indices, j_indices)
            resid = x_vals - b
            relResid = np.linalg.norm(resid) / normb
            print(f"Iteration {numiter}, rank: {r}, relative residual: {relResid:.1e}")

            # 更新 G：在观测位置上置为当前残差 resid
            update_sparse(G, resid, i_indices, j_indices)

            # 停止条件：相对残差小于 tol 时退出内层循环
            if relResid < tol:
                break
        print(f"Final relative residual for mu={mu:.3e}: {relResid:.3e}, rank: {r}")

    # 外层循环结束后，打印最终的相对误差和残差信息
    final_relative_error = relResid
    final_residual_norm = np.linalg.norm(resid)
    print("========================================")
    print(f"Final relative error: {final_relative_error:.3e}")
    print(f"Final residual norm: {final_residual_norm:.3e}")
    print("========================================")

    return U, sigma_vals, V, numiter

# =============================================================================
# 使用示例：
# =============================================================================
# 假设我们需要恢复一个 n1 x n2 的矩阵，其中 Omega 是观测位置（线性索引，列主序），
# b 为对应的观测数据，mu_final 为目标 mu 值。
#
# n = (n1, n2)
# Omega = np.array([...])  # 观测位置索引
# b = np.array([...])      # 观测数据
# mu_final = 0.1           # 最终 mu 值
#
# U, sigma, V, num_iter = FPC(n, Omega, b, mu_final, maxiter=500, tol=1e-4)
#
# 得到的低秩矩阵 X 可由 U @ np.diag(sigma) @ V.T 重构。