import sys
# from typing import Tuple, Union

# sys.path.append("../")

import numpy as np
import cupy as cp
# import cupyx.scipy.sparse as cpsp

from dataclasses import dataclass
import cupyx.scipy.linalg
import scipy

@dataclass
class DMClass:
    def __init__(
        self,
        data,
        f,
        epsilon: float,
        lambda_reg: float,
        mode: str = "rbf",
        distance_matrix = None,
    ):
        
        # self.multiple_param_flag = True
        epsilon = cp.asarray(epsilon)
        if epsilon.ndim == 0:
            # self.multiple_param_flag = False
            epsilon = epsilon[None]

        lambda_reg = cp.asarray(lambda_reg)
        if lambda_reg.ndim == 0:
            lambda_reg = lambda_reg[None]       
        # assert len(data.shape) == 2, "Data must be of form (n, d)"
        # assert len(f.shape) <= 2, "f must be of form (n,) or (n, k)"
        # assert epsilon > 0, "Epsilon must be positive"
        # assert lambda_reg >= 0, "Regularization parameter must be non-negative"

        # if k is not None:
        #     assert k > 0

        self.data_old = data
        self.epsilon = epsilon
        self.lambda_reg = lambda_reg
        
        self.f = f
        # self.DS = DS
        self.kernel_matrix, self.q1, self.q2, self.distance_matrix = (
            DMClass._compute_kernel_matrix_and_densities(data, epsilon, mode=mode, distance_matrix=distance_matrix)
        )

        self.mode = mode
        # self.k = k
        self.d = cp.shape(data)[1]

        self.q1 = cp.asarray(self.q1) if self.q1 is not None else None
        self.q2 = cp.asarray(self.q2) if self.q2 is not None else None
        # if self.DS is False:
        self._fit(self.f, lambda_reg)
        # else:
        #     self.SK_prepare()


    def predict(self, x):
        # print(f"x.shape = {x.shape}")

        assert x.ndim == 2 or x.ndim == 3

        kernel_matrix = self._kernel_eval(x)
        if x.ndim==3:
        #     output = cp.dot(kernel_matrix.T, self.beta)
            output = cp.einsum("knm,knd->kmd", kernel_matrix, self.beta)
        elif x.ndim==2:
            # output = cp.einsum("knm,knd->kmd", kernel_matrix, self.beta)
            output = cp.einsum("nm,knd->kmd", kernel_matrix, self.beta)
        else:
            ValueError("Unvalid epsilon shape. It has to be either scalar or 1D array.")

        return output


    def _fit(self, f, lambdaReg):
        eps = self.epsilon
        lam = cp.asarray(lambdaReg)
        P   = eps.shape[0]
        p, N, N2 = self.kernel_matrix.shape
        assert p == P and N == N2

        self.beta = DMClass._solve(cp.array(self.kernel_matrix) \
            + lam[:, None, None] * cp.eye(N)[None, :, :], cp.array(f))



    @staticmethod
    def _solve(psd_matrix, X):
        # solve random_psd_matrix_gpu @ z = X for z
        # psd_matrix must be positive semi-definite matrix or else this will fail
        psd_matrix = cp.asarray(psd_matrix)
        X = cp.asarray(X)

        if psd_matrix.ndim == 2:
            A = cp.linalg.cholesky(psd_matrix)
            Z0 = cupyx.scipy.linalg.solve_triangular(A, X, lower=True)
            Z1 = cupyx.scipy.linalg.solve_triangular(A, Z0, trans=1, lower=True)
            return Z1
        
        if psd_matrix.ndim == 3:
            P, N, N2 = psd_matrix.shape
            assert N == N2, "psd_matrix must be (K, N, N)"

            if X.ndim == 1:
                X = X[None, :, None]
                X = cp.broadcast_to(X, (P, N, 1))
            elif X.ndim == 2:
                # (N, d) -> (P, N, d)
                X = X[None, :, :]
                X = cp.broadcast_to(X, (P, N, X.shape[-1]))
            elif X.ndim == 3:
                # (P, N, d): assume already aligned
                assert X.shape[0] == P and X.shape[1] == N, \
                    "For batched solve, X must be (P, N, d) if 3D."
            else:
                raise ValueError(f"Unsupported X.ndim={X.ndim} for batched _solve")

            Z = cp.empty_like(X)
            A = cp.linalg.cholesky(psd_matrix)   # (P, N, N)

            for p in range(P):
                A_p = A[p]                                    # (N, N)
                Z0 = cupyx.scipy.linalg.solve_triangular(A_p, X[p], lower=True)   # (N, d)
                Z[p] = cupyx.scipy.linalg.solve_triangular(A_p, Z0, trans=1, lower=True)

            return Z

    
    def _kernel_eval(self, x):

        x = cp.asarray(x)
        assert x.ndim == 3

        eps = cp.asarray(self.epsilon)

        # if x.ndim == 2:
        #     print(f"x.ndim==2")
        #     distance_matrix = cp.linalg.norm(
        #         self.data_old[:, None, :] - x[None, :, :],
        #         axis=-1
        #     )

        #     kernel_matrix = cp.exp(-distance_matrix**2 / (4.0 * eps.squeeze()))

        #     if self.mode == "diffusion":
        #         assert self.q1 is not None
        #         assert self.q2 is not None

        #         # q from current kernel
        #         q = cp.power(cp.mean(kernel_matrix, axis=0), -1)  # (m,)
        #         kernel_matrix = cp.einsum("ij,j,i->ij", kernel_matrix, q, self.q1)

        #         if self.DS:
        #             q = cp.power(cp.dot(self.q2, kernel_matrix), -1)  # (m,)
        #             kernel_matrix = cp.einsum("ij,j,i->ij", kernel_matrix, q, self.q2)
        #         else:
        #             q = cp.power(cp.mean(kernel_matrix, axis=0), -1 / 2)  # (m,)
        #             kernel_matrix = cp.einsum("ij,j,i->ij", kernel_matrix, q, self.q2)

        #     return kernel_matrix  # (N, m)

        # if x.ndim == 3:
        # print(f"x.ndim==3")
        P, m, d = x.shape
        X = self.data_old  
        N = X.shape[0]
        
        Y = x.reshape(P * m, d)  

        XX = cp.sum(X * X, axis=1)[:, None]   # (N, 1)
        YY = cp.sum(Y * Y, axis=1)[None, :]   # # (1, Pm)

        dist2 = XX + YY - 2.0 * (X @ Y.T)     # (N, Pm)
        cp.maximum(dist2, 0.0, out=dist2)     # clip tiny negatives

        dist2 = dist2.reshape(N, P, m)        # (N, P, m)
        dist2 = dist2.transpose(1, 0, 2)      # (P, N, m)

        eps_b   = eps[:, None, None]         # (1, 1, 1)

        kernel_matrix = cp.exp(-dist2 / (4.0 * eps_b))  # (P, N, m)

        if self.mode == "diffusion":
            assert self.q1 is not None
            assert self.q2 is not None

            q = cp.power(cp.mean(kernel_matrix, axis=1), -1)  # (P, m)

            kernel_matrix = cp.einsum(
                "pnm,pm,pn->pnm",
                kernel_matrix,   # (p, N, m)
                q,               # (p, m)
                self.q1          # (p, N)
            )


            # mean over N, exponent -1/2
            q = cp.power(cp.mean(kernel_matrix, axis=1), -1 / 2)  # (p, m)
            kernel_matrix = cp.einsum(
                "knm,km,kn->knm",
                kernel_matrix,
                q,
                self.q2,
            )

        return kernel_matrix  # (p, N, m)

    @staticmethod
    def _compute_kernel_matrix_and_densities(
        data, epsilon, mode, distance_matrix=None, dtype=cp.float64,
    ):
        """
        Compute RBF or diffusion kernel matrix from data.
        Uses sparse matrix if k is provided (i.e., k-NN kernel).

        Args:
            data: (n, d) input data
            epsilon: float for Gaussian kernel
            mode: "rbf" or "diffusion"
            distance_matrix: optional (n, n) matrix of pairwise distances
            k: number of neighbors to keep (if None, full dense kernel is used)

        Returns:
            kernel_matrix: dense or sparse kernel
            q1, q2: normalization vectors (None if mode == "rbf")
            distance_matrix: full or kNN-masked
        """
        assert mode in {"rbf", "diffusion"}
        if epsilon.ndim == 0:
            epsilon = epsilon[None]

        epsilon = cp.asarray(epsilon)
        P = epsilon.shape[0]

        if distance_matrix is None:
            X = cp.asarray(data, dtype=cp.float64)  # (N, d)
            n = cp.sum(X * X, axis=1)                    # (N,)

            distance_matrix = X @ X.T
            distance_matrix *= -2.0
            distance_matrix += n[:, None]
            distance_matrix += n[None, :]
            cp.maximum(distance_matrix, 0.0, out=distance_matrix)
            cp.sqrt(distance_matrix, out=distance_matrix)
            cp.fill_diagonal(distance_matrix, 0.0)
            distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)
            base_dist = distance_matrix
        else:
            base_dist = distance_matrix

            
        base_dist = base_dist[None, :, :]
        epsilon = epsilon[:, None, None]
        kernel_matrix = cp.exp(-(base_dist**2) / (4 * epsilon))

        if mode == "rbf":
            return kernel_matrix, None, None, distance_matrix

        q1 = cp.power(cp.mean(kernel_matrix, axis=2), -1)
        kernel_matrix = cp.einsum("kij,ki,kj->kij", kernel_matrix, q1, q1)
        q2 = cp.power(cp.mean(kernel_matrix, axis=2), -1/2)
        kernel_matrix = cp.einsum("kij,ki,kj->kij", kernel_matrix, q2, q2)

        return kernel_matrix, q1, q2, distance_matrix





