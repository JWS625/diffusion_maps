from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
import cupy as cp
import cupyx.scipy.linalg

ArrayLike = Union[np.ndarray, cp.ndarray]


class DMClass:
    """
    Compatibility goals:
      - If P==1 and x.ndim==2: match legacy Code 2 numerics as closely as possible.
      - If P>1 and/or x.ndim==3: match legacy Code 1 batch numerics as closely as possible.

    Notes:
      - This class supports epsilon/lambda_reg as scalar or (P,) vector.
      - Training kernel:
          * P==1: uses cp.linalg.norm distances (Code 2 style)
          * P>1: uses a Gram-trick based distance->kernel path (Code 1 batch style)
      - Test kernel:
          * x.ndim==2 (requires P==1): uses cp.linalg.norm (Code 2 style)
          * x.ndim==3: uses Gram trick (Code 1 style)
    """

    def __init__(
        self,
        data: ArrayLike,                  # (N, d)
        f: ArrayLike,                     # (N,) or (N, out_dim)
        epsilon: Union[float, ArrayLike], # scalar or (P,)
        lambda_reg: Union[float, ArrayLike],  # scalar or (P,)
        mode: str = "rbf",
        distance_matrix: Optional[ArrayLike] = None,  # (N,N) Euclidean distances (not squared)
        dtype=cp.float64,
    ):
        if mode not in {"rbf", "diffusion"}:
            raise ValueError("mode must be 'rbf' or 'diffusion'")

        self.mode = mode

        self.data_old = cp.asarray(data, dtype=dtype)
        if self.data_old.ndim != 2:
            raise ValueError("data must be (N, d)")
        self.N, self.d = self.data_old.shape

        self.f = cp.asarray(f, dtype=dtype)
        if self.f.ndim == 1:
            self.f = self.f[:, None]
        if self.f.ndim != 2 or self.f.shape[0] != self.N:
            raise ValueError("f must be (N,) or (N, out_dim)")

        eps = cp.asarray(epsilon, dtype=dtype)
        lam = cp.asarray(lambda_reg, dtype=dtype)
        if eps.ndim == 0:
            eps = eps[None]
        if lam.ndim == 0:
            lam = lam[None]
        if eps.ndim != 1 or lam.ndim != 1:
            raise ValueError("epsilon and lambda_reg must be scalar or 1D")
        if eps.size != lam.size:
            raise ValueError("epsilon and lambda_reg must have the same length (P)")
        if cp.any(eps <= 0):
            raise ValueError("epsilon must be positive")
        if cp.any(lam < 0):
            raise ValueError("lambda_reg must be nonnegative")

        self.epsilon = eps
        self.lambda_reg = lam
        self.P = int(eps.size)

        self.kernel_matrix, self.q1, self.q2, self.distance_matrix = self._compute_kernel_matrix_and_densities(
            self.data_old,
            self.epsilon,
            mode=self.mode,
            distance_matrix=distance_matrix,
            dtype=dtype,
        )

        self._fit(self.f, self.lambda_reg)

    # ------------------------ public API ------------------------

    def predict(self, x: ArrayLike) -> cp.ndarray:
        """
        x:
          - (m, d)          -> returns (m, out_dim)        [only if P==1]
          - (P, m, d)       -> returns (P, m, out_dim)     [only if P>1]
        """
        x = cp.asarray(x, dtype=self.data_old.dtype)

        if x.ndim <= 2 :
            # compatibility with legacy: only allowed for P==1
            if self.P != 1:
                raise ValueError("For P>1, call predict with x shaped (P, m, d)")
            K = self._kernel_eval(x)          # (N, m)
            out = cp.dot(K.T, self.beta)      # (m, out_dim)
            return out

        elif x.ndim == 3:
            K = self._kernel_eval(x)          # (P, N, m)
            out = cp.einsum("pnm,pnd->pmd", K, self.beta)  # (P, m, out_dim)
            return out
        
        else:
            raise ValueError("x must be 2D (m,d) or 3D (P,m,d)")

    def _fit(self, f: cp.ndarray, lambda_reg: cp.ndarray) -> None:
        if self.P == 1:
            lam0 = lambda_reg[0]
            A = self.kernel_matrix + lam0 * cp.eye(self.N, dtype=self.kernel_matrix.dtype)
            self.beta = self._solve_2d(A, f)  # (N, out_dim)
        else:
            lam = cp.asarray(lambda_reg, dtype=self.kernel_matrix.dtype)  # (P,)
            I = cp.eye(self.N, dtype=self.kernel_matrix.dtype)[None, :, :]  # (1,N,N)
            A = self.kernel_matrix + lam[:, None, None] * I               # (P,N,N)
            self.beta = self._solve_batched(A, f)                         # (P,N,out_dim)

    @staticmethod
    def _solve_2d(A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
        A = cp.asarray(A)
        B = cp.asarray(B)
        if B.ndim == 1:
            B = B[:, None]
        L = cp.linalg.cholesky(A)
        Z0 = cupyx.scipy.linalg.solve_triangular(L, B, lower=True)
        Z1 = cupyx.scipy.linalg.solve_triangular(L, Z0, lower=True, trans=1)
        return Z1

    @staticmethod
    def _solve_batched(A: cp.ndarray, B: cp.ndarray) -> cp.ndarray:
        """
        A: (P, N, N)
        B: (N, out_dim) or (P, N, out_dim)
        returns: (P, N, out_dim)
        """
        A = cp.asarray(A)
        B = cp.asarray(B)

        P, N, N2 = A.shape
        if N != N2:
            raise ValueError("A must be (P,N,N)")

        if B.ndim == 1:
            B = B[:, None]         # (N,1)
            B = B[None, :, :]      # (1,N,1)
            B = cp.broadcast_to(B, (P, N, 1))
        elif B.ndim == 2:
            B = B[None, :, :]      # (1,N,out_dim)
            B = cp.broadcast_to(B, (P, N, B.shape[-1]))
        elif B.ndim == 3:
            if B.shape[0] != P or B.shape[1] != N:
                raise ValueError("For batched solve, B must be (P, N, out_dim)")
        else:
            raise ValueError("Unsupported B.ndim")

        L = cp.linalg.cholesky(A)  # (P,N,N)
        Z1 = cp.empty_like(B)
        for p in range(P):
            Z0 = cupyx.scipy.linalg.solve_triangular(L[p], B[p], lower=True)
            Z1[p] = cupyx.scipy.linalg.solve_triangular(L[p], Z0, lower=True, trans=1)
        return Z1

    def _kernel_eval(self, x: cp.ndarray) -> cp.ndarray:
        """
        Returns:
          - if P==1 and x is (m,d): (N,m)   [Code 2 style: direct norm]
          - if P>1 and x is (P,m,d): (P,N,m) [Code 1 style: Gram trick]
        """
        X = self.data_old  # (N,d)
        N, d = X.shape

        if x.ndim == 1:
            x = x[None]

        if x.ndim == 2:
            if self.P != 1:
                raise ValueError("x.ndim==2 is only supported for P==1")
            if x.shape[1] != d:
                raise ValueError(f"x has d={x.shape[1]} but training data has d={d}")

            dist = cp.linalg.norm(X[:, None] - x[None, :], axis=-1)  # (N,m) direct norm
            K = cp.exp(-(dist**2) / (4.0 * self.epsilon[0]))

            if self.mode == "diffusion":
                q1 = self.q1
                q2 = self.q2
                assert q1 is not None and q2 is not None

                q = cp.power(cp.mean(K, axis=0), -1)          # (m,)
                K = cp.einsum("ij,j,i->ij", K, q, q1)
                q = cp.power(cp.mean(K, axis=0), -1 / 2)      # (m,)
                K = cp.einsum("ij,j,i->ij", K, q, q2)

            return K  # (N,m)

        P, m, d2 = x.shape
        if d2 != d:
            raise ValueError(f"x has d={d2} but training data has d={d}")

        Y = x.reshape(P * m, d)  # (Pm,d)

        XX = cp.sum(X * X, axis=1)[:, None]      # (N,1)
        YY = cp.sum(Y * Y, axis=1)[None, :]      # (1,Pm)
        dist2 = XX + YY - 2.0 * (X @ Y.T)        # (N,Pm)
        cp.maximum(dist2, 0.0, out=dist2)

        dist2 = dist2.reshape(N, P, m).transpose(1, 0, 2)  # (P,N,m)
        eps = self.epsilon[:, None, None]                   # (P,1,1)
        K = cp.exp(-dist2 / (4.0 * eps))                    # (P,N,m)

        if self.mode == "diffusion":
            q1 = self.q1
            q2 = self.q2
            assert q1 is not None and q2 is not None  # (P,N)

            q = cp.power(cp.mean(K, axis=1), -1)              # (P,m)
            K = cp.einsum("pnm,pm,pn->pnm", K, q, q1)

            q = cp.power(cp.mean(K, axis=1), -1 / 2)          # (P,m)
            K = cp.einsum("pnm,pm,pn->pnm", K, q, q2)

        return K  # (P,N,m)

    @staticmethod
    def _compute_kernel_matrix_and_densities(
        data: cp.ndarray,
        epsilon: cp.ndarray,  # (P,)
        mode: str,
        distance_matrix: Optional[ArrayLike] = None,  # (N,N) Euclidean distances (not squared)
        dtype=cp.float64,
    ) -> Tuple[cp.ndarray, Optional[cp.ndarray], Optional[cp.ndarray], cp.ndarray]:
        """
        Training kernel builder.

        Compatibility rule:
          - P==1: match Code 2 => use cp.linalg.norm if distance_matrix not provided.
          - P>1: match Code 1 batch style => use Gram trick (dist2 = xx+yy-2xy).
        """
        X = cp.asarray(data, dtype=dtype)
        N, d = X.shape

        eps = cp.asarray(epsilon, dtype=dtype)
        if eps.ndim == 0:
            eps = eps[None]
        P = int(eps.size)

        if distance_matrix is not None:
            base_dist = cp.asarray(distance_matrix, dtype=dtype)
        else:
            if P == 1:
                base_dist = cp.linalg.norm(X[:, None] - X[None, :], axis=-1)  # (N,N)
            else:
                n = cp.sum(X * X, axis=1)                      # (N,)
                dist2 = n[:, None] + n[None, :] - 2.0 * (X @ X.T)
                cp.maximum(dist2, 0.0, out=dist2)
                base_dist = cp.sqrt(dist2)
                base_dist = 0.5 * (base_dist + base_dist.T)
                cp.fill_diagonal(base_dist, 0.0)

        # keep stored distance_matrix in Euclidean form (like Code 2/Code 1)
        distance_matrix_out = base_dist

        # ---- Build kernel ----
        if P == 1:
            K = cp.exp(-(base_dist**2) / (4.0 * eps[0]))  # (N,N)

            if mode == "rbf":
                return K, None, None, distance_matrix_out

            q1 = cp.power(cp.mean(K, axis=-1), -1)        # (N,)
            K = cp.einsum("ij,i,j->ij", K, q1, q1)
            q2 = cp.power(cp.mean(K, axis=-1), -1 / 2)    # (N,)
            K = cp.einsum("ij,i,j->ij", K, q2, q2)

            return K, q1, q2, distance_matrix_out

        # P>1 => (P,N,N)
        base2 = (base_dist**2)[None, :, :]                          # (1,N,N)
        K = cp.exp(-(base2) / (4.0 * eps[:, None, None]))           # (P,N,N)

        if mode == "rbf":
            return K, None, None, distance_matrix_out

        q1 = cp.power(cp.mean(K, axis=2), -1)                       # (P,N)
        K = cp.einsum("pij,pi,pj->pij", K, q1, q1)

        q2 = cp.power(cp.mean(K, axis=2), -1 / 2)                   # (P,N)
        K = cp.einsum("pij,pi,pj->pij", K, q2, q2)

        return K, q1, q2, distance_matrix_out
