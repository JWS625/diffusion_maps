import os
import numpy as np
import scipy.linalg as spl


# DMD helpers
def _svd_truncate(S, energy_threshold=None, r=None):
    """
    Decide truncation rank from singular values S.
    Priority: explicit r, then energy_threshold, else full.
    """
    if r is not None:
        return int(r)
    if energy_threshold is not None:
        s2 = S**2
        cum = np.cumsum(s2) / np.sum(s2)
        return int(np.searchsorted(cum, energy_threshold) + 1)
    return len(S)


def _filename(H):
    return f"omega_cube_phaseA_0_phaseH_{H}.npy"


def _find_file(basepath, H):
    p = os.path.join(basepath, _filename(H))
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"Could not find file for H={H} under {basepath}")


def _load_raw_sequence(H, basepath, N, N_trun, ds=1, dtype=np.float64):
    """
    Load one realization, return dat with shape (N_used, m) WITHOUT demeaning.
    """
    fn = _find_file(basepath, H)
    vort = np.load(fn)
    vort = vort[N_trun:]
    vort = vort.reshape(len(vort), -1).astype(dtype, copy=False)
    if ds is None or ds < 1:
        ds = 1
    vort = vort[::ds, :]
    N_used = vort.shape[0] if (N is None) else min(N, vort.shape[0])
    if N_used < 2:
        raise ValueError(f"Need at least 2 snapshots after truncation; got {N_used} for H={H}")
    return vort[:N_used, :]


def _exact_dmd(X, Y, r=None, energy_threshold=None, dt=None):
    """
    Exact DMD on X,Y with Y ≈ A X.
    X, Y : (m, T).

    Returns:
      lam, Phi, b0, omega, U_r, S_r, V_r
    """
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    r_eff = _svd_truncate(S, energy_threshold=energy_threshold, r=r)
    U_r = U[:, :r_eff]                    # (m, r)
    S_r = S[:r_eff]                       # (r,)
    V_r = Vh.conj().T[:, :r_eff]          # (T, r)

    Atilde = U_r.conj().T @ Y @ V_r @ np.diag(1.0 / S_r)
    lam, W = np.linalg.eig(Atilde)

    # Full DMD modes
    Phi = (Y @ V_r) @ np.diag(1.0 / S_r) @ W

    b0, *_ = np.linalg.lstsq(Phi, X[:, 0], rcond=None)

    omega = None if (dt is None or dt <= 0) else np.log(lam) / dt
    return lam, Phi, b0, omega, U_r, S_r, V_r


def _dmd_reconstruct(Phi, lam, b, timesteps):
    """
    Reconstruct snapshots from DMD modes.
    Phi: (m,r), lam:(r,), b:(r,), timesteps: 1D array of ints.
    Returns (m,K) complex.
    """
    k = np.asarray(timesteps)
    Vand = lam[:, None] ** k[None, :]
    return Phi @ (b[:, None] * Vand)


def _welford_update(mean, count, x):
    """
    Streaming mean update for a batch x: (k, m).
    """
    k = x.shape[0]
    if k == 0:
        return mean, count
    batch_mean = x.mean(axis=0)
    delta = batch_mean - mean
    new_count = count + k
    mean = mean + delta * (k / new_count)
    return mean, new_count


class ResDMD:
    """
    Exact DMD on multiple realizations + ResDMD residuals in reduced SVD space.

    Workflow:
      1) fit(...)      -> build global mean, stack X,Y, compute SVD and DMD eigensystem
      2) compute_residuals_sako()    -> one residual per eigenpair (r-dimensional)
      3) validate_residual_order_external(...)   -> pick best truncation order externally
      4) filter_by_residual(order=...)           -> activate chosen subset of modes
      5) evaluate(H_test, N_test) / reconstruct_sequence(...)
    """
    def __init__(self, basepath, N_trun=300, dt=None, ds=1,
                 r=None, energy_threshold=None, dtype=np.float64):
        self.basepath = basepath
        self.N_trun   = N_trun
        self.dt       = dt
        self.ds       = ds
        self.r        = r
        self.energy_threshold = energy_threshold
        self.dtype    = dtype

        # Learned after fit
        self.mean_vec_ = None      # (m,)
        self.Phi_      = None      # active DMD modes (m, r_active)
        self.lam_      = None      # active eigenvalues (r_active,)
        self.omega_    = None
        self.rank_     = None      # r_active
        self.m_        = None      # original dimension

        # Low-rank factors & training Y (for reduced ResDMD)
        self._U_r = None           # (m, r)
        self._S_r = None           # (r,)
        self._V_r = None           # (T_total, r)
        self._Y_tr = None          # (m, T_total)

        # Full (unfiltered) eigensystem + residuals
        self.Phi_full_   = None    # (m, r)
        self.lam_full_   = None    # (r,)
        self.omega_full_ = None    # (r,)
        self.residuals_  = None    # (r,)

    # Fit on training realizations
    def fit(self, H_sets_train, N_train):
        """
        Fit Exact DMD on all training realizations.

        H_sets_train : list of angles (or identifiers)
        N_train      : number of time steps per realization (after N_trun)
        """
        mean_vec = None
        count = 0
        raw_cache = {}

        for H in H_sets_train:
            dat = _load_raw_sequence(H, self.basepath, N_train, self.N_trun,
                                     self.ds, dtype=self.dtype)
            raw_cache[H] = dat                         # (N_train, m)
            if mean_vec is None:
                mean_vec = np.zeros(dat.shape[1], dtype=self.dtype)
            mean_vec, count = _welford_update(mean_vec, count, dat)

        if count == 0:
            raise RuntimeError("No training frames found.")

        self.mean_vec_ = mean_vec
        self.m_ = mean_vec.size

        X_list, Y_list = [], []
        for H in H_sets_train:
            dat_dm = raw_cache[H] - self.mean_vec_[None, :]
            X_list.append(dat_dm[:-1].T.astype(self.dtype, copy=False))
            Y_list.append(dat_dm[ 1:].T.astype(self.dtype, copy=False))
        X_tr = np.concatenate(X_list, axis=1)
        Y_tr = np.concatenate(Y_list, axis=1)
        self._Y_tr = Y_tr

        # 3) Exact DMD
        lam, Phi, _, omega, U_r, S_r, V_r = _exact_dmd(
            X_tr, Y_tr, r=self.r,
            energy_threshold=self.energy_threshold,
            dt=self.dt
        )

        self._U_r = U_r
        self._S_r = S_r
        self._V_r = V_r

        self.Phi_full_   = Phi.astype(np.complex128, copy=False)
        self.lam_full_   = lam.astype(np.complex128, copy=False)
        self.omega_full_ = None if omega is None else omega.astype(np.complex128, copy=False)

        self.Phi_   = self.Phi_full_.copy()
        self.lam_   = self.lam_full_.copy()
        self.omega_ = None if self.omega_full_ is None else self.omega_full_.copy()
        self.rank_  = self.Phi_.shape[1]

        return self

    def _build_Psi_reduced(self):
        """
        Build reduced (T x r) Psi0, Psi1 using SVD coordinates.
        Uses:
          X_tr ≈ U_r Σ_r V_r^H
          Z0 = U_r^* X_tr = Σ_r V_r^H
          Z1 = U_r^* Y_tr
        Returns:
          Psi0_r, Psi1_r : (T, r)
        """
        if self._U_r is None or self._S_r is None or self._V_r is None or self._Y_tr is None:
            raise RuntimeError("fit() must be called before computing residuals.")

        Vh_r = self._V_r.conj().T
        Z0   = np.diag(self._S_r) @ Vh_r
        Z1   = self._U_r.conj().T @ self._Y_tr

        Psi0_r = Z0.T
        Psi1_r = Z1.T
        return Psi0_r, Psi1_r

    def compute_residuals_sako(self, W=None, eps=1e-12, B=None):
        """
        ResDMD/SAKO residuals in the reduced DMD subspace (dimension r).

        Steps:
          - Build reduced Psi0_r, Psi1_r (T x r).
          - Build r x r inner-product matrices M00,M01,M10,M11.
          - Hermitianize M0,M1.
          - Solve generalized EVP (M1,M0) with left/right eigenvectors.
          - Biorthogonal scaling: vl^H vr ≈ I (or under metric B).
          - SAKO residual per eigenpair:
              r_i = sqrt( (g_i^H G(λ_i) g_i) / (g_i^H M00 g_i) ),
            where g_i is the scaled right generalized eigenvector.
          - Align residuals to lam_full_ (eigenvalues of Ã).

        Sets:
          self.residuals_ : shape (r,)
        Returns:
          residuals_, order (indices sorted by increasing residual)
        """
        if self.Phi_full_ is None or self.lam_full_ is None:
            raise RuntimeError("Model not fitted.")

        # 1) Reduced Psi matrices
        Psi0, Psi1 = self._build_Psi_reduced()
        T, r = Psi0.shape

        # 2) Time weights
        if W is None:
            W = np.ones(T, dtype=np.float64)
        W = W.reshape(-1)

        # 3) Inner products in reduced space (r x r)
        def _M(Pi, Pj):
            return (Pi.conj().T * W) @ Pj

        M00 = _M(Psi0, Psi0)
        M01 = _M(Psi0, Psi1)
        M10 = _M(Psi1, Psi0)
        M11 = _M(Psi1, Psi1)

        # Hermitianized pencil
        M0 = 0.5 * (M00 + M00.conj().T)
        M1 = 0.5 * (M01 + M10.conj().T)

        # 4) Generalized eigendecomposition (r x r)
        wd, vl, vr = spl.eig(M1, M0, left=True, right=True)

        # 5) Biorthogonal scaling: vl^H vr ≈ I (or with metric B)
        if B is None:
            S   = vl.conj().T @ vr
        else:
            S = vl.conj().T @ (B @ vr)

        scl = np.diag(S).copy()
        scl[np.isclose(scl, 0)] = eps
        sr  = np.sqrt(scl)
        sl  = sr.conj()

        vr = vr / sr.reshape(1, -1)
        vl = vl / sl.reshape(1, -1)

        # 6) G(lam) and residuals
        def _G_of(lam):
            G = M11 - lam * M10 - lam.conjugate() * M01 + (abs(lam) ** 2) * M00
            return 0.5 * (G + G.conj().T)

        res_eval = np.empty_like(wd.real)
        denom = np.sum((vr.conj().T @ M00) * vr.T, axis=1).real  # g^H M00 g
        denom = np.maximum(denom, eps)

        for i, lam in enumerate(wd):
            Gi = _G_of(lam)
            num = np.vdot(vr[:, i], Gi @ vr[:, i]).real
            res_eval[i] = np.sqrt(max(num, 0.0) / denom[i])

        lam_eval = wd

        # 7) Align to model eigenvalues (from reduced A)
        lam_model = self.lam_full_
        res_aligned = np.empty(lam_model.size, dtype=np.float64)
        for j, lm in enumerate(lam_model):
            k = int(np.argmin(np.abs(lam_eval - lm)))
            res_aligned[j] = res_eval[k]

        self.residuals_ = res_aligned
        order = np.argsort(self.residuals_)
        return self.residuals_, order

    # Residual filtering
    def filter_by_residual(self, order='full', keep_conjugates=True, conj_tol=1e-8):
        """
        Trim modes by residual (self.residuals_ must exist).

        order:
          'full'  -> keep all modes
          int k   -> keep k modes with smallest residuals
          float τ -> keep all modes with residual <= τ

        keep_conjugates:
          If True, attempt to keep complex-conjugate pairs together.
        """
        if self.residuals_ is None:
            raise RuntimeError("Call compute_residuals_sako() first.")

        res = self.residuals_
        lam = self.lam_full_
        Phi = self.Phi_full_
        omg = self.omega_full_

        # base selection
        if order == 'full':
            idx = np.arange(lam.size)
        elif isinstance(order, int):
            idx = np.argsort(res)[:max(1, order)]
        elif isinstance(order, float):
            idx = np.where(res <= order)[0]
            if idx.size == 0:
                idx = np.array([np.argmin(res)])
        else:
            raise ValueError("order must be 'full', int (k), or float (threshold).")

        # enforce conjugate pairs
        if keep_conjugates and idx.size > 0:
            chosen = set(idx.tolist())
            for i in idx.tolist():
                lam_i_conj = np.conj(lam[i])
                j = int(np.argmin(np.abs(lam - lam_i_conj)))
                if np.abs(lam[j] - lam_i_conj) <= conj_tol:
                    chosen.add(j)
            idx = np.array(sorted(chosen), dtype=int)

        # update active eigensystem
        self.Phi_   = Phi[:, idx]
        self.lam_   = lam[idx]
        self.omega_ = None if omg is None else omg[idx]
        self.rank_  = self.Phi_.shape[1]
        return idx, res[idx]

    # Rollout and evaluation
    def amplitudes(self, x0):
        """
        Compute DMD amplitudes for a de-meaned initial state x0 (m,).
        """
        if self.Phi_ is None:
            raise RuntimeError("Model not fitted.")
        b, *_ = np.linalg.lstsq(self.Phi_, x0.astype(self.dtype), rcond=None)
        return b.astype(np.complex128, copy=False)

    def reconstruct_sequence(self, H, N, add_back_mean=False):
        """
        Load realization H (N frames), subtract training mean, and roll out from first snapshot.

        Returns dict with keys:
          'X_demeaned_truth', 'Xrec', 'b', 'relative_error', 'rmse', 'H', 'K'
        """
        if self.mean_vec_ is None:
            raise RuntimeError("Model not fitted.")

        dat = _load_raw_sequence(H, self.basepath, N, self.N_trun,
                                 self.ds, dtype=self.dtype)   # (N, m)
        
        t_window = self.dt * np.arange(len(dat))
        weights = np.exp(t_window-t_window[-1])
        X_truth = dat - self.mean_vec_[None, :] #(N, m)
        x0 = X_truth[0]

        N = X_truth.shape[0]
        m = X_truth.shape[1]

        b_te, *_ = np.linalg.lstsq(self.Phi_, x0, rcond=None)
        Xrec = _dmd_reconstruct(self.Phi_, self.lam_, b_te, np.arange(N)).real.T  # (N, m)
        _err = X_truth - Xrec   # (N, m)
        
        rel_err = np.linalg.norm(weights[:, None]*_err) / np.linalg.norm(X_truth)
        rmse    = np.sqrt(np.mean((X_truth - Xrec) ** 2))

        if add_back_mean:
            Xrec_out = Xrec + self.mean_vec_[None, :]
        else:
            Xrec_out = Xrec

        return {
            "H": H,
            "ambient_dimension": m,
            "X_demeaned_truth": X_truth,
            "Xrec": Xrec_out,
            "b": b_te,
            "relative_error": float(rel_err),
            "rmse": float(rmse),
        }

    def evaluate(self, H, N):
        """
        Convenience: return (relative_error, rmse) on de-meaned space for (H,N).
        """
        res = self.reconstruct_sequence(H, N, add_back_mean=False)
        return res["relative_error"], res["rmse"]


# External validation of residual truncation order
def _snapshot_spectrum(model: ResDMD):
    return {
        "Phi_":   model.Phi_.copy(),
        "lam_":   model.lam_.copy(),
        "omega_": None if model.omega_ is None else model.omega_.copy(),
        "rank_":  model.rank_,
    }


def _restore_spectrum(model: ResDMD, snap):
    model.Phi_   = snap["Phi_"]
    model.lam_   = snap["lam_"]
    model.omega_ = snap["omega_"]
    model.rank_  = snap["rank_"]


def validate_residual_order_external(
    model: ResDMD,
    H_valid,
    N_valid,
    orders,
    *,
    keep_conjugates=True,
    metric="rmse",
    aggregate="mean",
    apply_best=True,
):
    """
    Sweep over 'orders' (e.g., ['full', 10, 20, 50, 1e-3]) and pick the best
    residual truncation order using validation trajectories.

    NOTE:
      - We assume model.fit(...) and model.compute_residuals_sako() have already been called.
      - We NEVER change the rank r of the reduced Ã; we only pick subsets of eigenpairs.

    Returns:
      best : dict { 'order', 'score', 'per_seq', 'kept', 'idx_kept' }
      results : list of such dicts, one per order
    """
    assert metric in ("rmse", "rel")
    assert aggregate in ("mean", "median", "max")

    if model.residuals_ is None:
        model.compute_residuals_sako()

    snap0 = _snapshot_spectrum(model)
    agg_fn = {"mean": np.mean, "median": np.median, "max": np.max}[aggregate]

    results = []
    for ord_val in orders:
        idx_kept, _ = model.filter_by_residual(order=ord_val,
                                               keep_conjugates=keep_conjugates)

        per_seq = []
        for H in H_valid:
            rel, rmse = model.evaluate(H, N_valid)
            per_seq.append(rmse if metric == "rmse" else rel)
        per_seq = np.asarray(per_seq, dtype=float)
        score = float(agg_fn(per_seq))

        results.append({
            "order":   ord_val,
            "score":   score,
            "per_seq": per_seq,
            "kept":    model.rank_,
            "idx_kept": idx_kept,
        })

    best = min(results, key=lambda d: d["score"])

    _restore_spectrum(model, snap0)
    if apply_best:
        model.filter_by_residual(order=best["order"],
                                 keep_conjugates=keep_conjugates)

    return best, results

