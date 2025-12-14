import numpy as np
import cupy as cp
from dm_main import DMClass

class modeler:
    def __init__(self, **kwargs):
        self.dm = None
        self.def_opt = {
            'map_type' : 'direct',   # 'direct' or 'skip-connection'
            'inp'      : None,       # X (raw)
            'out'      : None,       # Y (raw)
            'data'     : None,       # timeseries (X=data[:-1], Y=data[1:])
            'norm'     : False,      # bool only
        }
        for k, v in self.def_opt.items(): 
            setattr(self, k, v)
        for k, v in kwargs.items():
            if k in self.def_opt: 
                setattr(self, k, v)

        if self.map_type not in {"direct", "skip-connection"}:
            raise ValueError("map_type must be 'direct' or 'skip-connection'")
        if not isinstance(self.norm, bool):
            raise ValueError("norm must be a boolean: True or False")

        # Build raw X, Y
        if self.inp is not None and self.out is not None:
            X_raw = np.asarray(self.inp); Y_raw = np.asarray(self.out)
        elif self.data is not None:
            data_raw = np.asarray(self.data)
            X_raw = data_raw[:-1]; Y_raw = data_raw[1:]
        else:
            raise ValueError("Provide either (inp, out) or data.")

        self._normalization(X_raw, Y_raw, recompute_stats=True)

    def _normalization(self, X_raw: np.ndarray, Y_raw: np.ndarray, recompute_stats: bool):

        if not self.norm:

            T = Y_raw if self.map_type == 'direct' else (Y_raw - X_raw)
            self.inp = X_raw
            
            if not hasattr(self, "mean_state"): self.mean_state = None
            if not hasattr(self, "std_state"):  self.std_state  = None

            if recompute_stats:
                self.std_nmse = Y_raw.std(axis=0)
                self.std_nmse = np.where(self.std_nmse == 0.0, 1.0, self.std_nmse)

        else:

            if recompute_stats:
                # print(f"X_raw.dtype = {X_raw.dtype}, Y_raw.dtype = {Y_raw.dtype}")
                XY = np.vstack([X_raw, Y_raw[-1, :].reshape(1, -1)])
                self.mean_state = XY.mean(axis=0)
                # print(f"self.mean_state.dtype = {self.mean_state.dtype}")
                self.std_state  = XY.std(axis=0)
                self.std_state  = np.where(self.std_state == 0.0, 1.0, self.std_state)

                self.std_nmse = Y_raw.std(axis=0)
                self.std_nmse = np.where(self.std_nmse == 0.0, 1.0, self.std_nmse)
            else:
                if getattr(self, "mean_state", None) is None or getattr(self, "std_state", None) is None:
                    raise ValueError("Normalization stats are not initialized. "
                                     "They are computed in __init__ when norm=True; "
                                     "call __init__ with norm=True first or refit with recompute_stats=True.")

            Xn = (X_raw - self.mean_state) / self.std_state
            Yn = (Y_raw - self.mean_state) / self.std_state

            T = Yn if self.map_type == 'direct' else (Yn - Xn)
            self.inp = Xn
            
        self.label = T


    def fit_model(self, epsilon, lambda_reg, mode, distance_matrix=None):
    # make sure epsilon and lambda_reg are 1D arrays
        epsilon    = np.atleast_1d(np.asarray(epsilon))
        lambda_reg = np.atleast_1d(np.asarray(lambda_reg))

        self.epsilon    = epsilon
        self.lambda_reg = lambda_reg
        self.mode       = mode

        self.dm = DMClass(
            cp.array(self.inp),
            self.label,
            epsilon,
            lambda_reg,
            mode=mode,
            distance_matrix=distance_matrix,

        )


    def mapping(self, x0):
        return self.dm.predict(x0)

    def forecast(self, x0):
        """Accepts RAW x0 (NumPy or CuPy), returns prediction in RAW units."""
        xp = cp.get_array_module(x0)
        x0 = xp.array(x0)
        if self.dm is None:
            raise ValueError("Model not trained yet")

        if self.norm:
            mean_s = xp.array(self.mean_state); std_s = xp.array(self.std_state)
            x0n = (x0 - mean_s) / std_s
            t = self.mapping(x0n)
            y_norm = t if self.map_type == 'direct' else (x0n + t)
            return y_norm * std_s + mean_s
        else:
            t = self.mapping(x0)
            return x0 + t if self.map_type == 'skip-connection' else t

    def get_performance(self, test, dt, Lyapunov_time, error_threshold=0.3**2, return_pred=False):
        assert test.ndim == 2 or test.ndim == 3
        assert self.epsilon is not None
        assert self.lambda_reg is not None

        if test.ndim == 2:
            return self.get_vpt(test, dt, Lyapunov_time, error_threshold=error_threshold, return_pred=return_pred)
        else:
            return self.get_vpt_multipoint(test, dt, Lyapunov_time, error_threshold=error_threshold)
    
    def get_vpt_multipoint(self, test, dt, Lyapunov_time, error_threshold=0.3**2):
        """
        Multi-trajectory VPT / tau_f computation on GPU.

        Parameters
        ----------
        test : array-like, shape (n_traj, T, d)
            Multiple test trajectories in RAW units.
        dt : float
        Lyapunov_time : float
        error_threshold : float, optional
            Threshold for both NMSE and SE.
        return_pred : bool, optional
            If True, also returns predictions with shape (n_traj, T, d) as CuPy array.

        Returns
        -------
        tau_f_nmse : np.ndarray, shape (n_traj,)
            Tau_f based on NMSE per trajectory (in Lyapunov units)
        tau_f_se   : np.ndarray, shape (n_traj,)
            Tau_f based on SE per trajectory (in Lyapunov units)
        pred       : cp.ndarray, shape (n_traj, T, d) (only if return_pred=True)
            Forecast trajectories in RAW units.
        """
        test = cp.asarray(test)
        M, T, d = test.shape

        u_hat = test[:, 0, :]              # (M, d)
        u_hat = u_hat[:, None, :]

        nmse_flag  = cp.zeros(M, dtype=cp.bool_)
        se_flag    = cp.zeros(M, dtype=cp.bool_)
        tau_f_nmse = cp.zeros(M, dtype=cp.float64)
        tau_f_se   = cp.zeros(M, dtype=cp.float64)

        _scl = cp.asarray(self.std_nmse)[None, None, :]

        for i in range(1, T):
            u_hat = self.forecast(u_hat)

            # NaN handling: per trajectory
            nan_mask = cp.isnan(u_hat).any(axis=(1,2))   # (M,)
            if nan_mask.any():
                last_idx = i - 1

                upd_nmse = cp.logical_and(~nmse_flag, nan_mask)
                upd_se   = cp.logical_and(~se_flag, nan_mask)
                tau_f_nmse[upd_nmse] = last_idx
                tau_f_se[upd_se]     = last_idx
                nmse_flag = cp.logical_or(nmse_flag, nan_mask)
                se_flag   = cp.logical_or(se_flag, nan_mask)

            diff  = test[:, [i], :] - u_hat
            mdiff = diff / _scl             # (M, 1, d)

            nmse_ = cp.mean(mdiff**2, axis=(1,2))    # (M,)
            se_ = cp.linalg.norm(diff, axis=(1,2))**2 / cp.linalg.norm(test[:, i, :], axis=-1)**2                                           # (M,)

            cond_nmse = cp.logical_and(~nmse_flag, nmse_ > error_threshold)
            tau_f_nmse[cond_nmse] = i - 1
            nmse_flag = cp.logical_or(nmse_flag, cond_nmse)

            cond_se = cp.logical_and(~se_flag, se_ > error_threshold)
            tau_f_se[cond_se] = i - 1
            se_flag = cp.logical_or(se_flag, cond_se)

            if cp.logical_and(nmse_flag, se_flag).all():
                break

        last_step = T
        tau_f_nmse[~nmse_flag] = last_step
        tau_f_se[~se_flag]     = last_step

        tau_f_nmse = tau_f_nmse * dt / Lyapunov_time
        tau_f_se   = tau_f_se * dt / Lyapunov_time


        return cp.asnumpy(tau_f_nmse), cp.asnumpy(tau_f_se)

        
    def get_vpt(self, test, dt, Lyapunov_time, error_threshold=0.3**2,
            return_pred=False, verbose=False, ii=0):
        """
        Forecast times (Lyapunov units) using NMSE and SE.

        Always uses the multi-parameter pipeline internally.
        If epsilon is scalar, it's treated as a length-1 vector (P = 1).
        """
        test = cp.asarray(test)
        # number of hyperparameters (scalar epsilon -> P = 1)
        P = int(np.asarray(self.epsilon).size)
        assert P >= 1

        test_points = test.shape[0]
        d = test.shape[-1]

        # predictions: (T, P, d)
        pred = cp.zeros((test_points, P, d), dtype=test.dtype)

        # initial state (1, d)
        u0 = test[0].reshape(1, -1)
        pred[0] = u0  # broadcast into (P, d) later

        # initial states for each hyperparameter: (P, 1, d)
        u_hat = cp.ones((P, 1, 1)) * u0[None, :, :]  # (P, 1, d)

        nmse_flag  = cp.zeros(P, dtype=cp.bool_)
        se_flag    = cp.zeros(P, dtype=cp.bool_)
        tau_f_nmse = cp.zeros(P, dtype=cp.float64)
        tau_f_se   = cp.zeros(P, dtype=cp.float64)

        _scl = cp.asarray(self.std_nmse).reshape(1, -1)

        if verbose:
            print(self.dm.beta[ii])

        for i in range(1, test_points):
            # forecast for all hyperparams at once
            u_hat = self.forecast(u_hat)      # expected shape (P, 1, d) or (P, d)

            # ensure shape (P, d) no matter what forecast returns
            u_hat_squeezed = u_hat.reshape(P, -1)   # (P, d)
            pred[i] = u_hat_squeezed

            # NaN handling: per hyperparam
            nan_mask = cp.isnan(u_hat_squeezed).any(axis=1)  # (P,)
            if nan_mask.any():
                last_idx = i - 1
                # set tau_f for those that have not yet hit threshold
                upd_nmse = cp.logical_and(~nmse_flag, nan_mask)
                upd_se   = cp.logical_and(~se_flag,   nan_mask)
                tau_f_nmse[upd_nmse] = last_idx
                tau_f_se[upd_se]     = last_idx
                nmse_flag = cp.logical_or(nmse_flag, nan_mask)
                se_flag   = cp.logical_or(se_flag,   nan_mask)

            # error metrics
            # test[i] is (d,), broadcast to (P, d)
            diff  = (test[i] - u_hat_squeezed)       # (P, d)
            mdiff = diff / _scl                      # (P, d)

            nmse_ = cp.mean(mdiff**2, axis=-1)       # (P,)
            se_   = (cp.linalg.norm(diff, axis=1)**2
                    / (cp.linalg.norm(test[i])**2)) # (P,)

            # update nmse-based tau_f
            cond_nmse = cp.logical_and(~nmse_flag, nmse_ > error_threshold)
            tau_f_nmse[cond_nmse] = i - 1
            nmse_flag = cp.logical_or(nmse_flag, cond_nmse)

            # update se-based tau_f
            cond_se = cp.logical_and(~se_flag, se_ > error_threshold)
            tau_f_se[cond_se] = i - 1
            se_flag = cp.logical_or(se_flag, cond_se)

            # if all hyperparams are done and we don't need predictions, break
            if cp.logical_and(nmse_flag, se_flag).all() and not return_pred:
                break

        # any hyperparams that never crossed thresholds get full length
        last_step = test_points
        tau_f_nmse[~nmse_flag] = last_step
        tau_f_se[~se_flag]     = last_step

        factor = dt / Lyapunov_time
        tau_f_nmse = tau_f_nmse * factor
        tau_f_se   = tau_f_se * factor

        # ---- Output: if P == 1, squeeze back to scalar-like API ----
        tau_f_nmse_np = cp.asnumpy(tau_f_nmse)
        tau_f_se_np   = cp.asnumpy(tau_f_se)

        if P == 1:
            if return_pred:
                # pred: (T, P, d) -> (T, d)
                return float(tau_f_nmse_np[0]), float(tau_f_se_np[0]), pred[:, 0, :]
            else:
                return float(tau_f_nmse_np[0]), float(tau_f_se_np[0])
        else:
            if return_pred:
                return tau_f_nmse_np, tau_f_se_np, pred
            else:
                return tau_f_nmse_np, tau_f_se_np

