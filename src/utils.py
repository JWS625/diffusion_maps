'''
@package manifold.py

Manager of data on manifold.

@author Dr. Daning Huang
@date 06/15/2024
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize_scalar
import scipy.spatial as sps
from sklearn.preprocessing import PolynomialFeatures

class Manifold:
    def __init__(self, data, K=None, d=None, g=None, T=None, iforit=False, extT=None, verbose=True):
        self._data = np.array(data)
        self._Ndat, self._Ndim = self._data.shape
        self.verbose = verbose

        # Number of kNN points in local PCA
        tmp = int(np.sqrt(self._Ndat))
        self._Nknn = tmp if K is None else K

        # KD tree for kNN
        _leaf = max(20, self._Nknn)
        self._tree = sps.KDTree(self._data, leafsize=_leaf)

        # Intrinsic dimension
        self._Nman = d  # If None it will be estimated later

        # Order of GMLS
        self._Nlsq = 2 if g is None else g
        self._fphi = PolynomialFeatures(self._Nlsq, include_bias=False)
        self._fpsi = PolynomialFeatures(self._Nlsq, include_bias=True) # General GMLS

        # Order of tangent space estimation
        self._Ntan = 0 if T is None else T
        if self._Ntan > 0:
            self._ftau = PolynomialFeatures(self._Ntan, include_bias=False)
            self._estimate_tangent = self._estimate_tangent_ho
        else:
            self._estimate_tangent = self._estimate_tangent_1

        # Orientation of tangent vectors
        self._iforit = iforit

        if extT is None:
            self._ifprecomp = False  # Not precomputed yet
        else:
            self._ifprecomp = True
            self._T = np.array(extT)

        if self.verbose:
            print(
                f"Manifold info:\n" \
                f"    No. of data points: {self._Ndat}\n" \
                f"    No. of ambient dim: {self._Ndim}")

    def estimate_intrinsic_dim(self, mode='rbf', bracket=[-20, 5], tol=0.2, ifplt=False, ifest=False, ifeps=False):
        """
        Based on
            https://doi.org/10.1016/j.acha.2015.01.001
        See Fig. 6
        In implementation, we analytically evaluate
            d(log(S(e)))/d(log(e))
        """
        # Normalized squared distances
        dst = sps.distance.pdist(self._data)**2
        dmx = np.max(dst)
        dst /= dmx
        # Tuning functions
        def S(e):
            tmp = np.sum(np.exp(- dst / e))*2 + self._Ndat
            return tmp / self._Ndat**2
        def dS(e):
            _k = np.exp(- dst / e)
            _S = 2 * np.sum(_k) + self._Ndat
            _d = 2 * dst.dot(_k)
            return _d/_S/e
        # Find the intrinsic dimension as the max of slope
        func = lambda e: -2*dS(2.**e)
        res = minimize_scalar(func, bracket=bracket)
        est = -func(res.x)
        # For fractional dimensions, we do a biased rounding
        if np.ceil(est)-est <= tol:
            dim = int(np.ceil(est))
        else:
            dim = int(np.floor(est))
        
        bmp = (2**res.x * dmx)
        tmp = bmp**(1/est)

        if self.verbose:
            print(f"    Estimated intrinsic dim: {dim}/{est}")
            print(f"2**res.x * dmx = {2**res.x * dmx}")
            print(f"    Reference bandwidth {tmp}; ref L2 dist {dmx}; ref scalar {2**res.x}")

        if ifest:
            sol = (dim, est, bmp, tmp)
        else:
            sol = dim

        # Plotting for sanity check
        if ifplt:
            if isinstance(ifplt, bool):
                eps = 2.**np.arange(*bracket)
            else:
                eps = 2.**np.linspace(bracket[0], bracket[1], ifplt)
            val = [S(_e) for _e in eps]
            slp = [2*dS(_e) for _e in eps]

            f, ax = plt.subplots(nrows=2, sharex=True)
            ax[0].loglog(eps, val)
            ax[1].semilogx(eps, slp)
            ax[1].semilogx(2**res.x, est, 'bo', markerfacecolor='none', \
                label=f"Estimated dim: {est:4.3f}")
            ax[0].set_ylabel('$S(\epsilon)$')
            ax[1].set_xlabel('$\epsilon$')
            ax[1].set_ylabel('$d$')
            ax[1].legend()

            if ifeps:
                return sol, tmp, (f, ax)
            else:
                return sol, (f, ax)

        if ifeps:
            return sol, 2**res.x
        else:
            return sol

    def visualize_intrinsic_dim(self, K=None, ifref=True, ifnrm=True):
        _K = self._Nknn if K is None else K
        # Local PCA
        svs = []
        for _x in self._data:
            _, _i = self._tree.query(_x, k=_K)
            _V = self._data[_i] - _x
            _, _s, _ = np.linalg.svd(_V, full_matrices=False)
            svs.append(_s)
        _avr = np.mean(svs, axis=0)
        _std = np.std(svs, axis=0)
        # Global PCA, as reference
        _tmp = self._data - np.mean(self._data, axis=0)
        _, sv, _ = np.linalg.svd(_tmp, full_matrices=False)

        scl = np.max(_avr) if ifnrm else 1.0
        ds = np.arange(len(_avr))+1
        f = plt.figure()
        plt.plot(ds, _avr/scl)
        plt.fill_between(ds, (_avr+_std)/scl, (_avr-_std)/scl, alpha=0.4)
        if ifref:
            scl = np.max(sv) if ifnrm else 1.0
            plt.plot(np.arange(len(sv))+1, sv/scl, 'r--')
        plt.xlabel("SV Index")
        if ifnrm:
            plt.ylabel("Normalized SV")
        else:
            plt.ylabel("SV")

        return _avr, _std, f

    def precompute(self):
        print("  Precomputing")
        if self._ifprecomp:
            assert self._T.shape == (self._Ndat, self._Nman, self._Ndim)
            print("  Already done, or T supplied externally; skipping")
            return

        if self._Nman is None:
            self._Nman = self.estimate_intrinsic_dim()
        else:
            print(f"    Using intrinsic dimension: {self._Nman}")
        self._T = []
        for _d in self._data:
            self._T.append(self._estimate_tangent(_d))
        if self._iforit:
            print("  Orienting tangent vectors")
            if self._Nman == 1:
                rems = list(range(1,self._Ndat))
                curr = 0
                while len(rems) > 0:
                    _, _i = self._tree.query(self._data[curr], k=self._Nknn)
                    _T = self._T[curr]
                    for _j in _i:
                        if _j in rems:
                            _d = self._T[_j].dot(_T.T)
                            if _d < 0:
                                self._T[_j] = -self._T[_j]
                            rems.remove(_j)
                            curr = _j
            else:
                raise NotImplementedError("Orientation for higher-dim manifold not implemented; \
                    supply oriented T externally")
        self._ifprecomp = True
        print("  Done")

    def gmls(self, x, Y):
        _, _i = self._tree.query(x, k=self._Nknn)
        _T, _V = self._estimate_tangent(x, ret_V=True)
        _B = _V.dot(_T.T)
        _P = self._fpsi.fit_transform(_B)
        _C = np.linalg.pinv(_P).dot(Y[_i])
        _r = self._fpsi.fit_transform(np.zeros((1,self._Nman))).dot(_C)
        return _r

    def _estimate_normal(self, base, x):
        _T, _V = self._estimate_tangent(base, ret_V=True)
        _B = _V.dot(_T.T)
        _P = self._fphi.fit_transform(_B)
        _b = np.atleast_2d((x - base).dot(_T.T))
        _p = self._fphi.fit_transform(_b)
        _n = _p.dot(np.linalg.pinv(_P)).dot(_V - _B.dot(_T))
        return _n.squeeze()

    def _estimate_tangent_1(self, x, ret_V=False):
        _d, _i = self._tree.query(x, k=self._Nknn)
        _V = self._data[_i] - x
        _, _, _Vh = np.linalg.svd(_V, full_matrices=False)
        _T = _Vh.conj()[:self._Nman]
        if ret_V:
            return _T, _V
        return _T

    def _estimate_tangent_ho(self, x, ret_V=False):
        _T, _V = self._estimate_tangent_1(x, ret_V=True)
        _B = _V.dot(_T.T)
        _P = self._ftau.fit_transform(_B)
        _C = np.linalg.pinv(_P).dot(_V - _B.dot(_T))
        _T += _C[:self._Nman]
        _T = np.linalg.qr(_T.T, mode='reduced')[0].T
        if ret_V:
            return _T, _V
        return _T

    def plot2d(self, N, scl=1):
        assert self._Ndim == 2
        _d = self._data

        f, ax = plt.subplots(nrows=1, ncols=1)
        plt.plot(_d[:,0], _d[:,1], 'b.', markersize=1)
        for _i in range(N):
            _p = _d[_i] + scl*self._T[_i]
            _c = np.vstack([_d[_i], _p]).T
            plt.plot(_c[0], _c[1], 'k-')
        return f, ax

    def plot3d(self, N, scl=1):
        assert self._Ndim == 3
        _d = self._data

        f = plt.figure()
        ax = f.add_subplot(projection='3d')
        ax.plot(_d[:,0], _d[:,1], _d[:,2], 'b.', markersize=1)
        for _i in range(N):
            _p = _d[_i] + scl*self._T[_i]
            for _j in range(2):
                _c = np.vstack([_d[_i], _p[_j]]).T
                ax.plot(_c[0], _c[1], _c[2], 'k-')
        return f, ax

class ManifoldAnalytical(Manifold):
    def __init__(self, data, K=None, d=None, g=None, fT=None):
        self._data = np.array(data)
        self._Ndat, self._Ndim = self._data.shape

        # Number of kNN points in local PCA
        tmp = int(np.sqrt(self._Ndat))
        self._Nknn = tmp if K is None else K

        # KD tree for kNN
        _leaf = max(20, self._Nknn)
        self._tree = sps.KDTree(self._data, leafsize=_leaf)

        # Intrinsic dimension
        assert d is not None
        self._Nman = d

        # Order of GMLS
        self._Nlsq = 2 if g is None else g
        self._fphi = PolynomialFeatures(self._Nlsq, include_bias=False)
        self._fpsi = PolynomialFeatures(self._Nlsq, include_bias=True) # General GMLS

        # Order of tangent space estimation
        self._tangent_func = fT

        # Possible data members
        self._T = []   # Tangent space basis for every data point
        self._ifprecomp = False  # Not precomputed yet

        print(
            f"Manifold info:\n" \
            f"    No. of data points: {self._Ndat}\n" \
            f"    No. of ambient dim: {self._Ndim}")

    def precompute(self):
        print("  Precomputing")
        self._T = []
        for _d in self._data:
            self._T.append(self._estimate_tangent(_d))
        self._ifprecomp = True
        print("  Done")

    def _estimate_tangent(self, x, ret_V=False):
        _T = self._tangent_func(x)
        if ret_V:
            _, _i = self._tree.query(x, k=self._Nknn)
            _V = self._data[_i] - x
            return _T, _V
        return _T
