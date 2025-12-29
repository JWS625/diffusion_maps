"""
KS beating-travelling example

FFT-reduced MKRR

Detailed result comparison is done in ks_travel.py
"""

import sys
from pathlib import Path
root = Path.cwd().resolve().parents[1]
sys.path.insert(0, str(root))

data_dir = str(root) + "/diffusion_maps/data"


import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import rainbow
import numpy as np
from scipy import linalg
import time
import sys

from sklearn.manifold import Isomap

from gindy.src.manifold import Manifold
from gindy.src.kernelreg import MKRR, M2KRR
from gindy.src.gindy import DynRegMan
from gindy.src.utils import make_se

NT = 5000000
SKP = 500000
DT = 0.001
TS = 10
data = pickle.load(open(data_dir + f"/cached_data/ksdata_traveling_NT_{NT}_SKP_{SKP}_dt_{DT}_ts_{TS}.pkl", "rb"))
dt = data['dt']
nu = data['nu']
xx = data['x']
tt = data['t']
uu = data['udata']

Nx, Nt = len(xx), len(tt)
assert uu.shape == (Nt, Nx)

Ntrain = 100
Noffset = 6000
Ntest  = 14000

t_sim = tt[Noffset:Noffset+Ntest]
t_plt = t_sim - t_sim[0]

# Test data
data_test = [uu[Noffset:Noffset+Ntest]]

# ----------------------
# Processing by FFT
Xhat = np.fft.fft(uu)
phi = np.angle(Xhat[:, 1])
wav = np.concatenate((np.arange(33), np.arange(-31, 0))) # wavenumbers
XhatShift = Xhat*np.exp(-1j*np.outer(phi, wav))
Xshift = np.real(np.fft.ifft(XhatShift))

dphi = phi[1:] - phi[:-1]
dphi += (dphi < -np.pi)*2.0*np.pi - (dphi > np.pi)*2.0*np.pi
Xphi = dphi.reshape(-1,1)

phi_recon = np.hstack([phi[0], phi[0]+np.cumsum(dphi)])
tmp = np.fft.fft(Xshift) * np.exp(1j*np.outer(phi_recon, wav))
data_recon = np.real(np.fft.ifft(tmp))

a0_spc = Xshift[Noffset]
phi0 = np.angle(Xhat[Noffset, 1])
a0_phi = [phi[0]]

data_train = Xshift[:Ntrain]
phi_train = Xphi[:Ntrain-1]
# ----------------------

# Explore
ifvis = 0
ifdim = 0
ifman = 0
ifest = 0

# Training
ifkrr = 1

# Plots
iftim = 0


if ifvis:
    X, T = np.meshgrid(xx, tt)
    f = plt.figure()
    cc = plt.contourf(X, T, Xshift)
    plt.colorbar(cc)

    f = plt.figure()
    plt.plot(Xphi.reshape(-1))

    f, ax = plt.subplots(ncols=3, sharey=True, figsize=(10, 6))
    cs = ax[0].contourf(X, T, uu)
    ax[1].contourf(X, T, data_recon, levels=cs.levels)
    ax[2].contourf(X, T, uu-data_recon, levels=cs.levels)
    plt.colorbar(cs, ax=ax)

if ifdim:
    man = Manifold(Xshift, d=1)
    dim, (f, ax) = man.estimate_intrinsic_dim(bracket=[-40, 5], tol=0.2, ifplt=True)
    _, _, f = man.visualize_intrinsic_dim()

if ifman:
    Nm = 100
    dat = Xshift[:Nm]
    K = int(np.sqrt(Nm))

    isom = Isomap(n_neighbors=K, n_components=2)
    X = isom.fit_transform(dat).T
    f = plt.figure()
    plt.plot(X[0], X[1], 'b.')

if ifest:
    man = Manifold(Xshift, d=1)
    dim = man.estimate_intrinsic_dim(bracket=[-20, 10], tol=0.2, ifplt=False)

# ------------------
# GINDy
# ------------------
if ifkrr:
    t1 = time.time()
    man = Manifold(Xshift[:100], d=1)
    dim, __, eps = man.estimate_intrinsic_dim(bracket=[-20, 10], tol=0.2, ifest=True)
    gamma = 10 * eps
    print(f"gamma = {gamma}")
    manopt = {
        'd' : 1,
        'g' : 4,
        'T' : 0
    }
    regloc = {
        'man' : Manifold,
        'manopt' : manopt,
        'ker' : make_se(gamma),
        'nug' : 1e-6,
        'ifvec' : True
    }
    dloc = DynRegMan(M2KRR, regloc, fd='1', dt=dt)
    dloc.fit([data_train])

    mphi = Manifold(data_train[:-1], **manopt)
    mphi.precompute()
    t2 = time.time()

    a_spc = dloc.solve(a0_spc, t_sim)
    a_phi = np.array([
        mphi.gmls(_a, phi_train) for _a in a_spc[:-1]
    ])

    phi_recon = np.hstack([phi0, phi0+np.cumsum(a_phi)])
    tmp = np.fft.fft(a_spc) * np.exp(1j*np.outer(phi_recon, wav))
    data_recon = np.real(np.fft.ifft(tmp))
    t3 = time.time()

    print(t2-t1, t3-t2)
    pickle.dump([data_recon], open(f'./res/ks_trf_{Ntrain}.dat', 'wb'))

# ------------------
# Temporal response
# ------------------
if iftim:
    IDX = 0
    a_tru = data_test[IDX]
    a_krr = pickle.load(open(f'./res/ks_trf_{Ntrain}.dat', 'rb'))[IDX]

    X, T = np.meshgrid(xx, t_plt)

    f, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))
    cs = ax[0].contourf(X, T, a_tru)
    ax[1].contourf(X, T, a_krr, levels=cs.levels)
    ax[2].contourf(X, T, a_tru-a_krr, levels=cs.levels)
    plt.colorbar(cs, ax=ax)

    ax[0].set_title('Truth')
    ax[1].set_title('Prediction')
    ax[2].set_title(f'Error {np.linalg.norm(a_tru-a_krr):4.3e}')

    ax[0].set_xlabel('$x$')
    ax[0].set_ylabel('$t$')
    ax[1].set_xlabel('$x$')
    ax[2].set_xlabel('$x$')

plt.show()
