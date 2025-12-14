import os
import jax
jax.config.update("jax_enable_x64", True) 
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pickle

ifsav = 1
sol_type = 'train' # 'test'

DOMAIN_SIZE = 22.0
N_DOF = 64
mesh = jnp.linspace(0.0, DOMAIN_SIZE, N_DOF, endpoint=False)

NU = 1
if sol_type == 'train':
    u_0 = jnp.sin(16 * jnp.pi * mesh / DOMAIN_SIZE)
elif sol_type == 'test':
    u_0 = jnp.sin(8 * jnp.pi * mesh / DOMAIN_SIZE)

NT = 12500000
SKP = 500000
DT = 0.01
TS = 10

class KS_ETDRK4:
    def __init__(self, L, N, dt, nu):
        self.L = L; self.N = N; self.dt = dt
        k = jnp.fft.rfftfreq(N, d=L/(N*2*jnp.pi))
        self.ik = 1j * k
        Lhat = -(self.ik**2) - nu*(self.ik**4)
        self.E  = jnp.exp(dt * Lhat)
        self.E2 = jnp.exp(0.5 * dt * Lhat)

        # 2/3-rule mask
        self.mask = k < (2/3) * k.max()

        # Kassam–Trefethen coefficients (M=16 contour avg)
        M = 16
        r = jnp.exp(1j * jnp.pi * (jnp.arange(1, M+1) - 0.5) / M)
        LR = dt * Lhat[:, None] + r[None, :]
        av = lambda X: jnp.real(jnp.mean(X, axis=1))
        self.Q  = dt * av((jnp.exp(LR/2) - 1.0) / LR)
        self.f1 = dt * av((-4 - LR + jnp.exp(LR)*(4 - 3*LR + LR**2)) / LR**3)
        self.f2 = dt * av(( 2 + LR + jnp.exp(LR)*(-2 + LR))              / LR**3)
        self.f3 = dt * av((-4 - 3*LR - LR**2 + jnp.exp(LR)*(4 - LR))     / LR**3)

    def Nhat(self, u):
        # nonlinear term in Fourier
        uhat = jnp.fft.rfft(u); uhat = self.mask * uhat
        uf   = jnp.fft.irfft(uhat, n=self.N)
        nl   = -0.5 * uf**2
        nlhat = jnp.fft.rfft(nl); nlhat = self.mask * nlhat
        return self.ik * nlhat

    def step(self, u):
        uhat = self.mask * jnp.fft.rfft(u)

        Nv  = self.Nhat(jnp.fft.irfft(uhat, n=self.N))
        a   = self.E2 * uhat + self.Q * Nv
        Na  = self.Nhat(jnp.fft.irfft(a, n=self.N))
        b   = self.E2 * uhat + self.Q * Na
        Nb  = self.Nhat(jnp.fft.irfft(b, n=self.N))
        c   = self.E2 * a    + self.Q * (2*Nb - Nv)
        Nc  = self.Nhat(jnp.fft.irfft(c, n=self.N))

        uhat_next = (self.E * uhat
                     + self.f1 * Nv
                     + 2*self.f2 * (Na + Nb)
                     + self.f3 * Nc)
        return jnp.fft.irfft(self.mask * uhat_next, n=self.N)


ks = KS_ETDRK4(L=DOMAIN_SIZE, N=N_DOF, dt=DT, nu=NU)
ks_step = jax.jit(ks.step)
u_current = u_0
trj = [u_current]
for _ in range(1, NT + SKP):
    u_current = ks_step(u_current)
    trj.append(u_current)
trj = jnp.stack(trj)[SKP::TS]

ts = jnp.arange(NT + SKP) * DT
ts = ts[SKP::TS]

if ifsav:
    print(f"udata.shape = {np.array(trj).shape}")
    dat = {
        "x": np.array(mesh),
        "t": np.array(ts - ts[0]),
        "udata": np.array(trj),
        "nu": NU,
        "dt": DT * TS,
    }
    pickle.dump(dat, open(f"./../../../ks_utils/ksdata_chaotic_test_NT_{NT}_SKP_{SKP}_dt_{DT}_ts_{TS}.pkl", "wb"))