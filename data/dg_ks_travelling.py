"""
The solver is based on
    https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/fft_and_spectral_methods/ks_solver_etd_in_jax.ipynb
The solution setup is based on
    https://arxiv.org/pdf/2108.05928

Choose appropriate `cas` variables to run the simulations
"""

import jax
jax.config.update("jax_enable_x64", True) 
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import pickle

DOMAIN_SIZE = 2 * jnp.pi
N_DOF = 64

ifplt = 0
ifsav = 1


class KuramotoSivashinsky:
    def __init__(
        self,
        L,
        N,
        dt,
    ):
        self.L = L
        self.N = N
        self.dt = dt
        self.dx = L / N

        wavenumbers = jnp.fft.rfftfreq(N, d=L / (N * 2 * jnp.pi))
        self.derivative_operator = 1j * wavenumbers

        linear_operator = (
            -(self.derivative_operator**2) - NU * self.derivative_operator**4
        )
        self.exp_term = jnp.exp(dt * linear_operator)
        self.coef = jnp.where(
            linear_operator == 0.0,
            dt,
            (self.exp_term - 1.0) / linear_operator,
        )

        self.alias_mask = wavenumbers < 2 / 3 * jnp.max(wavenumbers)

    def __call__(self, u):

        u_hat = jnp.fft.rfft(u)
        u_hat = self.alias_mask * u_hat
        u_filt = jnp.fft.irfft(u_hat, n=self.N)

        u_nonlin = -0.5 * (u_filt ** 2)
        u_nonlin_hat = jnp.fft.rfft(u_nonlin)
        u_nonlin_hat = self.alias_mask * u_nonlin_hat
        u_nonlin_der_hat = self.derivative_operator * u_nonlin_hat

        u_next_hat = self.exp_term * u_hat + self.coef * u_nonlin_der_hat
        u_next = jnp.fft.irfft(u_next_hat, n=self.N)
        return u_next


mesh = jnp.linspace(0.0, DOMAIN_SIZE, N_DOF, endpoint=False)


NU = 4 / 87.0
data = loadmat("./cached_data/ksdataBeatingTraveling.mat")
u_0 = data["udata"][:, 0]
NT = 5000000
SKP = 500000
DT = 0.001
TS = 10

ks_stepper = KuramotoSivashinsky(
    L=DOMAIN_SIZE,
    N=N_DOF,
    dt=DT,
)

ks_stepper = jax.jit(ks_stepper)
u_current = u_0
trj = [u_current]
for i in range(1, NT + SKP):
    u_current = ks_stepper(u_current)
    trj.append(u_current)
trj = jnp.stack(trj)[SKP::TS]

ts = jnp.arange(NT + SKP) * DT
ts = ts[SKP::TS]

if ifplt:
    f = plt.figure(figsize=(20, 5))
    ax = f.add_subplot()
    ax.imshow(
        trj.T,
        cmap="RdBu",
        aspect="auto",
        origin="lower",
        extent=(ts[0], ts[-1], 0, DOMAIN_SIZE),
    )
    f.colorbar()
    ax.set_xlabel("time")
    ax.set_ylabel("space")



if ifsav:
    print(f"udata.shape = {np.array(trj).shape}")
    dat = {
        "x": np.array(mesh),
        "t": np.array(ts - ts[0]),
        "udata": np.array(trj),
        "nu": NU,
        "dt": DT * TS,
    }
    pickle.dump(dat, open(f"./cached_data/ksdata_travelling_NT_{NT}_SKP_{SKP}_dt_{DT}_ts_{TS}.pkl", "wb"))

plt.show()
