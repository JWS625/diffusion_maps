"""
KS beating-travelling example

Neural ODE with FFT-reduced data
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint
torch.set_default_dtype(torch.float64)

# Set fixed random seed for reproducibility
SEED = 42
random.seed(SEED)  # Set Python random seed
np.random.seed(SEED)  # Set NumPy random seed
torch.manual_seed(SEED)  # Set PyTorch random seed

# ----------------------
# Tunable parameters
# ----------------------
# Phase dynamics
phase_hidden_dim = 32
phase_num_epochs = 1000
phase_batch_size = 128
# Spatial dynamics
space_hidden_dim = 128
space_num_epochs = 2500
space_batch_size = 20
time_steps = 20

# ----------------------
# Load data
# ----------------------
NT = 5000000
SKP = 500000
DT = 0.001
TS = 10
data = pickle.load(open(f"/home/jzs6565/dm_code/dm_final/ks_utils/ksdata_traveling_NT_{NT}_SKP_{SKP}_dt_{DT}_ts_{TS}.pkl", "rb"))
dt = data['dt']
nu = data['nu']
xx = data['x']
tt = data['t']
uu = data['udata']

Nx, Nt = len(xx), len(tt)
assert uu.shape == (Nt, Nx)

Ntrain = 100
NOffset = 6000
Ntest  = 14000

t_sim = tt[NOffset:NOffset+Ntest]
t_plt = t_sim - t_sim[0]

# Test data
data_test = uu[NOffset:NOffset+Ntest]

# ----------------------
# Processing by FFT
# From CANDyMan
# ----------------------
Xhat = np.fft.fft(uu[:Ntrain])
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

a0_spc = Xshift[0]
a0_phi = [phi[0]]

data_train = Xshift[:Ntrain]
phi_train = Xphi[:Ntrain-1]

# ----------------------
# Part 1: Phase Dynamics
# A FCNN is used
# ----------------------
# Define the regression neural network
class RegressionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegressionNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.network(x)

scl = 100   # Normalization factor
spc_train = torch.tensor(data_train[:-1], dtype=torch.float64)
phi_train = torch.tensor(phi_train*scl, dtype=torch.float64)

dataset = TensorDataset(spc_train, phi_train)
train_loader = DataLoader(dataset, batch_size=phase_batch_size, shuffle=True)

# Define the model, loss function, and optimizer
input_dim = spc_train.shape[1]
output_dim = phi_train.shape[1]

model = RegressionNN(input_dim, phase_hidden_dim, output_dim)
criterion = nn.MSELoss()  # Mean squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
t1_phase = time.time()
for epoch in range(phase_num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_data, batch_targets in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_data)
        loss = criterion(predictions, batch_targets)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{phase_num_epochs}, Loss: {epoch_loss/len(train_loader):4.3e}")
t2_phase = time.time()

a_phi = model(torch.tensor(Xshift[:-1])).detach().numpy()/scl
phi_pred = np.hstack([phi[0], phi[0]+np.cumsum(a_phi)])

f = plt.figure()
plt.plot(tt[:100], phi_pred, 'b-')
plt.plot(tt[:100], phi_recon, 'k--')
plt.savefig('./training_node.png', bbox_inches='tight', dpi=500)

# ----------------------
# Part 2: Spatial Dynamics
# Neural ODE Model
# ----------------------
def get_batch(data, batch_size, time_steps, device):
    s = torch.from_numpy(np.random.choice(np.arange(len(data) - time_steps, dtype=np.int64), batch_size, replace=False))
    batch_y0 = data[s].to(device)
    batch_t = (torch.arange(time_steps)*dt).to(device)
    batch_y = torch.stack([data[s + i] for i in range(time_steps)], dim=0).to(device)  # (T, M, D)
    return batch_y0, batch_t, batch_y

class ODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1/hidden_dim)
                nn.init.constant_(m.bias, val=0)
    def forward(self, t, x):
        return self.net(x)

# Initialize model, loss, optimizer
input_dim = 64
output_dim = 64
odefunc = ODEFunc(input_dim, space_hidden_dim, output_dim)
optimizer = torch.optim.Adam(odefunc.parameters(), lr=1e-4)
data_train_tensor = torch.tensor(data_train, dtype=torch.float64)

t1_space = time.time()
loss_history = []
for epoch in range(space_num_epochs):
    epoch_loss = 0
    for _ in range(Ntrain // space_batch_size):
        batch_y0, batch_t, batch_y = get_batch(data_train_tensor, space_batch_size, time_steps, torch.device('cpu'))

        # Forward pass using ODE solver
        pred = odeint(odefunc, batch_y0, batch_t, method='rk4')

        # Compute loss (MSE)
        loss = ((pred - batch_y) ** 2).mean()
        epoch_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_history.append(epoch_loss)
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{space_num_epochs}, Loss: {epoch_loss:4.3e}')
t2_space = time.time()

# ----------------------
# Prediction
# ----------------------
ic_tensor = torch.tensor(a0_spc, dtype=torch.float64)
t_tensor = torch.tensor(t_sim, dtype=torch.float64)

t3 = time.time()
with torch.no_grad():
    a_spc = odeint(odefunc, ic_tensor, t_tensor, method='rk4')
a_phi = model(a_spc[:-1]).detach().numpy()/scl
phi_pred = np.hstack([phi[0], phi[0]+np.cumsum(a_phi)])
a_nde = np.real(np.fft.ifft(np.fft.fft(a_spc.numpy()) * np.exp(1j * np.outer(phi_pred, wav))))
t4 = time.time()

print(t2_phase-t1_phase, t2_space-t1_space, t4-t3)

# Save prediction results
pickle.dump([a_nde, phi_pred], open('./res/ks_trf_nde.pkl', 'wb'))

# ------------------
# Visualizations for sanity check
# ------------------
plt.figure(figsize=(10, 6))
plt.semilogy(loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.legend()

a_tru = data_test
X, T = np.meshgrid(xx, t_plt)

f, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))
cs = ax[0].contourf(X, T, a_tru)
ax[1].contourf(X, T, a_nde, levels=cs.levels)
ax[2].contourf(X, T, a_tru-a_nde, levels=cs.levels)
plt.colorbar(cs, ax=ax)

ax[0].set_title('Truth')
ax[1].set_title('Prediction')

ax[0].set_xlabel('$x$')
ax[0].set_ylabel('$t$')
ax[1].set_xlabel('$x$')
ax[2].set_xlabel('$x$')

plt.show()
