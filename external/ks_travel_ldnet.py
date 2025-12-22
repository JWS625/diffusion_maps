"""
KS beating-travelling example

LDNet with FFT-reduced data
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
tf.keras.backend.set_floatx('float64')

import utils
import optimization

# ----------------------
# Tunable parameters
# ----------------------
# Phase dynamics
phase_hidden_dim = 32
phase_num_epochs = 1500
phase_batch_size = 128
# Spatial dynamics
num_latent_states = 2
num_hidden_nodes = 8
num_epochs_Adam = 200
num_epochs_BFGS = 6000

# ----------------------
# Load data
# ----------------------
NT = 5000000
SKP = 500000
DT = 0.001
TS = 10
data = pickle.load(open(f"./../../../ks_utils/ksdata_traveling_NT_{NT}_SKP_{SKP}_dt_{DT}_ts_{TS}.pkl", "rb"))
dt = data['dt']
nu = data['nu']
xx = data['x']
tt = data['t']
uu = data['udata']
dt = tt[1]-tt[0]

Nx, Nt = len(xx), len(tt)
assert uu.shape == (Nt, Nx)

Ntrain = 100
Noffset = 6000
Ntest  = 14000

t_sim = tt[Noffset:Noffset+Ntest]
t_plt = t_sim - t_sim[0]

# Test data
data_test = uu[Noffset:Noffset+Ntest]

# ----------------------
# Processing by FFT
# From CANDyMan
# ----------------------
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
# Part 1: Phase Dynamics
# A FCNN is used
# ----------------------
# Define the regression neural network
class RegressionNN(Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegressionNN, self).__init__()
        self.dense1 = Dense(hidden_dim, activation='tanh')
        self.dense2 = Dense(output_dim, activation=None)
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

scl = 100   # Normalization factor
spc_train = tf.convert_to_tensor(data_train[:-1], np.float64)
phi_train = tf.convert_to_tensor(phi_train*scl, np.float64)

# Prepare the dataset for TensorFlow
train_dataset = tf.data.Dataset.from_tensor_slices((spc_train, phi_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(phase_batch_size)

# Define model, loss, and optimizer
input_dim = spc_train.shape[1]
output_dim = phi_train.shape[1]

model = RegressionNN(input_dim, phase_hidden_dim, output_dim)
criterion = tf.keras.losses.MeanSquaredError()
optimizer = Adam(learning_rate=1e-3)

# Training loop
t1_phase = time.time()
for epoch in range(phase_num_epochs):
    epoch_loss = 0.0
    for batch_data, batch_targets in train_dataset:
        with tf.GradientTape() as tape:
            predictions = model(batch_data)
            loss = criterion(batch_targets, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        epoch_loss += loss.numpy()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{phase_num_epochs}, Loss: {epoch_loss / len(train_dataset):4.3e}")
t2_phase = time.time()

a_phi = model(Xshift[:-1]).numpy()/scl
phi_pred = np.hstack([phi[0], phi[0]+np.cumsum(a_phi)])

f = plt.figure()
plt.plot(phi_pred, 'b-')
plt.plot(phi_recon, 'k--')

# ----------------------
# Part 2: Spatial Dynamics
# Neural ODE Model
# ----------------------
dataset_trn = {
    't' : tt[:Ntrain],
    'x' : xx,
    'output' : np.array([data_train])
}
dataset_tst = {
    't' : t_sim,
    'x' : xx,
    'output' : np.array([data_test])
}

# ----------------------------
# Settings adapted from TestCase_1a of the original paper
# ----------------------------
problem = {
    'space': {
        'dimension' : 1 # 1D problem
    },
    'input_parameters': [],
    'input_signals': [],
    'output_fields': [
        { 'name': 'z' }
    ]
}

normalization = {
    'space': { 'min' : [0], 'max' : [+2*np.pi]},
    'time': { 'time_constant' : 1.0 },
    'output_fields': {
        'z': { 'min': -12, 'max': +15 }
    }
}

samples_train = [0]
samples_valid = [0]
samples_tests = [0]
dataset_train = utils.ADR_create_dataset(dataset_trn, samples_train)
dataset_valid = utils.ADR_create_dataset(dataset_trn, samples_valid)
dataset_tests = utils.ADR_create_dataset(dataset_tst, samples_tests) 
# ----------------------------

# ----------------------------
# The rest is the same as TestCase_1a, except that the model architecture is simplified, and we added the timing
# ----------------------------
# We re-sample the time transients with timestep dt and we rescale each variable between -1 and 1.
utils.process_dataset(dataset_train, problem, normalization, dt = dt)
utils.process_dataset(dataset_valid, problem, normalization, dt = dt)
utils.process_dataset(dataset_tests, problem, normalization, dt = None)

# For reproducibility (delete if you want to test other random initializations)
np.random.seed(0)
tf.random.set_seed(0)

# dynamics network
input_shape = (num_latent_states + len(problem['input_parameters']) + len(problem['input_signals']),)
NNdyn = tf.keras.Sequential([
            tf.keras.layers.Dense(num_hidden_nodes, activation = tf.nn.tanh, input_shape = input_shape),
            tf.keras.layers.Dense(num_hidden_nodes, activation = tf.nn.tanh),
            tf.keras.layers.Dense(num_latent_states)
        ])

# summary
NNdyn.summary()

# reconstruction network
input_shape = (None, None, num_latent_states + problem['space']['dimension'])
NNrec = tf.keras.Sequential([
            tf.keras.layers.Dense(num_hidden_nodes, activation = tf.nn.tanh, input_shape = input_shape),
            tf.keras.layers.Dense(num_hidden_nodes, activation = tf.nn.tanh),
            tf.keras.layers.Dense(len(problem['output_fields']))
        ])

# summary
NNrec.summary()

def evolve_dynamics(dataset):
    # intial condition
    state = tf.zeros((dataset['num_samples'], num_latent_states), dtype=tf.float64)
    state_history = tf.TensorArray(tf.float64, size = dataset['num_times'])
    state_history = state_history.write(0, state)
    dt_ref = normalization['time']['time_constant']
    
    # time integration
    for i in tf.range(dataset['num_times'] - 1):
        state = state + dt/dt_ref * NNdyn(state)
        state_history = state_history.write(i + 1, state)

    return tf.transpose(state_history.stack(), perm=(1,0,2))

def reconstruct_output(dataset, states):    
    states_expanded = tf.broadcast_to(tf.expand_dims(states, axis = 2), 
        [dataset['num_samples'], dataset['num_times'], dataset['num_points'], num_latent_states])
    return NNrec(tf.concat([states_expanded, dataset['points_full']], axis = 3))

def LDNet(dataset):
    states = evolve_dynamics(dataset)
    return reconstruct_output(dataset, states)

def MSE(dataset):
    out_fields = LDNet(dataset)
    error = out_fields - dataset['out_fields']
    return tf.reduce_mean(tf.square(error))

def loss(): return MSE(dataset_train)
def MSE_valid(): return MSE(dataset_valid)

trainable_variables = NNdyn.variables + NNrec.variables
opt = optimization.OptimizationProblem(trainable_variables, loss, MSE_valid)

t1_space = time.time()
print('training (Adam)...')
opt.optimize_keras(num_epochs_Adam, tf.keras.optimizers.Adam(learning_rate=1e-2))
print('training (BFGS)...')
opt.optimize_BFGS(num_epochs_BFGS)
t2_space = time.time()

# Compute predictions.
t3 = time.time()
out_fields = LDNet(dataset_tests)
out_fields_app = utils.denormalize_output(out_fields, problem, normalization).numpy().squeeze()
a_phi = model(out_fields_app[:-1]).numpy()/scl
phi_pred = np.hstack([phi0, phi0+np.cumsum(a_phi)])
a_ldn = np.real(np.fft.ifft(np.fft.fft(out_fields_app) * np.exp(1j * np.outer(phi_pred, wav))))
t4 = time.time()

print(t2_phase-t1_phase, t2_space-t1_space, t4-t3)

# Save prediction results
pickle.dump([a_ldn, phi_pred], open('./ldnet_res/ks_trf_ldn.pkl', 'wb'))

# ------------------
# Visualizations for sanity check
# ------------------
f = plt.figure()
plt.loglog(opt.iterations_history, opt.loss_train_history, 'o-', label = 'training loss')
plt.loglog(opt.iterations_history, opt.loss_valid_history, 'o-', label = 'validation loss')
plt.axvline(num_epochs_Adam)
plt.xlabel('epochs'), plt.ylabel('MSE')
plt.legend()

a_tru = data_test
X, T = np.meshgrid(xx, t_plt)

f, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 6))
cs = ax[0].contourf(X, T, a_tru)
ax[1].contourf(X, T, a_ldn, levels=cs.levels)
ax[2].contourf(X, T, a_tru-a_ldn, levels=cs.levels)
plt.colorbar(cs, ax=ax)

ax[0].set_title('Truth')
ax[1].set_title('Prediction')
ax[2].set_title(f'Error {np.linalg.norm(a_tru-a_ldn):4.3e}')

ax[0].set_xlabel('$x$')
ax[0].set_ylabel('$t$')
ax[1].set_xlabel('$x$')
ax[2].set_xlabel('$x$')

# plt.show()
