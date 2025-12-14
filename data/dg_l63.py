# load necessary modules
import numpy as np 
from scipy.integrate import odeint
from matplotlib import pyplot as plt


def L63(u, alpha=10., rho=28., beta=8./3.):

    x, y, z = u 
    p = alpha * (y - x)
    q = (rho - z) * x - y
    r = x * y - beta * z
    return np.array([p, q, r])

def generate_trajectory(state0, dt, n_steps):
    return odeint(lambda x, t: L63(x), state0, np.arange(0, n_steps*dt, dt))

def gen_data(dt=0.01, train_size=int(1e5), test_num=500, test_size=2500):

    np.random.seed(train_seed)
    g0 =  np.random.normal(size=(3))
    state0 = generate_trajectory(g0, dt, int(40/dt))[-1]
    train = generate_trajectory(state0, dt, train_size)
    np.random.seed(test_seed)

    g0 =  np.random.normal(size=(3))
    state0 = generate_trajectory(g0, dt, int(40/dt))[-1]
    test = generate_trajectory(state0, dt, test_num*test_size)
    test = np.moveaxis(test.reshape(test_num, -1, 3), 1, 2)
    np.random.shuffle(test)

    return train, test

train_seed=22; test_seed=43
train, test = gen_data()

ifplt = 0
ifsav = 0


if ifplt:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(*(train[:1000].T))
    fig.savefig("./figs/l63_train.png")

if ifsav:
    np.save('./cached_data/lorenz_train.npy', train)
    np.save('./cached_data/lorenz_test.npy', test)