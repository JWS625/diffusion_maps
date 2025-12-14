import numpy as np
from scipy.signal import argrelmin, argrelmax, detrend
from numpy.typing import NDArray


def changetotorus_(xinner):
    padded = False
    if len(xinner.shape) == 2:
        xinner = xinner[None, ...]
        padded = True
    xin = np.copy(xinner)
    # xinner is fed in with shape
    # (path count, time steps, 3)
    xin = xin.transpose(1, 2, 0)

    # xin should have shape
    # (time steps, 3, path count)
    Ntr = xin.shape[2]

    theta_x = np.zeros((xin.shape[0], Ntr))
    phi_x = np.zeros((xin.shape[0], Ntr))

    for i in range(Ntr):
        theta_x[:, i] = np.arccos(xin[:, 2, i])
        phi_x[:, i] = np.sign(xin[:, 1, i]) * np.arccos(
            xin[:, 0, i] / np.sqrt(xin[:, 0, i] ** 2 + xin[:, 1, i] ** 2)
        )
        if np.isnan(phi_x[:, i].max()):
            phi_x[:, i] = 0

    phi_x2 = np.copy(phi_x)

    for j in range(Ntr):
        In = np.where(np.abs(np.diff(phi_x[:, j])) > 2.5)[0]
        for kk in In:
            if phi_x2[kk, j] < phi_x2[kk + 1, j]:
                phi_x2[kk + 1 :, j] -= 2 * np.pi
            else:
                phi_x2[kk + 1 :, j] += 2 * np.pi

    for j in range(Ntr):
        if len(np.where(np.abs(np.diff(phi_x2[:, j])) > 2.5)[0]) > 0:
            if phi_x2[1, j] > 0:
                phi_x2[0, j] = np.pi
            else:
                phi_x2[0, j] = -np.pi

    theta_x2 = 2 * theta_x

    xout = np.zeros(xin.shape)
    xout[:, 0, :] = (np.cos(theta_x2) + 2) * np.cos(phi_x2)
    xout[:, 1, :] = (np.cos(theta_x2) + 2) * np.sin(phi_x2)
    xout[:, 2, :] = np.sin(theta_x2)

    # xout has shape (time steps, 3, path count)
    # and must return(path count, time steps, 3)
    xout = xout.transpose(2, 0, 1)
    if padded:
        assert xout.shape[0] == 1
        assert theta_x2.shape[-1] == 1
        assert phi_x2.shape[-1] == 1
        return xout[0], theta_x2[:, 0], phi_x2[:, 0]
    return xout, theta_x2, phi_x2


def changetotorus(xinner_lst):
    xout_lst = []
    theta_lst = []
    phi_lst = []
    for xinner in xinner_lst:
        padded = False
        if len(xinner.shape) == 2:
            xinner = xinner[None, ...]
            padded = True
        xin = np.copy(xinner)
        # xinner is fed in with shape
        # (path count, time steps, 3)
        xin = xin.transpose(1, 2, 0)

        # xin should have shape
        # (time steps, 3, path count)
        Ntr = xin.shape[2]

        theta_x = np.zeros((xin.shape[0], Ntr))
        phi_x = np.zeros((xin.shape[0], Ntr))

        for i in range(Ntr):
            theta_x[:, i] = np.arccos(xin[:, 2, i])
            phi_x[:, i] = np.sign(xin[:, 1, i]) * np.arccos(
                xin[:, 0, i] / np.sqrt(xin[:, 0, i] ** 2 + xin[:, 1, i] ** 2)
            )
            if np.isnan(phi_x[:, i].max()):
                phi_x[:, i] = 0

        phi_x2 = np.copy(phi_x)

        for j in range(Ntr):
            In = np.where(np.abs(np.diff(phi_x[:, j])) > 2.5)[0]
            for kk in In:
                if phi_x2[kk, j] < phi_x2[kk + 1, j]:
                    phi_x2[kk + 1 :, j] -= 2 * np.pi
                else:
                    phi_x2[kk + 1 :, j] += 2 * np.pi

        for j in range(Ntr):
            if len(np.where(np.abs(np.diff(phi_x2[:, j])) > 2.5)[0]) > 0:
                if phi_x2[1, j] > 0:
                    phi_x2[0, j] = np.pi
                else:
                    phi_x2[0, j] = -np.pi

        theta_x2 = 2 * theta_x

        xout = np.zeros(xin.shape)
        xout[:, 0, :] = (np.cos(theta_x2) + 2) * np.cos(phi_x2)
        xout[:, 1, :] = (np.cos(theta_x2) + 2) * np.sin(phi_x2)
        xout[:, 2, :] = np.sin(theta_x2)

        # xout has shape (time steps, 3, path count)
        # and must return(path count, time steps, 3)
        xout = xout.transpose(2, 0, 1)
        if padded:
            assert xout.shape[0] == 1
            assert theta_x2.shape[-1] == 1
            assert phi_x2.shape[-1] == 1
            return xout[0], theta_x2[:, 0], phi_x2[:, 0]

        theta_lst.append(theta_x2)
        phi_lst.append(phi_x2)
    return xout, theta_lst, phi_lst


def computePeriodInner(signal: np.ndarray) -> int:
    if signal.shape[0] < 2:
        return 0
    if signal[0] == signal[-1]:
        return 0
    signal = np.array(signal)
    indices = argrelmin(signal)
    if len(indices[0]) < 2:
        indices = argrelmax(signal)
        if len(indices[0]) < 2:
            print("No minima or maxima found")
            return -1
    period = indices[0][1] - indices[0][0]
    return period


def computePeriod(path: np.ndarray) -> int:
    detrended = np.apply_along_axis(detrend, 0, path)
    periodArray = np.apply_along_axis(computePeriodInner, 0, detrended)
    periodArray = np.max(periodArray, axis=-1)
    period = np.max(periodArray)
    return period


def generateInitialConditions_(paths, random=False):
    if random:
        x0 = np.random.randn(paths, 3)
        index = x0[:, -1] < 0
        while np.any(index):
            x0[index] = np.random.randn(np.count_nonzero(index), 3)
            index = x0[:, -1] < 0
        x0 = x0 / np.linalg.norm(x0, axis=-1)[:, None]
        x0: NDArray[np.float64] = x0
        return x0
    phi = np.pi / 2
    arc1Length = paths // 2
    arc3Length = arc1Length // 2
    arc2Length = paths - arc1Length - arc3Length
    theta = np.linspace(0, np.pi / 2, arc1Length + 1, endpoint=False)[1:]

    x0 = np.zeros((paths, 3))
    x0[:arc1Length, 0] = np.sin(theta) * np.cos(phi)
    x0[:arc1Length, 1] = np.sin(theta) * np.sin(phi)
    x0[:arc1Length, 2] = np.cos(theta)

    theta = np.pi / 2
    phi = np.linspace(0, np.pi / 2, arc2Length + 1, endpoint=False)[1:]
    x0[arc1Length : arc1Length + arc2Length, 0] = np.sin(theta) * np.cos(phi)
    x0[arc1Length : arc1Length + arc2Length, 1] = np.sin(theta) * np.sin(phi)
    x0[arc1Length : arc1Length + arc2Length, 2] = np.cos(theta)

    theta = np.pi / 2
    phi = np.linspace(np.pi / 2, np.pi, arc3Length + 1, endpoint=False)[1:]
    x0[arc1Length + arc2Length :, 0] = np.sin(theta) * np.cos(phi)
    x0[arc1Length + arc2Length :, 1] = np.sin(theta) * np.sin(phi)
    x0[arc1Length + arc2Length :, 2] = np.cos(theta)

    return x0


def generateInitialConditions(paths, random=False):
    if random:
        x0 = np.random.randn(paths, 3)
        index = x0[:, -1] < 0
        while np.any(index):
            x0[index] = np.random.randn(np.count_nonzero(index), 3)
            index = x0[:, -1] < 0
        x0 = x0 / np.linalg.norm(x0, axis=-1)[:, None]
        x0: NDArray[np.float64] = x0
        return x0
    phi = np.pi / 2
    arc1Length = paths // 2
    arc3Length = arc1Length // 2
    arc2Length = paths - arc1Length - arc3Length
    theta = np.linspace(0, np.pi / 2, arc1Length + 1, endpoint=False)[1:]

    x0 = [] #np.zeros((paths, 3))
    x01 = np.zeros((arc1Length, 3))
    x02 = np.zeros((arc2Length, 3))
    x03 = np.zeros((arc3Length, 3))

    x01[:, 0] = np.sin(theta) * np.cos(phi)
    x01[:, 1] = np.sin(theta) * np.sin(phi)
    x01[:, 2] = np.cos(theta)
    x0.append(x01)

    theta = np.pi / 2
    phi = np.linspace(0, np.pi / 2, arc2Length + 1, endpoint=False)[1:]
    x02[:, 0] = np.sin(theta) * np.cos(phi)
    x02[:, 1] = np.sin(theta) * np.sin(phi)
    x02[:, 2] = np.cos(theta)
    x0.append(x02)

    theta = np.pi / 2
    phi = np.linspace(np.pi / 2, np.pi, arc3Length + 1, endpoint=False)[1:]
    x03[:, 0] = np.sin(theta) * np.cos(phi)
    x03[:, 1] = np.sin(theta) * np.sin(phi)
    x03[:, 2] = np.cos(theta)
    x0.append(x03)

    return x0

def rk4(x0, dt, f):
    k1 = f(x0)
    k2 = f(x0 + dt / 2 * k1)
    k3 = f(x0 + dt / 2 * k2)
    k4 = f(x0 + dt * k3)
    dx = (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x0 + dx

def generateData_(x, top, dt):
    x0 = np.copy(x)
    # ratio = int(DT / dt)
    # print(f"x0.shape[0] = {x0.shape[0]}")
    paths = x0.shape[0]
    steps = int(top / dt)
    xOutput = np.zeros((paths, steps + 1, 3))
    xOutput[:, 0, :] = np.copy(x0)
    for i in range(1, steps + 1):
        x0 = rk4(x0, dt, f)
        x0 /= np.linalg.norm(x0, axis=-1)[:, None]
        xOutput[:, i, :] = np.copy(x0)
    return xOutput

def generateData(x_lst, top, dt):
    out = []
    for x in x_lst:
        # print(f'x.shape = {x.shape}')
        x0 = np.copy(x)

        # ratio = int(DT / dt)
        paths = x0.shape[0]
        steps = int(top / dt)
        xOutput = np.zeros((paths, steps + 1, 3))
        xOutput[:, 0, :] = np.copy(x0)
        for i in range(1, steps + 1):
            x0 = rk4(x0, dt, f)
            x0 /= np.linalg.norm(x0, axis=-1)[:, None]
            xOutput[:, i, :] = np.copy(x0)
        out.append(xOutput)
    return out


def f(x0):
    I1 = 1.6
    I2 = 1
    I3 = 2 / 3

    alpha1 = 1 / I3 - 1 / I2
    alpha2 = 1 / I1 - 1 / I3
    alpha3 = 1 / I2 - 1 / I1

    assert x0.shape[-1] == 3
    x1, x2, x3 = x0.T
    dx1 = alpha1 * x2 * x3
    dx2 = alpha2 * x1 * x3
    dx3 = alpha3 * x1 * x2
    return np.array([dx1, dx2, dx3]).T


def derivativeApproxSix(xArray: np.ndarray, h):
    return (
        -xArray[:, :-6, :]
        + 9 * xArray[:, 1:-5, :]
        - 45 * xArray[:, 2:-4, :]
        + 45 * xArray[:, 4:-2, :]
        - 9 * xArray[:, 5:-1, :]
        + xArray[:, 6:, :]
    ) / (60 * h)


def derivativeApproxSixForward(xArray: np.ndarray, h):
    # Compute the forward difference approximation for all terms
    forward = (
        (-1 / 6) * xArray[:, 6:, :]
        + (6 / 5) * xArray[:, 5:-1, :]
        + (-15 / 4) * xArray[:, 4:-2, :]
        + (20 / 3) * xArray[:, 3:-3, :]
        + (-15 / 2) * xArray[:, 2:-4, :]
        + 6 * xArray[:, 1:-5, :]
        + (-49 / 20) * xArray[:, :-6, :]
    ) / (h)

    return forward


def derivativeApproxFirstOrderForward(xArray, h):
    return (xArray[:, 1:, :] - xArray[:, :-1, :]) / h


def computeFirstRepeatIndexArray_(path: np.ndarray) -> np.ndarray:
    path = np.array(path)
    periodArray = np.zeros(path.shape[0], dtype=int)
    for i in range(path.shape[0]):
        periodArray[i] = computePeriod(path[i])
    return periodArray

def computeFirstRepeatIndexArray(path_lst):
    period_lst = []
    for path in path_lst:
        path = np.array(path)
        periodArray = np.zeros(path.shape[0], dtype=int)
        for i in range(path.shape[0]):
            periodArray[i] = computePeriod(path[i])
        period_lst.append(periodArray)
    return period_lst


def truncatePath_(path: np.ndarray, periodArray: np.ndarray) -> list[np.ndarray]:
    path = np.array(path)
    assert len(path.shape) > 2
    pathOutput = []
    for i in range(path.shape[0]):
        pathOutput.append(path[i, : periodArray[i], :])
    return pathOutput

def truncatePath(path_lst, periodArray_lst):
    path_out = []
    for path, periodArray in zip(path_lst, periodArray_lst):
        path = np.array(path)
        assert len(path.shape) > 2
        pathOutput = []
        for i in range(path.shape[0]):
            pathOutput.append(path[i, : periodArray[i], :])
        path_out.append(pathOutput)
    return path_out
