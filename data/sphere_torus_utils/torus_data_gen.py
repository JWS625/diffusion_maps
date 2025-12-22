import os
import numpy as np
from . import sphere_torus_helpers as sth
from . import sphere_torus_data_gen as dg


def find_theta_and_phi(z):
    d = z.shape[-1]
    phi = np.arctan2(z[..., 1], z[..., 0])
    scale_factor = np.sqrt(
        np.sum(np.power(np.arange(1, np.floor((d - 1) / 2) + 1), -2))
    )
    cos_theta = np.clip(np.sqrt(z[..., 1] ** 2 + z[..., 0] ** 2) - 2, -1, 1)
    sin_theta = np.clip(z[..., -1] / scale_factor, -1, 1)
    theta = np.arctan2(sin_theta, cos_theta)
    return theta, phi

def map_to_torus_(theta, phi, d):
    x = np.zeros((*theta.shape, d))
    for i in range(0, d - 1, 2):
        factor = i // 2 + 1
        x[..., i] = (2 + np.cos(theta)) * np.cos(phi * factor) / factor
        x[..., i + 1] = (2 + np.cos(theta)) * np.sin(phi * factor) / factor
    scale_factor = np.sqrt(
        np.sum(np.power(np.arange(1, np.floor((d - 1) / 2) + 1), -2))
    )
    x[..., -1] = scale_factor * np.sin(theta)
    return x


def map_to_torus(theta_lst, phi_lst, d):
    xout = []
    for theta, phi in zip(theta_lst, phi_lst):
        x = np.zeros((*theta.shape, d))
        for i in range(0, d - 1, 2):
            factor = i // 2 + 1
            x[..., i] = (2 + np.cos(theta)) * np.cos(phi * factor) / factor
            x[..., i + 1] = (2 + np.cos(theta)) * np.sin(phi * factor) / factor
        scale_factor = np.sqrt(
            np.sum(np.power(np.arange(1, np.floor((d - 1) / 2) + 1), -2))
        )
        x[..., -1] = scale_factor * np.sin(theta)

        xout.append(x)
    return xout


def map_to_sphere(theta_lst, phi_lst):
    xout = []
    for theta, phi in zip(theta_lst, phi_lst):
        x = np.zeros((*theta.shape, 3))
        x[..., 0] = np.cos(theta) * np.cos(phi)
        x[..., 1] = np.cos(theta) * np.sin(phi)
        x[..., 2] = np.sin(theta)
        xout.append(x)

    return xout


def generate_truncated_data(NUM_POINTS, d=3):
    AMBIENT_DIM = d
    dt = 0.01
    top = 100
    paths = 100
    filename = (
        f"cached_data/torus_truncated_raw_data_{dt}_{top}_{paths}_{AMBIENT_DIM}.npy"
    )
    filename2 = (
        f"cached_data/torus_truncated_raw_data_f_{dt}_{top}_{paths}_{AMBIENT_DIM}.npy"
    )
    if os.path.exists(filename) and os.path.exists(filename2):
        data = np.load(filename)
        globalCoordField = np.load(filename2)
    else:
        x0 = sth.generateInitialConditions(paths, random=False)
        data_ = sth.generateData(x0, top, dt)

        data_, theta, phi = sth.changetotorus(data_)
        data_ = map_to_torus(theta, phi, AMBIENT_DIM)
        data_ = data_.transpose(1, 0, 2)
        globalCoordField_ = sth.derivativeApproxSixForward(data_, dt)
        data_ = data_[:, :-6, :]

        assert data_.shape == globalCoordField_.shape
        periodArray = sth.computeFirstRepeatIndexArray(data_)
        globalCoordField_ = sth.truncatePath(globalCoordField_, periodArray)
        data_ = sth.truncatePath(data_, periodArray)

        globalCoordField = dg.pathsToCloud(globalCoordField_, AMBIENT_DIM)
        data = dg.pathsToCloud(data_, AMBIENT_DIM)
        np.save(filename, data)
        np.save(filename2, globalCoordField)

    randomMask = np.full(data.shape[0], False)
    randomMask[:NUM_POINTS] = True
    np.random.shuffle(randomMask)
    data = data[randomMask]
    globalCoordField = globalCoordField[randomMask]
    return data, globalCoordField




def generate_truncated_data_pair(NUM_POINTS, d=3, map_val_gap=1, n_pts_lst=None, dt=0.01):
    AMBIENT_DIM = d
    top = 100
    paths = 100
    filename = (
        f"cached_data/torus_truncated_raw_data_{dt}_{top}_{paths}_{AMBIENT_DIM}.npy"
    )
    filename2 = (
        f"cached_data/torus_truncated_raw_data_f_{dt}_{top}_{paths}_{AMBIENT_DIM}.npy"
    )
    if os.path.exists(filename) and os.path.exists(filename2):
        data_ = np.load(filename)
        globalCoordField_ = np.load(filename2)
    else:
        x0 = sth.generateInitialConditions(paths, random=False)
        data_ = sth.generateData(x0, top, dt)

        data_, theta, phi = sth.changetotorus(data_)
        data_ = map_to_torus(theta, phi, AMBIENT_DIM)

        data_ = [data_[i].transpose(1, 0, 2) for i in range(len(data_))]
        data_ = [data_[i][:, :-6, :] for i in range(len(data_))]

        periodArray = sth.computeFirstRepeatIndexArray(data_)
        data_ = sth.truncatePath(data_, periodArray)  # [3, path, [time, d]]
        data_size = []
        for _r in data_:
            _ds = []
            for _p in _r:
                _ds.append(_p.shape[0])
            data_size.append(_ds)

        data_ratio = []
        for _ds in data_size:
            _ds = np.array(_ds)
            _sum = np.sum(_ds)
            _ds = _ds/_sum
            data_ratio.append(_ds)

    X_, Y_, Yv_ = [], [], []
    for path_data in data_:
        XX, YY, Yv = [], [], []
        for _d in path_data:
            XX.append(_d[:-map_val_gap, :])
            YY.append(_d[1:-map_val_gap+1, :])
            Yv.append(_d[map_val_gap:, :])
        X_.append(XX)
        Y_.append(YY)
        Yv_.append(Yv)


    if n_pts_lst == None:
        n_pts_per_region = NUM_POINTS // 3
        n_pts_lst = 3*[n_pts_per_region]
        if NUM_POINTS != 3*n_pts_per_region:
            rem = NUM_POINTS % 3
            if rem == 1:
                n_pts_lst[0] += 1
            elif rem == 2:
                n_pts_lst[0] += 1
                n_pts_lst[1] += 1

    #print(n_pts_lst)
    sample_lst = []
    for _i, _dr in enumerate(data_ratio):
        _data_samples = np.floor(_dr  * n_pts_lst[_i]).astype(int)
        total_samples = np.sum(_data_samples)
        #print(f"calculated samples at region {_i+1} = {total_samples}")
        # if the sum is not equal to the target sample number, simply samples more points at the end
        if total_samples != n_pts_lst[_i]:
            _data_samples[-1] += np.abs(total_samples - n_pts_lst[_i])
        total_samples = np.sum(_data_samples)
        #print(f"fixed samples at region {_i+1} = {total_samples}")
        sample_lst.append(_data_samples)


    X, Y, Yv = [], [], []
    for i, (XX, YY, YYv) in enumerate(zip(X_, Y_, Yv_)):
        _npts_arr = sample_lst[i]
        _X, _Y, _Yv = [], [], []
        for j, (__X, __Y, __Yv) in enumerate(zip(XX, YY, YYv)):
            _ss = _npts_arr[j]
            randomMask = np.full(__X.shape[0], False)
            randomMask[:_ss] = True
            np.random.shuffle(randomMask)

            __X = __X[randomMask]
            __Y = __Y[randomMask]
            __Yv = __Yv[randomMask]
            _X.append(__X)
            _Y.append(__Y)
            _Yv.append(__Yv)
        _X = np.concatenate(_X, axis=0)
        _Y = np.concatenate(_Y, axis=0)
        _Yv = np.concatenate(_Yv, axis=0)
        
        # print(_X.shape)
        # print(_Y.shape)
        # print(_Yv.shape)
        # print("----")
        X.append(_X)
        Y.append(_Y)
        Yv.append(_Yv)
    X_arr = np.concatenate(X, axis=0)
    Y_arr = np.concatenate(Y, axis=0)
    Yv_arr = np.concatenate(Yv, axis=0)
    # print(Yv[0].shape)
    return X_arr, Y_arr, Yv_arr, X, Y, Yv
