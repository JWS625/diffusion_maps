import os

# set wkdir to dm_final
# os.chdir("dm_final")

import itertools
import pickle
import krr_model
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import cupy as cp
import polars as pl
from tqdm import tqdm

from gindy.src.manifold import Manifold
from dm_main import DMClass

from matplotlib import pyplot as plt
np.random.seed(42)
cp.random.seed(42)

def _to_float(x):
    # Handle CuPy scalars/arrays, NumPy scalars/arrays, and Python floats
    try:
        if isinstance(x, cp.ndarray):
            return float(cp.asnumpy(x))
    except Exception:
        pass
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    return float(x)


def free_gpu_memory():
    import gc
    from multiprocessing import active_children

    active = active_children()
    for child in active:
        child.terminate()
    for child in active:
        child.join()

    for obj in gc.get_objects():
        if isinstance(obj, cp.ndarray):
            del obj

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    
    gc.collect()
    cp.cuda.runtime.deviceSynchronize()

def main(mode):
    epsilon_array, lambda_array, rmse_array = compute_cv_rmse_with_parallelization(
        mode, dt, data_train_x, data_train_y, data_val
    )
    pldf = pl.DataFrame(
        {
            "epsilon":    list(np.asarray(epsilon_array, dtype=float)),
            "lambda":     list(np.asarray(lambda_array,  dtype=float)),
            "rmse":       list(np.asarray(rmse_array,    dtype=float)),
            "epsilon_c":  [float(eps)] * len(epsilon_array),
            "lambda_min": [float(lambda_min)] * len(epsilon_array),
        }
    )
    pldf.write_parquet(
        f"./numerical_results/{title}_cv_{mode}_{map_type}.parquet"
    )

    free_gpu_memory()

def test(mode):

    cv_results = pl.read_parquet(
        f"./numerical_results/{title}_cv_{mode}_{map_type}.parquet"
    )
    cv_rmse = cv_results["rmse"].to_numpy()
    eps = cv_results["epsilon"].to_numpy()
    lam =  cv_results["lambda"].to_numpy()

    index = np.nanargmin(cv_rmse)
    epsilon = eps[index]
    lambda_reg = lam[index]

    opts["inp"] = data_train_x
    opts["out"] = data_train_y
    data_test_arr = np.asarray(data_test).squeeze()  # [steps, Nx]
    steps = data_test_arr.shape[0]

    model = krr_model.modeler(**opts)
    model.fit_model(epsilon, lambda_reg, mode)

    _truePath     = cp.array(data_test_arr) #[steps, trials, Nx]
    truePath = _truePath

    test_point = cp.asarray(truePath[0])
    pred_path = [cp.array(test_point).get()]
    # mean_cp = cp.asarray(mean)
    # std_cp = cp.asarray(std)

    for i in tqdm(range(steps - 1)):
        test_point = model.forecast(test_point)
        # test_point = std_cp * test_point + mean_cp
        pred_path.append(test_point.get())

    pred_path = cp.array(pred_path) #@ Q_cp.T  # [steps, trials, Nx]
    print(f"pred_path.shape = {pred_path.shape}")
    print(f"_truePath.shape = {_truePath.shape}")
    nrmse = cp.linalg.norm(pred_path - _truePath) / cp.linalg.norm(_truePath)
    nrmse = cp.asnumpy(nrmse)
    
    t_plot = dt * np.arange(N_test)
    plot_modes = 10
    fig, ax = plt.subplots(nrows=plot_modes//2 + plot_modes%2, ncols=2, sharex=True, sharey=True)
    for _ in range(plot_modes):
        row = _ // 2
        col = _ % 2
        l1, = ax[row, col].plot(t_plot, _truePath[:, _].get())
        l2, = ax[row, col].plot(t_plot, pred_path[:, _].get())
        ax[row, col].grid()
    ax[0, 1].legend([l1, l2], ["Truth", "Pred"])
    ax[-1, 0].set_xlabel('$t$'); ax[-1, 1].set_xlabel('$t$')
    plt.suptitle(f"NRMSE = {nrmse}")
    # fig.savefig(f'./journal_pics/ppplate_pred_{mode}_{map_type}_weights.png')

    fig, ax = plt.subplots(nrows=plot_modes//2 + plot_modes%2, ncols=2, sharex=True, sharey=True)
    for _ in range(plot_modes):
        row = _ // 2
        col = _ % 2
        l1, = ax[row, col].plot(t_plot, _truePath[:, r-1-_].get())
        l2, = ax[row, col].plot(t_plot, pred_path[:, r-1-_].get())
        ax[row, col].grid()
    ax[0, 1].legend([l1, l2], ["Truth", "Pred"])
    ax[-1, 0].set_xlabel('$t$'); ax[-1, 1].set_xlabel('$t$')
    plt.suptitle(f"NRMSE = {nrmse}")
    # fig.savefig(f'./journal_pics/ppplate_pred_last10_{mode}_{map_type}_weights.png')


    print(
        f"Test max RMSE: {np.nanmax(nrmse)}, Val RMSE: {cv_rmse[index]}, "
        f"epsilon: {epsilon}, lambda: {lambda_reg}"
    )

    results_df = pl.DataFrame(
        {
            "epsilon": [epsilon],
            "lambda": [lambda_reg],
            "rmse": [nrmse.tolist()],
            "path": [pred_path.get().tolist()],
            # "Q"   : [Q.tolist()],
        }
    )

    results_df.write_parquet(
        f"./numerical_results/{title}_{mode}_{map_type}.parquet"
    )

    free_gpu_memory()


def compute_cv_rmse_with_parallelization(mode, dt, data_train_x, data_train_y, data_val):
    total_parallel_trials = 4

    import time

    start = time.time()
    rmse = Parallel(n_jobs=total_parallel_trials)(
        delayed(batched_compute_rmse_inner_cv)(
            mode,
            dt,
            data_train_x,
            data_train_y,
            data_val,
            np.array_split(epsilon_array, total_parallel_trials)[i],
            np.array_split(lambda_array, total_parallel_trials)[i],
            i % devices,
        )
        for i in range(total_parallel_trials)
    )
    print(f"Time taken: {time.time() - start}")

    rmse_array   = np.concatenate([np.asarray(x, dtype=float) for x in rmse], axis=0) 
    return epsilon_array, lambda_array, rmse_array


def batched_compute_rmse_inner_cv(
    mode, dt, data_train_x, data_train_y, data_val,
    epsilon_array, lambda_array, device,
):
    with cp.cuda.Device(device):
        rmse_lst = []
        try:
            for epsilon, lambda_reg in tqdm(zip(epsilon_array, lambda_array)):

                rmse = compute_rmse_inner_cv(
                    mode, dt, data_train_x, data_train_y,
                    data_val, epsilon, lambda_reg, device,
                )

                rmse_lst.append(rmse)
            return rmse_lst
        finally:
            free_gpu_memory()

def compute_rmse_inner_cv(
    mode,
    dt,
    data_train_x,
    data_train_y,
    data_val,     # shape: (T, n_traj, r)
    epsilon,
    lambda_reg,
    device=0,
):
    with cp.cuda.Device(device):
        opts["inp"] = data_train_x
        opts["out"] = data_train_y
        T, n_traj, rdim = data_val.shape

        t_window = dt * np.arange(T)
        weights = np.exp(t_window-t_window[-1])
        weights = weights / np.linalg.norm(weights)

        model = krr_model.modeler(**opts)
        model.fit_model(epsilon, lambda_reg, mode)

        _data_val_cp = cp.asarray(data_val)
        data_val_cp  = _data_val_cp
        weights_cp = cp.asarray(weights)[:, None, None]

        # initial condition
        test_point = data_val_cp[0]          # (n_traj, r)

        pred_list = [test_point]

        for _ in range(1, validation_horizon):
            test_point = model.forecast(test_point)

            if cp.isnan(test_point).any():
                return float("nan")

            pred_list.append(test_point)


        path = cp.stack(pred_list, axis=0)

        target = cp.asarray(_data_val_cp[:T])         # (T, n_traj, r)
        error  = path[:T] - target

        nrmse_scalar = cp.linalg.norm(weights_cp*error) / cp.linalg.norm(target)

        return _to_float(nrmse_scalar)



def optimalSig(s, thr=0.99):
    n = len(s)
    find = 0
    ind = 0
    _ = 0
    sigsum = np.cumsum(s**2)
    sig = sigsum / sigsum[-1]
    while (_ < n) and (find == 0):
        if sig[_] >= thr:
          ind = _
          find = 1
        _ = _ + 1
    return ind

if __name__ == "__main__":

    dt = 0.1
    map_type_lst = ["direct"]#, "skip-connection"]
    mode_lst = ["diffusion", "rbf"]
    cv_trials = 8192
    title = "ppplate"

    # data collection
    training_sets = [30*i for i in range(12)]
    validation_sets = [45, 225, 315]
    n_training_sets = len(training_sets)
    n_validation_sets = len(validation_sets)
    test_sets = [135]

    N = 200
    N_test = 801
    validation_horizon = N_test
    N_trun = 200
    t = dt*np.arange(0, N)
    training_dat_lst = []
    for _i, _h in enumerate(training_sets):
        filename = f'./cached_data/omega_cube_phaseA_0.0000_phaseH_{_h}.0000.pkl'
        with open(filename, 'rb') as pickle_file:
            vort = pickle.load(pickle_file)['omega'][N_trun:]

        vort = vort.reshape(len(vort), -1)
        _dat = vort[:N]
        training_dat_lst.append(_dat)
        
    training_dat_vstack = np.vstack(training_dat_lst)
    vortex_mean = np.mean(training_dat_vstack, axis=0)
    training_dat_vstack = training_dat_vstack - vortex_mean
    u, s, vh = np.linalg.svd(training_dat_vstack, full_matrices=False)
    r = optimalSig(s, 0.991) + 1
    ur, sr, vhr = u[:, :r], s[:r], vh[:r]
    sr_norm = sr / sr[0]

    data_train = ur @ np.diag(sr_norm)

    print(f"svd order = {r}")

    data_train_x, data_train_y = [], []
    for _i, _h in enumerate(training_sets):
        _d = data_train[_i*N:(_i+1)*N]
        data_train_x.append(_d[:-1])
        data_train_y.append(_d[1:])
        # data_train_y.append((_d[1:] - mean) / std)

    data_train_x, data_train_y = np.vstack(data_train_x), np.vstack(data_train_y)
    print(f"data_train_x.shape = {data_train_x.shape}")
    print(f"data_train_y.shape = {data_train_y.shape}")

    # validation
    validation_dat_lst = []
    for _i, _h in enumerate(validation_sets):
        filename = f'./cached_data/omega_cube_phaseA_0.0000_phaseH_{_h}.0000.pkl'
        with open(filename, 'rb') as pickle_file:
            vort = pickle.load(pickle_file)['omega'][N_trun:]

        vort = vort.reshape(len(vort), -1)
        vort = vort - vortex_mean
        _dat = vort[:N]
        validation_dat_lst.append(_dat)
    validation_dat_vstack = np.vstack(validation_dat_lst)
    validation_dat = validation_dat_vstack @ vhr.T @ np.diag(sr_norm/sr)

    data_val = []
    for _j in range(len(validation_sets)):
        print(validation_dat[_j*N:(_j+1)*N].shape)
        data_val.append(validation_dat[_j*N:(_j+1)*N])

    data_val = np.array(data_val)
    data_val = data_val.transpose(1, 0, 2)
    print(f"data_val.shape = {data_val.shape}")

    # test data
    test_dat_lst = []
    for _i, _h in enumerate(test_sets):
        filename = f'./cached_data/omega_cube_phaseA_0.0000_phaseH_{_h}.0000.pkl'
        with open(filename, 'rb') as pickle_file:
            vort = pickle.load(pickle_file)['omega'][N_trun:]

        vort = vort.reshape(len(vort), -1)
        vort = vort - vortex_mean
        _dat = vort[:N_test]
        test_dat_lst.append(_dat)
    test_dat_vstack = np.vstack(test_dat_lst)
    test_dat = test_dat_vstack @ vhr.T @ np.diag(sr_norm/sr)

    data_test = []
    for _j in range(len(test_sets)):
        data_test.append(test_dat[_j*N_test:(_j+1)*N_test])

    data_test = np.array(data_test)
    data_test = data_test.transpose(1, 0, 2)
    print(f"data_test.shape = {data_test.shape}")

    # kernel solver setup
    for map_type in map_type_lst:
        opts = {
            "map_type": map_type,
            "norm": False,
        }
        _man_data = data_train_x
        man = Manifold(_man_data)
        dim, __, _eps, eps = man.estimate_intrinsic_dim(bracket=[-20, 10], tol=0.2, ifest=True)
        eps = 2.5 * eps / r
        eps_min = np.log10(eps) - 3
        eps_max = np.log10(eps) + 3
        print(f"eps min = {eps_min}, eps max = {eps_max}")

        devices = 4
        for mode in mode_lst:
            print(f"map type: {map_type}, mode = {mode}")
            dmK, _q1, _q2, _dist = DMClass._compute_kernel_matrix_and_densities(cp.asarray(_man_data), epsilon=eps, mode=mode)
            rbfK = cp.exp(-(_dist**2) / (4 * eps))
            eig, eigvec = np.linalg.eig(rbfK.get())
            mineig = np.min(eig.real)
            lambda_min = np.log10(np.abs(mineig))
            print(f"lambda min power = {lambda_min}")

            epsilon_array = np.power(10, np.random.uniform(eps_min, eps_max, cv_trials))
            lambda_array = np.power(10, np.random.uniform(lambda_min, lambda_min+4, cv_trials))

            print(mode)
            main(mode)
            test(mode)
