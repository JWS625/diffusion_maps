import sys
from pathlib import Path
root = Path.cwd().resolve().parents[1]
sys.path.insert(0, str(root))

from diffusion_maps import model_dir, data_dir

import pickle
from src.krr_model import Modeler
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import cupy as cp
import polars as pl

from src.utils import Manifold
from src.dm_main import DMClass

np.random.seed(42)
cp.random.seed(42)

def _to_float(x):
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

def run_val(mode):
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
        model_dir + f"/ppplate/{title}_cv_{mode}_{map_type}.parquet"
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
            for epsilon, lambda_reg in tqdm(zip(epsilon_array, lambda_array), total=len(epsilon_array)):

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

        model = Modeler(**opts)
        distance_matrix = cp.linalg.norm(
                cp.array(model.inp[:, None]) - cp.array(model.inp[None]), axis=-1
            )
        model.fit_model(epsilon, lambda_reg, mode, distance_matrix=distance_matrix)

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
        free_gpu_memory()

        return _to_float(nrmse_scalar)

def test(mode):

    cv_results = pl.read_parquet(
        model_dir + f"/ppplate/{title}_cv_{mode}_{map_type}.parquet"
    )
    cv_rmse = cv_results["rmse"].to_numpy()
    eps = cv_results["epsilon"].to_numpy()
    lam =  cv_results["lambda"].to_numpy()

    index = np.nanargmin(cv_rmse)
    epsilon = eps[index]
    lambda_reg = lam[index]

    opts["inp"] = data_train_x
    opts["out"] = data_train_y
    data_test_arr = np.asarray(data_test).squeeze()
    steps = data_test_arr.shape[0]

    model = Modeler(**opts)
    distance_matrix = cp.linalg.norm(
            cp.array(model.inp[:, None]) - cp.array(model.inp[None]), axis=-1
        )
    model.fit_model(epsilon, lambda_reg, mode, distance_matrix=distance_matrix)

    truePath = cp.array(data_test_arr)
    test_point = cp.asarray(truePath[0]).reshape(1, -1)
    pred_path = [cp.array(test_point).get()]

    for i in tqdm(range(steps - 1)):
        test_point = model.forecast(test_point)
        pred_path.append(test_point.get())

    pred_path = cp.array(pred_path).squeeze()
    nrmse = cp.linalg.norm(pred_path - truePath) / cp.linalg.norm(truePath)
    nrmse = cp.asnumpy(nrmse)

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
        }
    )
    free_gpu_memory()
    
    results_df.write_parquet(
        model_dir + f"/ppplate/{title}_{mode}_{map_type}.parquet"
    )
    


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
    map_type_lst = ["direct"]
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
    print("Training data processing...")
    for _i, _h in enumerate(training_sets):
        filename = str(data_dir) + f'/cached_data/omega_cube_phaseA_0_phaseH_{_h}.npy'
        vort = np.load(filename)[N_trun:]

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

    data_train_x, data_train_y = np.vstack(data_train_x), np.vstack(data_train_y)
    print("Done.")

    # validation
    print("Validation data processing...")
    validation_dat_lst = []
    for _i, _h in enumerate(validation_sets):
        filename = str(data_dir) + f'/cached_data/omega_cube_phaseA_0_phaseH_{_h}.npy'
        vort = np.load(filename)[N_trun:]

        vort = vort.reshape(len(vort), -1)
        vort = vort - vortex_mean
        _dat = vort[:N]
        validation_dat_lst.append(_dat)
    validation_dat_vstack = np.vstack(validation_dat_lst)
    validation_dat = validation_dat_vstack @ vhr.T @ np.diag(sr_norm/sr)

    data_val = []
    for _j in range(len(validation_sets)):
        data_val.append(validation_dat[_j*N:(_j+1)*N])

    data_val = np.array(data_val)
    data_val = data_val.transpose(1, 0, 2)
    print("Done.")

    # test
    print("Test data processing...")
    test_dat_lst = []
    for _i, _h in enumerate(test_sets):
        filename = str(data_dir) + f'/cached_data/omega_cube_phaseA_0_phaseH_{_h}.npy'
        vort = np.load(filename)[N_trun:]

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
    print("Done.")

    for map_type in map_type_lst:
        opts = {
            "map_type": map_type,
            "pipeline": "single"
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
            run_val(mode)
            test(mode)
