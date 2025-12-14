import os

# set wkdir to dm_final
# os.chdir("dm_final")

import itertools
import pickle
import general_utils
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import cupy as cp
import polars as pl
from tqdm import tqdm

from gindy.src.manifold import Manifold
from dm_main import DMClass

# np.random.seed(42)
np.random.seed(1442)

def to_numpy(x):
    # cupy -> numpy, numpy stays numpy, python scalars -> numpy scalar
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            x = cp.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)

def to_pylist(x, dtype=np.float64):
    x = to_numpy(x)
    if x.dtype == object:
        # ensure numeric, not object
        x = x.astype(dtype, copy=False)
    return x.tolist()

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
            "epsilon": list(epsilon_array),
            "lambda": list(lambda_array),
            "epsilon_c": eps,
            "lambda_min":lambda_min,
            "rmse": list(rmse_array),
        }
    )

    pldf.write_parquet(
        f"./numerical_results/ks_traveling_cv_{mode}_{num_points}_{map_type}_dt_0.02_seed_1442.parquet"
    )

    free_gpu_memory()



def load_data():

    NT = 5000000
    SKP = 500000
    DT = 0.001
    TS = 10
    data = pickle.load(open(f"./ks_utils/ksdata_traveling_NT_{NT}_SKP_{SKP}_dt_{DT}_ts_{TS}.pkl", "rb"))
    dt = data["dt"]
    nu = data["nu"]
    xx = data["x"]
    tt = data["t"]
    uu = data["udata"].astype(np.float64)
    
    Nx, Nt = len(xx), len(tt)
    assert uu.shape == (Nt, Nx)

    data_train = uu[:num_points:2]
    print(data_train.shape)
    dt = dt * 2

    data_train_x = data_train[:-1]
    data_train_y = data_train[1:]

    data_val = data_train.copy()

    data_test = []
    start0 = num_points
    for _ in range(test_trials):
        s = start0 + _ * steps
        e = s + steps
        data_test.append([uu[s:e:2]])
    return dt, data_train_x, data_train_y, data_val, data_test

def test(mode):

    cv_results = pl.read_parquet(
        f"./numerical_results/ks_traveling_cv_{mode}_{num_points}_{map_type}_dt_0.02_seed_1442.parquet"
    )
    cv_rmse = cv_results["rmse"]
    index = cv_rmse.arg_min()
    epsilon, lambda_reg = (
        cv_results["epsilon"][index],
        cv_results["lambda"][index],
    )

    opts["inp"] = data_train_x
    opts["out"] = data_train_y

    model = general_utils.modeler(**opts)
    model.fit_model(epsilon, lambda_reg, mode)

    data_test_arr = np.asarray(data_test).squeeze().reshape(test_trials, steps//2, d)  # [test_trials, steps, Nx]
    truePath = cp.array(data_test_arr).transpose(1, 0, 2)  # [steps, trials, Nx]

    test_point = cp.asarray(truePath[0])
    pred_path = [cp.array(test_point).get()]

    for i in range(steps//2 - 1):
        test_point = model.forecast(test_point)
        pred_path.append(test_point.get())

    pred_path = cp.array(pred_path)
    rmse = cp.max(
            cp.sqrt(
                cp.mean(
                        ((pred_path - truePath))**2, axis=-1)
                )
            , axis=0)
    error = cp.abs((pred_path - truePath))
    print(
        f"Test max RMSE: {np.nanmax(rmse)}, Val RMSE: {cv_rmse[index]}, "
        f"epsilon: {epsilon}, lambda: {lambda_reg}"
    )
    rmse = np.asarray(rmse.get())
    error = np.asarray(error.get())
    pred_path = np.asarray(pred_path.get())

    results = {}
    results["epsilon"] = epsilon
    results["lambda"] = lambda_reg
    results["rmse"] = rmse
    results["errr"] = error
    results["path"] = pred_path

    results_path = (
        f"./numerical_results/ks_traveling_{mode}_{num_points}_{map_type}_dt_0.02_seed_1442.pkl"
    )
    with open(results_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    free_gpu_memory()


def compute_cv_rmse_with_parallelization(mode, dt, data_train_x, data_train_y, data_val):
    total_parallel_trials = 4

    import time

    start = time.time()
    rmse_array = Parallel(n_jobs=total_parallel_trials)(
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

    rmse_array = np.concatenate(rmse_array)
    return epsilon_array, lambda_array, rmse_array


def batched_compute_rmse_inner_cv(
    mode, dt, data_train_x, data_train_y, data_val,
    epsilon_array, lambda_array, device,
):
    with cp.cuda.Device(device):
        rmse_array = []
        try:
            for epsilon, lambda_reg in tqdm(zip(epsilon_array, lambda_array)):
                try:
                    rmse = 0.0
                    for i in range(validation_repeats):
                        _data_val = data_val[:num_points]
                        rmse += compute_rmse_inner_cv(
                            mode, dt, data_train_x, data_train_y,
                            _data_val, epsilon, lambda_reg, device,
                        )
                    rmse /= validation_repeats
                except Exception as e:
                    print(f"[worker-{device}] trial failed: {e}")
                    rmse = np.nan
                rmse_array.append(rmse)
            return rmse_array
        finally:
            free_gpu_memory()



def compute_rmse_inner_cv(
    mode,
    dt,
    data_train_x,
    data_train_y,
    data_val,
    epsilon,
    lambda_reg,
    device=0,
):
    with cp.cuda.Device(device):
        
        mask = np.zeros(data_train_x.shape[0]).astype(bool)
        mask[:-num_points//(validation_repeats)] = 1
        np.random.shuffle(mask)
        opts["inp"] = data_train_x[mask]
        opts["out"] = data_train_y[mask]
        model = general_utils.modeler(**opts)

        try:
            model.fit_model(epsilon, lambda_reg, mode)

            test_point = cp.asarray(data_val[0])
            path = [cp.array(test_point).get()]

            for i in range(num_points//2 - 1):
                test_point = model.forecast(test_point)
                path.append(test_point.get().flatten())

            path = cp.asarray(path)  # [validation_horizon, Nx]
            error = path - cp.asarray(data_val[:len(path)])
            rmse = cp.sqrt(cp.mean((error)**2))
        except Exception as e:
            rmse = float("nan")
            print(e)
    return _to_float(rmse)


if __name__ == "__main__":

    num_points_lst = [6000]
    steps = 14000
    validation_repeats = 5
    test_trials = 1
    map_type_lst = ["skip-connection"]
    mode_lst = ["diffusion", "rbf"]
    cv_trials = 4096

    for map_type in map_type_lst:
        opts = {
            "map_type": map_type,
            "norm": False,
        }
        for i_n, num_points in enumerate(num_points_lst):
            dt, data_train_x, data_train_y, data_val, data_test = load_data()
            d = data_train_x.shape[-1]
            man = Manifold(data_train_x)
            dim, dimf, _eps, eps = man.estimate_intrinsic_dim(bracket=[-20, 10], tol=0.2, ifest=True)

            eps = 10 * eps**(dimf) / 4 / d
            eps_min = np.log10(eps) - 2
            eps_max = np.log10(eps) + 2

            with cp.cuda.Device(1):
                devices = 4
                for mode in mode_lst:
                    print(f"map type: {map_type}, num points = {num_points}, mode = {mode}")
                    dmK, _q1, _q2, _dist = DMClass._compute_kernel_matrix_and_densities(cp.asarray(data_train_x), epsilon=eps, mode=mode)
                    rbfK = cp.exp(-(_dist**2) / (4 * eps))
                    eig, eigvec = np.linalg.eig(rbfK.get())
                    mineig = np.min(eig.real)
                    lambda_min = np.log10(np.abs(mineig))
                    print(f"lmabda from rbfK = {lambda_min}")
                    
                    epsilon_array = np.power(10, np.random.uniform(eps_min, eps_max, cv_trials))
                    lambda_array = np.power(10, np.random.uniform(lambda_min, lambda_min+4, cv_trials))

                    print(mode)
                    main(mode)
                    test(mode)
