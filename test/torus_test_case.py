import os


import cupy as cp
import krr_model
import numpy as np

# sphere
import sphere_torus_utils.torus_data_gen as tdg
import sphere_torus_utils.sphere_torus_helpers as sth

from dm_main import DMClass
from gindy.src.manifold import Manifold

from joblib import Parallel, delayed
import polars as pl
import time
from tqdm import tqdm
import cupy as cp


prob = "torus"
# np.random.seed(12)
np.random.seed(42)

filename_test = lambda d: f"cached_data/{prob}_test_data_{d}.npy"
filename_training = lambda u, N, d: f"cached_data/{prob}_training_data_{u}_{N}_{d}.npy"


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

def compute_error_inner(
    epsilon_array,
    lambda_array,
    mode,
    device,
):
    with cp.cuda.Device(device):
        model = krr_model.modeler(**opts)
        l2_err_array = np.zeros_like(epsilon_array)

        for i, (epsilon, lambda_reg) in tqdm(
            enumerate(zip(epsilon_array, lambda_array)), total=len(epsilon_array)
        ):
            try:
                model.fit_model(epsilon, lambda_reg, mode)

                x0 = sth.generateInitialConditions_(12)
                truePath = sth.generateData_(x0, validation_horizon * dt, dt)
                # convert to torus
                truePath, theta, phi = sth.changetotorus_(truePath)
                truePath = tdg.map_to_torus_(theta, phi, d)
                x0 = cp.array(np.copy(truePath[0]))
                pred_path = [cp.copy(x0)]
                hit_nan = False
                for _ in range(validation_horizon*2):
                    x0 = model.forecast(x0)
                    if cp.isnan(x0).any():
                        hit_nan = True
                        break
                    pred_path.append(cp.copy(x0))

                if hit_nan:
                    l2_err_array[i] = np.nan
                    continue

                pred_path = cp.array(pred_path)[:validation_horizon+1] # [T, Va, d]
                truePath = cp.array(truePath)
                l2_err_array[i] = cp.sqrt(cp.mean((pred_path-truePath)**2, axis=(0, 2))).mean()
            except Exception as e:
                l2_err_array[i] = np.nan
                print(e)
    return l2_err_array


def compute_error(epsilon_array, lambda_array, mode, devices):
    parted_epsilon_array = np.array_split(epsilon_array, devices)
    parted_lambda_array = np.array_split(lambda_array, devices)

    with Parallel(n_jobs=4) as parallel:
        output = parallel(
            delayed(compute_error_inner)(
                parted_epsilon_array[i],
                parted_lambda_array[i],
                mode,
                i,
            )
            for i in range(devices)
        )
    return np.concatenate(output)


def run_cv():
    cv_rmse_array = compute_error(
        epsilon_array, lambda_array, mode, devices
    )
    # print(eps)
    cv_result = pl.DataFrame(
        {
            "epsilon": epsilon_array,
            "lambda_reg": lambda_array,
            "epsilon_c": eps,
            "lambda_min":lambda_min,
            "rmse": cv_rmse_array,
        }
    )
    cv_result.write_parquet(
        f"numerical_results/{title}_cv_result_{mode}_{num_points}_{map_type}_dt_{dt}.parquet"
    )


def run_test():

    cv_results = pl.read_parquet(
        f"numerical_results/{title}_cv_result_{mode}_{num_points}_{map_type}_dt_{dt}.parquet"
    )

    epsilon_array = cv_results["epsilon"].to_numpy()
    lambda_reg_array = cv_results["lambda_reg"].to_numpy()
    rmse_array = cv_results["rmse"].to_numpy()

    best_idx = np.nanargmin(rmse_array)
    best_epsilon = epsilon_array[best_idx]
    best_lambda_reg = lambda_reg_array[best_idx]
    print(f'best epsilon = {best_epsilon}, best_lambda_reg = {best_lambda_reg}')

    results_df_list = []
    for j in range(test_trials):
        with Parallel(n_jobs=devices) as parallel:
            test_err = parallel(
                delayed(evaluate_performance)(
                    best_epsilon,
                    best_lambda_reg,
                    i,
                    j
                )
                for i in range(devices)
            )
            print(test_err)

        results_df = pl.DataFrame(
            {
                "epsilon": best_epsilon * np.ones(devices),
                "lambda_reg": best_lambda_reg * np.ones(devices),
                "rmse": [float(err) for err in test_err]
                #"final_rmse": [i[-1] for i in results_array],
            }
        )
        results_df_list.append(results_df)
        free_gpu_memory()
        
    results_df = pl.concat(results_df_list)

    results_df.write_parquet(
        f"numerical_results/{title}_test_result_{mode}_{num_points}_{map_type}_dt_{dt}.parquet"
    )
    return results_df


def evaluate_performance(epsilon, lambda_reg, device, trial_loop):
    trial_ind = 4 * trial_loop + device
    _x0 = x0_lst[trial_ind]
    with cp.cuda.Device(device):
        model = krr_model.modeler(**opts)
        model.fit_model(epsilon, lambda_reg, mode)

        truePath = sth.generateData_(_x0, steps * dt, dt)
        # convert to torus
        truePath, theta, phi = sth.changetotorus_(truePath)
        truePath = tdg.map_to_torus_(theta, phi, d)
        x0 = cp.array(np.copy(truePath[0]))
        pred_path = [cp.copy(x0)]
        for i in tqdm(range(steps)):
            x0 = model.forecast(x0)
            pred_path.append(cp.copy(x0))

        pred_path = cp.array(pred_path)
        truePath = cp.array(truePath)
        error = cp.sqrt(cp.mean((pred_path - truePath)**2, axis=(0, 2))).mean()

    return cp.asnumpy(error)



if __name__ == "__main__":

    d_lst = [3, 7, 15]
    num_points_lst = [1024, 2048, 4096, 8192]

    devices = 4
    dt = 0.01
    N_validation = 1000
    steps = 1000
    n_test = 1000
    cv_trials_per_device = 1024
    test_trials = 8
    mode_list = ["diffusion", "rbf"]
    x0_test = np.load(f'./cached_data/sphere_test_data.npy')
    map_type_lst = ["skip-connection"]

    x0_lst = []
    for _ in range(devices * test_trials):
        x0_lst.append(x0_test[_*n_test:(_+1)*n_test])

    for mode in mode_list:
        validation_horizon = 300

        for i_d, d in enumerate(d_lst):
            title=f"{prob}_{d}d"

            for map_type in map_type_lst:
                opts = {
                    "map_type": map_type,
                    "norm": False,
                }

                for i_n, num_points in enumerate(num_points_lst):
                    print(f"mode = {mode}, d = {d}, map type = {map_type}, N = {num_points}")
                    
                    training_data_x = np.load(filename_training('x', num_points, d))
                    training_data_y = np.load(filename_training('y', num_points, d))

                    man = Manifold(training_data_x)
                    dim, __, eps2, eps1 = man.estimate_intrinsic_dim(bracket=[-20, 10], tol=0.2, ifest=True)
                    eps = 10 * eps1 / 4 / d
                    eps_min = np.log10(eps) - 2
                    eps_max = np.log10(eps) + 2
                    
                    kernel_matrix, _q1, _q2, _dist = DMClass._compute_kernel_matrix_and_densities(cp.asarray(training_data_x), epsilon=eps, mode=mode)
                    # rbfK = cp.exp(-(_dist**2) / (4 * eps))
                    eig, eigvec = np.linalg.eig(kernel_matrix.get())
                    
                    mineig = np.min(eig.real)
                    lambda_min = np.log10(np.abs(mineig))

                    epsilon_range = (eps_min, eps_max)
                    lambda_range = (lambda_min, lambda_min+4)

                    epsilon_array = np.power(10, np.random.uniform(eps_min, eps_max, cv_trials_per_device * devices))
                    lambda_array = np.power(10, np.random.uniform(lambda_min, lambda_min+4, cv_trials_per_device * devices))

                    opts['inp'] = training_data_x
                    opts['out'] = training_data_y
                    run_cv()
                    free_gpu_memory()
                    run_test()
                    free_gpu_memory()


