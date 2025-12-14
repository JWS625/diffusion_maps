import os

import cupy as cp
import general_utils
import numpy as np

# sphere
import sphere_torus_utils.torus_data_gen as tdg
import sphere_torus_utils.sphere_torus_helpers as sth

# lorenz
from lorenz import generateData as lorenz_datagen
from lorenz import f as lorenz_f

from joblib import Parallel, delayed
import polars as pl
import time
from tqdm import tqdm

import pickle
import scipy.io as sio

from gindy.src.manifold import Manifold
from dm_main import DMClass

np.random.seed(12)
cp.random.seed(12)


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


def find_safe_chunk_size(N, device_id=0, dtype=cp.float64,
                            safety=0.5, overhead_factor=3.0):
    with cp.cuda.Device(device_id):
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()

    print(f"[GPU device ({device_id})]  {free_bytes/1e9:2.2f} GB available.")

    bytes_per_elem = np.dtype(dtype).itemsize 
    bytes_per_matrix = N * N * bytes_per_elem

    return int((free_bytes * safety) / (overhead_factor * bytes_per_matrix))


def batched_compute_inner_cv(
    k_train,
    k_valid,
    epsilon_array,
    lambda_array,
    mode,
    device,
):
    with cp.cuda.Device(device):

        tau_f_array = np.zeros(cv_trials_per_device, dtype=float)
        vpt_array   = np.zeros(cv_trials_per_device, dtype=float)
        

        for j in range(validation_repeats):
            j_valid = k_valid[
                validation_indices[j] : validation_indices[j] + validation_horizon
            ]

            opts["data"] = k_train
            model = general_utils.modeler(**opts)

            X = cp.asarray(model.inp, dtype=cp.float64)  # (N, d)
            n = cp.sum(X * X, axis=1)                    # (N,)

            distance_matrix = X @ X.T
            distance_matrix *= -2.0
            distance_matrix += n[:, None]
            distance_matrix += n[None, :]
            cp.maximum(distance_matrix, 0.0, out=distance_matrix)
            cp.sqrt(distance_matrix, out=distance_matrix)
            cp.fill_diagonal(distance_matrix, 0.0)
            distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)

            chunk_size = n_chunk_lst[device]

            for start in range(0, cv_trials_per_device, chunk_size):
                end = min(cv_trials_per_device, start + chunk_size)

                eps_chunk = epsilon_array[start:end]
                lam_chunk = lambda_array[start:end]

                model.fit_model(eps_chunk, lam_chunk, mode, distance_matrix=distance_matrix)
                # if np.isnan(model.dm.beta).any():
                #     print("Nan found")
                _vpt_chunk, _tau_f_chunk = model.get_performance(
                    j_valid,
                    dt=dt,
                    Lyapunov_time=1 / Lyapunov_exp,
                    error_threshold=error_threshold,
                )

                _vpt_chunk   = cp.asnumpy(_vpt_chunk)
                _tau_f_chunk = cp.asnumpy(_tau_f_chunk)

                vpt_array[start:end]   += _vpt_chunk
                tau_f_array[start:end] += _tau_f_chunk

            free_gpu_memory()

        vpt_array   /= validation_repeats
        tau_f_array /= validation_repeats

        
        return tau_f_array, vpt_array

        



def compute_cv_with_parallelization(k_train, k_valid, k_epsilon_array, k_lambda_array, mode, devices):

    parted_epsilon_array = np.array_split(k_epsilon_array, devices)
    parted_lambda_array = np.array_split(k_lambda_array, devices)

    with Parallel(n_jobs=4) as parallel:
        results = parallel(
            delayed(batched_compute_inner_cv)(
                k_train,
                k_valid,
                parted_epsilon_array[i],
                parted_lambda_array[i],
                mode,
                i,
            )
            for i in range(devices)
        )
    free_gpu_memory()

    tau_fs, vpts = zip(*results)
    
    return np.concatenate(tau_fs), np.concatenate(vpts)

def run_cv():

    print("Starting to find reference hyper-parameters.")
    ref_param_file = f"./numerical_results/lorenz_3d_ref_params_{num_points}.pkl"
    epsilon_array = np.zeros((test_trials, cv_trials_per_device*devices))
    lambda_array = np.zeros((test_trials, cv_trials_per_device*devices))
    try:
        with open(ref_param_file, "rb") as f:
            ref_params = pickle.load(f)
            ref_epsilon = ref_params["epsilon"]
            ref_lambda = ref_params["lambda_reg"]
        print("Existing referene params found")

        for k in tqdm(range(test_trials)):
            k_eps = ref_epsilon[k]
            lambda_min = ref_lambda[k]

            eps_min = k_eps - 2
            eps_max = k_eps + 2

            np.random.seed(1442)
            epsilon_array[k, :] = np.power(10, np.random.uniform(eps_min, eps_max, cv_trials_per_device * devices)) 
            lambda_array[k, :] = np.power(10, np.random.uniform(lambda_min, lambda_min+4, cv_trials_per_device * devices))

    except:

        print(f"No existing reference hyper-parameters. \nFinding reference....")
        # Finding reference hyperparameters
        
        ref_epsilon = np.zeros(test_trials)
        ref_lambda = np.zeros(test_trials)

        for k in tqdm(range(test_trials)):
            k_train = train[data_indices[k]:data_indices[k]+num_points]
            man = Manifold(k_train, verbose=False)
            dim, __, eps_c, eps = man.estimate_intrinsic_dim(bracket=[-20, 10], tol=0.2, ifest=True)

            eps = 1000 * eps / 4
            epsilon_mid = np.log10(eps)
            eps_min = epsilon_mid - 2
            eps_max = epsilon_mid + 2

            kernel_matrix, _q1, _q2, _dist = DMClass._compute_kernel_matrix_and_densities(cp.array(k_train), epsilon=eps, mode=mode)
            rbfK = cp.exp(-(_dist**2) / (4 * eps))

            eig, eigvec = np.linalg.eig(rbfK.get())
            mineig = np.min(eig.real)
            lambda_min = np.log10(np.abs(mineig))

            ref_epsilon[k] = epsilon_mid
            ref_lambda[k] = lambda_min

            np.random.seed(1442)
            epsilon_array[k, :] = np.power(10, np.random.uniform(eps_min, eps_max, cv_trials_per_device * devices)) 
            lambda_array[k, :] = np.power(10, np.random.uniform(lambda_min, lambda_min+4, cv_trials_per_device * devices))


        ref_params = {
            'epsilon': ref_epsilon,
            'lambda_reg': ref_lambda,
        }
        with open(ref_param_file, "wb") as f:
            pickle.dump(ref_params, f)

       
    print(f"eps_min = {eps_min}, eps_max = {eps_max}, lambda min = {lambda_min}")
    print("Finding reference is done.")

    print("Starting validation process.")
    cv_tau_f_array = []; cv_vpt_array = []
    for k in tqdm(range(test_trials)):
        k_tau_f_array, k_vpt_array = compute_cv_with_parallelization( 
            train[data_indices[k]:data_indices[k]+num_points],
            train[data_indices[k]+num_points:data_indices[k]+num_points+validation_length],
            epsilon_array[k, :], 
            lambda_array[k, :],
            mode, 
            devices,
        )


        # predictions.append(k_preds)
        cv_tau_f_array.append(k_tau_f_array)
        cv_vpt_array.append(k_vpt_array)

    cv_tau_f_array = np.array(cv_tau_f_array)
    cv_vpt_array = np.array(cv_vpt_array)


    cv_results = {
                "epsilon": epsilon_array,
                "lambda": lambda_array,
                "epsilon_c": ref_epsilon,
                "lambda_min": ref_lambda,
                "tau_f": cv_tau_f_array,
                "vpt": cv_vpt_array,
                "data_indices": data_indices,
                "validation_length": validation_length,
                "validation_repeats": validation_repeats,
                "validation_horizon":validation_horizon,
                }
    
    cv_filename = f"numerical_results/{title}_cv_result_{mode}_{num_points}_{map_type}_vl_{validation_length}.pkl"
    with open(cv_filename, "wb") as f:
        pickle.dump(cv_results, f)
    


def run_test():

    train = np.load("./cached_data/lorenz_train.npy").T
    test = np.load('./cached_data/lorenz_test.npy').transpose(0, 2, 1)
    n_test, T, d = test.shape

    # Load CV results
    path = f"numerical_results/{title}_cv_result_{mode}_{num_points}_{map_type}_vl_{validation_length}.pkl"
    with open(path, "rb") as f:
        cv_results = pickle.load(f)

    epsilon_array_loc   = cv_results["epsilon"]      # shape (test_trials, n_trials)
    lambda_reg_array_loc = cv_results["lambda"]
    data_indices        = cv_results["data_indices"]
    score               = cv_results["vpt"]
    rows = np.arange(epsilon_array_loc.shape[0])

    best_idx       = np.nanargmax(score, axis=1)
    best_epsilon   = epsilon_array_loc[rows, best_idx]     # shape (n_test,)
    best_lambda_reg = lambda_reg_array_loc[rows, best_idx] # shape (n_test,)

    vpts    = np.zeros(n_test)
    tau_fs  = np.zeros(n_test)

    all_indices = np.arange(n_test)
    index_splits = np.array_split(all_indices, devices)  # ~equal chunks per GPU

    def _run_test_block(index_block, device):

        with cp.cuda.Device(device):
            block_size = len(index_block)
            chunk_size = n_chunk_lst[device]

            vpts_block   = np.zeros(block_size)
            taufs_block  = np.zeros(block_size)

            for local_idx, i in tqdm(enumerate(index_block), total=block_size):

                i_epsilon    = best_epsilon[i]
                i_lambda_reg = best_lambda_reg[i]
                
                i_train = train[data_indices[i] : data_indices[i] + num_points]
                local_opts = dict(opts)
                local_opts["data"] = i_train

                model = general_utils.modeler(**local_opts)

                X = cp.asarray(model.inp, dtype=cp.float64)  # (N, d)
                n = cp.sum(X * X, axis=1)                    # (N,)

                distance_matrix = X @ X.T
                distance_matrix *= -2.0
                distance_matrix += n[:, None]
                distance_matrix += n[None, :]
                cp.maximum(distance_matrix, 0.0, out=distance_matrix)
                cp.sqrt(distance_matrix, out=distance_matrix)
                cp.fill_diagonal(distance_matrix, 0.0)
                distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)
                model.fit_model(i_epsilon, i_lambda_reg, mode, distance_matrix=distance_matrix)

                vpt_sum = 0.0
                tau_sum = 0.0
                
                for start in range(0, n_test, chunk_size):
                    
                    end = min(n_test, start + chunk_size)
                    vpt_j, tau_f_j = model.get_performance(
                        test[start:end],
                        dt=dt,
                        Lyapunov_time=1 / Lyapunov_exp,
                        error_threshold=error_threshold,
                    )
                    vpt_sum  += np.sum(vpt_j)
                    tau_sum  += np.sum(tau_f_j)

                vpts_block[local_idx]   = vpt_sum / n_test
                taufs_block[local_idx]  = tau_sum / n_test

            free_gpu_memory()
            return index_block, vpts_block, taufs_block

    results = Parallel(n_jobs=devices)(
        delayed(_run_test_block)(index_splits[dev], dev)
        for dev in range(devices)
    )

    for index_block, v_block, tau_block in results:
        vpts[index_block]   = v_block
        tau_fs[index_block] = tau_block
    # print(np.sort(vpts))

    free_gpu_memory()
    print(f"mean VPT = {np.mean(vpts)}, min VPT = {np.min(vpts)}, max VPT = {np.max(vpts)}.")
    performance = {
        "vpts": vpts,
        "tau_fs": tau_fs,
        "epsilon": best_epsilon,
        "lambda_reg": best_lambda_reg,
    }

    print(f"test mean vpt: = {np.mean(vpts)}")

    outp = f"./numerical_results/lorenz_3d_test_result_{mode}_{map_type}_{num_points}_vl_{validation_length}.pkl"
    with open(outp, "wb") as f:
        pickle.dump(performance, f, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":

    num_points_lst = [512, 1024]
    devices = 4
    dt = 0.01
    cv_trials_per_device = 1024
    mode_list = ["diffusion", "rbf"]
    map_type = 'skip-connection'
    test_trials = 500
    error_threshold = 0.3 ** 2
    Lyapunov_exp = 0.91

    title = f"lorenz_3d"

    # Datasets 
    train = np.load("./cached_data/lorenz_train.npy").T
    validation_size_multiplier = 2
    validation_horizon_dict = {
        '512': [200*i+500 for i in range(8)],
        '1024': [200*i+1100 for i in range(5)]
    }
    validation_repeats = 3
    
    opts = {
        "map_type": map_type,
        "norm": False,
    }

    for num_points in num_points_lst:
        validation_horizon_lst = validation_horizon_dict[str(num_points)]
        for validation_horizon in validation_horizon_lst:
            validation_length = int(validation_horizon * validation_size_multiplier)
            n_chunk_lst = []
            for dev in range(devices):
                n_chunk_lst.append(find_safe_chunk_size(num_points, device_id=dev, dtype=cp.float64,
                                    safety=0.4, overhead_factor=3.0))

            np.random.seed(142)
            data_indices = np.random.randint(
                train.shape[0] - (num_points + validation_length), size=test_trials
            )

            # validation starting indices
            if validation_length > validation_horizon:
                np.random.seed(132)
                validation_indices = np.random.randint(
                    int(validation_horizon_lst[0]*(validation_size_multiplier-1)),
                    size=validation_repeats,
                )
            else:
                validation_indices = np.zeros(validation_repeats, dtype=int)
            print(f"validation_indices = {validation_indices}")

            for mode in mode_list:
                print(f"mode = {mode}, validation horizon = {validation_horizon}, N = {num_points}")

                run_cv()
                free_gpu_memory()

                run_test()
                free_gpu_memory()
