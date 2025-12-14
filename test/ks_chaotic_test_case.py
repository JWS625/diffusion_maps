import pickle
import general_utils
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import cupy as cp
import time

from gindy.src.manifold import Manifold
from dm_main import DMClass


np.random.seed(42)
cp.random.seed(42)

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

    print(f"[GPU device ({device_id})]  {free_bytes/1e9:2.2f}/{total_bytes/1e9:2.2f} GB available.")

    bytes_per_elem = np.dtype(dtype).itemsize 
    bytes_per_matrix = N * N * bytes_per_elem

    return int((free_bytes * safety) / (overhead_factor * bytes_per_matrix))


def run_val(mode):

    
    print("Starting to find reference hyper-parameters.")
    ref_param_file = f"./numerical_results/ks_chaotic_ref_params_{num_points}.pkl"
    epsilon_array = np.zeros((n_models, cv_trials_per_device*devices))
    lambda_array = np.zeros((n_models, cv_trials_per_device*devices))
    try:
        with open(ref_param_file, "rb") as f:
            ref_params = pickle.load(f)
            ref_epsilon = ref_params["epsilon"]
            ref_lambda = ref_params["lambda_reg"]
        print("Existing referene params found")
            
        for k in tqdm(range(n_models)):
            k_eps = ref_epsilon[k]
            lambda_min = ref_lambda[k]

            eps_min = k_eps - 1
            eps_max = k_eps + 1
            
            np.random.seed(1442)
            epsilon_array[k, :] = np.power(10, np.random.uniform(eps_min, eps_max, cv_trials_per_device * devices)) 
            lambda_array[k, :] = np.power(10, np.random.uniform(lambda_min, lambda_min+2, cv_trials_per_device * devices))

    except:

        print(f"No existing reference hyper-parameters. \nFinding reference....")
        # Finding reference hyperparameters
        
        ref_epsilon = np.zeros(test_trials)
        ref_lambda = np.zeros(test_trials)

        for k in tqdm(range(n_models)):
            k_train = train[data_indices[k]:data_indices[k]+num_points]
            man = Manifold(k_train, verbose=False)
            dim, __, eps_c, eps = man.estimate_intrinsic_dim(bracket=[-20, 10], tol=0.2, ifest=True)

            eps = 1000 * eps / 4
            epsilon_mid = np.log10(eps)
            eps_min = epsilon_mid - 1
            eps_max = epsilon_mid + 1

            kernel_matrix, _q1, _q2, _dist = DMClass._compute_kernel_matrix_and_densities(cp.array(k_train), epsilon=eps, mode=mode)
            rbfK = cp.exp(-(_dist**2) / (4 * eps))

            eig, eigvec = np.linalg.eig(rbfK.get())
            mineig = np.min(eig.real)
            lambda_min = np.log10(np.abs(mineig))

            ref_epsilon[k] = epsilon_mid
            ref_lambda[k] = lambda_min

            np.random.seed(1442)
            epsilon_array[k, :] = np.power(10, np.random.uniform(eps_min, eps_max, cv_trials_per_device * devices)) 
            lambda_array[k, :] = np.power(10, np.random.uniform(lambda_min, lambda_min+2, cv_trials_per_device * devices))

        ref_params = {
            'epsilon': ref_epsilon,
            'lambda_reg': ref_lambda,
        }
        with open(ref_param_file, "wb") as f:
            pickle.dump(ref_params, f)

    print("Starting validation process.")
    cv_tau_f_array = []; cv_vpt_array = []
    for k in tqdm(range(n_models)):
        k_tau_f_array, k_vpt_array = compute_cv_with_parallelization( 
            train[data_indices[k]:data_indices[k]+num_points],
            train[data_indices[k]+num_points:data_indices[k]+num_points+validation_length],
            epsilon_array[k, :], 
            lambda_array[k, :],
            mode, 
            devices
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

    cv_filename = f"./numerical_results/ks_chaotic_cv_results_{mode}_{num_points}_{map_type}_vl_{validation_length}.pkl"
    with open(cv_filename, "wb") as f:
        pickle.dump(cv_results, f)



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


def batched_compute_inner_cv(
    k_train,
    k_valid,
    epsilon_array,
    lambda_array,
    mode,
    device,
):
    with cp.cuda.Device(device):

        opts["data"] = k_train
        model = general_utils.modeler(**opts)

        X = cp.asarray(model.inp, dtype=cp.float64)
        n = cp.sum(X * X, axis=1) 

        distance_matrix = X @ X.T
        distance_matrix *= -2.0
        distance_matrix += n[:, None]
        distance_matrix += n[None, :]
        cp.maximum(distance_matrix, 0.0, out=distance_matrix)
        cp.sqrt(distance_matrix, out=distance_matrix)
        cp.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)

        tau_f_array = np.zeros(cv_trials_per_device, dtype=float)
        vpt_array   = np.zeros(cv_trials_per_device, dtype=float)


        if k_valid.shape[0] > validation_horizon:
            np.random.seed(132)
            validation_indices = np.random.randint(
                k_valid.shape[0] - validation_horizon,
                size=validation_repeats,
            )
        else:
            validation_indices = np.zeros(validation_horizon, dtype=int)

        for j in range(validation_repeats):
            j_valid = k_valid[
                validation_indices[j] : validation_indices[j] + validation_horizon
            ]

            chunk_size = n_chunk_lst[device]

            for start in range(0, cv_trials_per_device, chunk_size):
                end = min(cv_trials_per_device, start + chunk_size)

                eps_chunk = epsilon_array[start:end]
                lam_chunk = lambda_array[start:end]

                # tic = time.time()
                model.fit_model(eps_chunk, lam_chunk, mode, distance_matrix)
                # toc = time.time()
                # print(f"Fitting takes {toc-tic}sec")

                # tic = time.time()
                _vpt_chunk, _tau_f_chunk = model.get_performance(
                    j_valid,
                    dt=dt,
                    Lyapunov_time=1 / Lyapunov_exp,
                    error_threshold=error_threshold,
                )
                # toc = time.time()
                # print(f"VPT takes {toc-tic}sec")

                _vpt_chunk   = cp.asnumpy(_vpt_chunk)
                _tau_f_chunk = cp.asnumpy(_tau_f_chunk)

                vpt_array[start:end]   += _vpt_chunk
                tau_f_array[start:end] += _tau_f_chunk

            free_gpu_memory()

        vpt_array   /= validation_repeats
        tau_f_array /= validation_repeats

        return tau_f_array, vpt_array




def run_test(mode):

    cv_path = f"./numerical_results/ks_chaotic_cv_results_{mode}_{num_points}_{map_type}_vl_{validation_length}.pkl"
    with open(cv_path, "rb") as f:
        cv_results = pickle.load(f)

    epsilon_array_loc = cv_results["epsilon"]
    lambda_reg_array_loc = cv_results["lambda"]
    data_indices = cv_results["data_indices"]
    score = cv_results["vpt"]
    rows = np.arange(epsilon_array_loc.shape[0])

    best_idx = np.nanargmax(score, axis=1)
    best_epsilon = epsilon_array_loc[rows, best_idx]
    best_lambda_reg = lambda_reg_array_loc[rows, best_idx]


    NT = 15_000_000
    SKP = 500_000
    DT = 0.01
    TS = 10
    data = pickle.load(open(f"./ks_utils/ksdata_chaotic_training_NT_{NT}_SKP_{SKP}_dt_{DT}_ts_{TS}.pkl", "rb"))
    dt = data["dt"]
    train = data["udata"].astype(np.float64)

    NT = 12_500_000
    SKP = 500_000
    DT = 0.01
    TS = 10
    test_load = pickle.load(open(f"./ks_utils/ksdata_chaotic_test_NT_{NT}_SKP_{SKP}_dt_{DT}_ts_{TS}.pkl", "rb"))
    uu_test = test_load["udata"].astype(np.float64)
    test = uu_test[:steps*test_trials].reshape(test_trials, steps, -1)    

    vpts_all = np.zeros((n_models, test_trials))
    vpts = np.zeros(n_models); tau_fs =  np.zeros(n_models)

    all_indices = np.arange(n_models)
    index_splits = np.array_split(all_indices, devices)

    def _run_test_block(index_block, device):

        with cp.cuda.Device(device):
            block_size = len(index_block)
            chunk_size = n_chunk_lst[device]

            vpts_block   = np.zeros(block_size)
            taufs_block  = np.zeros(block_size)
            vpts_all_block  = np.zeros((block_size, test_trials))

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

                for start in range(0, test_trials, chunk_size):
                    end = min(test_trials, start + chunk_size)
                    vpt_j, tau_f_j = model.get_performance(
                        test[start:end],
                        dt=dt,
                        Lyapunov_time=1 / Lyapunov_exp,
                        error_threshold=error_threshold,
                    )
                    vpts_all_block[local_idx, start:end] = vpt_j
                    vpt_sum  += np.sum(vpt_j)
                    tau_sum  += np.sum(tau_f_j)

                vpts_block[local_idx]   = vpt_sum / test_trials
                taufs_block[local_idx]  = tau_sum / test_trials

            free_gpu_memory()
            return index_block, vpts_block, taufs_block, vpts_all_block
        
    results = Parallel(n_jobs=devices)(
        delayed(_run_test_block)(index_splits[dev], dev)
        for dev in range(devices)
    )

    for index_block, v_block, tau_block, v_all_block in results:
        vpts[index_block]   = v_block
        tau_fs[index_block] = tau_block
        vpts_all[index_block]  = v_all_block

    free_gpu_memory()

    performance = {
        "vpts": vpts,
        "tau_fs": tau_fs,
        "vpts_all": vpts_all,
        "epsilon": best_epsilon,
        "lambda_reg": best_lambda_reg,
    }

    print(f"test mean vpt: = {np.nanmean(vpts)}")

    outp = f'./numerical_results/ks_chaotic_test_result_{mode}_{num_points}_{map_type}_vl_{validation_length}.pkl'
    with open(outp, "wb") as f:
        pickle.dump(performance, f, protocol=pickle.HIGHEST_PROTOCOL)







if __name__ == "__main__":

    num_points_lst = [2048, 4096, 8192]#, 16384]
    safety_lst = [0.3, 0.4, 0.5]
    steps = 2500
    cv_trials_per_device = 256
    devices = 4
    
    n_models = 50
    test_trials = 500
    map_type_lst = ['direct']
    mode_lst = ['diffusion', 'rbf']
    
    error_threshold = 0.5 ** 2
    Lyapunov_exp = 0.043
    

    # data load
    NT = 15_000_000
    SKP = 500_000
    DT = 0.01
    TS = 10
    data = pickle.load(open(f"./ks_utils/ksdata_chaotic_training_NT_{NT}_SKP_{SKP}_dt_{DT}_ts_{TS}.pkl", "rb"))
    dt = data["dt"]
    xx = data["x"]

    train = data["udata"].astype(np.float64)
    validation_repeats = 3
    validation_horizon = 2000
    validation_size_multiplier = 2

    validation_length = int(validation_horizon * validation_size_multiplier)
    print(f"validation_length = {validation_length}")
    
    for map_type in map_type_lst:
        opts = {
            "map_type": map_type,
            "norm": False,
            }

        for i_n, num_points in enumerate(num_points_lst):

            n_chunk_lst = []
            for dev in range(devices):
                n_chunk_lst.append(find_safe_chunk_size(num_points, device_id=dev, dtype=cp.float64,
                                    safety=safety_lst[i_n], overhead_factor=3.0))
            print(f"chunk size = {n_chunk_lst}")
            
            np.random.seed(142)
            data_indices = np.random.randint(
                    train.shape[0] - (num_points + validation_length), size=test_trials
                )
            
            print(f"  num_points={num_points}, map_type={map_type}")
                    
            for mode in mode_lst:
                print(f"mode = {mode}")

                # run_val(mode)
                # free_gpu_memory()

                run_test(mode)
                free_gpu_memory()
