import numpy as np
import sphere_torus_utils.sphere_torus_helpers as sth
from sphere_torus_utils.torus_data_gen import generate_truncated_data_pair

# sphere
import sphere_torus_utils.torus_data_gen as tdg
import sphere_torus_utils.sphere_torus_helpers as sth
import copy
np.random.seed(72)

save_bool = False

d = 3  # ambient dimension

N_total_training = 8192
N_validation = 1000
test_total_trials = 32
test_samples = 1000
total_N_samples = N_total_training + N_validation

#test set
x0_test = sth.generateInitialConditions_(test_total_trials * test_samples, random=True)
filename_test = (f"./cached_data/sphere_test_data.npy")
if save_bool:
    np.save(filename_test, x0_test)

data_pts_lst = [N_total_training//8, N_total_training//4, N_total_training//2, N_total_training]

x_tr_ds_lst = []
y_tr_ds_lst = []

x_vl_ds_lst = []
y_vl_ds_lst = []

tr_size = [4096, 2048, 2048]
tr_size_diff = [tr_size, [2048, 1024, 1024], [1024, 512, 512], [512, 256, 256]]

np.random.seed(224)
dt = 0.01

n_pts_lst = copy.deepcopy(tr_size)
n_pts_lst[0] += N_validation //3 + 1
n_pts_lst[1] += N_validation //3 + 1
n_pts_lst[2] += N_validation //3
data_x, data_y, data_yv, x_lst, y_lst, yv_lst = generate_truncated_data_pair(total_N_samples, d=d, map_val_gap=10,  n_pts_lst=n_pts_lst, dt=dt)

for i, (xi, yi, yvi) in enumerate(zip(x_lst, y_lst, yv_lst)):
    print(f'random sampling from {xi.shape[0]} to {tr_size[i]}...')

    ind_tv = np.random.choice(xi.shape[0], tr_size[i], replace=False).tolist()
    ind_tv.sort()

    all_indices = set(range(xi.shape[0]))
    ind_val = list(all_indices - set(ind_tv))
    # Optional: Sort if order matters
    ind_val.sort()

    data_x_training, data_y_training = xi[ind_tv], yi[ind_tv]
    data_x_validation, data_y_validation = xi[ind_val], yvi[ind_val]
    x_tr_ds_lst.append(data_x_training)
    y_tr_ds_lst.append(data_y_training)

    x_vl_ds_lst.append(data_x_validation)
    y_vl_ds_lst.append(data_y_validation)


N_list = [8192, 4096, 2048, 1024]
for _ in range(4):

    _N = N_list[_]

    if _ == 0:
        _x_re1, _y_re1 = x_tr_ds_lst[0], y_tr_ds_lst[0]
        _x_re2, _y_re2 = x_tr_ds_lst[1], y_tr_ds_lst[1]
        _x_re3, _y_re3 = x_tr_ds_lst[2], y_tr_ds_lst[2]
    else:
        _ind_re1 = np.random.choice(tr_size_diff[_-1][0], tr_size_diff[_][0], replace=False).tolist()
        _ind_re2 = np.random.choice(tr_size_diff[_-1][1], tr_size_diff[_][1], replace=False).tolist()
        _ind_re3 = np.random.choice(tr_size_diff[_-1][2], tr_size_diff[_][2], replace=False).tolist()

        _x_re1, _y_re1 = _x_re1[_ind_re1], _y_re1[_ind_re1]
        _x_re2, _y_re2 = _x_re2[_ind_re2], _y_re2[_ind_re2]
        _x_re3, _y_re3 = _x_re3[_ind_re3], _y_re3[_ind_re3]

    filename_training_x = (f"cached_data/torus_training_data_x_{_N}_{d}.npy")  
    filename_training_y = (f"cached_data/torus_training_data_y_{_N}_{d}.npy")   

    data_x_training = np.concatenate([_x_re1, _x_re2, _x_re3], axis=0)
    data_y_training = np.concatenate([_y_re1, _y_re2, _y_re3], axis=0)
    
    if save_bool:
        np.save(filename_training_x, data_x_training)
        np.save(filename_training_y, data_y_training)