import numpy as np
import sphere_torus_utils.sphere_torus_helpers as sth
from sphere_torus_utils.torus_data_gen import generate_truncated_data_pair
from matplotlib import pyplot as plt

# sphere
import sphere_torus_utils.torus_data_gen as tdg
import sphere_torus_utils.sphere_torus_helpers as sth
import copy
#np.random.seed(1442)
np.random.seed(72)

save_bool = False

d = 15  # ambient dimension

N_total_training = 8192
N_validation = 1000
test_total_trials = 32
test_samples = 1000
total_N_samples = N_total_training + N_validation

data_pts_lst = [N_total_training//8, N_total_training//4, N_total_training//2, N_total_training]

# x0_test = sth.generateInitialConditions_(test_total_trials * test_samples, random=True)

#test set
# filename_test = (f"cached_data/sphere_test_data.npy")

# if save_bool:
#     np.save(filename_test, x0_test)


# print(f"Total N samples = {total_N_samples}.")



x_tr_ds_lst = []
y_tr_ds_lst = []

x_vl_ds_lst = []
y_vl_ds_lst = []

#if d == 3:
tr_size = [4096, 2048, 2048]
tr_size_diff = [tr_size, [2048, 1024, 1024], [1024, 512, 512], [512, 256, 256]]

#elif d == 7:
#    tr_size = [910, 3641, 3641]
#    tr_size_diff = [tr_size, [452, 1820, 1820], [228, 910, 910], [114, 455, 455]]

#elif d == 15:
#    tr_size = [910, 3641, 3641]
#    tr_size_diff = [tr_size, [452, 1820, 1820], [228, 910, 910], [114, 455, 455]]

np.random.seed(224)
dt = 0.01
#data_total, f_total = generate_truncated_data(total_N_samples, d=d)
n_pts_lst = copy.deepcopy(tr_size)
n_pts_lst[0] += N_validation //3 + 1
n_pts_lst[1] += N_validation //3 + 1
n_pts_lst[2] += N_validation //3
data_x, data_y, data_yv, x_lst, y_lst, yv_lst = generate_truncated_data_pair(total_N_samples, d=d, map_val_gap=10,  n_pts_lst=n_pts_lst, dt=dt)


print(f'n_pts_lst = {n_pts_lst}')
print(f'tr_size = {tr_size}')
for i, (xi, yi, yvi) in enumerate(zip(x_lst, y_lst, yv_lst)):
    print(f'random sampling from {xi.shape[0]} to {tr_size[i]}...')
    print(xi.shape)
    #print(yi.shape)
    #print(yvi.shape)
    ind_tv = np.random.choice(xi.shape[0], tr_size[i], replace=False).tolist()
    ind_tv.sort()

    all_indices = set(range(xi.shape[0]))
    ind_val = list(all_indices - set(ind_tv))
    # Optional: Sort if order matters
    ind_val.sort()

    data_x_training, data_y_training = xi[ind_tv], yi[ind_tv]
    data_x_validation, data_y_validation = xi[ind_val], yvi[ind_val]
    print(f'region = {i+1} training x shape = {data_x_training.shape}')
    print(f'region = {i+1} validation x shape = {data_x_validation.shape}')

    x_tr_ds_lst.append(data_x_training)
    y_tr_ds_lst.append(data_y_training)

    x_vl_ds_lst.append(data_x_validation)
    y_vl_ds_lst.append(data_y_validation)


#data_x_training = np.concatenate(x_tr_ds_lst, axis=0)
#data_y_training = np.concatenate(y_tr_ds_lst, axis=0)
#data_x_validation = np.concatenate(x_vl_ds_lst, axis=0)
#data_y_validation = np.concatenate(y_vl_ds_lst, axis=0)


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

    print(data_x_training.shape)
    print(data_y_training.shape)

    if save_bool:
        np.save(filename_training_x, data_x_training)
        np.save(filename_training_y, data_y_training)

#ind_2 = np.random.choice(N_total_training, N_total_training//2, replace=False).tolist()
#ind_4 = np.random.choice(N_total_training//2, N_total_training//4, replace=False).tolist()
#ind_8 = np.random.choice(N_total_training//4, N_total_training//8, replace=False).tolist()

#data_x_training_2, data_y_training_2 = data_x_training[ind_2, :], data_y_training[ind_2, :]
#data_x_training_4, data_y_training_4 = data_x_training_2[ind_4, :], data_y_training_2[ind_4, :]
#data_x_training_8, data_y_training_8 = data_x_training_4[ind_8, :], data_y_training_4[ind_8, :]



# training set
#filename_training_x_8192 = (f"cached_data/torus_training_data_x_8192_{d}.npy")
#filename_training_y_8192 = (f"cached_data/torus_training_data_y_8192_{d}.npy")

#filename_training_x_4096 = (f"cached_data/torus_training_data_x_4096_{d}.npy")
#filename_training_y_4096 = (f"cached_data/torus_training_data_y_4096_{d}.npy")

#filename_training_x_2048 = (f"cached_data/torus_training_data_x_2048_{d}.npy")
#filename_training_y_2048 = (f"cached_data/torus_training_data_y_2048_{d}.npy")

#filename_training_x_1024 = (f"cached_data/torus_training_data_x_1024_{d}.npy")
#filename_training_y_1024 = (f"cached_data/torus_training_data_y_1024_{d}.npy")


#if save_bool:
#    # 1024 samples
#    np.save(filename_training_x_1024, data_x_training_8)
#    np.save(filename_training_y_1024, data_y_training_8)
#
#    # 2048 samples
#    np.save(filename_training_x_2048, data_x_training_4)
#    np.save(filename_training_y_2048, data_y_training_4)
#
#    # 4096 samples
#    np.save(filename_training_x_4096, data_x_training_2)
#    np.save(filename_training_y_4096, data_y_training_2)
#
#    # 8192 samples
#    np.save(filename_training_x_8192, data_x_training)
#    np.save(filename_training_y_8192, data_y_training)

# validation set
# data_x_validation = np.concatenate(x_vl_ds_lst, axis=0)
# data_y_validation = np.concatenate(y_vl_ds_lst, axis=0)
# filename_validation_x = (f"cached_data/torus_validation_data_x_{N_validation}_{d}_dt_{dt}.npy")
# filename_validation_y = (f"cached_data/torus_validation_data_y_{N_validation}_{d}_dt_{dt}.npy")

# if save_bool:
#     np.save(filename_validation_x, data_x_validation)
#     np.save(filename_validation_y, data_y_validation)

steps = 2000
_x0 = np.load(f'./cached_data/sphere_test_data.npy')[:1000]
print(f'_x0.shape = {_x0.shape}')
truePath = sth.generateData_(_x0, steps * dt, dt)
# convert to torus
truePath, theta, phi = sth.changetotorus(truePath)
truePath = tdg.map_to_torus_(theta, phi, d)
# x0t = truePath[0]
print(f"truePath.shape = {truePath.shape}")

idx = 0
fig = plt.figure(figsize=(15, 15))
for i in range(3):
    ax = fig.add_subplot(2,2,i+1,projection='3d')
    ax.scatter(x_tr_ds_lst[i][:, 3*idx], x_tr_ds_lst[i][:, 3*idx+1], x_tr_ds_lst[i][:, 3*idx+2], facecolors='none', edgecolors='blue',s=10,zorder=1)
    ax.scatter(y_tr_ds_lst[i][:, 3*idx], y_tr_ds_lst[i][:, 3*idx+1], y_tr_ds_lst[i][:, 3*idx+2], facecolors='none', edgecolors='red',s=10,zorder=1)
    #ax.scatter(data_x_training[:, 3*i], data_x_training[:, 3*i+1], data_x_training[:, 3*i+2], facecolors='none', edgecolors='red',s=10,zorder=1)
    #ax.scatter(data_y_training[:, 0], data_y_training[:, 1], data_y_training[:, 2], facecolors='none', edgecolors='blue',s=10, zorder=1)
    m =3
    for _ in range(m):
        ax.scatter(truePath[_, 0], truePath[_, 1], truePath[_, 2], color='black', s=40, zorder=2)
ax = fig.add_subplot(2,2,4,projection='3d')
for i in range(3):
    ax.scatter(x_tr_ds_lst[i][:, 3*idx], x_tr_ds_lst[i][:, 3*idx+1], x_tr_ds_lst[i][:, 3*idx+2], facecolors='none', edgecolors='blue',s=10,zorder=1)
    ax.scatter(y_tr_ds_lst[i][:, 3*idx], y_tr_ds_lst[i][:, 3*idx+1], y_tr_ds_lst[i][:, 3*idx+2], facecolors='none', edgecolors='red',s=10,zorder=1)
#ax.view_init(elev=20,azim=120) 
# fig.savefig(f'./pics/torus_{d}d_{idx}_dt_{dt}.png', bbox_inches='tight', dpi=500)
