# Diffusion Maps

## Description
This repositories contains code implementation of the paper "Learning solution operator of dynamical systems with diffusion maps kernel ridge regression", which can be found in https://arxiv.org/abs/2512.17203.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Training/validation](#training/validation)
  - [Testing](#testing)
- [Citation](#citation)


## Installation
This project uses a Conda environment defined in `environment.yml` (Python 3.9 + NumPy/SciPy/Scikit-learn/Matplotlib, with optional GPU support via CuPy).

### 1) Create the environment
From the repository root:

```text
conda env create -f environment.yml
```

### 2) Activate the environment
```text
conda activate dm_env
```


## Usage
In each test case script, there are usually two stages; training/validation and test.
For KRR model, training simply denotes finding kernel coefficients through kernel matrix inversion (done by Cholesky decomposition)

```text
diffusion_maps/
├── data/
│   ├── cached_data/
│   │   ├── ksdata_chaotic_test_NT_12500000_SKP_500000_dt_0.01_ts_10.pkl
│   │   ├── ksdata_chaotic_training_NT_15000000_SKP_500000_dt_0.01_ts_10.pkl
│   │   ├── ksdata_traveling_NT_5000000_SKP_500000_dt_0.001_ts_10.pkl
│   │   ├── ksdataBeatingTraveling.mat
│   │   ├── lorenz_test.npy
│   │   ├── lorenz_train.npy
│   ├── figs/
│   │   └── l63_train.png
│   ├── sphere_torus_utils/
│   │   ├── __init__.py
│   │   ├── sphere_data_gen.py
│   │   ├── sphere_torus_data_gen.py
│   │   ├── sphere_torus_helpers.py
│   │   └── torus_data_gen.py
│   ├── dg_ks_chaotic.py
│   ├── dg_ks_traveling.py
│   ├── dg_l63.py
│   └── dg_torus.py
├── external/
│   ├── data/
│   │   └── deepskip_best_vpts.csv
│   ├── envs/
│   │   ├── candyman_environment.yml
│   │   ├── ldnet_environment.yml
│   │   └── node_environment.yml
│   ├── ks_travel_candyman.py
│   ├── ks_travel_ldnet.py
│   └── ks_travel_node.py
├── src/
│   ├── dm_main.py
│   ├── krr_model.py
│   ├── resDMD.py
│   └── utils.py
├── test/
│   ├── pics/
│   │   ├── lorenz_3d_violin_skip-connection.png
│   │   ├── lorenz_cv_results.png
│   │   ├── torus_cv_results.png
│   │   └── torus_test_rmse.png
│   ├── ks_chaotic_test_case.py
│   ├── ks_figures.ipynb
│   ├── ks_traveling_test_case.py
│   ├── lorenz_figures.ipynb
│   ├── lorenz_test_case.py
│   ├── lorenz_test_case_sensitivity.py
│   ├── ppplate.ipynb
│   ├── ppplate_test_case.py
│   ├── torus_figures.ipynb
│   └── torus_test_case.py
├── trained_mdls/
│   ├── ks_traveling/
│   ├── l63/
│   └── torus/
│
├── .gitignore
├── __init__.py
├── environment.yml
└── README.md
```

