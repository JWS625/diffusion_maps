# Diffusion Maps

## Description
This repositories contains code implementation of the paper "Learning solution operator of dynamical systems with diffusion maps kernel ridge regression", which can be found in https://arxiv.org/abs/2512.17203.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Training/validation](#training/validation)
  - [Testing](#testing)
- [Project Structure](#project-structure)
- [Results](#results)
- [Citation](#citation)


## Installation
This project uses a Conda environment defined in `environment.yml` (Python 3.9 + NumPy/SciPy/Scikit-learn/Matplotlib, with optional GPU support via CuPy).

### 1) Create the environment
From the repository root:

conda env create -f environment.yml

### 2) Activate the environment
conda activate dm_env


## Usage
