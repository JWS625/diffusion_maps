import sys
from pathlib import Path
root = Path.cwd().resolve().parents[1]
sys.path.insert(0, str(root))
from diffusion_maps import model_dir, data_dir

from src.resDMD import ResDMD
import numpy as np
import time

from matplotlib import pyplot as plt

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

nx = 599
ny = 299

basepath = data_dir + "/cached_data"
H_train  = [30 * i for i in range(12)]

N_trun   = 200
N_train  = 200
dt       = 0.1
r = 81*2

model = ResDMD(basepath, N_trun=N_trun, dt=dt, ds=1,
                r=r, energy_threshold=None)
tic = time.time()
model.fit(H_train, N_train)
toc = time.time()
print(f"DMD fitting took {toc-tic:.3f} sec.")


def _snapshot_spectrum(model):
    """
    Save the current *active* spectrum of the model so we can restore it.
    """
    return {
        "Phi_":   model.Phi_.copy(),
        "lam_":   model.lam_.copy(),
        "omega_": None if model.omega_ is None else model.omega_.copy(),
        "rank_":  model.rank_,
    }

def _restore_spectrum(model, snap):
    """
    Restore a previously snapshot spectrum to the model.
    """
    model.Phi_   = snap["Phi_"]
    model.lam_   = snap["lam_"]
    model.omega_ = snap["omega_"]
    model.rank_  = snap["rank_"]

def validate_residual_order_external(
    model,
    H_valid,
    N_valid,
    orders,
    *,
    keep_conjugates=True,
    metric="rmse",
    aggregate="mean",
    apply_best=True,
):

    assert metric in ("rmse", "rel", "wrmse")
    assert aggregate in ("mean", "median", "max")

    if model.residuals_ is None:
        raise RuntimeError("Call model.compute_residuals_sako() before validation.")

    snap0 = _snapshot_spectrum(model)
    agg_fn = {"mean": np.mean, "median": np.median, "max": np.max}[aggregate]

    results = []

    for ord_val in orders:
        idx_kept, _ = model.filter_by_residual(order=ord_val,
                                               keep_conjugates=keep_conjugates)

        per_seq = []
        for H in H_valid:
            rel, rmse = model.evaluate(H, N_valid)
            per_seq.append(rmse if metric == "rmse" else rel)

        per_seq = np.asarray(per_seq, dtype=float)
        score = float(agg_fn(per_seq))

        results.append({
            "order":    ord_val,
            "score":    score,
            "per_seq":  per_seq,
            "kept":     model.rank_,
            "idx_kept": idx_kept,
        })

    best = min(results, key=lambda d: d["score"])

    _restore_spectrum(model, snap0)
    if apply_best:
        model.filter_by_residual(order=best["order"],
                                 keep_conjugates=keep_conjugates)

    return best, results

H_valid  = [45, 225, 315]
N_valid  = 200
H_test   = 135
N_test = 801

tic = time.time()
model.compute_residuals_sako()
toc = time.time()
print(f"Residual calculations took {toc-tic:.3f} sec.")

full_rank = model.Phi_full_.shape[1]
orders = ["full"] + list(range(9, 201, 2))

tic = time.time()
best, all_results = validate_residual_order_external(
    model,
    H_valid=H_valid,
    N_valid=N_valid,
    orders=orders,
    keep_conjugates=True,
    metric="rmse",
    aggregate="mean",
    apply_best=True,
)
toc = time.time()
print(f"validation took {toc-tic:.3f} sec.")


print(f"[Validation] best order={best['order']}, score={best['score']:.3e}, kept={best['kept']}")
print("Per-seq scores (H_valid):", best["per_seq"])

# final test
N_test = 80
rel, rmse = model.evaluate(H_test, N_test)
print(f"[Test {H_test}°] rel={rel:.3e}, rmse={rmse:.3e}")

H_test = 135
N_test = 801

res = model.reconstruct_sequence(H_test, N_test, add_back_mean=True)

X_test_pred  = res["Xrec"].T
X_test_truth = res["X_demeaned_truth"].T  
X_test_truth_full = X_test_truth + model.mean_vec_[:, None]
error = np.abs(X_test_truth_full - X_test_pred)

order = all_results[69]['order'] # best order by looking at the "all_results"
filename = model_dir + f"/ppplate/ppplate_ResDMD_rank_{order}.npy"
np.save(filename, X_test_pred.T)

idx_kept, _ = model.filter_by_residual(order="full",
                                               keep_conjugates=True)

markersize = None
fontsize=12

fig, ax = plt.subplots(figsize=(6, 6))
_t = np.linspace(0, 2*np.pi, 101)
ax.plot(np.cos(_t), np.sin(_t), 'k--')
l1, = ax.plot(model.lam_.real, model.lam_.imag, 'bo', markersize=markersize,  markerfacecolor='none')

best_order = 145
idx_kept, _ = model.filter_by_residual(order=best_order,
                                               keep_conjugates=True)

l2, = ax.plot(model.lam_.real, model.lam_.imag, 'rx', markersize=markersize)                                         

ax.set_aspect("equal")
ax.set_xlabel(r"$\text{Re}(\lambda)$", fontsize=fontsize)
ax.set_ylabel(r"$\text{Im}(\lambda)$", fontsize=fontsize)


xticks = [-1, -0.5, 0.0, 0.5, 1.0]
yticks = [-1, -0.5, 0.0, 0.5, 1.0]

ax.set_xticks(xticks)
ax.set_yticks(yticks)

ax.set_xticklabels(xticks)
ax.set_yticklabels(yticks)

ax.legend([l1, l2], ["Full order", r"$\kappa=145$"], loc=2, fontsize=fontsize)
fig.savefig("./pics/ppplate_spectrum.png", bbox_inches='tight', dpi=600, transparent=True)