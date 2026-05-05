"""Side-by-side comparison of notebook vs script benchmarking pipelines.

Goal: pinpoint where the discrepancy comes from by running the same model
on the same systems with controlled variations of:
  (A) IC sampling: random (notebook) vs deterministic stride (script)
  (B) Prediction extraction: forecast[0,4,:] (notebook) vs qs.index(0.5) (script)
  (C) Metric implementations: scripts.utils-style vs metrics.py
"""
import os
import sys
import warnings
import numpy as np
import torch

BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(BENCHMARK_DIR)
sys.path.insert(0, BENCHMARK_DIR)

from dysts.metrics import smape, estimate_kl_divergence
from dysts.analysis import gp_dim

# Script-style metrics
from metrics import (
    smape_rolling as smape_rolling_script,
    mse_rolling as mse_rolling_script,
    mae_rolling as mae_rolling_script,
    vpt_smape as vpt_smape_script,
)
# Notebook-style metrics (from in-repo utils.py — closest proxy for scripts.utils)
from utils import (
    smape_rolling as smape_rolling_nb,
    vpt_smape as vpt_smape_nb,
)


# Notebook-style mse/mae (redefined in cell 3 with np.mean — matches script)
def mse_nb(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_rolling_nb(ts1, ts2):
    n = min(ts1.shape[0], ts2.shape[0])
    return np.array([mse_nb(ts1[:i], ts2[:i]) for i in range(1, n + 1)])


def mae_nb(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mae_rolling_nb(ts1, ts2):
    n = min(ts1.shape[0], ts2.shape[0])
    return np.array([mae_nb(ts1[:i], ts2[:i]) for i in range(1, n + 1)])


# ---- shared model
print("Loading ChronosBolt...", flush=True)
from chronos import BaseChronosPipeline

PIPE = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)
print("Model loaded. Quantiles:", list(PIPE.quantiles), flush=True)


# ---- two prediction extraction paths
def predict_notebook_style(traj_context_dim, horizon):
    """Notebook: forecast[0,4,:].numpy() — index 4 hardcoded."""
    forecast = PIPE.predict(
        context=torch.tensor(traj_context_dim),
        prediction_length=horizon,
        limit_prediction_length=False,
    )
    return forecast[0, 4, :].numpy()


def predict_script_style(traj_context_dim, horizon):
    """Script: arr[qs.index(0.5)] via ChronosBoltModel._predict_univariate."""
    x = torch.tensor(traj_context_dim, dtype=torch.float32)
    forecast = PIPE.predict(x, prediction_length=horizon)
    arr = forecast[0].detach().cpu().numpy()  # (num_quantiles, H)
    qs = list(PIPE.quantiles)
    if 0.5 in qs:
        return arr[qs.index(0.5)]
    return arr[arr.shape[0] // 2]


# ---- IC samplers
def ics_notebook(traj_len, context_length, forecast_length, num_ic, rng):
    """Random uniform sampling (notebook)."""
    return [rng.integers(0, traj_len - context_length - forecast_length) for _ in range(num_ic)]


def ics_script(traj_len, context_length, forecast_length, num_ic, rolling_window):
    """Deterministic stride (script)."""
    out = []
    for i in range(0, traj_len - context_length - forecast_length, rolling_window):
        if i < rolling_window * num_ic:
            out.append(i)
    return out


# ---- driver
def run_one(traj, context_length, forecast_length, granularity, ic_indices,
            predict_fn, smape_rolling_fn, vpt_smape_fn, mse_rolling_fn, mae_rolling_fn):
    per_ic = []
    for i in ic_indices:
        traj_context = traj[i : i + context_length, :]
        traj_true = traj[i + context_length : i + context_length + forecast_length, :]
        traj_pred = np.zeros_like(traj_true)
        for dim in range(traj_true.shape[1]):
            traj_pred[:, dim] = predict_fn(traj_context[:, dim], forecast_length)

        # per-channel
        vpt_per_dim = []
        smape_last_per_dim = []
        for d in range(traj_true.shape[1]):
            vpt_per_dim.append(vpt_smape_fn(traj_pred[:, d], traj_true[:, d]) / granularity)
            smape_last_per_dim.append(smape_rolling_fn(traj_true[:, d], traj_pred[:, d])[-1])
        # joint
        vpt_2 = vpt_smape_fn(traj_pred.squeeze(), traj_true.squeeze()) / granularity
        smape_2_last = np.array(smape_rolling_fn(traj_true, traj_pred))[-1]
        kl = estimate_kl_divergence(traj_true, traj_pred)
        if np.isinf(kl):
            kl = np.nan
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            cdim_pred = gp_dim(traj_pred)
            cdim_true = gp_dim(traj_true)

        per_ic.append({
            "i": i,
            "vpt_mean_dim": float(np.mean(vpt_per_dim)),
            "smape_last_mean_dim": float(np.mean(smape_last_per_dim)),
            "vpt_2": float(vpt_2),
            "smape_2_last": float(smape_2_last),
            "kl": float(kl) if not np.isnan(kl) else np.nan,
            "cdim_pred": float(cdim_pred),
            "cdim_true": float(cdim_true),
            "pred_first10": traj_pred[:10, 0].tolist(),
        })
    return per_ic


def aggregate(per_ic):
    if not per_ic:
        return {}
    keys = ["vpt_mean_dim", "smape_last_mean_dim", "vpt_2", "smape_2_last", "kl",
            "cdim_pred", "cdim_true"]
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in per_ic], dtype=float)
        out[f"{k}_mean"] = float(np.nanmean(vals))
        out[f"{k}_median"] = float(np.nanmedian(vals))
    return out


def main():
    granularity = 30
    forecast_length = 300
    context_length = 512  # one context length to keep runtime bounded
    num_ic = 5  # small but enough to see agreement / disagreement
    rolling_window = 1024  # script uses context_lengths[-1] = 1024

    systems = ["Lorenz", "Rossler", "Chua"]
    traj_dir = os.path.join(REPO, "data", "good_trajectories")

    rng = np.random.default_rng(seed=12345)  # seeded so notebook-style is reproducible

    summary = {}
    for sysname in systems:
        path = os.path.join(traj_dir, f"{sysname}.npy")
        traj = np.load(path)
        if traj.ndim == 1:
            traj = traj[:, None]
        std_val = np.std(traj, axis=0)
        std_val[std_val == 0] = 1
        traj = (traj - np.mean(traj, axis=0)) / std_val
        print(f"\n=== {sysname}  traj.shape={traj.shape} ===", flush=True)

        ics_nb = ics_notebook(len(traj), context_length, forecast_length, num_ic, rng)
        ics_sc = ics_script(len(traj), context_length, forecast_length, num_ic, rolling_window)
        print(f"  notebook ICs (random, seeded): {ics_nb}")
        print(f"  script   ICs (deterministic):  {ics_sc}")

        # CONFIG A — notebook code path on notebook ICs
        A = run_one(traj, context_length, forecast_length, granularity, ics_nb,
                    predict_notebook_style, smape_rolling_nb, vpt_smape_nb,
                    mse_rolling_nb, mae_rolling_nb)
        # CONFIG B — script code path on script ICs
        B = run_one(traj, context_length, forecast_length, granularity, ics_sc,
                    predict_script_style, smape_rolling_script, vpt_smape_script,
                    mse_rolling_script, mae_rolling_script)
        # CONFIG C — script code path on notebook ICs (isolates IC effect)
        C = run_one(traj, context_length, forecast_length, granularity, ics_nb,
                    predict_script_style, smape_rolling_script, vpt_smape_script,
                    mse_rolling_script, mae_rolling_script)
        # CONFIG D — notebook predict on script ICs (isolates predict-fn effect)
        D = run_one(traj, context_length, forecast_length, granularity, ics_sc,
                    predict_notebook_style, smape_rolling_nb, vpt_smape_nb,
                    mse_rolling_nb, mae_rolling_nb)

        summary[sysname] = {"A_nbAll": A, "B_scriptAll": B,
                            "C_scriptCode_nbICs": C, "D_nbCode_scriptICs": D}

        for tag, runs in [("A nb code+nb ICs ", A), ("B script code+script ICs", B),
                          ("C script code+nb ICs", C), ("D nb code+script ICs ", D)]:
            agg = aggregate(runs)
            print(f"  [{tag}] vpt_mean={agg['vpt_mean_dim_mean']:.3f}  "
                  f"smape_last_mean={agg['smape_last_mean_dim_mean']:.2f}  "
                  f"kl_mean={agg['kl_mean']:.3f}  "
                  f"cdim_pred_mean={agg['cdim_pred_mean']:.2f}")

        # bit-identical predictions check (A vs D differ only in IC; C vs B differ only in IC)
        # predictions on the SAME IC index between code paths:
        # Compare predict_notebook_style vs predict_script_style on the FIRST notebook IC for dim 0
        i0 = ics_nb[0]
        ctx0 = traj[i0 : i0 + context_length, 0]
        p_nb = predict_notebook_style(ctx0, forecast_length)
        p_sc = predict_script_style(ctx0, forecast_length)
        diff = np.max(np.abs(p_nb.astype(np.float64) - p_sc.astype(np.float64)))
        print(f"  same-IC prediction max|nb-script| = {diff:.3e}  "
              f"(notebook[0:5]={p_nb[:5]}, script[0:5]={p_sc[:5]})")

    out_path = os.path.join(os.path.dirname(__file__), "compare_pipelines_results.npz")
    np.savez(out_path, summary=np.array(summary, dtype=object))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
