"""Quantify how much of the discrepancy is just IC-sampling variance.

Run the script's exact code path multiple times, each time with a different
random seed for IC selection, all using num_ic=20 like the notebook. This shows
the spread of the notebook estimator. Then compare to the script's deterministic
result and to the saved notebook result.
"""
import os
import sys
import warnings
import numpy as np
import torch

BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(BENCHMARK_DIR)
sys.path.insert(0, BENCHMARK_DIR)

from dysts.metrics import estimate_kl_divergence
from dysts.analysis import gp_dim
from metrics import smape_rolling, vpt_smape

print("Loading ChronosBolt...", flush=True)
from chronos import BaseChronosPipeline

PIPE = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base", device_map="cpu", torch_dtype=torch.bfloat16,
)


def predict(traj_context_dim, horizon):
    x = torch.tensor(traj_context_dim, dtype=torch.float32)
    forecast = PIPE.predict(x, prediction_length=horizon)
    arr = forecast[0].detach().cpu().numpy()
    qs = list(PIPE.quantiles)
    return arr[qs.index(0.5)]


def benchmark_with_ics(traj, ic_indices, context_length, forecast_length, granularity):
    vpt_2_list = []
    smape_2_last_list = []
    kl_list = []
    cdim_pred_list = []
    for i in ic_indices:
        ctx = traj[i:i+context_length, :]
        true = traj[i+context_length:i+context_length+forecast_length, :]
        pred = np.zeros_like(true)
        for d in range(true.shape[1]):
            pred[:, d] = predict(ctx[:, d], forecast_length)
        vpt_2_list.append(vpt_smape(pred.squeeze(), true.squeeze()) / granularity)
        smape_2_last_list.append(np.array(smape_rolling(true, pred))[-1])
        kl = estimate_kl_divergence(true, pred)
        kl_list.append(np.nan if np.isinf(kl) else kl)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            cdim_pred_list.append(gp_dim(pred))
    return {
        "vpt_2_mean": float(np.mean(vpt_2_list)),
        "smape_2_last_mean": float(np.mean(smape_2_last_list)),
        "kl_mean": float(np.nanmean(kl_list)),
        "cdim_pred_mean": float(np.mean(cdim_pred_list)),
    }


def main():
    granularity = 30
    forecast_length = 300
    context_length = 512
    num_ic = 20
    rolling_window = 1024
    n_random_runs = 8  # bootstrap the random IC estimator

    systems = ["Lorenz", "Rossler", "Chua"]
    traj_dir = os.path.join(REPO, "data", "good_trajectories")

    for sysname in systems:
        path = os.path.join(traj_dir, f"{sysname}.npy")
        traj = np.load(path)
        if traj.ndim == 1:
            traj = traj[:, None]
        std_val = np.std(traj, axis=0)
        std_val[std_val == 0] = 1
        traj = (traj - np.mean(traj, axis=0)) / std_val

        # 1) script's deterministic ICs
        script_ics = [i for i in range(0, len(traj) - context_length - forecast_length, rolling_window) if i < rolling_window * num_ic]
        script_res = benchmark_with_ics(traj, script_ics, context_length, forecast_length, granularity)

        # 2) repeated runs with random ICs (notebook style, varying seeds)
        random_runs = []
        for seed in range(n_random_runs):
            rng = np.random.default_rng(seed=seed)
            ics = [int(rng.integers(0, len(traj) - context_length - forecast_length)) for _ in range(num_ic)]
            random_runs.append(benchmark_with_ics(traj, ics, context_length, forecast_length, granularity))

        def sp(key, runs):
            vals = np.array([r[key] for r in runs])
            return f"{vals.mean():.3f} +/- {vals.std():.3f}  (min {vals.min():.3f}, max {vals.max():.3f})"

        print(f"\n=== {sysname}, ctx={context_length}, num_ic={num_ic}, n_random_runs={n_random_runs} ===")
        print(f"  script-deterministic:  vpt={script_res['vpt_2_mean']:.3f}  smape_last={script_res['smape_2_last_mean']:.3f}  kl={script_res['kl_mean']:.3f}  cdim_pred={script_res['cdim_pred_mean']:.3f}")
        print(f"  notebook-random  vpt:        {sp('vpt_2_mean', random_runs)}")
        print(f"  notebook-random  smape_last: {sp('smape_2_last_mean', random_runs)}")
        print(f"  notebook-random  kl:         {sp('kl_mean', random_runs)}")
        print(f"  notebook-random  cdim_pred:  {sp('cdim_pred_mean', random_runs)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*prediction length.*")
    main()
