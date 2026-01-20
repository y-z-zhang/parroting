import glob
import os
from typing import Dict, Iterable

import numpy as np
from dysts.analysis import gp_dim
from dysts.metrics import estimate_kl_divergence

from forecast_models import create_model, list_models
from metrics import mae_rolling, mse_rolling, smape_rolling, vpt_smape


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _init_stats():
    return {
        "vpt": {"average": {}, "median": {}},
        "smape": {"average": {}, "median": {}},
        "vpt_2": {"average": {}, "median": {}},
        "smape_2": {"average": {}, "median": {}},
        "cdim": {"average": {}, "median": {}},
        "kl": {"average": {}, "median": {}},
        "mse": {"average": {}, "median": {}},
        "mae": {"average": {}, "median": {}},
    }


def _init_metric_buffers():
    return {
        "vpt": [],
        "smape": [],
        "vpt_2": [],
        "smape_2": [],
        "cdim": [],
        "kl": [],
        "mse": [],
        "mae": [],
    }


def _store_metric(stats_dict, metric, key, values, axis=None, nan=False):
    if len(values) == 0:
        stats_dict[metric]["average"][key] = np.array([])
        stats_dict[metric]["median"][key] = np.array([])
        return
    arr = np.array(values)
    if nan:
        avg = np.nanmean(arr, axis=axis)
        med = np.nanmedian(arr, axis=axis)
    else:
        avg = np.mean(arr, axis=axis)
        med = np.median(arr, axis=axis)
    stats_dict[metric]["average"][key] = avg
    stats_dict[metric]["median"][key] = med


def _save_stats(output_dir: str, stats: Dict[str, Dict[str, Dict]]):
    os.makedirs(output_dir, exist_ok=True)
    for metric_name, buckets in stats.items():
        for agg_name, data in buckets.items():
            np.save(os.path.join(output_dir, f"{agg_name}_{metric_name}.npy"), data)


def run_model_benchmarks(
    model_name: str,
    trajectory_glob: str,
    context_lengths: np.ndarray,
    forecast_length: int,
    rolling_window: int,
    num_ic: int,
    granularity: int,
    output_dir: str,
):
    stats = _init_stats()
    model = create_model(model_name)

    for trajectory in sorted(glob.glob(trajectory_glob)):
        equation_name = os.path.basename(trajectory).split(".")[0]
        print(f"{model_name}: {equation_name}", flush=True)

        traj = np.load(trajectory, allow_pickle=True)
        traj = (traj - np.mean(traj, axis=0)) / np.std(traj, axis=0)

        try:
            for context_length in context_lengths:
                metrics = _init_metric_buffers()

                for i in range(
                    0, len(traj) - context_length - forecast_length, rolling_window
                ):
                    if i < rolling_window * num_ic:
                        traj_context = traj[i : i + context_length, :]
                        traj_true = traj[
                            i + context_length : i + context_length + forecast_length, :
                        ]

                        traj_pred_full = model.forecast(traj_context, forecast_length)
                        traj_pred_full = _ensure_2d(traj_pred_full)

                        for dim in range(traj_true.shape[1]):
                            traj_pred = traj_pred_full[:, dim]

                            vpt = vpt_smape(traj_pred, traj_true[:, dim]) / granularity
                            smape_val = np.array(
                                smape_rolling(traj_true[:, dim], traj_pred)
                            )

                            metrics["vpt"].append(vpt)
                            metrics["smape"].append(smape_val)

                            mse_val = mse_rolling(traj_true[:, dim], traj_pred)
                            metrics["mse"].append(mse_val)

                            mae_val = mae_rolling(traj_true[:, dim], traj_pred)
                            metrics["mae"].append(mae_val)

                        vpt = (
                            vpt_smape(traj_pred_full.squeeze(), traj_true.squeeze())
                            / granularity
                        )
                        smape_val = np.array(smape_rolling(traj_true, traj_pred_full))
                        metrics["vpt_2"].append(vpt)
                        metrics["smape_2"].append(smape_val)

                        kl_dist = estimate_kl_divergence(traj_true, traj_pred_full)
                        if np.isinf(kl_dist):
                            kl_dist = np.nan
                        metrics["kl"].append(kl_dist)

                        cdim_pred = gp_dim(traj_pred_full)
                        cdim_true = gp_dim(traj_true)
                        metrics["cdim"].append(np.array([cdim_pred, cdim_true]))

                key = (equation_name, context_length)
                _store_metric(stats, "vpt", key, metrics["vpt"])
                _store_metric(stats, "smape", key, metrics["smape"], axis=0)
                _store_metric(stats, "vpt_2", key, metrics["vpt_2"])
                _store_metric(stats, "smape_2", key, metrics["smape_2"], axis=0)
                _store_metric(stats, "cdim", key, metrics["cdim"], axis=0)
                _store_metric(stats, "kl", key, metrics["kl"], nan=True)
                _store_metric(stats, "mse", key, metrics["mse"], axis=0)
                _store_metric(stats, "mae", key, metrics["mae"], axis=0)

        except Exception as exc:
            print(exc)
            print(f"{model_name}: skipping {equation_name}", flush=True)
            continue

    _save_stats(output_dir, stats)


def main():
    granularity = 30
    context_lengths = 2 ** np.arange(6, 11)
    forecast_length = 300
    rolling_window = context_lengths[-1]
    num_ic = 20

    trajectory_glob = "../data/long_trajectories/*.npy"

    for model_name in list_models():
        output_dir = f"./benchmark_results/{model_name}_dysts"
        run_model_benchmarks(
            model_name=model_name,
            trajectory_glob=trajectory_glob,
            context_lengths=context_lengths,
            forecast_length=forecast_length,
            rolling_window=rolling_window,
            num_ic=num_ic,
            granularity=granularity,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
