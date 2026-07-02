"""Parrot embedding-dimension sweep over context lengths (Figs. 4, 9, 10).

Re-runs the context-parroting benchmark for several embedding dimensions D and
context lengths 2^6..2^16, saving per-(equation, context_length) aggregates of
VPT, rolling sMAPE, and the matching-motif distance (min L2). Replaces the
earlier sweep that was generated with a half-scale sMAPE (see
analysis/legacy/README.md); this version uses the repo-pinned 0-200 sMAPE from
benchmark/metrics.py.

Output format matches the historical files, one set per D:
    D={D}_{average,median}_{vpt,smape,l2}.npy   dicts keyed (equation, CL)

Usage:
    python benchmark/run_parrot_D_sweep.py --D 5 10 15 20 25 30 \
        --num-ic 100 --seed 0 --n-jobs 10
"""
import argparse
import glob
import os
import sys

import numpy as np
from joblib import Parallel, delayed

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
for p in (_here, _root):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.parrot import context_parroting_forecast  # noqa: E402


def smape_rolling_fast(y_true, y_pred, eps=1e-10):
    """Rolling 0-200 sMAPE via prefix means; O(T) instead of O(T^2).

    Identical to applying benchmark.metrics.smape to every prefix.
    """
    r = np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)
    return 200 * np.cumsum(r) / np.arange(1, len(r) + 1)


def vpt_from_rolling(rolling, granularity, threshold=30):
    exceed = np.nonzero(rolling > threshold)[0]
    tind = exceed[0] if len(exceed) else len(rolling)
    return tind / granularity


def process_equation(traj_path, D, context_lengths, forecast_length, num_ic,
                     seed, granularity):
    eq = os.path.basename(traj_path).split('.')[0]
    traj = np.load(traj_path, allow_pickle=True)
    if not isinstance(traj, np.ndarray) or traj.ndim == 0 or traj.size == 0:
        return eq, None
    if traj.ndim == 1:
        traj = traj[:, None]
    std = np.std(traj, axis=0)
    std[std == 0] = 1
    traj = (traj - np.mean(traj, axis=0)) / std

    out = {}
    for cl in context_lengths:
        max_start = len(traj) - cl - forecast_length
        if max_start <= 0:
            continue
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, max_start, size=num_ic)
        vpts, curves, l2s = [], [], []
        for i in starts:
            i = int(i)
            ctx = traj[i:i + cl]
            true = traj[i + cl:i + cl + forecast_length]
            for d in range(true.shape[1]):
                _, min_l2, pred = context_parroting_forecast(
                    ctx[:, d], D=D, forecast_total_length=forecast_length)
                rolling = smape_rolling_fast(true[:, d], pred)
                vpts.append(vpt_from_rolling(rolling, granularity))
                curves.append(rolling)
                l2s.append(min_l2)
        curves = np.array(curves)
        out[cl] = {
            'average_vpt': float(np.mean(vpts)),
            'median_vpt': float(np.median(vpts)),
            'average_smape': np.mean(curves, axis=0),
            'median_smape': np.median(curves, axis=0),
            'average_l2': float(np.mean(l2s)),
            'median_l2': float(np.median(l2s)),
        }
    return eq, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--D', nargs='+', type=int, default=[5, 10, 15, 20, 25, 30])
    ap.add_argument('--context-lengths', nargs='+', type=int,
                    default=list(2 ** np.arange(6, 17)))
    ap.add_argument('--forecast-length', type=int, default=300)
    ap.add_argument('--num-ic', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--granularity', type=int, default=30)
    ap.add_argument('--n-jobs', type=int, default=10)
    ap.add_argument('--output-dir', default=os.path.join(_root, 'analysis', 'parrot_D_sweep'))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    traj_paths = sorted(glob.glob(os.path.join(_root, 'data', 'good_trajectories', '*.npy')))
    print(f'{len(traj_paths)} systems, D={args.D}, CLs={args.context_lengths}', flush=True)

    for D in args.D:
        print(f'=== D={D} ===', flush=True)
        results = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(process_equation)(p, D, args.context_lengths,
                                      args.forecast_length, args.num_ic,
                                      args.seed, args.granularity)
            for p in traj_paths
        )
        stats = {k: {} for k in ['average_vpt', 'median_vpt', 'average_smape',
                                 'median_smape', 'average_l2', 'median_l2']}
        for eq, out in results:
            if out is None:
                print(f'D={D}: skipped {eq} (invalid trajectory)', flush=True)
                continue
            for cl, metrics in out.items():
                for k, v in metrics.items():
                    stats[k][(eq, cl)] = v
        for k, d in stats.items():
            np.save(os.path.join(args.output_dir, f'D={D}_{k}.npy'), d)
        print(f'D={D}: saved ({len(stats["average_vpt"])} (eq, CL) entries)', flush=True)


if __name__ == '__main__':
    main()
