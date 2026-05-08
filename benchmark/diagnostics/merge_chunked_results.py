"""Merge chunked benchmark output dirs into the main model output dir.

Used after splitting one model's run across N parallel processes via
``run_dysts_benchmarks.py --output-suffix _chunkN --equations ...``. Each chunk
writes a complete set of metric .npy files containing only its assigned
equations. This script unions the per-equation entries across chunks (plus an
optional base dir holding earlier results) into a single output dir.

Usage:
    python merge_chunked_results.py --base benchmark/benchmark_results/chronos_dysts \\
        --chunks benchmark/benchmark_results/chronos_dysts_chunk{0..5}
"""
import argparse
import os
import numpy as np


METRIC_FILES = [
    "average_vpt.npy", "average_smape.npy", "average_vpt_2.npy", "average_smape_2.npy",
    "average_cdim.npy", "average_kl.npy", "average_mse.npy", "average_mae.npy",
    "median_vpt.npy", "median_smape.npy", "median_vpt_2.npy", "median_smape_2.npy",
    "median_cdim.npy", "median_kl.npy", "median_mse.npy", "median_mae.npy",
]


def _load_dict(path):
    if not os.path.exists(path):
        return {}
    return np.load(path, allow_pickle=True).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True,
                    help="Output dir to write merged dicts into. Existing entries here are kept and overwritten by chunks.")
    ap.add_argument("--chunks", nargs="+", required=True,
                    help="Chunk dirs to merge in (later chunks override earlier on key collision).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.base, exist_ok=True)

    for fname in METRIC_FILES:
        merged = _load_dict(os.path.join(args.base, fname))
        before = len(merged)
        for chunk_dir in args.chunks:
            cd = _load_dict(os.path.join(chunk_dir, fname))
            for k, v in cd.items():
                merged[k] = v
        added = len(merged) - before
        print(f"{fname}: {before} -> {len(merged)} (+{added})")
        if not args.dry_run:
            np.save(os.path.join(args.base, fname), merged)

    if args.dry_run:
        print("(dry run — no files written)")


if __name__ == "__main__":
    main()
