# Context parroting

Context parroting and time series foundation models for zero-shot forecasting of
dynamical systems.

Context parroting is an informative baseline for time series foundation models. It
generates naive forecasts by copying contiguous sequences of context points.

## Repository layout

| Path | What's in it |
|---|---|
| `models/` | The context parroting model (`parrot.py`), including the simplex variant. |
| `demo/` | Minimal end-to-end demo of context parroting on a Lorenz trajectory (`context_parroting_demo.ipynb`). |
| `data/` | Test datasets used by the benchmarks. See [Datasets](#datasets) below. |
| `benchmark/` | Benchmarking driver, model registry, metrics, benchmark notebooks, and results. See [Benchmarks](#benchmarks). |
| `analysis/` | Figure notebooks and the statistics they read; see [`analysis/README.md`](analysis/README.md). |

Superseded material lives in `benchmark/legacy/` and `analysis/legacy/`, each
with a README explaining what it is and what replaced it.

## Datasets

The dysts zero-shot forecasting dataset was studied in earlier work. To avoid data
fragmentation, we point to the
[original source files](https://github.com/GilpinLab/dysts_data/tree/main/benchmark_results/zero-shot/chronos_benchmarks_context_512_granularity_30)
associated with that paper.

`data/` includes:

| Subdirectory | Contents |
|---|---|
| `good_trajectories/` | High-resolution dysts trajectories (100k points, 30 pts/period). Used by `run_dysts_benchmarks.py`. |
| `long_trajectories/` | Longer/older dysts trajectories. |
| `trajectories/` | Shorter (10k) dysts trajectories. |
| `electrocardiogram/`, `electronic_circuit/`, `von_karman_street/`, `kuramoto/` | Real-world and physical-system test datasets. |

## Benchmarks

The unified driver runs any model in the registry against the full dysts dataset:

```bash
# Run all models
python benchmark/run_dysts_benchmarks.py

# Run specific models at a single context length
python benchmark/run_dysts_benchmarks.py chronos timemoe --context-lengths 512

# Split one slow model across parallel processes
python benchmark/run_dysts_benchmarks.py chronos --equations Lorenz Rossler --output-suffix _chunk0

# Available models: arima, chronos, chronos_bolt, dynamix, moirai2, panda_patchtst,
#                   parrot, simplex, timemoe, timesfm (2.0, paper version), timesfm_2p5
```

Results are written to `benchmark/benchmark_results/<model>_dysts/` as
`{average,median}_{vpt,smape,vpt_2,smape_2,cdim,kl,mse,mae}.npy`, each a dict
keyed by `(equation_name, context_length)`.
[`benchmark/forecast_models.py`](benchmark/forecast_models.py) is the model registry —
add a new model by subclassing `ForecastModel` and registering it. DynaMix
additionally requires cloning
[DynaMix-python](https://github.com/DurstewitzLab/DynaMix-python) into
`benchmark/DynaMix/`.

The benchmarks can also be run with `uv`:

```bash
uv run python benchmark/run_dysts_benchmarks.py
```

### Conventions (important for comparing numbers)

- **sMAPE** is on the 0–200 scale (predicting the mean of white noise gives
  ~200), defined locally in [`benchmark/metrics.py`](benchmark/metrics.py).
  It is deliberately *not* imported from dysts, whose default switched to a
  0–100 scale in April 2025 — numbers produced against the wrong convention
  are silently half-sized (see `analysis/legacy/README.md` for a cautionary
  example).
- **Initial conditions**: for each `(equation, context_length)` pair the driver
  samples `num_ic` (default 20) starts uniformly across the full trajectory
  with a deterministic per-pair seed, so runs are reproducible and unbiased.

### Contents of `benchmark/`

| Item | Purpose |
|---|---|
| `run_dysts_benchmarks.py` | The benchmark driver (see `--help`). |
| `forecast_models.py` | Model registry with a unified `forecast(context, horizon)` interface. |
| `metrics.py`, `utils.py` | sMAPE/VPT/MSE/MAE metric implementations. |
| `benchmark_results/` | Driver output: the Fig. 2 data (all seven models, CL=512, unified protocol). |
| `benchmark_plotting.ipynb` | Comparison figures from `benchmark_results/`. |
| `benchmark_SciML.ipynb`, `benchmark_Dynamix_SciML.ipynb`, `benchmark_Moirai_SciML.ipynb` | Benchmarks on the SciML datasets (turbulence, ECG, circuits, ...). |
| `benchmark_vary_context.ipynb`, `benchmark_long_rollouts.ipynb`, `benchmark_longhorizon.ipynb`, `invariant_properties_longhorizons.ipynb` | Context-length scaling, long-rollout, and invariant-property analyses. |
| `diagnostics/` | Validation scripts: notebook-vs-driver equivalence, IC-sampling variance, chunk merging. |
| `legacy/` | Superseded per-model notebooks and the previous generation of results. |

## Reference

Information about the experiments can be found in the accompanying paper.

```bibtex
@inproceedings{zhang2026context,
  title={Context parroting: A simple but tough-to-beat baseline for foundation models in scientific machine learning},
  author={Yuanzhao Zhang and William Gilpin},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```
