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
| `benchmark/` | Benchmarking driver, model registry, metrics, and analysis notebooks. See [Benchmarks](#benchmarks). |
| `analysis/` | Saved benchmark statistics and figures used in the paper. |

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

# Run specific models only
python benchmark/run_dysts_benchmarks.py chronos timemoe

# Available models: arima, chronos, chronos_bolt, dynamix, moirai2, panda_patchtst,
#                   parrot, simplex, timemoe, timesfm
```

Results are written to `benchmark/benchmark_results/<model>_dysts/`.
[`benchmark/forecast_models.py`](benchmark/forecast_models.py) is the model registry —
add a new model by subclassing `ForecastModel` and registering it.

The benchmarks can also be run with `uv`, which is recommended given the
overlapping-but-incompatible dependencies of the various foundation models:

```bash
uv run python benchmark/run_dysts_benchmarks.py
```

### Initial-condition sampling

For each `(equation, context_length)` pair, the driver samples `num_ic=20` random
initial-condition starts uniformly across the full trajectory, using a deterministic
seed so runs are reproducible. To change the seed or number of ICs, edit `main()` in
[`benchmark/run_dysts_benchmarks.py`](benchmark/run_dysts_benchmarks.py).

### Notebooks under `benchmark/`

| Notebook | Purpose |
|---|---|
| `benchmark_plotting.ipynb` | Generates the comparison figures from saved `benchmark_results/`. |
| `benchmark_ChronosBolt_normalized.ipynb` | Reference implementation of the ChronosBolt benchmark loop (matches the script's results when seeded the same way — see [`diagnostics/`](benchmark/diagnostics/)). |
| `benchmark_Dynamix_dysts_forecast_models.ipynb` | DynaMix benchmark using the unified `forecast_models.py` interface. |
| `benchmark_arima.ipynb`, `benchmark_panda.ipynb`, `benchmark_parrot_dysts.ipynb`, `benchmark_simplex.ipynb` | Per-model exploratory notebooks. |
| `benchmark_SciML.ipynb`, `benchmark_Dynamix_SciML.ipynb`, `benchmark_Moirai_SciML.ipynb` | Benchmarks on SciML datasets (not covered by the dysts driver). |
| `benchmark_vary_context.ipynb`, `benchmark_long_rollouts.ipynb`, `benchmark_longhorizon.ipynb`, `invariant_properties_longhorizons.ipynb` | Special-purpose analyses. |
| `legacy/` | Older per-model `_dysts.ipynb` notebooks predating the unified driver. Kept for reference. |
| `diagnostics/` | Scripts that validate the script and notebook pipelines compute the same metrics on identical inputs, and quantify IC-sampling variance. |

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
