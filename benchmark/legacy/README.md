# Legacy benchmark notebooks and data

Everything here is superseded by the unified driver:

```bash
# from the repo root — see `python benchmark/run_dysts_benchmarks.py --help`
python benchmark/run_dysts_benchmarks.py chronos_bolt --context-lengths 512
```

with models defined in `../forecast_models.py`. Kept for reference and to
document how earlier data was produced.

## Notebooks

Per-model dysts benchmarks that each re-implement the benchmarking loop with
model-specific loading code inline:

- `benchmark_Chronos_dysts.ipynb`, `benchmark_ChronosBolt_dysts.ipynb`,
  `benchmark_Dynamix_dysts.ipynb`, `benchmark_TimeMoE_dysts.ipynb`,
  `benchmark_TimesFM_dysts.ipynb` — oldest generation (un-normalized
  trajectories).
- `benchmark_ChronosBolt_normalized.ipynb` — newer ChronosBolt loop on
  normalized trajectories; validated to produce bit-identical forecasts to the
  driver given the same initial conditions (see `../diagnostics/`).
- `benchmark_parrot_dysts.ipynb`, `benchmark_arima.ipynb`,
  `benchmark_panda.ipynb`, `benchmark_simplex.ipynb` — per-model loops whose
  models are all registered in `../forecast_models.py`.
- `benchmark_Dynamix_dysts_forecast_models.ipynb` — DynaMix through the
  unified `forecast_models.py` interface; the driver does the same thing.

Note: these notebooks write statistics into `../../analysis/` directories.
Some of those directories are still read by the active context-length-sweep
notebooks; regenerating that data should now go through the driver instead.

## `benchmark_results_old/`

The previous generation of driver output (context lengths 64–1024, fixed-stride
IC sampling biased toward the start of each trajectory). Superseded for the
Fig. 2 models by `../benchmark_results/`, which uses seeded random IC sampling
across the full trajectory (see `run_dysts_benchmarks.py`). Still the data
source for the supplementary baselines (panda, simplex, arima) in
`analysis_fixed_context_length.ipynb`, and useful for old-vs-new comparisons.
