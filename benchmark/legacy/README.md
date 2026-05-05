# Legacy per-model benchmark notebooks

These notebooks predate the unified `run_dysts_benchmarks.py` driver and the
`forecast_models.py` model registry. Each one re-implements the dysts benchmarking
loop for a single model, with model-specific loading code inline.

**For new benchmarks, use the script instead:**

```bash
# from the repo root
python benchmark/run_dysts_benchmarks.py chronos_bolt   # or any model in forecast_models.py
```

Notebooks kept here for reference (and to reproduce the saved figures in
`analysis/`):

- `benchmark_Chronos_dysts.ipynb` - Chronos T5
- `benchmark_ChronosBolt_dysts.ipynb` - ChronosBolt (older version; see
  `../benchmark_ChronosBolt_normalized.ipynb` for the current one)
- `benchmark_Dynamix_dysts.ipynb` - DynaMix (older; see
  `../benchmark_Dynamix_dysts_forecast_models.ipynb` for the version that uses
  the unified `forecast_models.py` interface)
- `benchmark_TimeMoE_dysts.ipynb` - TimeMoE
- `benchmark_TimesFM_dysts.ipynb` - TimesFM
