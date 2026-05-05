# Benchmark diagnostics

These scripts were used to validate that the script pipeline (`run_dysts_benchmarks.py`)
and the per-model notebooks (e.g. `benchmark_ChronosBolt_normalized.ipynb`) compute the
same metrics when given the same initial conditions, and to quantify how much of the
observed run-to-run variation is due to IC sampling alone.

## `compare_pipelines.py`

Runs ChronosBolt under a 2x2 design on Lorenz, Rossler, and Chua at `context_length=512`:

|                          | Notebook ICs (random) | Script ICs (deterministic stride) |
|--------------------------|-----------------------|-----------------------------------|
| Notebook code path       | A                     | D                                 |
| Script code path         | C                     | B                                 |

Result: with the same ICs, both code paths produce **bit-identical** predictions and
identical sMAPE / VPT / cdim. The only residual is in KL, which is from
`dysts.metrics.estimate_kl_divergence` being a stochastic k-NN estimator (std ~ 0.02 on
identical inputs).

## `ic_variance_bootstrap.py`

Bootstraps the notebook estimator: 8 random-seed runs of 20 ICs each on the same
3 systems. Reports mean +/- std and (min, max) ranges. Used to confirm that the
observed differences between the script and the saved notebook results are within
natural sampling variance at `num_ic=20`.

## How to run

```bash
# from the repo root
python benchmark/diagnostics/compare_pipelines.py
python benchmark/diagnostics/ic_variance_bootstrap.py
```

Both scripts load `data/good_trajectories/{Lorenz,Rossler,Chua}.npy` and require the
ChronosBolt model to be reachable through `BaseChronosPipeline.from_pretrained`.
