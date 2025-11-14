# Context parroting

Context parroting and time series foundation models for zero-shot forecasting of dynamical systems

Context parroting is an informative baseline for time series foundation models. It consists of generating naive forecasts by copying contiguous sequences of context points.

## Repository structure

+ The directory `data` includes test datasets used for benchmarks.
+ The directory `demo` includes a minimal demo of the context parroting model.
+ The directory `benchmark` includes the benchmarks used in the accompanying paper.

## Datasets

+ The dysts zero-shot forecasting dataset was studied in an earler work. In order to avoid data fragmentation, we point to the [original source files associated with that paper.](https://github.com/GilpinLab/dysts_data/tree/main/benchmark_results/zero-shot/chronos_benchmarks_context_512_granularity_30)

+ In this repository, the directory `data` includes test datasets corresponding to an electrocardiogram (ECG), an electronic circuit, the von Karman vortex street, and coupled oscillators.

## Reference

Information about the experiments can be found in the accompanying paper.

```bibtex
@article{parroting2025,
  title={Context parroting: A simple but tough-to-beat baseline for foundation models in scientific machine learning},
  author={Anonymous},
  journal={Under review},
  year={2025}
}
```