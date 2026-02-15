# Context parroting

Context parroting and time series foundation models for zero-shot forecasting of dynamical systems

Context parroting is an informative baseline for time series foundation models. It consists of generating naive forecasts by copying contiguous sequences of context points.

## Repository structure

+ The directory `data` includes test datasets used for benchmarks.
+ The directory `demo` includes a minimal demo of the context parroting model.
+ The directory `benchmark` includes notebooks for benchmarking the context parroting model and the baseline models.

## Datasets

+ The dysts zero-shot forecasting dataset was studied in an earler work. In order to avoid data fragmentation, we point to the [original source files associated with that paper.](https://github.com/GilpinLab/dysts_data/tree/main/benchmark_results/zero-shot/chronos_benchmarks_context_512_granularity_30)

+ In this repository, the directory `data` includes test datasets corresponding to an electrocardiogram (ECG), an electronic circuit, the von Karman vortex street, and coupled oscillators.

## Running benchmarks

To run the benchmarks, use the following command:
```bash
# Run all models
  python benchmark/run_dysts_benchmarks.py

# Run specific models only
  python benchmark/run_dysts_benchmarks.py chronos timemoe
```
This will run the benchmarks for all or a subset of the models in the `forecast_models.py` file. The results will be saved in the `benchmark/benchmark_results` directory. Plot the existing benchmark results for all the models in the `benchmark/benchmark_plotting.ipynb` notebook.

The benchmarks can also be run using `uv`, which recommended due to the many dependencies of the different models.
```bash
  uv run python benchmark/run_dysts_benchmarks.py
```


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