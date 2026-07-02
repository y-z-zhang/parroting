# Legacy analysis data

Superseded data kept for provenance. Nothing in the active notebooks reads
from this directory.

## WARNING: `parrot_statistics_new/` is on a different sMAPE scale

This data was generated in an environment whose dysts snapshot reports sMAPE
on a **0–100 scale** (dysts changed its default in April 2025,
GilpinLab/dysts@4b4d9de). Every other statistics directory in this repository
uses the paper's **0–200 convention**. As a result:

- all sMAPE values here are **half** of what the rest of the repo reports;
- all VPT values are inflated (~2x), because the threshold-30 crossing on a
  half-scale metric corresponds to threshold 60 on the standard scale.

Do not compare these numbers against anything else in the repo. The benchmark
code now defines sMAPE locally (`benchmark/metrics.py`) precisely so this
cannot happen again.

## Other contents

- `ChronosBolt_statistics/`, `TimeMoE_statistics/`, `TimesFM_statistics/` —
  benchmark statistics from un-normalized trajectory runs; superseded by the
  `*_statistics_normalized/` directories one level up.
- `fig_dynamix.pdf`, `fig_longhorizon.pdf`, `parrot_psd.pdf`,
  `smape_rolling_compare_normalized_0.pdf` — figures whose generating code no
  longer exists in the repository.
