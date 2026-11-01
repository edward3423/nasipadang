# Experiments

Exploratory scripts behind the method choices in `nasipadang.ipynb`. They are kept for reference and are not part of the notebook run. They were
written as one-off scripts, so some expect their outputs or bundle files in a working folder given on the command line, and some contain paths from the machine they ran on.

| Path | What it compares |
|---|---|
| `strict_forecast.py` | Forecasting models on 2000-2010, trained only on 1990-2000 (Ridge, KNN, SVR, Random Forest, Extra Trees, gradient boosting, MLP) |
| `pipeline_2030.py` | Forecast to 2030 and clustering comparison on 2030 values |
| `cluster_search/` | Wider clustering search (24,246 settings across 31 algorithm variants, including K-means as a reference) |
| `ranking_search/` | Ranking methods without clustering (TOPSIS, PROMETHEE, VIKOR, DEA, Pareto, SMAA and others) |
| `gmm5_check.py` | Settings, neighbours and types for the five-type GMM |

## Main results

- **Forecasting:** no model was best on every indicator, and exact market picks did not improve on assuming no change. SVR was the most consistent machine-learning model.
- **Clustering:** on 2030 values, the GMM passes the checks, but it did not beat every alternative. K-means (k=4) and self-tuning spectral clustering (k=4) scored at least as well on several measures, including silhouette. The GMM was chosen for its membership probabilities and per-group variance.
- **Ranking:** no ranking method clearly beat the percentile score, which is also the simplest.
- **Limit:** the fake-data test for the wide clustering search was shortened (20 datasets, 77 settings), so it under-adjusts for the search size.
