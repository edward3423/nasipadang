# TODO

1. **Justify the autoencoder choice.** Look through past Claude Code sessions and
   write up why the denoising autoencoder was chosen over other imputation
   techniques (mean/median, kNN, MICE, matrix factorisation).

2. **Make the missing-data chart readable.** The table-style gap/coverage chart is
   extremely hard to understand at a glance - redesign it so the message lands
   quickly.

3. **Add a world map to the clustering section.** Place it directly under the GMM
   cluster profile table.

4. **Justify the SVR choice.** Explain why SVR was picked over other forecasting
   techniques.

5. **No free lunch - SVR comparison.** Benchmark SVR against other regression
   algorithms, not just against the baseline.

6. **No free lunch - GMM comparison.** Benchmark the Gaussian mixture against
   other clustering algorithms (k-means, agglomerative, DBSCAN).
