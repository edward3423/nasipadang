# Where should I open my nasi padang shop in 2060?

A data science mini-project (MA0218) that ranks countries by how good a place they would be to open a nasi padang
shop in 2060, once a hotter, wetter and more regulated climate is priced in.

**Dataset.** World Bank climate change data (the course dataset): 58 indicators for 218 countries, 1990-2011, plus
climate projections for 2045-2065. It is bundled at `data/climate_change.xls`. No external data is used.

**Approach.**
1. **Imputation** - genuinely missing values are filled by a denoising autoencoder (a neural network), benchmarked
   against mean filling, KNN and an iterative MLP. Blanks that mean "no" or "about 100%" are filled by rule first.
2. **Feature engineering** - demand, rice and beef supply, climate and policy (including beef-tax risk and a
   rendang advantage index), and kitchen and operations.
3. **Clustering** - K-means groups countries into five market types (selected over Gaussian mixtures by a fixed
   stability and separation rule).
4. **Ranking** - a transparent score, checked against 5,000 random weightings.

## How to run

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the repository and enter the project folder:
   ```
   git clone https://github.com/edward3423/nasipadang.git
   cd nasipadang
   ```
3. Install the dependencies (Python 3.13+, about 1 GB including PyTorch):
   ```
   uv sync
   ```
4. Open the notebook interactively:
   ```
   uv run jupyter lab nasipadang.ipynb
   ```
   or run it end to end without opening it:
   ```
   uv run jupyter nbconvert --to notebook --execute --inplace nasipadang.ipynb
   ```

Run it from inside the `nasipadang` folder: the dataset is loaded from `data/climate_change.xls` relative to it.
A full run takes about 3 minutes and regenerates `figures/` and `nasipadang_ranking.csv`.

## Contents

| Path | What it is |
|---|---|
| `nasipadang.ipynb` | The full analysis, with outputs |
| `data/climate_change.xls` | The World Bank climate change dataset |
| `figures/` | Slide-ready figures, numbered in presentation order |
| `nasipadang_ranking.csv` | Final table: rank, score, cluster, group scores and every feature for 156 countries |
| `pyproject.toml`, `uv.lock` | Pinned dependencies for `uv` |
