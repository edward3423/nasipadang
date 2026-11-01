# Where should I open my nasi padang shop by 2030?

A data science mini-project (MA0218) that forecasts World Bank indicators to 2030, ranks countries as places to open a nasi padang
shop, and groups them into market types.

**Dataset.** World Bank climate change data (the course dataset): 58 indicators for 218 countries, 1990-2011 (most yearly indicators end in
2008-2010), plus climate projections for 2045-2065. It is bundled at `data/climate_change.xls`. No external data is used.

**Approach.**
1. **Imputation** - genuinely missing values are filled by a denoising autoencoder (a neural network), benchmarked
   against mean filling, KNN and an iterative MLP. Blanks that mean "no" or "about 100%" are filled by rule first.
2. **Feature engineering** - demand, rice and beef supply, climate and policy (including beef-tax risk and a
   rendang advantage index), and kitchen and operations.
3. **Forecasting** - a support vector regressor predicts 10-year changes and is applied twice to reach 2030, tested on an unseen decade.
4. **Ranking** - a transparent score, checked against 5,000 random weightings.
5. **Clustering** - a Gaussian mixture groups countries into five market types using their 2030 values. The clusters do not affect the score.

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
A full run takes about 2 minutes on a typical multi-core laptop and regenerates `figures/` and `nasipadang_ranking.csv`.

## Contents

| Path | What it is |
|---|---|
| `nasipadang.ipynb` | The full analysis, with outputs |
| `data/climate_change.xls` | The World Bank climate change dataset |
| `figures/` | Slide-ready figures, numbered in presentation order |
| `nasipadang_ranking.csv` | Final table: 2030 rank, current rank, score, market type, type probability, group scores and every feature for 156 countries |
| `experiments/` | Exploratory scripts used to compare forecasting, clustering and ranking methods |
| `pyproject.toml`, `uv.lock` | Pinned dependencies for `uv` |
