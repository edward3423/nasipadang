"""Verify the last few claims in the write-up that were not printed exactly by earlier scripts."""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import *

b = load_bundle()
out = pd.read_csv(HERE / "final_ranking_2030.csv", index_col=0)
print("Scores, ranks 7 to 14:", out[["name", "score"]].iloc[6:14].round(5).to_string())
print("Rank 8 minus rank 11:", round(float(out["score"].iloc[7] - out["score"].iloc[10]), 5), "| rank 10 minus rank 11:", round(float(out["score"].iloc[9] - out["score"].iloc[10]), 5))
F = b["features_2030"]
r = F[F["population"] >= 1e6]
print("Rendang exactly zero share, ranked:", round(float((r["rendang_advantage"] == 0).mean()), 3), "| at most 1e-3:", round(float((r["rendang_advantage"] <= 1e-3).mean()), 3))
print("Duplicate-value share:", round(float(1 - r["rendang_advantage"].nunique() / len(r)), 3))
print("Yield volatility, ranked: median", round(float(r["yield_volatility"].median()), 1), "max", round(float(r["yield_volatility"].max()), 1))
print("Spearman rendang vs income, ranked:", round(spear(r["rendang_advantage"], r["log_gni_pc"]), 3))
A0 = b["backtest_actual_2000"]
ix = set(A0.index[A0["population"] >= np.log(1e6)])
for c in ["HKG", "QAT", "MMR", "CAN", "PRI", "CUB", "SAU", "OMN"]:
    print(c, "in the 157:", c in A0.index, "| in the 132:", c in ix, "| 2000 population, millions:", round(float(np.exp(A0.loc[c, "population"]) / 1e6), 2) if c in A0.index else "n/a")
