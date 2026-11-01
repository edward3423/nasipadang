"""For each country in the recommended top 25: how many distinct method families agree, on the ensemble of forecast tables?

Families: percentile (ordinal, additive), Copeland (ordinal, head-to-head), z-score mean (cardinal, additive),
PROMETHEE V 2 SD (capped cardinal, pairwise), TOPSIS min-max (distance to ideal), BoD with assurance region (country's best case).
Each method's score is z-normalised within a forecast table, averaged over tables, then ranked.
"""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import *

pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 300, "display.max_colwidth", 30)
b = load_bundle()
cols, signs, groups = b["feature_cols"], b["signs"], b["groups"]
g_of = {f: i for i, v in enumerate(groups.values()) for f in v}
g = np.array([g_of[c] for c in cols])
sgn = np.array([signs[c] for c in cols])
extra = pickle.load(open(HERE / "extra_2030.pkl", "rb"))
FX = extra["features"]
FAM = ["percentile", "copeland", "zscore_mean", "promethee_v", "topsis_minmax", "bod_ar"]
out = pd.read_csv(HERE / "final_ranking_2030.csv", index_col=0)
ids = out.index
Z = {}
for m in FAM:
    per = []
    for k, df in FX.items():
        s = METHODS[m](df.loc[ids, cols].values.astype(float) * sgn, g)
        per.append((s - s.mean()) / s.std())
    Z[m] = np.mean(per, axis=0)
R = pd.DataFrame({m: ranks(Z[m]) for m in FAM}, index=ids)
t = pd.DataFrame({"name": out["name"], "rank": out["rank"], "in_top10_of_6": (R <= 10).sum(axis=1), "in_top20_of_6": (R <= 20).sum(axis=1), "best": R.min(axis=1), "worst": R.max(axis=1)}).join(R)
t.to_csv(HERE / "method_agreement.csv")
print(t.sort_values("rank").head(25).to_string())
print("\nSanity rows:"); print(t.loc[["IDN", "MYS", "SGP", "PRI", "HKG", "CUB", "MMR"]].to_string())
print("\nIn the top 20 of all 6:", t.index[t["in_top20_of_6"] == 6].map(out["name"]).tolist())
print("In the top 10 of all 6:", t.index[t["in_top10_of_6"] == 6].map(out["name"]).tolist())
print("In the top 20 of at least 5:", t.index[t["in_top20_of_6"] >= 5].map(out["name"]).tolist())
print("In the top 10 of at least 4:", t.index[t["in_top10_of_6"] >= 4].map(out["name"]).tolist())
print("Pairwise: ensemble percentile vs each family, (rho, top10, top20)")
for m in FAM[1:]:
    print(" ", m, round(spear(Z["percentile"], Z[m]), 3), overlap(Z["percentile"], Z[m], 10), overlap(Z["percentile"], Z[m], 20))
