"""Print the top 20 by each family, for reading what actually changes."""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import pandas as pd
from pathlib import Path
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 300, "display.max_colwidth", 22)
HERE = Path(__file__).resolve().parent
r = pd.read_csv(HERE / "ranks_all_methods_svr.csv", index_col=0)
show = ["percentile", "copeland", "smaa_meanrank", "geomean_groups", "zscore_mean", "promethee_v", "topsis_minmax", "critic", "kemeny", "pca", "bod_ar"]
out = {}
for m in show:
    out[m] = r.sort_values(m)["name"].head(20).str.slice(0, 16).values
print(pd.DataFrame(out, index=range(1, 21)).to_string())
print(r[show].corr(method="spearman").round(2).to_string())
