import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import pickle
import numpy as np, pandas as pd
pd.set_option("display.width", 230, "display.max_columns", 40, "display.max_rows", 80)
P = r"C:\Users\Edward\AppData\Local\Temp\claude\C--Users-Edward-OneDrive---Nanyang-Technological-University-Materials-MA0218-miniproject-nasipadang\b1936c1a-544e-48af-b48d-65146663a38e\scratchpad\run3\bundle_2030.pkl"
b = pickle.load(open(P, "rb"))
for k, v in b.items():
    print("==", k, type(v).__name__, getattr(v, "shape", ""))
print(b["feature_cols"]); print(b["signs"]); print(b["groups"])
f = b["features_2030"]
print(f.head(3).T)
print(f.describe().T)
print("missing", f[b["feature_cols"]].isna().sum().to_dict())
print("missing knn", b["features_2030_knn"][b["feature_cols"]].isna().sum().to_dict())
print("missing now", b["features_now"][b["feature_cols"]].isna().sum().to_dict())
print("pop>=1M", (f["population"] >= 1e6).sum())
print(b["rank_2030"].sort_values("rank").head(25))
print(b["rank_2030"].shape, b["rank_2030_knn"].shape, b["rank_now"].shape)
print(b["rank_2030_knn"].sort_values("rank").head(12))
print(b["rank_now"].sort_values("rank").head(12))
for k in ["backtest_actual_2000", "backtest_actual_2010", "backtest_svr_forecast_2010"]:
    print(k); print(b[k].head(3)); print(b[k].isna().sum().to_dict())
print(f.index.equals(b["features_2030_knn"].index), f.index.equals(b["features_now"].index))
print(f[b["feature_cols"]].corr(method="spearman").round(2))
