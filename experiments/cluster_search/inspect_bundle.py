import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import pickle
import numpy as np, pandas as pd, sklearn

if __name__ == "__main__":
    p = r"C:\Users\Edward\AppData\Local\Temp\claude\C--Users-Edward-OneDrive---Nanyang-Technological-University-Materials-MA0218-miniproject-nasipadang\b1936c1a-544e-48af-b48d-65146663a38e\scratchpad\run3\bundle_2030.pkl"
    b = pickle.load(open(p, "rb"))
    print("sklearn", sklearn.__version__)
    for k, v in b.items():
        t = type(v).__name__
        s = getattr(v, "shape", None)
        print(k, t, s if s is not None else (len(v) if hasattr(v, "__len__") else v))
    print(b["feature_cols"])
    print(b["signs"])
    print(b["groups"])
    C = b["C_2030"]
    print("C_2030 mean", np.round(C.mean(0), 3))
    print("C_2030 std", np.round(C.std(0), 3))
    print("nan", np.isnan(C).sum(), np.isnan(b["C_2030_knn"]).sum(), np.isnan(b["C_now"]).sum())
    print("C_2030_knn mean", np.round(b["C_2030_knn"].mean(0), 3))
    print("C_2030_knn std", np.round(b["C_2030_knn"].std(0), 3))
    print(type(b["index"]), list(b["index"])[:5], list(b["names"])[:5])
    print(b["features_2030"].head())
    print(b["features_2030"].describe().T)
    print(b["rank_2030"].head(15))
    print(b["backtest_actual_2010"].head())
    print(b["backtest_actual_2000"].head())
    print(b["backtest_svr_forecast_2010"].head())
    from sklearn.decomposition import PCA
    pca = PCA().fit(C)
    print("evr", np.round(pca.explained_variance_ratio_, 3), np.round(np.cumsum(pca.explained_variance_ratio_), 3))
    print("same index?", list(b["features_2030"].index) == list(b["index"]))
    print("bt idx subset", set(b["backtest_actual_2010"].index) <= set(b["index"]))
    print("bt align", (b["backtest_actual_2010"].index == b["backtest_actual_2000"].index).all(), (b["backtest_actual_2010"].index == b["backtest_svr_forecast_2010"].index).all())
    print("bt cols", list(b["backtest_actual_2010"].columns), list(b["backtest_actual_2000"].columns), list(b["backtest_svr_forecast_2010"].columns))
    print("bt nan", b["backtest_actual_2010"].isna().sum().sum(), b["backtest_actual_2000"].isna().sum().sum(), b["backtest_svr_forecast_2010"].isna().sum().sum())
