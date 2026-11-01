"""Read the raw World Bank panel once (read-only) and cache it in this folder. Forecast helpers for the backtest."""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import pickle
from lab import *
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

CACHE = HERE / "panels.pkl"
S = {"population": "Population", "urban": "Urban population", "gni": "GNI per capita (Atlas $)", "yield": "Cereal yield (kg per hectare)",
     "co2": "CO2 emissions per capita (metric tons)", "u5": "Under-five mortality rate (per 1,000)"}


def load_panels():
    if CACHE.exists():
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    raw = pd.read_excel(XLS, sheet_name="Data")
    meta = pd.read_excel(XLS, sheet_name="Country")
    meta = meta[meta["Region"] != "Aggregates"].set_index("Country code")
    raw = raw[raw["Country code"].isin(meta.index)]
    years = [c for c in raw.columns if str(c).isdigit()]
    w = raw.set_index(["Country code", "Series name"])[years].replace("..", np.nan)
    L = {k: np.log(w.xs(v, level="Series name").apply(pd.to_numeric, errors="coerce").reindex(meta.index).rename(columns=int).pipe(lambda d: d.where(d > 0))) for k, v in S.items()}
    out = {"L": L, "income": pd.Series(pd.Categorical(meta["Income group"]).codes, index=meta.index), "region": pd.Series(pd.Categorical(meta["Region"]).codes, index=meta.index),
           "name": meta["Country name"], "income_label": meta["Income group"], "region_label": meta["Region"]}
    with open(CACHE, "wb") as fh:
        pickle.dump(out, fh)
    return out


def makers():
    return {
        "svr": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVR(C=1.0, epsilon=0.05)),
        "knn": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsRegressor(10, weights="distance")),
        "ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(5.0)),
        "rf": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(500, min_samples_leaf=3, random_state=42, n_jobs=1)),
    }


def features_at(pan, t, income=True):
    d = {"region": pan["region"]}
    if income:
        d["income"] = pan["income"]
    d.update({k: v[t] for k, v in pan["L"].items()})
    return pd.DataFrame(d)


def forecast_targets(pan, train_pairs, origin, horizon_scale, targets, maker, index, income=True):
    """Fit log change on train_pairs, predict change from origin for each target."""
    out = {}
    for k in targets:
        v = pan["L"][k]
        Xs, ys = [], []
        for a, b in train_pairs:
            y = (v[b] - v[a]) * horizon_scale / (b - a)
            ok = y.notna()
            Xs.append(features_at(pan, a, income)[ok])
            ys.append(y[ok])
        m = maker().fit(pd.concat(Xs), pd.concat(ys))
        out[k] = v[origin].loc[index] + m.predict(features_at(pan, origin, income).loc[index])
    return pd.DataFrame(out, index=index)
