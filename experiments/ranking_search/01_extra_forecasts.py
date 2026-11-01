"""Rebuild the 2030 feature table under extra forecast models, so SVR vs KNN is not the only sensitivity estimate.

Runs the notebook cells in a sandbox copy (cwd = ranking_search/nbrun), never in the project folder.
Forecast logic is copied from scratchpad/pipeline_2030.py.
"""
import json
import os
import pickle
import shutil
import sys
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

HERE = Path(__file__).resolve().parent
MAKERS = {
    "SVR": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVR(C=1.0, epsilon=0.05)),
    "KNN": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsRegressor(10, weights="distance")),
    "Ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(5.0)),
    "RandomForest": lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(500, min_samples_leaf=3, random_state=42, n_jobs=-1)),
    "ExtraTrees": lambda: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(500, min_samples_leaf=3, random_state=42, n_jobs=-1)),
    "GradBoost": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=10, random_state=42),
    "MLP": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPRegressor(hidden_layer_sizes=(32,), alpha=1.0, max_iter=3000, random_state=42)),
}


def main(project):
    project = Path(project)
    work = HERE / "nbrun"
    (work / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(project / "data/climate_change.xls", work / "data")
    os.chdir(work)
    nb = json.load(open(project / "nasipadang.ipynb", encoding="utf-8"))
    ns = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        exec(compile(src, "cell", "exec"), ns)
        if 'R["rank"]' in src:
            break
    d, ts, midpoint, meta, names, FEATURES, pct, zscore, fossil, was_imputed = (ns[k] for k in ["d", "ts", "midpoint", "meta", "names", "FEATURES", "pct", "zscore", "fossil", "was_imputed"])
    idx = d.index
    SERIES = {"population": "Population", "urban": "Urban population", "gni": "GNI per capita (Atlas $)", "yield": "Cereal yield (kg per hectare)", "co2": "CO2 emissions per capita (metric tons)", "u5": "Under-five mortality rate (per 1,000)"}
    L = {k: np.log(ts(v).rename(columns=int).pipe(lambda x: x.where(x > 0))).reindex(meta.index) for k, v in SERIES.items()}
    inc = pd.Series(pd.Categorical(meta["Income group"]).codes, index=meta.index)
    reg = pd.Series(pd.Categorical(meta["Region"]).codes, index=meta.index)
    KEYS = list(SERIES)

    def F(t):
        return pd.DataFrame({"income": inc, "region": reg, **{k: L[k][t] for k in KEYS}})

    PAIRS = {k: [(1990, 2000), (2000, 2010)] for k in ["population", "urban", "gni", "u5"]}
    PAIRS["yield"] = [(1990, 2000), (1999, 2009)]
    PAIRS["co2"] = [(1990, 2000), (1998, 2008)]
    start = pd.DataFrame({"income": inc.loc[idx], "region": reg.loc[idx],
                          "population": np.log(d["population"]), "urban": np.log(d["urban_pop"]), "gni": np.log(d["gni_pc"]),
                          "yield": L["yield"].ffill(axis=1).iloc[:, -1].loc[idx], "co2": np.log(d["co2_pc"]), "u5": np.log(d["u5_mortality"])})
    start["yield"] = start["yield"].fillna(start["yield"].median())

    def forecast(maker):
        models = {}
        for k, pairs in PAIRS.items():
            Xs, ys = [], []
            for a, b in pairs:
                y = L[k][b] - L[k][a]
                ok = y.notna()
                Xs.append(F(a)[ok])
                ys.append(y[ok])
            models[k] = maker().fit(pd.concat(Xs), pd.concat(ys))
        state = start.copy()
        for _ in range(2):
            change = pd.DataFrame({k: models[k].predict(state) for k in KEYS}, index=state.index)
            state = state.copy()
            state[KEYS] = state[KEYS] + change
        state["urban"] = np.minimum(state["urban"], state["population"])
        return state

    def build_2030(state):
        x = d.copy()
        x["population"], x["urban_pop"], x["gni_pc"] = np.exp(state["population"]), np.exp(state["urban"]), np.exp(state["gni"])
        x["co2_pc"], x["u5_mortality"] = np.exp(state["co2"]), np.exp(state["u5"])
        x["yield_trend"] = (state["yield"] - start["yield"]) / 20 * 100
        inputs_now = pd.DataFrame({"gni": np.log(d["gni_pc"]), "u5": np.log(d["u5_mortality"]), "co2": np.log(d["co2_pc"]), "share": d["urban_pop"] / d["population"]})
        inputs_30 = pd.DataFrame({"gni": state["gni"], "u5": state["u5"], "co2": state["co2"], "share": x["urban_pop"] / x["population"]})
        for c, kind in {"electricity": "pct", "paved_roads": "pct", "water": "pct", "sanitation": "pct", "physicians": "log", "nurses": "log", "malaria": "log1p"}.items():
            y = {"pct": d[c], "log": np.log(d[c]), "log1p": np.log1p(d[c])}[kind]
            m = make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05 * y.std())).fit(inputs_now, y)
            val = y + (m.predict(inputs_30) - m.predict(inputs_now))
            x[c] = {"pct": val.clip(0, 100), "log": np.exp(val), "log1p": np.expm1(val).clip(lower=0)}[kind]
        frac = (2030 - 1975) / (2055 - 1975)
        base = midpoint("Average daily min/max temperature (1961-1990, Celsius)", "/").reindex(idx)
        change = midpoint("Projected annual temperature change (2045-2065, Celsius)", "to").reindex(idx)
        x["temp_2060"] = (base + frac * change).fillna(d["temp_2060"])
        x["rain_shift"] = d["rain_shift"] * frac
        x["beef_tax_risk"] = pct(x["methane_pc"]) * x["commitment"]
        x["rendang_advantage"] = pct(x["temp_2060"]) * (1 - x["electricity"] / 100)
        infra = StandardScaler().fit_transform(x[["paved_roads", "water", "sanitation"]])
        pc1 = PCA(1).fit_transform(infra)[:, 0]
        x["infrastructure"] = pc1 * np.sign(np.corrcoef(pc1, x["water"])[0, 1])
        x["staff_health"] = (zscore(np.log(x["physicians"])) + zscore(np.log(x["nurses"])) - zscore(np.log1p(x["malaria"]))) / 3
        x["log_urban_pop"], x["log_gni_pc"] = np.log10(x["urban_pop"]), np.log10(x["gni_pc"])
        x["methane_supply"] = x["methane_pc"].where(~fossil, x["methane_pc"].median())
        return x

    feature_cols = [c for g in FEATURES.values() for c in g]
    out = {}
    for name, mk in MAKERS.items():
        x = build_2030(forecast(mk))
        out[name] = x[feature_cols + ["population", "name"]]
        print(name, "done", flush=True)
    imputed = was_imputed.reindex(idx).sum(axis=1)
    pickle.dump({"features": out, "imputed_count": imputed, "imputed_flags": was_imputed.reindex(idx), "impute_cols": list(was_imputed.columns), "income_group": meta["Income group"].reindex(idx), "region": meta["Region"].reindex(idx)}, open(HERE / "extra_2030.pkl", "wb"))
    print("saved", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
