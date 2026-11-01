"""Scratch only: SVR forecast to 2030, 2030 score, recluster, compare with current, and test cluster forecasting on 2000->2010."""
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
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score as ari, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

SEED = 42
pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 80, "display.max_colwidth", 80)


def svr():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVR(C=1.0, epsilon=0.05))


def knn():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsRegressor(10, weights="distance"))


def fit_gmm(cfg, X, seed=SEED):
    return GaussianMixture(cfg[0], covariance_type=cfg[1], reg_covar=cfg[2], init_params=cfg[3], n_init=5, max_iter=500, random_state=seed)


def lab(cfg, C, seed=SEED):
    Z = PCA(8, random_state=SEED).fit_transform(C)
    if cfg[0] == "K":
        return KMeans(cfg[1], n_init=20, random_state=seed).fit_predict(Z)
    if cfg[0] == "H":
        return HDBSCAN(min_cluster_size=cfg[1], min_samples=cfg[2], cluster_selection_epsilon=cfg[3], cluster_selection_method=cfg[4], metric=cfg[5]).fit(Z.copy()).labels_
    return fit_gmm(cfg[1:], Z, seed).fit(Z).predict(Z)


def sizes_ok(l, k):
    n = np.bincount(l[l >= 0])
    n = n[n > 0]
    return len(n) == k and n.min() >= 8 and n.max() <= 0.5 * len(l)


def stab(cfg, C, base):
    ref = lab(cfg, C, base)
    rng = np.random.default_rng(base)
    out = []
    for r in range(25):
        i = rng.choice(len(C), int(0.8 * len(C)), replace=False)
        out.append(ari(ref[i], lab(cfg, C[i], base * 100 + r)))
    return np.mean(out)


def bic_row(Z, k, cov, reg, init):
    try:
        m = GaussianMixture(k, covariance_type=cov, reg_covar=reg, init_params=init, n_init=5, max_iter=500, random_state=SEED).fit(Z)
    except Exception:
        return None
    return {"cfg": ("G", k, cov, reg, init), "k": k, "bic": m.bic(Z), "ok": sizes_ok(m.predict(Z), k)}


def hdb_row(C, mcs, ms, eps, me, mt):
    try:
        l = lab(("H", mcs, ms, eps, me, mt), C)
    except TypeError:
        return None
    ok = l >= 0
    g = len(set(l[ok]))
    return {"cfg": ("H", mcs, ms, eps, me, mt), "groups": g, "noise": 1 - ok.mean(), "usable": 3 <= g <= 7 and sizes_ok(l, g)}


def main(project, work):
    project, work = Path(project), Path(work)
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
    d, f, ts, midpoint, latest, meta, names, FEATURES, pct, zscore, fossil, R0 = (ns[k] for k in ["d", "f", "ts", "midpoint", "latest", "meta", "names", "FEATURES", "pct", "zscore", "fossil", "R"])
    par = Parallel(n_jobs=-1)
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

    def scores(x):
        r = x[x["population"] >= 1e6].copy()
        g = pd.DataFrame(index=r.index)
        for group, feats in FEATURES.items():
            parts = [pct(r[c]) if s > 0 else 1 - pct(r[c]) + 1 / len(r) for c, s in feats.items()]
            g[group] = pd.concat(parts, axis=1).mean(axis=1)
        r["score"] = g.mean(axis=1)
        r["rank"] = r["score"].rank(ascending=False).astype(int)
        return r.sort_values("rank")

    def matrix(x):
        return StandardScaler().fit_transform(pd.DataFrame({c: x[c] * s for g in FEATURES.values() for c, s in g.items()}, index=x.index))

    d2 = build_2030(forecast(svr))
    d3 = build_2030(forecast(knn))
    R30, R30k = scores(d2), scores(d3)
    print("== Forecast sanity: median values, now vs 2030 ==")
    print(pd.DataFrame({"now": d[["population", "urban_pop", "gni_pc", "u5_mortality", "electricity", "water", "sanitation"]].median(), "2030": d2[["population", "urban_pop", "gni_pc", "u5_mortality", "electricity", "water", "sanitation"]].median()}).round(1).to_string())
    print("Largest 2030 urban populations:", (d2["urban_pop"] / 1e6).nlargest(6).round(0).rename(index=names).to_dict())
    print("Highest 2030 GNI per person:", d2["gni_pc"].nlargest(5).round(0).rename(index=names).to_dict())
    print(f"Countries in 2030 ranking {len(R30)}, now {len(R0)}")

    print("\n== 2030 ranking vs current ==")
    both = R30.index.intersection(R0.index)
    print("Rank correlation:", round(spearmanr(R30.loc[both, "rank"], R0.loc[both, "rank"])[0], 3))
    print("Top 10 kept:", len(set(R30.head(10).index) & set(R0.head(10).index)), "| Top 20 kept:", len(set(R30.head(20).index) & set(R0.head(20).index)))
    print("Top 10 in 2030:", R30.head(10)[["name", "score"]].round(3).values.tolist())
    print("Top 10 now:", R0.head(10)["name"].tolist())
    print("Top 20 entrants:", R30.head(20).loc[~R30.head(20).index.isin(R0.head(20).index), "name"].tolist())
    move = (R0.loc[both, "rank"] - R30.loc[both, "rank"]).rename(index=names)
    print("Biggest climbers:", move.nlargest(8).to_dict())
    print("Biggest fallers:", move.nsmallest(8).to_dict())
    print("Indonesia, Malaysia, Singapore 2030 rank:", {names[c]: int(R30.loc[c, "rank"]) for c in ["IDN", "MYS", "SGP"]})
    bothk = R30.index.intersection(R30k.index)
    print("SVR vs KNN 2030 rank correlation:", round(spearmanr(R30.loc[bothk, "rank"], R30k.loc[bothk, "rank"])[0], 3), "| top 10 shared:", len(set(R30.head(10).index) & set(R30k.head(10).index)))

    print("\n== Reclustering on 2030 values ==")
    C = matrix(d2)
    Z = PCA(8, random_state=SEED).fit_transform(C)
    KS = list(range(1, 11))
    ine = [KMeans(k, n_init=20, random_state=SEED).fit(Z).inertia_ for k in KS]
    x_, y_ = (np.array(KS) - 1) / 9, (np.array(ine) - ine[-1]) / (ine[0] - ine[-1])
    ke = KS[int(np.argmax(1 - x_ - y_))]
    import itertools
    B = pd.DataFrame([r for r in par(delayed(bic_row)(Z, *a) for a in itertools.product(range(2, 11), ["spherical", "diag", "tied", "full"], [1e-6, 1e-3, 1e-2, 0.1, 0.5, 1.0], ["kmeans", "k-means++", "random"])) if r])
    V = B[B["ok"]].sort_values("bic")
    print("Elbow k:", ke, "| Lowest valid BIC per k:", V.groupby("k")["bic"].min().round(0).to_dict())
    gcfg = V.iloc[0]["cfg"]
    H = pd.DataFrame([r for r in par(delayed(hdb_row)(C, *a) for a in itertools.product([3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30], [1, 2, 3, 5, 8, 10, None], [0.0, 0.25, 0.5, 1.0], ["eom", "leaf"], ["euclidean", "manhattan"])) if r])
    U = H[H["usable"]].sort_values("noise")
    hcfg = U.iloc[0]["cfg"] if len(U) else None

    def report(name, cfg):
        k = cfg[1]
        starts = par(delayed(lab)(cfg, C, s) for s in range(30))
        l = lab(cfg, C)
        groups = len(set(l[l >= 0]))
        return {"model": name, "cfg": str(cfg[1:]), "left_out": round(float((l < 0).mean()), 3), "sizes": sorted(np.bincount(l[l >= 0]).tolist(), reverse=True),
                "sizes_all_starts": all(sizes_ok(s, groups) for s in starts), "stability": round(float(np.mean(par(delayed(stab)(cfg, C, b) for b in range(1, 6)))), 3),
                "silhouette": round(float(np.mean([silhouette_score(C, s) for s in starts if len(set(s)) > 1])), 3)}

    rows = [report("K-means", ("K", ke)), report("GMM", gcfg)]
    if hcfg:
        try:
            rows.append(report("HDBSCAN", hcfg))
        except TypeError:
            rows.append({"model": "HDBSCAN", "cfg": str(hcfg), "left_out": round(float(U.iloc[0]["noise"]), 3)})
    print(pd.DataFrame(rows).to_string(index=False))

    l = lab(gcfg, C)
    labels = [c for g in FEATURES.values() for c in g]
    prof = pd.DataFrame(C, columns=labels).groupby(l).mean().round(1)
    prof.insert(0, "size", np.bincount(l))
    print(prof.T.to_string())
    pop = d2["population"].values
    nm = names.loc[idx].values
    for t in sorted(set(l)):
        m = l == t
        print(f"  Type {t} ({m.sum()}): {', '.join(nm[m][np.argsort(-pop[m])][:12])}")
    s = pd.Series(l, index=idx)
    print("Top 10 mix:", s[R30.head(10).index].value_counts().to_dict(), [(names[c], int(s[c])) for c in R30.head(10).index])
    current_cfg = ("G", 6, "spherical", 1e-6, "random")
    Cnow = matrix(d)
    now = lab(current_cfg, Cnow)
    print("Agreement with today's six types (ARI):", round(ari(l, now), 3))
    same6 = lab(current_cfg, C)
    print("Same settings as today's GMM, k=6, on 2030: ARI vs today", round(ari(same6, now), 3), "sizes", sorted(np.bincount(same6).tolist(), reverse=True))
    print("GMM on KNN-based 2030 vs SVR-based 2030 (ARI):", round(ari(lab(gcfg, matrix(d3)), l), 3))

    print("\n== Can a forecast predict cluster membership? 2000 -> 2010, six indicators, train 1990 -> 2000 ==")
    ix = meta.index[(L["population"][2000].notna()) & L["urban"][2000].notna() & L["gni"][2000].notna() & L["u5"][2000].notna() & L["gni"][2010].notna() & L["u5"][2010].notna() & L["urban"][2010].notna() & L["population"][2010].notna()]
    use = ["population", "urban", "gni", "u5"]
    A0 = pd.DataFrame({k: L[k][2000] for k in use}).loc[ix]
    A1 = pd.DataFrame({k: L[k][2010] for k in use}).loc[ix]
    P = pd.DataFrame(index=ix)
    for k in use:
        y = L[k][2000] - L[k][1990]
        ok = y.notna()
        P[k] = A0[k] + svr().fit(F(1990)[ok], y[ok]).predict(F(2000).loc[ix])
    scaler = StandardScaler().fit(A1)
    feature_cols = [c for g in FEATURES.values() for c in g]
    signs = {c: s_ for g in FEATURES.values() for c, s_ in g.items()}
    groups = {g: list(v) for g, v in FEATURES.items()}
    pickle.dump({"C_2030": C, "C_now": Cnow, "C_2030_knn": matrix(d3), "index": list(idx), "names": names.loc[idx].tolist(), "feature_cols": feature_cols, "signs": signs, "groups": groups,
                 "features_2030": d2[feature_cols + ["population", "name"]], "features_2030_knn": d3[feature_cols + ["population", "name"]], "features_now": d[feature_cols + ["population", "name"]],
                 "rank_2030": R30[["name", "rank", "score"]], "rank_2030_knn": R30k[["name", "rank", "score"]], "rank_now": R0[["name", "rank", "score"]],
                 "backtest_actual_2000": A0, "backtest_actual_2010": A1, "backtest_svr_forecast_2010": P}, open(work / "bundle_2030.pkl", "wb"))
    print("saved bundle")
    res = []
    for k in [3, 4, 5, 6]:
        km = KMeans(k, n_init=20, random_state=SEED).fit(scaler.transform(A1))
        truth = km.labels_
        res.append({"k": k, "countries": len(ix), "ARI no change": round(ari(truth, km.predict(scaler.transform(A0))), 3), "ARI SVR forecast": round(ari(truth, km.predict(scaler.transform(P))), 3)})
    print(pd.DataFrame(res).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
