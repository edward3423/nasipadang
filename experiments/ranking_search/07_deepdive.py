"""Follow-ups raised by the first comparison: the yield_trend forecast, PROMETHEE threshold, why odd entries rank high, sources of uncertainty."""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import itertools, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import *
from panels import *

pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 300, "display.max_colwidth", 90)
b = load_bundle()
cols, signs, groups = b["feature_cols"], b["signs"], b["groups"]
GN = list(groups)
g_of = {f: i for i, v in enumerate(groups.values()) for f in v}
g = np.array([g_of[c] for c in cols])
sgn = np.array([signs[c] for c in cols])
extra = pickle.load(open(HERE / "extra_2030.pkl", "rb"))
FX = extra["features"]
Fnow = b["features_now"]


def prep(df):
    r = df[df["population"] >= 1e6]
    return r[cols].values.astype(float) * sgn, r.index


def cmp(a, ia, c, ic):
    sh = ia.intersection(ic)
    sa, sc = pd.Series(a, index=ia), pd.Series(c, index=ic)
    return {"spearman": spear(sa[sh], sc[sh]), "top10": len(set(ia[order(a)[:10]]) & set(ic[order(c)[:10]])), "top20": len(set(ia[order(a)[:20]]) & set(ic[order(c)[:20]]))}


def frozen(df):
    x = df.copy()
    x["yield_trend"] = Fnow["yield_trend"].reindex(x.index)
    return x


def yield_skill():
    print("== Is the yield forecast worth anything? Test 1999 -> 2009, trained on 1990 -> 1999 ==")
    pan = load_panels()
    L = pan["L"]["yield"]
    A0 = b["backtest_actual_2000"]
    ix = A0.index[A0["population"] >= np.log(1e6)]
    ok = L[1999].reindex(ix).notna() & L[2009].reindex(ix).notna() & L[1990].reindex(ix).notna()
    ids = ix[ok.values]
    y = (L[2009] - L[1999]).loc[ids]
    rows = []
    mk = makers()
    for nm in ["svr", "knn", "ridge", "rf"]:
        yy = (L[1999] - L[1990]) * 10 / 9
        o2 = yy.notna()
        m = mk[nm]().fit(features_at(pan, 1990)[o2], yy[o2])
        p = pd.Series(m.predict(features_at(pan, 1999).loc[ids]), index=ids)
        rows.append({"forecast": nm, "rmse": float(np.sqrt(((y - p) ** 2).mean())), "spearman_with_actual_change": spear(p, y)})
    own = ((L[1999] - L[1990]) * 10 / 9).loc[ids]
    rows.append({"forecast": "own 1990s trend repeats", "rmse": float(np.sqrt(((y - own) ** 2).mean())), "spearman_with_actual_change": spear(own, y)})
    avg = float(((L[1999] - L[1990]) * 10 / 9).dropna().mean())
    rows.append({"forecast": "same average change for all", "rmse": float(np.sqrt(((y - avg) ** 2).mean())), "spearman_with_actual_change": np.nan})
    rows.append({"forecast": "no change", "rmse": float(np.sqrt((y ** 2).mean())), "spearman_with_actual_change": np.nan})
    print(f"countries {len(ids)}")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def yield_freeze():
    print("\n== Keep yield_trend at its observed 1990-2011 value, forecast everything else (baseline method) ==")
    rows = []
    names = list(FX)
    for tag, fn in [("forecast yield_trend (current)", lambda d: d), ("observed yield_trend", frozen)]:
        T = {k: prep(fn(v)) for k, v in FX.items()}
        S = {k: (m_percentile(x, g), i) for k, (x, i) in T.items()}
        ref = cmp(*S["SVR"], *S["KNN"])
        pairs = [cmp(*S[u], *S[v]) for u, v in itertools.combinations(names, 2)]
        Xn, In = prep(Fnow)
        sn = m_percentile(Xn, g)
        rows.append({"variant": tag, "svr_knn_spearman": ref["spearman"], "svr_knn_top10": ref["top10"], "svr_knn_top20": ref["top20"],
                     "7models_spearman": np.mean([p["spearman"] for p in pairs]), "7models_top10_mean": np.mean([p["top10"] for p in pairs]), "7models_top10_min": np.min([p["top10"] for p in pairs]), "7models_top20_mean": np.mean([p["top20"] for p in pairs]),
                     **{f"svr_now_{k}": v for k, v in cmp(*S["SVR"], sn, In).items()}})
    print(pd.DataFrame(rows).round(3).T.to_string())
    S0, I0 = prep(b["features_2030"])
    S1, I1 = prep(frozen(b["features_2030"]))
    print("Current SVR vs observed-yield SVR:", cmp(m_percentile(S0, g), I0, m_percentile(S1, g), I1))
    print("Countries in observed-yield SVR top 20:", b["features_2030"]["name"].loc[I1[order(m_percentile(S1, g))[:20]]].tolist())


def promethee_p():
    print("\n== PROMETHEE V-shape threshold: a dial between percentile (small p) and z-score mean (large p) ==")
    X, idx = prep(b["features_2030"])
    Xk, ik = prep(b["features_2030_knn"])
    base = m_percentile(X, g)
    z = m_zmean(X, g)
    rows = []
    for p_sd in [0.1, 0.25, 0.5, 1, 2, 3, 5, 100]:
        def flows(Y):
            F = np.empty(Y.shape)
            for k in range(Y.shape[1]):
                d = Y[:, k][:, None] - Y[:, k][None, :]
                P = np.clip(d / (p_sd * Y[:, k].std()), 0, 1)
                F[:, k] = (P - P.T).sum(1) / (len(Y) - 1)
            return F @ equal_group_w(g)
        s, sk = flows(X), flows(Xk)
        nm = pd.Series(ranks(s), index=idx)
        rows.append({"p_in_sd": p_sd, "spearman_vs_percentile": spear(s, base), "top10_vs_percentile": overlap(s, base, 10), "top20_vs_percentile": overlap(s, base, 20), "spearman_vs_zmean": spear(s, z), "top10_vs_zmean": overlap(s, z, 10), **{f"svr_knn_{k}": v for k, v in cmp(s, idx, sk, ik).items()}, "MMR": nm["MMR"], "HKG": nm["HKG"], "PRI": nm["PRI"], "CUB": nm["CUB"]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def why_odd():
    print("\n== Why do the questionable entries rank high? Percentile by feature, baseline 2030 (SVR) ==")
    X, idx = prep(b["features_2030"])
    P = pd.DataFrame(pct(X), index=idx, columns=cols)
    show = ["USA", "GBR", "DEU", "HKG", "KOR", "JPN", "MYS", "SGP", "IDN", "MMR", "PRI", "CUB"]
    print(P.loc[show].round(2).T.to_string())
    S, _ = group_scores(pct(X), g)
    print(pd.DataFrame(S, index=idx, columns=GN).loc[show].round(2).T.to_string())
    flags = extra["imputed_flags"]
    print("\nImputed inputs:", {c: [k for k, v in flags.loc[c].items() if v] for c in show})
    print("Imputed input count over ranked countries:", extra["imputed_count"].loc[idx].value_counts().sort_index().to_dict())
    print("Ranked countries with at least 4 imputed inputs:", b["features_2030"]["name"].loc[idx[extra["imputed_count"].loc[idx] >= 4]].tolist())
    # raw commitment check: beef_tax_risk = 0 is the rendering of 'no pledge'
    print("beef_tax_risk raw:", b["features_2030"]["beef_tax_risk"].loc[show].round(3).to_dict())
    print("Countries at beef_tax_risk = 0 among ranked:", int((b["features_2030"]["beef_tax_risk"].loc[idx] == 0).sum()))


def score_gaps():
    print("\n== How close are the top scores? (baseline, SVR) ==")
    X, idx = prep(b["features_2030"])
    s = m_percentile(X, g)
    o = order(s)
    print("score at rank 1, 5, 10, 11, 20, 21, 40:", [round(float(s[o[r - 1]]), 3) for r in [1, 5, 10, 11, 20, 21, 40]])
    print("score SD across countries:", round(float(s.std()), 3))
    rng = np.random.default_rng(42)
    # sampling noise of the score if one feature's percentile moves by the typical SVR-vs-KNN shift of yield_trend: 0.207
    print("A 0.207 percentile shift in yield_trend alone moves a score by:", round(0.207 / 16, 4))


def uncertainty_sources():
    print("\n== Top-10 and top-20 retention by source of uncertainty, baseline percentile, 2030 SVR as reference ==")
    X, idx = prep(b["features_2030"])
    base = m_percentile(X, g)
    rng = np.random.default_rng(42)
    Wg = rng.dirichlet(np.ones(4), size=5000)
    sizes = np.array([(g == t).sum() for t in range(4)])
    Wf = Wg[:, g] / sizes[g]
    rows = []
    t10 = [overlap(pct(X) @ w, base, 10) for w in Wf]
    t20 = [overlap(pct(X) @ w, base, 20) for w in Wf]
    rows.append({"source": "group weights, Dirichlet(1,1,1,1)", "top10": np.mean(t10), "top10_p5_p95": tuple(np.quantile(t10, [.05, .95])), "top20": np.mean(t20)})
    Wc = rng.dirichlet(np.full(4, 10.0), size=5000)
    Wcf = Wc[:, g] / sizes[g]
    t10 = [overlap(pct(X) @ w, base, 10) for w in Wcf]
    t20 = [overlap(pct(X) @ w, base, 20) for w in Wcf]
    rows.append({"source": "group weights, tighter Dirichlet(10,10,10,10) (each group about 15-35%)", "top10": np.mean(t10), "top10_p5_p95": tuple(np.quantile(t10, [.05, .95])), "top20": np.mean(t20)})
    t10, t20 = [], []
    for k, v in FX.items():
        if k == "SVR":
            continue
        x, i = prep(v)
        c = cmp(base, idx, m_percentile(x, g), i)
        t10.append(c["top10"]); t20.append(c["top20"])
    rows.append({"source": "forecast model swap, 6 other models vs SVR", "top10": np.mean(t10), "top10_p5_p95": (min(t10), max(t10)), "top20": np.mean(t20)})
    x, i = prep(frozen(b["features_2030"]))
    Sx = {}
    t10, t20 = [], []
    for k, v in FX.items():
        if k == "SVR":
            continue
        x2, i2 = prep(frozen(v))
        c = cmp(m_percentile(x, g), i, m_percentile(x2, g), i2)
        t10.append(c["top10"]); t20.append(c["top20"])
    rows.append({"source": "same, with yield_trend held at observed values, vs observed-yield SVR", "top10": np.mean(t10), "top10_p5_p95": (min(t10), max(t10)), "top20": np.mean(t20)})
    x, i = prep(Fnow)
    c = cmp(base, idx, m_percentile(x, g), i)
    rows.append({"source": "no forecast: today's table vs 2030 SVR", "top10": c["top10"], "top10_p5_p95": None, "top20": c["top20"]})
    t10, t20 = [], []
    for s in range(1000):
        Xn = X + rng.normal(0, 0.25, X.shape) * X.std(0)
        sc = m_percentile(Xn, g)
        t10.append(overlap(sc, base, 10)); t20.append(overlap(sc, base, 20))
    rows.append({"source": "value noise 0.25 SD on every value", "top10": np.mean(t10), "top10_p5_p95": tuple(np.quantile(t10, [.05, .95])), "top20": np.mean(t20)})
    t10, t20 = [], []
    for j in range(len(cols)):
        keep = np.array([k for k in range(len(cols)) if k != j])
        sc = m_percentile(X[:, keep], g[keep])
        t10.append(overlap(sc, base, 10)); t20.append(overlap(sc, base, 20))
    rows.append({"source": "drop any one feature (11 runs)", "top10": np.mean(t10), "top10_p5_p95": (min(t10), max(t10)), "top20": np.mean(t20)})
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    # Population threshold
    print("\nPopulation threshold (baseline, 2030 SVR):")
    F = b["features_2030"]
    for th in [1e6, 2e6, 5e6, 1e7]:
        r = F[F["population"] >= th]
        Y = r[cols].values.astype(float) * sgn
        sc = pd.Series(m_percentile(Y, g), index=r.index)
        top = sc.sort_values(ascending=False).index[:10]
        print(f"  {th/1e6:.0f}M: {len(r)} countries, overlap with 1M top 10 = {len(set(top) & set(idx[order(base)[:10]]))}; top 10 = {F['name'].loc[top].tolist()}")


if __name__ == "__main__":
    yield_skill()
    yield_freeze()
    promethee_p()
    why_odd()
    score_gaps()
    uncertainty_sources()
