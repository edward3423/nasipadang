"""Candidate final procedure: percentile score averaged over forecast models, with Monte Carlo weight sampling used for the uncertainty report.

Score = mean over 7 forecast models of the equal-group-weight percentile score.
Because a symmetric Dirichlet has mean weight 1/4 per group, this score is exactly the expected score over the weight draws,
so the Monte Carlo layer changes nothing in the order: it only adds P(top 10), P(top 20) and rank ranges.
"""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import itertools, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import *
from panels import *
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 300, "display.max_colwidth", 90)
b = load_bundle()
cols, signs, groups = b["feature_cols"], b["signs"], b["groups"]
GN = list(groups)
g_of = {f: i for i, v in enumerate(groups.values()) for f in v}
g = np.array([g_of[c] for c in cols])
sgn = np.array([signs[c] for c in cols])
extra = pickle.load(open(HERE / "extra_2030.pkl", "rb"))
FX = extra["features"]
MODELS = list(FX)
SHORT = {"Demand": "Demand", "Rice and beef supply": "Supply", "Climate and policy": "Climate", "Kitchen and operations": "Kitchen"}
B = 2000


def final_set():
    votes = sum((FX[m]["population"] >= 1e6).astype(int) for m in MODELS)
    return votes.index[votes >= 4]


def signed(df, ids):
    return df.loc[ids, cols].values.astype(float) * sgn


def cmp_s(a, c):
    return spear(a, c), overlap(a, c, 10), overlap(a, c, 20)


def boot_svr_knn(seed, Xa, Xb):
    rng = np.random.default_rng(seed)
    r = rng.integers(0, len(Xa), len(Xa))
    return cmp_s(m_percentile(Xa[r], g), m_percentile(Xb[r], g))


def boot_split(seed, Xs):
    rng = np.random.default_rng(seed)
    n = len(Xs[0])
    r = rng.integers(0, n, n)
    perm = rng.permutation(len(Xs))
    a = perm[:3]
    c = perm[3:]
    sa = np.mean([m_percentile(Xs[i][r], g) for i in a], axis=0)
    sc = np.mean([m_percentile(Xs[i][r], g) for i in c], axis=0)
    return cmp_s(sa, sc)


def smaa_pool(Ps, W):
    """Ranks for every forecast table x every weight draw. Returns (mean rank, P top10, P top20, p5, p95, by-model P top10)."""
    allr, per = [], []
    for P in Ps:
        S, _ = group_scores(P, g)
        sc = S @ W.T
        o = np.argsort(-np.round(sc, 10), axis=0, kind="stable")
        r = np.empty_like(o)
        np.put_along_axis(r, o, np.arange(1, S.shape[0] + 1)[:, None].repeat(W.shape[0], 1), axis=0)
        allr.append(r)
        per.append((r <= 10).mean(1))
    R = np.concatenate(allr, axis=1)
    return R.mean(1), (R <= 10).mean(1), (R <= 20).mean(1), np.quantile(R, 0.05, axis=1), np.quantile(R, 0.95, axis=1), np.column_stack(per)


def backtest_ensemble():
    print("\n== Backtest of the ensemble forecast. Train 1990 -> 2000, test 2000 -> 2010, percentile score, 132 countries ==")
    pan = load_panels()
    A0, A1, P = b["backtest_actual_2000"], b["backtest_actual_2010"], b["backtest_svr_forecast_2010"]
    ix = A0.index[A0["population"] >= np.log(1e6)]
    T = ["urban", "gni", "u5"]
    mk = makers()
    mk["et"] = lambda: make_pipeline(SimpleImputer(strategy="median"), ExtraTreesRegressor(500, min_samples_leaf=3, random_state=42, n_jobs=-1))
    mk["gb"] = lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=10, random_state=42)
    mk["mlp"] = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), MLPRegressor(hidden_layer_sizes=(32,), alpha=1.0, max_iter=3000, random_state=42))
    F = {k: forecast_targets(pan, [(1990, 2000)], 2000, 10, T, m, A0.index) for k, m in mk.items()}
    mat = lambda d: np.column_stack([d.loc[ix, "urban"], d.loc[ix, "gni"], -d.loc[ix, "u5"]]).astype(float)
    X1, X0 = mat(A1), mat(A0)
    Xf = {k: mat(v) for k, v in F.items()}
    g3 = np.arange(3)
    t = pct(X1).mean(1)
    nc = pct(X0).mean(1)
    n = len(ix)
    rows = [{"forecast": "no change", "spearman": spear(nc, t), "top10": overlap(nc, t, 10), "top20": overlap(nc, t, 20)}]
    for k, v in Xf.items():
        s = pct(v).mean(1)
        rows.append({"forecast": k, "spearman": spear(s, t), "top10": overlap(s, t, 10), "top20": overlap(s, t, 20)})
    ens = np.mean([pct(v).mean(1) for v in Xf.values()], axis=0)
    rows.append({"forecast": "ensemble of 7", "spearman": spear(ens, t), "top10": overlap(ens, t, 10), "top20": overlap(ens, t, 20)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    rng = np.random.default_rng(42)
    rr = rng.integers(0, n, (B, n))
    gaps = {k: [] for k in ["spearman", "top10", "top20"]}
    gaps_svr = {k: [] for k in gaps}
    mv, mv_mr = [], []
    keys = list(Xf)
    for r in rr:
        tb, nb = pct(X1[r]).mean(1), pct(X0[r]).mean(1)
        eb = np.mean([pct(Xf[k][r]).mean(1) for k in keys], axis=0)
        sb = pct(Xf["svr"][r]).mean(1)
        gaps["spearman"].append(spear(eb, tb) - spear(nb, tb)); gaps["top10"].append(overlap(eb, tb, 10) - overlap(nb, tb, 10)); gaps["top20"].append(overlap(eb, tb, 20) - overlap(nb, tb, 20))
        gaps_svr["spearman"].append(spear(eb, tb) - spear(sb, tb)); gaps_svr["top10"].append(overlap(eb, tb, 10) - overlap(sb, tb, 10)); gaps_svr["top20"].append(overlap(eb, tb, 20) - overlap(sb, tb, 20))
        r0 = ranks(nb)
        da = (r0 - ranks(tb)).astype(float)
        de = (r0 - ranks(eb)).astype(float)
        mv.append(spear(de, da)); mv_mr.append(spear(r0 - (n + 1) / 2, da))
    for k in gaps:
        print(f"  ensemble minus no-change, {k}: CI {np.nanquantile(gaps[k], .025):.3f} to {np.nanquantile(gaps[k], .975):.3f}; ensemble minus SVR: CI {np.nanquantile(gaps_svr[k], .025):.3f} to {np.nanquantile(gaps_svr[k], .975):.3f}")
    r0 = ranks(nc); da = (r0 - ranks(t)).astype(float); de = (r0 - ranks(ens)).astype(float)
    print(f"  movers: ensemble rho {spear(de, da):.3f} (CI {np.nanquantile(mv, .025):.3f} to {np.nanquantile(mv, .975):.3f}); mean-reversion rule {spear(r0 - (n + 1) / 2, da):.3f}; gap CI {np.nanquantile(np.array(mv) - np.array(mv_mr), .025):.3f} to {np.nanquantile(np.array(mv) - np.array(mv_mr), .975):.3f}")

    # yield: ensemble skill
    L = pan["L"]["yield"]
    ok = L[1999].reindex(ix).notna() & L[2009].reindex(ix).notna() & L[1990].reindex(ix).notna()
    ids = ix[ok.values]
    y = (L[2009] - L[1999]).loc[ids]
    preds = []
    yy = (L[1999] - L[1990]) * 10 / 9
    o2 = yy.notna()
    for k, m_ in mk.items():
        preds.append(m_().fit(features_at(pan, 1990)[o2], yy[o2]).predict(features_at(pan, 1999).loc[ids]))
    ep = np.mean(preds, axis=0)
    rng = np.random.default_rng(1)
    bs = [spear(ep[r], y.values[r]) for r in rng.integers(0, len(y), (B, len(y)))]
    bs2 = []
    print(f"  yield change 1999 -> 2009, ensemble: RMSE {np.sqrt(((y.values - ep) ** 2).mean()):.3f}, Spearman with actual {spear(ep, y.values):.3f} (CI {np.quantile(bs, .025):.3f} to {np.quantile(bs, .975):.3f}), {len(y)} countries")


def main():
    t0 = time.time()
    ids = final_set()
    print("Ranked countries (>=1M in at least 4 of 7 forecast models):", len(ids))
    print("Membership differs across models for:", sorted(set().union(*[set(FX[m].index[FX[m]['population'] >= 1e6]) for m in MODELS]) - set(FX['SVR'].index[(FX['SVR']['population'] >= 1e6)]) ), sorted(set(FX['SVR'].index[(FX['SVR']['population'] >= 1e6)]) - set(ids)))
    Xs = {m: signed(FX[m], ids) for m in MODELS}
    Ps = {m: pct(Xs[m]) for m in MODELS}
    sc = {m: Ps[m] @ equal_group_w(g) for m in MODELS}
    ens = np.mean(list(sc.values()), axis=0)
    names = FX["SVR"]["name"].loc[ids]

    print("\n== Single forecast models vs ensembles ==")
    rows = []
    for u, v in itertools.combinations(MODELS, 2):
        rows.append(cmp_s(sc[u], sc[v]))
    a = np.array(rows)
    print(f"Single model vs single model, 21 pairs: rho mean {a[:,0].mean():.3f} (min {a[:,0].min():.3f}); top10 mean {a[:,1].mean():.2f} (min {a[:,1].min():.0f}); top20 mean {a[:,2].mean():.2f} (min {a[:,2].min():.0f})")
    rows = []
    for comb in itertools.combinations(range(7), 3):
        rest = [i for i in range(7) if i not in comb]
        rows.append(cmp_s(np.mean([sc[MODELS[i]] for i in comb], axis=0), np.mean([sc[MODELS[i]] for i in rest], axis=0)))
    a = np.array(rows)
    print(f"Ensemble of 3 vs ensemble of the other 4, 35 splits: rho mean {a[:,0].mean():.3f} (min {a[:,0].min():.3f}); top10 mean {a[:,1].mean():.2f} (min {a[:,1].min():.0f}); top20 mean {a[:,2].mean():.2f} (min {a[:,2].min():.0f})")
    rows = []
    for h in MODELS:
        others = [m for m in MODELS if m != h]
        e = np.mean([sc[m] for m in others], axis=0)
        ce = cmp_s(e, sc[h])
        cs = np.mean([cmp_s(sc[m], sc[h]) for m in others], axis=0)
        cv = cmp_s(sc["SVR"], sc[h]) if h != "SVR" else (np.nan, np.nan, np.nan)
        rows.append({"held_out": h, "ens6_rho": ce[0], "ens6_top10": ce[1], "ens6_top20": ce[2], "avg_single_rho": cs[0], "avg_single_top10": cs[1], "avg_single_top20": cs[2], "svr_rho": cv[0], "svr_top10": cv[1], "svr_top20": cv[2]})
    t = pd.DataFrame(rows)
    print("Predicting a forecast model that was left out: ensemble of the other 6 vs one single model")
    print(t.round(3).to_string(index=False))
    print(t.drop(columns="held_out").mean().round(3).to_dict())

    par = Parallel(n_jobs=-1, batch_size=8)
    bs = np.array(par(delayed(boot_svr_knn)(s, Xs["SVR"], Xs["KNN"]) for s in range(B)))
    print(f"\nBootstrap over countries. SVR vs KNN (percentile): rho {cmp_s(sc['SVR'], sc['KNN'])[0]:.3f} CI {np.quantile(bs[:,0],.025):.3f}-{np.quantile(bs[:,0],.975):.3f}; top10 {cmp_s(sc['SVR'], sc['KNN'])[1]} CI {np.quantile(bs[:,1],.025):.0f}-{np.quantile(bs[:,1],.975):.0f}; top20 {cmp_s(sc['SVR'], sc['KNN'])[2]} CI {np.quantile(bs[:,2],.025):.0f}-{np.quantile(bs[:,2],.975):.0f}")
    bs = np.array(par(delayed(boot_split)(s, [Xs[m] for m in MODELS]) for s in range(B)))
    print(f"Bootstrap over countries and random model split, 3 vs 4: rho mean {bs[:,0].mean():.3f} CI {np.quantile(bs[:,0],.025):.3f}-{np.quantile(bs[:,0],.975):.3f}; top10 mean {bs[:,1].mean():.2f} CI {np.quantile(bs[:,1],.025):.0f}-{np.quantile(bs[:,1],.975):.0f}; top20 mean {bs[:,2].mean():.2f} CI {np.quantile(bs[:,2],.025):.0f}-{np.quantile(bs[:,2],.975):.0f}")

    # ---- Weight sampling
    rng = np.random.default_rng(42)
    W = rng.dirichlet(np.ones(4), size=5000)
    W10 = rng.dirichlet(np.full(4, 10.0), size=5000)
    Pl = [Ps[m] for m in MODELS]
    mr, p10, p20, lo, hi, per = smaa_pool(Pl, W)
    mr10, p10t, p20t, lo10, hi10, per10 = smaa_pool(Pl, W10)
    _, p10_svr, p20_svr, _, _, _ = smaa_pool([Ps["SVR"]], W)
    # ---- Retention under weights for ensemble vs baseline
    S_e = np.mean([group_scores(P, g)[0] for P in Pl], axis=0)
    def ret(Sx, Wx, base):
        t10, t20 = [], []
        for w in Wx:
            s = Sx @ w
            t10.append(overlap(s, base, 10)); t20.append(overlap(s, base, 20))
        return np.mean(t10), np.quantile(t10, .05), np.mean(t20)
    print("\nRetention of nominal top 10 / top 20 under weights:")
    print("  baseline SVR, Dirichlet(1):", np.round(ret(group_scores(Ps['SVR'], g)[0], W, sc['SVR']), 2), " Dirichlet(10):", np.round(ret(group_scores(Ps['SVR'], g)[0], W10, sc['SVR']), 2))
    print("  ensemble, Dirichlet(1):", np.round(ret(S_e, W, ens), 2), " Dirichlet(10):", np.round(ret(S_e, W10, ens), 2))
    # common value noise, applied identically to all tables
    t10b, t20b, t10e, t20e = [], [], [], []
    for s in range(300):
        rg = np.random.default_rng(s)
        N = rg.normal(0, 0.25, Xs["SVR"].shape)
        nb = m_percentile(Xs["SVR"] + N * Xs["SVR"].std(0), g)
        ne = np.mean([m_percentile(Xs[m] + N * Xs[m].std(0), g) for m in MODELS], axis=0)
        t10b.append(overlap(nb, sc["SVR"], 10)); t20b.append(overlap(nb, sc["SVR"], 20)); t10e.append(overlap(ne, ens, 10)); t20e.append(overlap(ne, ens, 20))
    print(f"Value noise 0.25 SD (same noise on all tables): baseline top10 {np.mean(t10b):.2f}, top20 {np.mean(t20b):.2f}; ensemble top10 {np.mean(t10e):.2f}, top20 {np.mean(t20e):.2f}")

    # ---- Observed yield variant
    Ps_o = {}
    for m in MODELS:
        x = FX[m].copy()
        x["yield_trend"] = b["features_now"]["yield_trend"].reindex(x.index)
        Ps_o[m] = pct(signed(x, ids))
    ens_o = np.mean([Ps_o[m] @ equal_group_w(g) for m in MODELS], axis=0)
    # ---- Without yield_trend
    keep = np.array([i for i, c in enumerate(cols) if c != "yield_trend"])
    ens_d = np.mean([pct(Xs[m][:, keep]) @ equal_group_w(g[keep]) for m in MODELS], axis=0)
    print("\nEnsemble vs variants: (rho, top10, top20)")
    print("  vs baseline SVR:", np.round(cmp_s(ens, sc['SVR']), 3), "| vs KNN table:", np.round(cmp_s(ens, sc['KNN']), 3))
    print("  vs observed yield_trend in all tables:", np.round(cmp_s(ens, ens_o), 3))
    print("  vs yield_trend removed:", np.round(cmp_s(ens, ens_d), 3))
    print("  vs SMAA mean rank pooled over models:", np.round(cmp_s(ens, -mr), 3), "| vs P(top 10):", np.round(cmp_s(ens, p10 - 1e-6 * mr), 3))
    print("  vs today's table:", end=" ")
    now = b["features_now"]
    sh = ids.intersection(now.index[now["population"] >= 1e6])
    sn = pd.Series(m_percentile(signed(now, now.index[now["population"] >= 1e6]), g), index=now.index[now["population"] >= 1e6])
    print(round(spear(pd.Series(ens, index=ids)[sh], sn[sh]), 3), "top10", len(set(ids[order(ens)[:10]]) & set(sn.index[order(sn.values)[:10]])), "top20", len(set(ids[order(ens)[:20]]) & set(sn.index[order(sn.values)[:20]])))

    # ---- Labels
    Se = pd.DataFrame(S_e, index=ids, columns=GN)
    Z = Se - Se.mean(0)
    strength = Z.idxmax(axis=1).map(SHORT)
    weakness = Z.idxmin(axis=1).map(SHORT)
    imp = extra["imputed_count"].reindex(ids)
    out = pd.DataFrame({"name": names, "score": ens, "rank": ranks(ens), "p_top10": p10, "p_top20": p20, "rank_p5": lo, "rank_p95": hi, "mean_rank": mr,
                        "p_top10_tight": p10t, "p_top20_tight": p20t, "rank_p5_tight": lo10, "rank_p95_tight": hi10,
                        "p10_min_model": per.min(1), "p10_max_model": per.max(1), "baseline_rank": ranks(sc["SVR"]), "knn_rank": ranks(sc["KNN"]),
                        "strength": strength, "weakness": weakness, "imputed_inputs": imp}, index=ids).sort_values("rank")
    out = pd.concat([out, Se.rename(columns=SHORT).loc[out.index].round(3)], axis=1)
    out.to_csv(HERE / "final_ranking_2030.csv")
    print("\n== Top 25, recommended order ==")
    print(out.head(25).round(3).to_string())
    print("\nSanity:")
    print(out.loc[["IDN", "MYS", "SGP", "PRI", "HKG", "CUB", "MMR"]].round(3).to_string())
    print("\nCounts: P(top 10) >= 0.5:", int((out.p_top10 >= .5).sum()), "; >= 0.25:", int((out.p_top10 >= .25).sum()), "; P(top 20) >= 0.5:", int((out.p_top20 >= .5).sum()), "; countries ever in a top 10:", int((out.p_top10 > 0).sum()))
    print("With tighter weights, Dirichlet(10): P(top 10) >= 0.5:", int((out.p_top10_tight >= .5).sum()), "; P(top 20) >= 0.5:", int((out.p_top20_tight >= .5).sum()), "; ever in top 10:", int((out.p_top10_tight > 0).sum()))
    base_top = set(ids[order(sc['SVR'])[:20]])
    ens_top = set(ids[order(ens)[:20]])
    print("Enter top 20 vs baseline:", names.loc[sorted(ens_top - base_top)].tolist(), "| leave:", names.loc[sorted(base_top - ens_top)].tolist())
    base_top10 = set(ids[order(sc['SVR'])[:10]])
    ens_top10 = set(ids[order(ens)[:10]])
    print("Enter top 10 vs baseline:", names.loc[sorted(ens_top10 - base_top10)].tolist(), "| leave:", names.loc[sorted(base_top10 - ens_top10)].tolist())
    print("Top 10, by baseline, with imputed>=4 excluded:", [names[c] for c in out.index if out.loc[c, 'imputed_inputs'] < 4][:10])
    print("Elapsed", round(time.time() - t0))
    backtest_ensemble()


if __name__ == "__main__":
    main()
