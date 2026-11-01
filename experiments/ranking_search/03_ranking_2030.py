"""Criteria 1, 2, 5, 6 plus extra common stress tests, for every method, on the 2030 feature tables."""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import itertools, pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import *
from joblib import Parallel, delayed

pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 200, "display.max_colwidth", 60)
b = load_bundle()
cols, signs, groups = b["feature_cols"], b["signs"], b["groups"]
GN = list(groups)
g_of = {f: i for i, v in enumerate(groups.values()) for f in v}
g = np.array([g_of[c] for c in cols])
sgn = np.array([signs[c] for c in cols])
SEED = 42
SLOW = {"bod_ar", "kemeny", "schulze"}
NO_WEIGHT = {"bod_pure", "pareto_features", "pareto_groups"}
ALL = list(METHODS) + ["kemeny", "schulze"]


def prep(df, min_pop=1e6):
    r = df[df["population"] >= min_pop]
    return r[cols].values.astype(float) * sgn, r.index


def scores_df(df):
    X, idx = prep(df)
    return pd.DataFrame(all_scores(X, g), index=idx)


def expand_group(W):
    sizes = np.array([(g == t).sum() for t in range(4)])
    return W[:, g] / sizes[g]


def one_cheap(nm, X, w):
    return METHODS[nm](X, g, w)


def cheap_chunk(nm, X, W, base):
    tops10, tops20, sp, counts = [], [], [], np.zeros(len(X))
    b10, b20 = topk(base, 10), topk(base, 20)
    for w in W:
        s = METHODS[nm](X, g, w)
        o = order(s)
        t10 = set(o[:10].tolist())
        tops10.append(len(t10 & b10))
        tops20.append(len(set(o[:20].tolist()) & b20))
        sp.append(spear(s, base))
        counts[o[:10]] += 1
    return tops10, tops20, sp, counts


def slow_one(X, w, base):
    sc = all_scores(X, g, w, names=VOTERS)
    out = {}
    for nm in SLOW:
        s = sc[nm]
        o = order(s)
        c = np.zeros(len(X))
        c[o[:10]] = 1
        out[nm] = (len(set(o[:10].tolist()) & topk(base[nm], 10)), len(set(o[:20].tolist()) & topk(base[nm], 20)), spear(s, base[nm]), c)
    return out


def weight_robust(X, base, W, W_slow, label, par):
    rows = {}
    chunks = np.array_split(W, 48)
    cheap = [m for m in METHODS if m not in SLOW and m not in NO_WEIGHT]
    for nm in cheap:
        res = par(delayed(cheap_chunk)(nm, X, c, base[nm]) for c in chunks)
        t10 = np.concatenate([r[0] for r in res])
        t20 = np.concatenate([r[1] for r in res])
        sp = np.concatenate([r[2] for r in res])
        cnt = sum(r[3] for r in res) / len(W)
        rows[nm] = {"top10_mean": t10.mean(), "top10_p5": np.quantile(t10, 0.05), "top20_mean": t20.mean(), "spearman_mean": np.nanmean(sp),
                    "n_ge50pct": int((cnt >= 0.5).sum()), "n_ever": int((cnt > 0).sum()), "draws": len(W)}
    res = par(delayed(slow_one)(X, w, base) for w in W_slow)
    for nm in SLOW:
        t10 = np.array([r[nm][0] for r in res])
        t20 = np.array([r[nm][1] for r in res])
        sp = np.array([r[nm][2] for r in res])
        cnt = sum(r[nm][3] for r in res) / len(W_slow)
        rows[nm] = {"top10_mean": t10.mean(), "top10_p5": np.quantile(t10, 0.05), "top20_mean": t20.mean(), "spearman_mean": np.nanmean(sp),
                    "n_ge50pct": int((cnt >= 0.5).sum()), "n_ever": int((cnt > 0).sum()), "draws": len(W_slow)}
    out = pd.DataFrame(rows).T
    out.columns = [f"{label}_{c}" for c in out.columns]
    return out


def jack_one(X, keep):
    gk = g[keep]
    return all_scores(X[:, keep], gk)


def noise_one(X, base, seed, sd):
    rng = np.random.default_rng(seed)
    Xn = X + rng.normal(0, sd, X.shape) * X.std(0)
    sc = all_scores(Xn, g)
    return {k: (overlap(sc[k], base[k], 10), overlap(sc[k], base[k], 20), spear(sc[k], base[k])) for k in sc}


def iia_one(X, base, seed):
    rng = np.random.default_rng(seed)
    n = len(X)
    keep = np.sort(rng.choice(n, int(0.9 * n), replace=False))
    sc = all_scores(X[keep], g)
    return {k: (overlap(sc[k], np.asarray(base[k])[keep], 10), overlap(sc[k], np.asarray(base[k])[keep], 20), spear(sc[k], np.asarray(base[k])[keep])) for k in sc}


def main():
    t0 = time.time()
    par = Parallel(n_jobs=-1, batch_size=1)
    R = {"SVR": scores_df(b["features_2030"]), "KNN": scores_df(b["features_2030_knn"]), "NOW": scores_df(b["features_now"])}
    extra = pickle.load(open(HERE / "extra_2030.pkl", "rb"))
    for nm, df in extra["features"].items():
        if nm not in ("SVR", "KNN"):
            R[nm] = scores_df(df)
    pickle.dump(R, open(HERE / "scores_2030.pkl", "wb"))
    # reproduction check of the rebuilt forecasts
    for nm in ["SVR", "KNN"]:
        key = {"SVR": "features_2030", "KNN": "features_2030_knn"}[nm]
        diff = (extra["features"][nm][cols] - b[key][cols]).abs().max().max()
        print(f"rebuilt {nm} vs bundle max abs feature diff: {diff:.3g}")
        print(f"  rebuilt top 10 vs bundle top 10 overlap: {overlap(scores_df(extra['features'][nm])['percentile'].reindex(R[nm].index).fillna(-9).values, R[nm]['percentile'].values, 10)}")

    svr, knn, now = R["SVR"], R["KNN"], R["NOW"]
    X, idx = prep(b["features_2030"])
    base = {k: svr[k].values for k in svr}
    names = b["features_2030"]["name"]

    # ---- Criterion 1: forecast sensitivity
    rows = {}
    shared = svr.index.intersection(knn.index)
    shared_now = svr.index.intersection(now.index)
    fm = [m for m in R if m != "NOW"]
    for m in ALL:
        a, c = svr[m], knn[m]
        row = {"svr_knn_spearman": spear(a.loc[shared], c.loc[shared]),
               "svr_knn_top10": len(set(a.nlargest(10).index) & set(c.nlargest(10).index)) if False else len(set(idx[order(a.values)[:10]]) & set(knn.index[order(c.values)[:10]])),
               "svr_knn_top20": len(set(idx[order(a.values)[:20]]) & set(knn.index[order(c.values)[:20]])),
               "svr_now_spearman": spear(a.loc[shared_now], now[m].loc[shared_now]),
               "svr_now_top10": len(set(idx[order(a.values)[:10]]) & set(now.index[order(now[m].values)[:10]]))}
        pairs = []
        for u, v in itertools.combinations(fm, 2):
            su, sv = R[u][m], R[v][m]
            sh = su.index.intersection(sv.index)
            pairs.append((spear(su.loc[sh], sv.loc[sh]), len(set(su.index[order(su.values)[:10]]) & set(sv.index[order(sv.values)[:10]])), len(set(su.index[order(su.values)[:20]]) & set(sv.index[order(sv.values)[:20]]))))
        pairs = np.array(pairs)
        row.update({"models7_spearman_mean": pairs[:, 0].mean(), "models7_top10_mean": pairs[:, 1].mean(), "models7_top10_min": pairs[:, 1].min(), "models7_top20_mean": pairs[:, 2].mean()})
        rows[m] = row
    c1 = pd.DataFrame(rows).T
    print("\n== Criterion 1: forecast sensitivity =="); print(c1.round(3).to_string())

    # ---- Criterion 2: weights
    rng = np.random.default_rng(SEED)
    Wg = expand_group(rng.dirichlet(np.ones(4), size=5000))
    Wf = rng.dirichlet(np.ones(len(cols)), size=5000)
    c2g = weight_robust(X, base, Wg, Wg[:300], "grp", par)
    c2f = weight_robust(X, base, Wf, Wf[:300], "feat", par)
    c2 = pd.concat([c2g, c2f], axis=1)
    print("\n== Criterion 2: Dirichlet weights (group level, feature level) =="); print(c2.astype(float).round(3).to_string())
    print(f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- Common stress tests, valid for every method including weight-free ones
    drops = [("feature:" + cols[i], np.array([j for j in range(len(cols)) if j != i])) for i in range(len(cols))]
    drops += [("group:" + GN[t], np.where(g != t)[0]) for t in range(4)]
    jr = par(delayed(jack_one)(X, k) for _, k in drops)
    jrows = {}
    for m in ALL:
        ov = np.array([overlap(s[m], base[m], 10) for s in jr])
        ov20 = np.array([overlap(s[m], base[m], 20) for s in jr])
        sp = np.array([spear(s[m], base[m]) for s in jr])
        jrows[m] = {"dropfeat_top10_mean": ov[:11].mean(), "dropfeat_top10_min": ov[:11].min(), "dropfeat_top20_mean": ov20[:11].mean(), "dropfeat_spearman": sp[:11].mean(),
                    "dropgroup_top10_mean": ov[11:].mean(), "dropgroup_top10_min": ov[11:].min(), "dropgroup_spearman": sp[11:].mean()}
    c_j = pd.DataFrame(jrows).T
    print("\n== Stress: leave one feature / group out (11 + 4 reruns) =="); print(c_j.round(3).to_string())
    print("Baseline top-10 overlap by dropped item:", {drops[i][0]: overlap(jr[i]["percentile"], base["percentile"], 10) for i in range(len(drops))})

    nr = par(delayed(noise_one)(X, base, s, 0.25) for s in range(200))
    ir = par(delayed(iia_one)(X, base, s) for s in range(200))
    nrows, irows = {}, {}
    for m in ALL:
        a = np.array([r[m] for r in nr]); nrows[m] = {"noise25_top10_mean": a[:, 0].mean(), "noise25_top20_mean": a[:, 1].mean(), "noise25_spearman": a[:, 2].mean()}
        a = np.array([r[m] for r in ir]); irows[m] = {"drop10pct_top10_mean": a[:, 0].mean(), "drop10pct_top20_mean": a[:, 1].mean(), "drop10pct_spearman": a[:, 2].mean()}
    c_n, c_i = pd.DataFrame(nrows).T, pd.DataFrame(irows).T
    print("\n== Stress: value noise 0.25 SD on every value (200 runs); remove a random 10% of countries (200 runs) ==")
    print(pd.concat([c_n, c_i], axis=1).round(3).to_string())
    print(f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- Criterion 5 + 6
    rows = {}
    wanted = ["IDN", "MYS", "SGP", "PRI", "HKG", "CUB", "MMR", "USA"]
    imp = extra["imputed_count"]
    inc = extra["income_group"]
    for m in ALL:
        s = svr[m].values
        rk = pd.Series(ranks(s), index=idx)
        t10, t20 = idx[order(s)[:10]], idx[order(s)[:20]]
        rows[m] = {"spearman_base": spear(s, base["percentile"]), "top10_base": overlap(s, base["percentile"], 10), "top20_base": overlap(s, base["percentile"], 20),
                   **{c: int(rk[c]) for c in wanted}, "low_inc_in_top20": int(inc.loc[t20].isin(["Low income"]).sum()),
                   "imputed_mean_top20": float(imp.loc[t20].mean()), "ties_at_top": int((np.round(s, 6) == np.round(s, 6).max()).sum())}
    c5 = pd.DataFrame(rows).T
    print("\n== Criteria 5, 6: difference from baseline, sanity (rank out of 160) =="); print(c5.round(3).to_string())
    print("Imputed input count (of 16), mean over all ranked:", round(float(imp.loc[idx].mean()), 2), "and by country:", imp.loc[wanted].to_dict())

    # ---- Which feature carries the SVR vs KNN disagreement in the baseline?
    Fs, Fk = b["features_2030"], b["features_2030_knn"]
    rowsf = {}
    Xs, ids = prep(Fs)
    Xk = Fk.loc[ids, cols].values.astype(float) * sgn  # same countries
    Ps, Pk = pct(Xs), pct(Xk)
    bs = m_percentile(Xs, g)
    for j, c in enumerate(cols):
        X2 = Xs.copy(); X2[:, j] = Xk[:, j]
        rowsf[c] = {"spearman_svr_knn": spear(Xs[:, j], Xk[:, j]), "mean_abs_pct_shift": np.abs(Ps[:, j] - Pk[:, j]).mean(), "max_abs_pct_shift": np.abs(Ps[:, j] - Pk[:, j]).max(),
                    "weight": equal_group_w(g)[j], "swap_top10_overlap": overlap(m_percentile(X2, g), bs, 10), "swap_top20_overlap": overlap(m_percentile(X2, g), bs, 20),
                    "svr_vs_now_spearman": spear(Fs.loc[ids, c].values, b["features_now"].reindex(ids)[c].fillna(Fs[c]).loc[ids].values)}
    fa = pd.DataFrame(rowsf)
    fa = fa.T
    print("\n== Which feature carries the forecast disagreement? (swap = put the KNN version of this one feature into the SVR table) ==")
    print(fa.round(3).to_string())

    allrank = pd.DataFrame({m: pd.Series(ranks(svr[m].values), index=idx) for m in ALL})
    allrank.insert(0, "name", names.loc[idx])
    allrank.sort_values("percentile").to_csv(HERE / "ranks_all_methods_svr.csv")
    full = pd.concat([c1, c2.astype(float), c_j, c_n, c_i, c5], axis=1)
    full.to_csv(HERE / "criteria_2030.csv")
    fa.to_csv(HERE / "feature_disagreement.csv")
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
