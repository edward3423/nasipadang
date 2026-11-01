"""Criteria 3 and 4: backtest every method on 2000 -> 2010.

Indicators: urban (+), gni (+), u5 (-). Each indicator is its own group, equal weight.
Population is not scored (urban population already measures market size). It only applies the
1 million filter, using the 2000 value, which was known at forecast time.
Truth for each method = the same method on actual 2010. So this measures predictability, not rightness.
"""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import *
from panels import *
from joblib import Parallel, delayed

pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 300, "display.max_colwidth", 60)
b = load_bundle()
TARGETS = ["urban", "gni", "u5"]
g3 = np.arange(3)
ALL = list(METHODS) + ["kemeny", "schulze"]
B = 2000
CHANCE_K = 20


def mat(df):
    return np.column_stack([df["urban"], df["gni"], -df["u5"]]).astype(float)


def partial_spear(a, b, c):
    """Spearman of a and b after removing the linear effect of rank c from both."""
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    Z = np.column_stack([np.ones(len(rc)), rc])
    ea = ra - Z @ np.linalg.lstsq(Z, ra, rcond=None)[0]
    eb = rb - Z @ np.linalg.lstsq(Z, rb, rcond=None)[0]
    if ea.std() == 0 or eb.std() == 0:
        return np.nan
    return float(np.corrcoef(ea, eb)[0, 1])


def evaluate(sc, n, noise_seed=7):
    """sc: dict scenario -> dict method -> score. Returns a flat dict of metrics."""
    rng = np.random.default_rng(noise_seed)
    tie_truth = rng.uniform(0, 1e-3, n)
    tie_pred = rng.uniform(0, 1e-3, n)
    out = {}
    truth = sc["actual"]
    nc = sc["nochange"]
    for m in truth:
        t = truth[m]
        r0 = ranks(nc[m])
        rt = ranks(t)
        d_act = (r0 - rt).astype(float)
        hit_truth = set(np.argsort(-(d_act + tie_truth), kind="stable")[:CHANCE_K].tolist())
        for f in sc:
            if f == "actual":
                continue
            s = sc[f][m]
            out[(m, f, "spearman")] = spear(s, t)
            out[(m, f, "top10")] = overlap(s, t, 10)
            out[(m, f, "top20")] = overlap(s, t, 20)
            if f != "nochange":
                d = (r0 - ranks(s)).astype(float)
                out[(m, f, "mover_spearman")] = spear(d, d_act)
                out[(m, f, "mover_partial")] = partial_spear(d, d_act, r0)
                out[(m, f, "mover_hits20")] = len(set(np.argsort(-(d + tie_pred), kind="stable")[:CHANCE_K].tolist()) & hit_truth)
        d_mr = r0 - (n + 1) / 2
        out[(m, "mean_reversion", "mover_spearman")] = spear(d_mr, d_act)
        out[(m, "mean_reversion", "mover_hits20")] = len(set(np.argsort(-(d_mr + tie_pred), kind="stable")[:CHANCE_K].tolist()) & hit_truth)
    return out


def run_scores(mats, rows=None, use_slow=True):
    res = {}
    for f, X in mats.items():
        XX = X if rows is None else X[rows]
        res[f] = all_scores(XX, g3)
    return res


def joint_rows(mats, rows, n, W):
    """Forecast ensembles: scenario SMAA over SVR and KNN, and the mean of the two percentile scores."""
    Xt, Xs, Xk = mats["actual"][rows], mats["svr"][rows], mats["knn"][rows]
    Pt, Ps, Pk = pct(Xt), pct(Xs), pct(Xk)
    def meanrank(*Ps_):
        rk = []
        for P_ in Ps_:
            sc = P_ @ W.T
            o = np.argsort(-sc, axis=0, kind="stable")
            r = np.empty_like(o)
            np.put_along_axis(r, o, np.arange(1, P_.shape[0] + 1)[:, None].repeat(W.shape[0], 1), axis=0)
            rk.append(r)
        return -np.concatenate(rk, axis=1).mean(1)
    truth = meanrank(Pt)
    joint = meanrank(Ps, Pk)
    tp = Pt.mean(1)
    mp = (Ps.mean(1) + Pk.mean(1)) / 2
    return {("smaa_joint_svr_knn", "spearman"): spear(joint, truth), ("smaa_joint_svr_knn", "top10"): overlap(joint, truth, 10), ("smaa_joint_svr_knn", "top20"): overlap(joint, truth, 20),
            ("percentile_mean_svr_knn", "spearman"): spear(mp, tp), ("percentile_mean_svr_knn", "top10"): overlap(mp, tp, 10), ("percentile_mean_svr_knn", "top20"): overlap(mp, tp, 20)}


def boot_iter(seed, mats, n, W):
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, n, n)
    sc = run_scores(mats, rows)
    out = evaluate(sc, n, noise_seed=seed + 1)
    out.update({("joint", *k): v for k, v in joint_rows(mats, rows, n, W).items()})
    return out


def table(point, boot, n):
    rows = []
    for m in ALL:
        r = {"method": m}
        for metric in ["spearman", "top10", "top20"]:
            r[f"svr_{metric}"] = point[(m, "svr", metric)]
            r[f"nc_{metric}"] = point[(m, "nochange", metric)]
            gap = np.array([x[(m, "svr", metric)] - x[(m, "nochange", metric)] for x in boot])
            r[f"gap_{metric}"] = point[(m, "svr", metric)] - point[(m, "nochange", metric)]
            r[f"gap_{metric}_lo"], r[f"gap_{metric}_hi"] = np.nanquantile(gap, [0.025, 0.975])
            r[f"gap_{metric}_pgt0"] = np.nanmean(gap > 0)
        for f in ["knn", "trend"]:
            for metric in ["spearman", "top10", "top20"]:
                r[f"{f}_{metric}"] = point[(m, f, metric)]
        rows.append(r)
    return pd.DataFrame(rows).set_index("method")


def movers_table(point, boot, n):
    rows = []
    for m in ALL:
        r = {"method": m, "chance_hits": CHANCE_K ** 2 / n}
        for f in ["svr", "knn", "trend", "mean_reversion"]:
            r[f"{f}_rho"] = point[(m, f, "mover_spearman")]
            r[f"{f}_hits"] = point[(m, f, "mover_hits20")]
        for f in ["svr", "knn", "trend"]:
            r[f"{f}_partial"] = point[(m, f, "mover_partial")]
        a = lambda f, k: np.array([x[(m, f, k)] for x in boot], dtype=float)
        r["svr_rho_lo"], r["svr_rho_hi"] = np.nanquantile(a("svr", "mover_spearman"), [0.025, 0.975])
        r["svr_minus_mr_lo"], r["svr_minus_mr_hi"] = np.nanquantile(a("svr", "mover_spearman") - a("mean_reversion", "mover_spearman"), [0.025, 0.975])
        r["svr_minus_trend_lo"], r["svr_minus_trend_hi"] = np.nanquantile(a("svr", "mover_spearman") - a("trend", "mover_spearman"), [0.025, 0.975])
        r["svr_partial_lo"], r["svr_partial_hi"] = np.nanquantile(a("svr", "mover_partial"), [0.025, 0.975])
        r["svr_hits_lo"], r["svr_hits_hi"] = np.nanquantile(a("svr", "mover_hits20"), [0.025, 0.975])
        rows.append(r)
    return pd.DataFrame(rows).set_index("method")


def main():
    t0 = time.time()
    pan = load_panels()
    A0, A1, P = b["backtest_actual_2000"], b["backtest_actual_2010"], b["backtest_svr_forecast_2010"]
    ix_all = A0.index
    keep = A0["population"] >= np.log(1e6)
    ix = ix_all[keep]
    print(f"Backtest countries: {len(ix_all)} in bundle, {len(ix)} with at least 1 million people in 2000")
    mk = makers()
    # Reproduce the bundle's SVR forecast, then build other forecasts the same way (train only on 1990 -> 2000)
    Fsvr = forecast_targets(pan, [(1990, 2000)], 2000, 10, TARGETS, mk["svr"], ix_all)
    print("Bundle SVR reproduced, max abs log diff:", float((Fsvr[TARGETS] - P[TARGETS]).abs().max().max()))
    Fknn = forecast_targets(pan, [(1990, 2000)], 2000, 10, TARGETS, mk["knn"], ix_all)
    Fridge = forecast_targets(pan, [(1990, 2000)], 2000, 10, TARGETS, mk["ridge"], ix_all)
    Frf = forecast_targets(pan, [(1990, 2000)], 2000, 10, TARGETS, mk["rf"], ix_all)
    Fsvr_noinc = forecast_targets(pan, [(1990, 2000)], 2000, 10, TARGETS, mk["svr"], ix_all, income=False)
    L = pan["L"]
    tr = pd.DataFrame({k: A0[k] + ((L[k][2000] - L[k][1995]) * 2).reindex(ix_all).fillna((L[k][2000] - L[k][1990]).mean()) for k in TARGETS})
    mean_sk = (P[TARGETS] + Fknn[TARGETS]) / 2

    def mats(ids):
        return {"actual": mat(A1.loc[ids]), "nochange": mat(A0.loc[ids]), "svr": mat(P.loc[ids]), "knn": mat(Fknn.loc[ids]), "trend": mat(tr.loc[ids])}

    M = mats(ix)
    n = len(ix)
    rng = np.random.default_rng(42)
    W = rng.dirichlet(np.ones(3), size=500)

    # indicator-level forecast skill, for context
    print("\n== Indicator-level skill, 2000 -> 2010 (no-change RMSE of log vs forecast RMSE, rank corr of change) ==")
    rows = []
    for k in TARGETS:
        y = (A1[k] - A0[k]).loc[ix]
        for nm, F_ in [("no_change", A0), ("svr", P), ("svr_no_income_input", Fsvr_noinc), ("knn", Fknn), ("trend", tr), ("ridge", Fridge), ("rf", Frf)]:
            pr = (F_[k] - A0[k]).loc[ix]
            rows.append({"indicator": k, "forecast": nm, "rmse": float(np.sqrt(((y - pr) ** 2).mean())), "rank_corr_change": spear(y, pr) if pr.std() > 0 else np.nan})
    print(pd.DataFrame(rows).pivot(index="forecast", columns="indicator", values=["rmse", "rank_corr_change"]).round(3).to_string())

    # ---- Point estimates, then the bootstrap
    sc = run_scores(M)
    point = evaluate(sc, n)
    jp = joint_rows(M, np.arange(n), n, W)
    print(f"[{time.time()-t0:.0f}s] start bootstrap B={B}", flush=True)
    boot = Parallel(n_jobs=-1, batch_size=4)(delayed(boot_iter)(s, M, n, W) for s in range(B))
    pickle.dump({"point": point, "boot": boot, "joint_point": jp, "ix": list(ix)}, open(HERE / "backtest_boot.pkl", "wb"))
    print(f"[{time.time()-t0:.0f}s] bootstrap done", flush=True)

    T = table(point, boot, n)
    MV = movers_table(point, boot, n)
    T.to_csv(HERE / "backtest_table.csv")
    MV.to_csv(HERE / "movers_table.csv")
    print("\n== Criterion 3: forecast vs no-change against the method's own actual-2010 ranking ==")
    print(T[["svr_spearman", "nc_spearman", "gap_spearman", "gap_spearman_lo", "gap_spearman_hi", "gap_spearman_pgt0"]].round(4).to_string())
    print(T[["svr_top10", "nc_top10", "gap_top10", "gap_top10_lo", "gap_top10_hi", "svr_top20", "nc_top20", "gap_top20", "gap_top20_lo", "gap_top20_hi"]].round(2).to_string())
    print(T[["knn_spearman", "knn_top10", "knn_top20", "trend_spearman", "trend_top10", "trend_top20"]].round(3).to_string())
    print("\n== Criterion 4: rank-change skill. Chance hits for top-20 risers =", round(CHANCE_K ** 2 / n, 2), "==")
    print(MV.round(3).to_string())
    print("\n== Forecast ensembles, point and bootstrap gap vs no-change (baseline family) ==")
    for k in ["smaa_joint_svr_knn", "percentile_mean_svr_knn"]:
        base_m = "smaa_meanrank" if k.startswith("smaa") else "percentile"
        for metric in ["spearman", "top10", "top20"]:
            gap = np.array([x[("joint", k, metric)] - x[(base_m, "nochange", metric)] for x in boot], dtype=float)
            print(f"  {k} {metric}: {jp[(k, metric)]:.3f} vs no-change {point[(base_m, 'nochange', metric)]:.3f}; gap CI {np.nanquantile(gap, .025):.3f} to {np.nanquantile(gap, .975):.3f}")

    # ---- Other forecasts, point estimate only
    print("\n== Baseline family, more forecasts, point estimates ==")
    extra = {"ridge": Fridge, "rf": Frf, "svr_no_income_input": Fsvr_noinc, "mean_svr_knn": mean_sk}
    rows = []
    t_ = m_percentile(M["actual"], g3)
    for nm, F_ in [("nochange", A0), ("svr", P), ("knn", Fknn), ("trend", tr), *extra.items()]:
        s = m_percentile(mat(F_.loc[ix]), g3)
        rows.append({"forecast": nm, "spearman": spear(s, t_), "top10": overlap(s, t_, 10), "top20": overlap(s, t_, 20)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---- Momentum baseline (needs complete 1990 data): the method's own 1990 -> 2000 rank change repeats
    ok = (L["urban"][1990].notna() & L["gni"][1990].notna() & L["u5"][1990].notna()).reindex(ix)
    sub = ix[ok.values]
    X90 = np.column_stack([L["urban"][1990].loc[sub], L["gni"][1990].loc[sub], -L["u5"][1990].loc[sub]])
    MS = {k: mats(sub)[k] for k in ["actual", "nochange", "svr"]}
    sc90 = all_scores(X90, g3)
    sc_ = run_scores(MS)
    rows = []
    for m in ALL:
        r0 = ranks(sc_["nochange"][m])
        act = (r0 - ranks(sc_["actual"][m])).astype(float)
        mom = (ranks(sc90[m]) - r0).astype(float)
        sv = (r0 - ranks(sc_["svr"][m])).astype(float)
        rows.append({"method": m, "n": len(sub), "rho_svr": spear(sv, act), "rho_momentum": spear(mom, act)})
    mom_t = pd.DataFrame(rows).set_index("method")
    print("\n== Momentum baseline: repeat the 1990 -> 2000 rank change. Countries with complete 1990 data ==")
    print(mom_t.round(3).to_string())
    mom_t.to_csv(HERE / "momentum.csv")

    # ---- Sensitivities, point estimates
    print("\n== Sensitivity: no population filter (157 countries), point estimates ==")
    M157 = mats(ix_all)
    s157 = run_scores(M157)
    p157 = evaluate(s157, len(ix_all))
    print(pd.DataFrame([{ "method": m, **{f"{f}_{k}": p157[(m, f, k)] for f in ["svr", "nochange"] for k in ["spearman", "top10", "top20"]}} for m in ALL]).set_index("method").round(3).to_string())

    print("\n== Sensitivity: four indicators, adding cereal yield (+). Yield 1999 -> 2009, SVR trained on 1990 -> 1999 ==")
    y0, y1 = L["yield"][1999], L["yield"][2009]
    ok4 = (y0.reindex(ix).notna() & y1.reindex(ix).notna()).values
    ix4 = ix[ok4]
    m_y = mk["svr"]().fit(features_at(pan, 1990)[(y0 - L["yield"][1990]).notna()], (y0 - L["yield"][1990]).dropna() * 10 / 9)
    yP = y0.loc[ix4] + m_y.predict(features_at(pan, 1999).loc[ix4])
    def m4(df, yv):
        return np.column_stack([df.loc[ix4, "urban"], df.loc[ix4, "gni"], -df.loc[ix4, "u5"], yv.loc[ix4]]).astype(float)
    M4 = {"actual": m4(A1, y1), "nochange": m4(A0, y0), "svr": m4(P, yP)}
    g4 = np.arange(4)
    s4 = {f: all_scores(X, g4) for f, X in M4.items()}
    print(f"countries: {len(ix4)}")
    print(pd.DataFrame([{"method": m, **{f"{f}_{k}": v for f in ["svr", "nochange"] for k, v in zip(["spearman", "top10", "top20"], [spear(s4[f][m], s4["actual"][m]), overlap(s4[f][m], s4["actual"][m], 10), overlap(s4[f][m], s4["actual"][m], 20)])}} for m in ALL]).set_index("method").round(3).to_string())

    print("\n== Sensitivity: rolling 5-year windows, SVR trained only on earlier windows. Not independent of the main test except 1995 -> 2000 ==")
    W5 = [([(1990, 1995)], 1995, 2000), ([(1990, 1995), (1995, 2000)], 2000, 2005), ([(1990, 1995), (1995, 2000), (2000, 2005)], 2005, 2010)]
    rows = []
    for pairs, o, e in W5:
        ok5 = pd.concat([L[k][o].notna() & L[k][e].notna() for k in TARGETS], axis=1).all(axis=1) & (L["population"][o] >= np.log(1e6))
        ids = ok5.index[ok5]
        Fc = forecast_targets(pan, pairs, o, 5, TARGETS, mk["svr"], ids)
        act = pd.DataFrame({k: L[k][e].loc[ids] for k in TARGETS})
        nc = pd.DataFrame({k: L[k][o].loc[ids] for k in TARGETS})
        sa, sf, sn = all_scores(mat(act), g3), all_scores(mat(Fc), g3), all_scores(mat(nc), g3)
        for m in ["percentile", "topsis_minmax", "promethee_v", "critic", "bod_ar", "smaa_meanrank", "kemeny"]:
            rows.append({"window": f"{o}->{e}", "n": len(ids), "method": m, "svr_spearman": spear(sf[m], sa[m]), "nc_spearman": spear(sn[m], sa[m]), "svr_top10": overlap(sf[m], sa[m], 10), "nc_top10": overlap(sn[m], sa[m], 10), "svr_top20": overlap(sf[m], sa[m], 20), "nc_top20": overlap(sn[m], sa[m], 20)})
    rw = pd.DataFrame(rows)
    print(rw.round(4).to_string(index=False))
    rw.to_csv(HERE / "rolling_windows.csv", index=False)
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
