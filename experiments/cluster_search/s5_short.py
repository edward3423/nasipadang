"""Deep checks for a shortlist: neighbouring settings, per-type recovery, check 8 usefulness, and data-version drift.

Shortlist rule, fixed before looking at usefulness: passes checks 1-5, positive forecast skill in the backtest,
then the best setting per algorithm under each of stability, silhouette margin and forecast sensitivity.
K-means look-alikes (ARI >= 0.95 with K-means at the same k and version) are kept but flagged."""
import json, os, sys, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["PYTHONPATH"] = HERE
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score as ARI
import cs_lib as L

ORDERED = {"ver": [3, 5, 8, "raw"]}
SKIP = {"fam", "merge_k"}


def neighbours(s, G):
    """One-parameter-at-a-time neighbours taken from the screened grid.
    Ordered parameters step to the adjacent value. Categorical parameters switch to every other value."""
    out = []
    for p in s:
        if p in SKIP:
            continue
        alts = []
        for g in G:
            if g["fam"] != s["fam"] or p not in g or set(g) != set(s):
                continue
            if all(g[q] == s[q] for q in s if q != p and not (p == "k" and q == "merge_k")):
                alts.append(g)
        vals = []
        for g in alts:
            if g[p] not in vals:
                vals.append(g[p])
        if p in ORDERED:
            vals = [v for v in ORDERED[p] if v in vals]
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            vals = sorted(vals)
        else:
            for g in alts:
                if g[p] != s[p]:
                    out.append((p, g))
            continue
        i = vals.index(s[p])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                for g in alts:
                    if g[p] == vals[j]:
                        out.append((p, g))
    # a k step on merged SOM moves merge_k with it
    return out


def recovery(s, base_lab, runs=100):
    """Per-type stability: best Jaccard of each type against the refit groups on an 80% sample."""
    C = L.B()["C_2030"]
    rng = np.random.default_rng(99)
    ids = np.unique(base_lab[base_lab >= 0])
    J = {i: [] for i in ids}
    for r in range(runs):
        idx = rng.choice(len(C), int(0.8 * len(C)), replace=False)
        lab = L.run(s, C[idx], 5000 + r)
        for i in ids:
            A = set(np.where(base_lab[idx] == i)[0])
            best = 0.0
            if lab is not None:
                for j in np.unique(lab[lab >= 0]):
                    B_ = set(np.where(lab == j)[0])
                    best = max(best, len(A & B_) / max(len(A | B_), 1))
            J[i].append(best)
    return {int(i): (float(np.mean(v)), float(np.mean(np.array(v) >= 0.6))) for i, v in J.items()}


def describe(s, r):
    b = L.B()
    C = b["C_2030"]
    F = b["features_2030"]
    rank = b["rank_2030"]
    codes = list(b["index"])
    cols = b["feature_cols"]
    lab = r["labels"]
    top = rank.sort_values("rank").index[:10]
    pos = {c: i for i, c in enumerate(codes)}
    tt = [int(lab[pos[c]]) for c in top]
    lines = [f"top10 span {len(set(t for t in tt if t >= 0))} types: " + ", ".join(f"{rank.loc[c, 'name']}={lab[pos[c]]}" for c in top)]
    prof = pd.DataFrame(C, columns=cols).groupby(lab).mean().round(2)
    lines.append(prof.to_string())
    ranked = set(rank.index)
    for t in np.unique(lab):
        mem = np.where(lab == t)[0]
        pop = F.iloc[mem].sort_values("population", ascending=False)
        rk = rank.reindex([codes[i] for i in mem])["rank"].dropna()
        med = rk.median() if len(rk) else float("nan")
        lines.append(f"type {t} n={len(mem)} unranked={sum(codes[i] not in ranked for i in mem)} median_rank={med} best_rank={rk.min() if len(rk) else 'na'} largest: " + ", ".join(pop["name"].head(8)))
    return "\n".join(lines)


def deep(name, s, r, G):
    out = {"name": name, "s": s}
    b = L.B()
    C = b["C_2030"]
    lab = r["labels"]
    nb = neighbours(s, G)
    res = Parallel(n_jobs=-1, batch_size=1)(delayed(L.full_check)(g, 100, False) for _p, g in nb)
    tab = []
    for (p, g), x in zip(nb, res):
        same = ARI(lab, x["labels"]) if x["labels"] is not None else np.nan
        tab.append(dict(param=p, value=g[p], all5=x["all5"], cov=x["cov"], size=x["size"], stab=x["stab"], sil=x["sil"], p95=x["fake_p95"], knn=x["knn_ari"], ari_to_cand=same, sizes=x["sizes"]))
    out["neighbours"] = pd.DataFrame(tab)
    out["recovery"] = recovery(s, lab)
    out["describe"] = describe(s, r)
    now = [L.run(s, b["C_now"], sd) for sd in ([0] if L.is_det(s) else range(10))]
    out["ari_now"] = float(np.mean([ARI(lab, x) for x in now if x is not None]))
    out["sizes_knn"] = L.sizes(L.run(s, b["C_2030_knn"], 0))
    km = L.run(dict(fam="kmeans_ref", ver=8, k=4), C, 0)
    out["ari_km4"] = float(ARI(lab, km))
    out["crosstab_km4"] = pd.crosstab(pd.Series(lab, name="candidate"), pd.Series(km, name="kmeans4")).to_string()
    return out


if __name__ == "__main__":
    d = pd.read_pickle(os.path.join(HERE, "full_df.pkl"))
    G = L.grid()
    p = d[d.all5 & (d.fam != "kmeans_ref") & (d.bt_svr > d.bt_nochange)].copy()
    picks = {}
    for algo, g in p.groupby("algo"):
        for col in ("stab", "margin", "knn_ari"):
            picks[g.sort_values(col, ascending=False).iloc[0].key] = f"{algo} best {col}"
    # overall balanced picks: rank sum across the four signals
    for col in ("stab", "margin", "knn_ari", "bt_svr"):
        p["r_" + col] = p[col].rank(ascending=False)
    p["rs"] = p[["r_stab", "r_margin", "r_knn_ari", "r_bt_svr"]].sum(1)
    for k_ in p.sort_values("rs").head(15).key:
        picks.setdefault(k_, "balanced top 15")
    extra = sys.argv[1] if len(sys.argv) > 1 else None
    print("shortlist size", len(picks), flush=True)
    sh = p[p.key.isin(picks)].copy()
    sh["why"] = sh.key.map(picks)
    cols = ["algo", "ver", "groups", "det", "sizes", "stab", "sil", "fake_p95", "margin", "knn_ari", "bt_svr", "bt_nochange", "ari_km_samek", "ari_km4", "rs"]
    print(sh.sort_values("rs")[cols + ["why"]].round(3).to_string(index=False), flush=True)
    sh.to_pickle(os.path.join(HERE, "short.pkl"))
