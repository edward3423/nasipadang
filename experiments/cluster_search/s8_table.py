"""Final read-outs from saved results. Only cheap reference fits for the chosen setting, no new search."""
import os, sys, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import cs_lib as L

pd.set_option("display.width", 300, "display.max_columns", 40, "display.max_rows", 500, "display.max_colwidth", 200)
F = dict(fam="spectral_st", ver=5, k=4, nn=10, assign="cluster_qr")
H = dict(fam="fcm", ver=8, k=5, m=1.5)
A = dict(fam="gmm_ward", ver=8, k=6, cov="spherical", reg=1e-3)
KM = dict(fam="kmeans_ref", ver=8, k=4)

if __name__ == "__main__":
    b = L.B()
    C, codes, names = b["C_2030"], list(b["index"]), list(b["names"])
    rank = b["rank_2030"]
    pos = {c: i for i, c in enumerate(codes)}
    top = [pos[c] for c in rank.sort_values("rank").index[:10]]
    d = pd.read_pickle(os.path.join(HERE, "full_df.pkl"))
    sh = pd.read_pickle(os.path.join(HERE, "short.pkl"))
    nf = pd.read_pickle(os.path.join(HERE, "null_fast.pkl"))
    tab = d.set_index("key")

    print("in 77 shortlist: F", L.key(F) in set(sh.key), "H", L.key(H) in set(sh.key), "A", L.key(A) in set(sh.key))
    best = d.sort_values(["all5", "margin"], ascending=False).groupby("algo", sort=False).head(1)
    extra = d[d.key == L.key(F)]
    print("\n| Algorithm | Fit on, k/groups | Det | Unassigned | Worst min/max size | Stability | Silhouette | Fake mean / p95 | Checks 1-5 | KNN ARI | Backtest SVR / no change | Top-10 types |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in pd.concat([best, extra]).itertuples():
        if r.fam == "kmeans_ref":
            nm = "K-means (reference, not a candidate)"
        elif r.key == L.key(F):
            nm = "Spectral self-tuning (recommended setting)"
        else:
            nm = r.algo
        t10 = len({int(r.labels[i]) for i in top if r.labels[i] >= 0}) if r.labels is not None else "-"
        f = lambda v: "-" if v is None or not np.isfinite(v) else f"{v:.3f}"
        print(f"| {nm} | {r.ver}, {r.groups} | {'yes' if r.det else 'no'} | {r.unassigned:.0%} | {r.sizes_worst_min}/{r.sizes_worst_max} | {f(r.stab)} | {f(r.sil)} | {f(r.fake_mean)} / {f(r.fake_p95)} | {'PASS' if r.all5 else 'fail'} | {f(r.knn_ari)} | {f(r.bt_svr)} / {f(r.bt_nochange)} | {t10} |")

    print("\nBest-row settings")
    for r in pd.concat([best, extra]).itertuples():
        print(r.algo, "|", r.txt)

    print("\nReduced null, margin reading: each fake silhouette minus that setting's own fake p95, best per dataset")
    p95 = sh.set_index("key").fake_p95
    nf = nf[nf.passes_12].copy()
    nf["m"] = nf.sil - nf.key.map(p95)
    mm = nf.groupby("fake").m.max()
    print("best margin per fake dataset: mean %.3f p95 %.3f max %.3f" % (mm.mean(), mm.quantile(0.95), mm.max()), mm.round(3).tolist())
    sx = nf.groupby("fake").sil.max()
    for nm, s in (("F", F), ("H", H), ("A", A), ("K-means", KM)):
        r = tab.loc[L.key(s)]
        print(nm, "sil %.3f margin %.3f | fake datasets at or above: silhouette %d/20, margin %d/20" % (r.sil, r.margin, (sx >= r.sil).sum(), (mm >= r.margin).sum()))

    lab, km = tab.loc[L.key(F)].labels, tab.loc[L.key(KM)].labels
    ct = pd.crosstab(lab, km)
    ri, ci = linear_sum_assignment(-ct.values)
    mp = {ct.index[i]: ct.columns[j] for i, j in zip(ri, ci)}
    same = np.array([mp[x] == y for x, y in zip(lab, km)])
    print("\nF vs K-means: identical type after best label match", int(same.sum()), "of", len(same), "; ARI", round(L.adjusted_rand_score(lab, km), 3))
    print("differ:", [(names[i], "spectral type", int(lab[i]), "kmeans type", int(km[i])) for i in np.where(~same)[0]])
    print("stability by reference seed", np.round(tab.loc[L.key(F)].stab_by_base, 3))
    for t in np.unique(lab):
        m = lab == t
        ctr = C[m].mean(0)
        print(f"type {t}: n={m.sum()} rms distance to centre {np.sqrt(((C[m] - ctr) ** 2).sum(1).mean()):.2f}; top-10 members:", [names[i] for i in top if lab[i] == t])
    print("share of all stage-2 passers with ARI>=.95 to K-means at same k", round((d[d.all5].ari_km_samek >= 0.95).mean(), 3))
