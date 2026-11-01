"""Cheap read-outs for contenders. No new full checks. Neighbour results are looked up in the stage 2 and screen
tables, so a neighbour that stopped early shows NaN for the checks it never reached."""
import json, os, sys, time, warnings
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
import s5_short as S

pd.set_option("display.width", 300, "display.max_columns", 40, "display.max_rows", 500, "display.max_colwidth", 200)

CONTENDERS = {
    "A GMM Ward-init spherical k=6 8PC": dict(fam="gmm_ward", ver=8, k=6, cov="spherical", reg=1e-3),
    "B GMM Ward-init spherical k=5 raw": dict(fam="gmm_ward", ver="raw", k=5, cov="spherical", reg=0.1),
    "C GMM spherical random k=5 raw": dict(fam="gmm", ver="raw", k=5, cov="spherical", reg=1e-3, init="random"),
    "E Trimmed GMM spherical k=4 raw": dict(fam="trim_gmm", ver="raw", k=4, cov="spherical", reg=0.01, alpha=0.05),
    "F Spectral self-tuning k=4 5PC": dict(fam="spectral_st", ver=5, k=4, nn=10, assign="cluster_qr"),
    "G Fuzzy c-means k=5 m=1.75 5PC": dict(fam="fcm", ver=5, k=5, m=1.75),
    "H Fuzzy c-means k=5 m=1.5 8PC": dict(fam="fcm", ver=8, k=5, m=1.5),
    "I Bayesian GMM tied k=6 5PC": dict(fam="bgmm", ver=5, k=6, cov="tied", wtype="dirichlet_distribution", wcp=100.0, reg=0.1),
    "J Ward k=4 3PC": dict(fam="agglo", ver=3, k=4, linkage="ward", metric="euclidean"),
    "K-means reference": dict(fam="kmeans_ref", ver=8, k=4),
}


def work(name, s, lab):
    b = L.B()
    rec = S.recovery(s, lab, runs=50)
    now = [L.run(s, b["C_now"], sd) for sd in ([0] if L.is_det(s) else range(5))]
    km = L.run(dict(fam="kmeans_ref", ver=8, k=4), b["C_2030"], 0)
    return name, rec, float(np.mean([ARI(lab, x) for x in now if x is not None])), L.sizes(L.run(s, b["C_2030_knn"], 0)), float(ARI(lab, km)), pd.crosstab(pd.Series(lab, name="cand"), pd.Series(km, name="km4")).to_string()


if __name__ == "__main__":
    t0 = time.time()
    d = pd.read_pickle(os.path.join(HERE, "full_df.pkl"))
    scr = pd.read_pickle(os.path.join(HERE, "screen.pkl")).set_index("key")
    G = L.grid()
    tab = d.set_index("key")
    rows = {n: tab.loc[L.key(s)].to_dict() for n, s in CONTENDERS.items()}
    out = Parallel(n_jobs=6, batch_size=1)(delayed(work)(n, s, rows[n]["labels"]) for n, s in CONTENDERS.items())
    p = d[d.all5 & (d.fam != "kmeans_ref")]
    print("GMM passers by covariance", p[p.fam == "gmm"].s.map(lambda s: s["cov"]).value_counts().to_dict(), "| Ward-init", p[p.fam == "gmm_ward"].s.map(lambda s: s["cov"]).value_counts().to_dict(), flush=True)
    print("all GMM settings passing screen sizes+coverage by covariance", pd.read_pickle(os.path.join(HERE, "screen.pkl")).query("fam=='gmm' and cov_ok and size_ok").key.map(lambda k: json.loads(k)["cov"]).value_counts().to_dict(), flush=True)
    for (name, rec, now, kn, ak, ct) in out:
        s, r = CONTENDERS[name], rows[name]
        print("\n" + "=" * 100 + f"\n{name}  {s}")
        print({k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in ("all5", "det", "sizes", "sizes_worst_min", "sizes_worst_max", "stab", "sil", "fake_mean", "fake_p95", "knn_ari", "bt_svr", "bt_nochange", "bt_native", "start_agreement", "ari_km_samek")})
        print("ARI with K-means reference labels (k=4, 8 PCs, seed 0):", round(ak, 3), "| ARI with latest actuals:", round(now, 3), "| sizes on KNN forecast:", kn)
        print(ct)
        print("per-type recovery over 50 samples (mean Jaccard, share >= .6):", {k: (round(a, 2), round(b, 2)) for k, (a, b) in rec.items()})
        print(S.describe(s, r))
        if name != "K-means reference":
            nb = []
            for pn, g in S.neighbours(s, G):
                k_ = L.key(g)
                if k_ in tab.index:
                    x = tab.loc[k_]
                    nb.append(dict(param=pn, value=g[pn], src="stage2", all5=x.all5, size=x["size"], stab=x.stab, sil=x.sil, p95=x.fake_p95, knn=x.knn_ari, ari_to_cand=ARI(r["labels"], x.labels) if x.labels is not None else np.nan, sizes=x.sizes))
                else:
                    x = scr.loc[k_]
                    nb.append(dict(param=pn, value=g[pn], src="screen only", all5=False, size=bool(x.size_ok and x.cov_ok), stab=x.stab, sil=x.sil, p95=np.nan, knn=np.nan, ari_to_cand=np.nan, sizes=None))
            nb = pd.DataFrame(nb)
            print("neighbours passing 1-5:", int(nb.all5.sum()), "of", len(nb))
            print(nb.round(3).to_string(index=False))
    print("DONE", round(time.time() - t0), flush=True)
