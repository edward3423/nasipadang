"""Deep checks on a fixed list of contenders, chosen from the stage 2 table before any usefulness was inspected."""
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
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import cs_lib as L
import s5_short as S

pd.set_option("display.width", 300, "display.max_columns", 40, "display.max_rows", 500, "display.max_colwidth", 200)

CONTENDERS = {
    "A GMM Ward-init spherical k=6 8PC": dict(fam="gmm_ward", ver=8, k=6, cov="spherical", reg=1e-3),
    "B GMM Ward-init spherical k=5 raw": dict(fam="gmm_ward", ver="raw", k=5, cov="spherical", reg=0.1),
    "C GMM spherical random k=5 raw": dict(fam="gmm", ver="raw", k=5, cov="spherical", reg=1e-3, init="random"),
    "D Consensus GMM spherical k=5 5PC": dict(fam="consensus", ver=5, k=5, base="gmm_spherical", final="spectral"),
    "E Trimmed GMM spherical k=4 raw": dict(fam="trim_gmm", ver="raw", k=4, cov="spherical", reg=0.01, alpha=0.05),
    "F Spectral self-tuning k=4 5PC": dict(fam="spectral_st", ver=5, k=4, nn=10, assign="cluster_qr"),
    "G Fuzzy c-means k=5 m=1.75 5PC": dict(fam="fcm", ver=5, k=5, m=1.75),
    "H Fuzzy c-means k=5 m=1.5 8PC": dict(fam="fcm", ver=8, k=5, m=1.5),
    "I Bayesian GMM tied k=6 5PC": dict(fam="bgmm", ver=5, k=6, cov="tied", wtype="dirichlet_distribution", wcp=100.0, reg=0.1),
    "J Ward k=4 3PC": dict(fam="agglo", ver=3, k=4, linkage="ward", metric="euclidean"),
}


def bic_gmm():
    """Second calibration against the stated BIC-chosen GMM: stability 0.677, silhouette 0.158, k=9."""
    C = L.B()["C_2030"]
    Z = PCA(8, random_state=0).fit_transform(C)

    def one(k, cov, reg, init):
        try:
            m = GaussianMixture(k, covariance_type=cov, reg_covar=reg, init_params=init, n_init=5, max_iter=500, random_state=0).fit(Z)
            return m.bic(Z), k, cov, reg, init
        except Exception:
            return np.inf, k, cov, reg, init

    res = Parallel(n_jobs=-1)(delayed(one)(k, cov, reg, init) for k in range(2, 11) for cov in ("spherical", "diag", "tied", "full") for reg in (1e-6, 1e-3, 0.01, 0.1, 0.5, 1.0) for init in ("kmeans", "k-means++", "random"))
    res.sort(key=lambda x: x[0])
    return res[:5]


def km_comparators():
    out = []
    C = L.B()["C_2030"]
    base = L.run(dict(fam="kmeans_ref", ver=8, k=4), C, 0)
    specs = [dict(fam="kmeans_ref", ver=8, k=k) for k in (3, 5, 6)] + [dict(fam="kmeans_ref", ver=v, k=4) for v in (3, 5, "raw")]
    res = Parallel(n_jobs=-1, batch_size=1)(delayed(L.full_check)(s, 100, False) for s in specs)
    for s, r in zip(specs, res):
        out.append(dict(ver=s["ver"], k=s["k"], all5=r["all5"], size=r["size"], stab=r["stab"], sil=r["sil"], p95=r["fake_p95"], knn=r["knn_ari"], bt_svr=r["bt_svr"], bt_nc=r["bt_nochange"], ari_to_ref=ARI(base, r["labels"]), sizes=r["sizes"]))
    return pd.DataFrame(out)


if __name__ == "__main__":
    t0 = time.time()
    d = pd.read_pickle(os.path.join(HERE, "full_df.pkl"))
    G = L.grid()
    print("BIC calibration, top settings", bic_gmm(), flush=True)
    results = {}
    for name, s in CONTENDERS.items():
        row = d[d.key == L.key(s)]
        if len(row) == 0:
            print("NOT IN FULL TABLE", name, flush=True)
            continue
        r = row.iloc[0].to_dict()
        print("\n" + "=" * 110 + f"\n{name}\n{s}", flush=True)
        print({k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in ("all5", "det", "sizes", "sizes_worst_min", "sizes_worst_max", "stab", "sil", "fake_mean", "fake_p95", "knn_ari", "bt_svr", "bt_nochange", "bt_native", "start_agreement", "ari_km_samek", "ari_km4")}, flush=True)
        out = S.deep(name, s, r, G)
        results[name] = out
        print("neighbours:\n" + out["neighbours"].round(3).to_string(index=False), flush=True)
        print("neighbours passing 1-5:", int(out["neighbours"].all5.sum()), "of", len(out["neighbours"]), flush=True)
        print("per-type recovery (mean Jaccard, share of samples >= .6):", {k: (round(a, 2), round(b, 2)) for k, (a, b) in out["recovery"].items()}, flush=True)
        print(out["describe"], flush=True)
        print("ARI with latest actuals (C_now)", round(out["ari_now"], 3), "| sizes on C_2030_knn", out["sizes_knn"], "| ARI with K-means k=4", round(out["ari_km4"], 3), flush=True)
        print(out["crosstab_km4"], flush=True)
        pd.to_pickle(results, os.path.join(HERE, "deep.pkl"))
    print("\nK-means comparators\n" + km_comparators().round(3).to_string(index=False), flush=True)
    # K-means reference usefulness and recovery, for comparison
    km = dict(fam="kmeans_ref", ver=8, k=4)
    r = d[d.key == L.key(km)].iloc[0].to_dict()
    print("\nK-means reference recovery", {k: (round(a, 2), round(b, 2)) for k, (a, b) in S.recovery(km, r["labels"]).items()}, flush=True)
    print(S.describe(km, r), flush=True)
    print("DONE", time.time() - t0, flush=True)
