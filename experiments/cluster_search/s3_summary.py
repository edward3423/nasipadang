"""Summarise stage 2: pass counts, best row per algorithm, closeness of each passer to K-means at the same k."""
import json, os, sys, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["PYTHONPATH"] = HERE
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ARI
import cs_lib as L

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500, "display.max_colwidth", 120)


def settings_text(s):
    return " ".join(f"{k}={v}" for k, v in s.items() if k not in ("fam",))


if __name__ == "__main__":
    rows = pd.read_pickle(os.path.join(HERE, "full.pkl"))
    d = pd.DataFrame(rows)
    d["algo"] = d.s.map(L.algo_name)
    d["k"] = d.s.map(lambda s: s.get("k"))
    d["margin"] = d.sil - d.fake_p95
    d["txt"] = d.s.map(settings_text)
    C = L.B()["C_2030"]
    km = {}
    for ver in L.VERS:
        for k in range(3, 16):
            km[(str(ver), k)] = L.run(dict(fam="kmeans_ref", ver=ver, k=k), C, 0)
    ref = km[("8", 4)]
    d["ari_km_samek"] = [ARI(r.labels, km[(r.ver, r.groups)]) if r.labels is not None and (r.ver, r.groups) in km else np.nan for r in d.itertuples()]
    d["ari_km4"] = [ARI(r.labels, ref) if r.labels is not None else np.nan for r in d.itertuples()]
    d.to_pickle(os.path.join(HERE, "full_df.pkl"))

    print("rows", len(d), "all5", int(d.all5.sum()))
    print("\nCounts by algorithm: full-checked, pass sizes, pass fake, pass all five")
    a = d.groupby("algo").agg(n=("key", "size"), cov=("cov", "sum"), size=("size", "sum"), fake=("fake_pass", "sum"), all5=("all5", "sum"))
    print(a.sort_values("all5", ascending=False).to_string())

    p = d[d.all5 & (d.fam != "kmeans_ref")]
    print("\nPassers by groups realised and version")
    print(pd.crosstab(p.groups, p.ver).to_string())
    print("\nPassers by algorithm and groups")
    print(pd.crosstab(p.algo, p.groups).to_string())

    cols = ["algo", "ver", "groups", "det", "unassigned", "sizes_worst_min", "sizes_worst_max", "stab", "sil", "fake_mean", "fake_p95", "margin", "knn_ari", "bt_svr", "bt_nochange", "ari_km_samek", "txt"]
    print("\nBest row per algorithm (passes first, then largest silhouette margin)")
    best = d.sort_values(["all5", "margin"], ascending=False).groupby("algo", sort=False).head(1)
    print(best[cols].round(3).to_string(index=False))

    print("\nTop 30 passers by stability, excluding K-means reference")
    print(p.sort_values("stab", ascending=False).head(30)[cols].round(3).to_string(index=False))
    print("\nTop 30 passers by silhouette margin")
    print(p.sort_values("margin", ascending=False).head(30)[cols].round(3).to_string(index=False))
    print("\nPassers furthest from K-means at the same k, with stability >= .75")
    print(p[p.stab >= 0.75].sort_values("ari_km_samek").head(30)[cols].round(3).to_string(index=False))
    print("\nK-means reference")
    print(d[d.fam == "kmeans_ref"][cols].round(3).to_string(index=False))
