"""Which passers match or beat the K-means reference on everything except stability, and are not K-means copies?"""
import os, sys, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import cs_lib as L

pd.set_option("display.width", 300, "display.max_columns", 40, "display.max_rows", 500, "display.max_colwidth", 90)

if __name__ == "__main__":
    d = pd.read_pickle(os.path.join(HERE, "full_df.pkl"))
    screen = pd.read_pickle(os.path.join(HERE, "screen.pkl"))
    print("screen rows", len(screen), "fit failures in screen (min_size 0, no silhouette)", int(((screen.min_size == 0) & screen.sil.isna()).sum()))
    print("hdbscan screen failures with no labels", int(((screen.fam == "hdbscan") & (screen.groups == 0) & (screen.unassigned == 1.0)).sum()))
    print(screen[(screen.groups == 0) & (screen.unassigned == 1.0)].groupby("fam").size().to_string())
    ref = d[d.fam == "kmeans_ref"].iloc[0]
    p = d[d.all5 & (d.fam != "kmeans_ref")].copy()
    print("\nnon-K-means passers", len(p), "families", p.algo.nunique())
    print("look-alikes ARI>=.95 with K-means at same k/version:", int((p.ari_km_samek >= 0.95).sum()), " ARI>=.8:", int((p.ari_km_samek >= 0.8).sum()), " ARI<.6:", int((p.ari_km_samek < 0.6).sum()))
    p["skill"] = p.bt_svr - p.bt_nochange
    print("backtest: SVR beats no change in", int((p.skill > 0).sum()), "of", len(p), "; K-means skill", round(ref.bt_svr - ref.bt_nochange, 3))
    cols = ["algo", "ver", "groups", "det", "sizes", "sizes_worst_min", "sizes_worst_max", "stab", "sil", "fake_p95", "margin", "knn_ari", "bt_svr", "bt_nochange", "ari_km_samek", "ari_km4", "txt"]
    q = p[(p.stab >= 0.75) & (p.margin >= ref.margin) & (p.knn_ari >= 0.8) & (p.skill > 0) & (p.ari_km_samek < 0.95)]
    print("\nstab>=.75, margin>=K-means, knn>=.8, positive skill, not a copy:", len(q))
    print(q.groupby(["algo", "groups"]).size().to_string())
    print(q.sort_values(["groups", "margin"], ascending=[False, False]).head(60)[cols].round(3).to_string(index=False))
    q2 = p[(p.groups >= 5) & (p.stab >= 0.8) & (p.skill > 0) & (p.knn_ari >= 0.7)]
    print("\nk>=5, stab>=.8, knn>=.7:", len(q2))
    print(q2.sort_values("margin", ascending=False).head(40)[cols].round(3).to_string(index=False))
    print("\nSpectral self-tuning shortlist settings")
    print(p[(p.fam == "spectral_st") & (p.groups == 4) & (p.ver == "5")].sort_values("stab", ascending=False).head(12)[cols].round(3).to_string(index=False))
