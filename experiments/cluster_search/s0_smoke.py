"""Smoke test: harness calibration against known K-means numbers, then one timed setting per family."""
import os, sys, time, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["PYTHONPATH"] = HERE
import numpy as np
import pandas as pd
import cs_lib as L

if __name__ == "__main__":
    C = L.B()["C_2030"]
    G = L.grid()
    cnt = pd.Series([g["fam"] for g in G]).value_counts()
    print(cnt.to_string())
    print("TOTAL settings", len(G), "unique", len({L.key(g) for g in G}))

    ref = dict(fam="kmeans_ref", ver=8, k=4)
    t = time.time()
    labs = [L.run(ref, C, s) for s in range(30)]
    print("K-means k=4 8PC sizes", L.sizes(labs[0]), "silhouette", np.mean([L.sil(C, l) for l in labs]))
    print("K-means stability", np.mean([L.stability_ref(ref, C, b) for b in range(1, 6)]), "time", time.time() - t)
    fk = [L.fake_sil(ref, s) for s in range(20)]
    print("fake mean (20)", np.mean(fk), "p95", np.quantile(fk, .95))

    # one sample per family, timed; report a failure explicitly
    seen = {}
    for g in G:
        seen.setdefault(g["fam"], []).append(g)
    rng = np.random.default_rng(0)
    for fam, lst in seen.items():
        tm = []
        fails = 0
        ng = []
        for i in rng.choice(len(lst), min(8, len(lst)), replace=False):
            t = time.time()
            lab = L.run(lst[i], C, 1)
            tm.append(time.time() - t)
            if lab is None:
                fails += 1
            else:
                ng.append(L.n_groups(lab))
        # also surface the exception text for the first failure
        err = ""
        if fails:
            for i in range(len(lst)):
                try:
                    Z, _ = L.embed(C, lst[i]["ver"])
                    L.fit_method(lst[i], Z, 1)
                except Exception as e:
                    err = repr(e)[:120]
                    break
        print(f"{fam:14s} n={len(lst):5d} mean_s={np.mean(tm):.3f} max_s={np.max(tm):.3f} fails={fails} groups={ng} {err}")
    # backtest sanity
    for s in [ref, dict(fam="agglo", ver="raw", k=4, linkage="ward", metric="euclidean"), dict(fam="gmm", ver=8, k=4, cov="spherical", reg=0.01, init="kmeans")]:
        print(s["fam"], L.backtest_one(s, 0))
