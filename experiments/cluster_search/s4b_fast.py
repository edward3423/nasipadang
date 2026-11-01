"""Reduced selection-adjusted null: 20 fake datasets, searched over the 77-setting shortlist only, no stability.
Under-adjusts for the original 24,246-setting search, and a 95th percentile from 20 values is rough.
Stops starting new datasets after 280 seconds and reports how many were finished."""
import os, sys, time, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["PYTHONPATH"] = HERE
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import cs_lib as L


def one(s, fs):
    Y = L.fake_data(fs)
    lab = L.run(s, Y, fs)
    ok = lab is not None and L.coverage_ok(lab) and L.sizes_ok(lab, s)
    return (fs, L.key(s), s["fam"], bool(ok), L.sil(Y, lab) if ok else np.nan)


if __name__ == "__main__":
    t0 = time.time()
    sh = pd.read_pickle(os.path.join(HERE, "short.pkl"))
    settings = list(sh.s)
    print("settings per fake dataset", len(settings), flush=True)
    rows = []
    with Parallel(n_jobs=-1, batch_size=1) as par:
        for fs in range(20):
            if time.time() - t0 > 280:
                break
            rows += par(delayed(one)(s, fs) for s in settings)
            df = pd.DataFrame(rows, columns=["fake", "key", "fam", "passes_12", "sil"])
            df.to_pickle(os.path.join(HERE, "null_fast.pkl"))
            print(f"finished fake dataset {fs + 1}/20 secs={time.time() - t0:.0f}", flush=True)
    mx = df[df.passes_12].groupby("fake").sil.max()
    mx2 = df[df.passes_12 & (df.fam != "trim_gmm")].groupby("fake").sil.max()
    print("datasets finished", df.fake.nunique(), "of 20")
    print("best silhouette per fake dataset, all shortlist: mean %.3f p95 %.3f max %.3f" % (mx.mean(), mx.quantile(0.95), mx.max()))
    print("best silhouette per fake dataset, without trimmed GMM: mean %.3f p95 %.3f max %.3f" % (mx2.mean(), mx2.quantile(0.95), mx2.max()))
    print("share of shortlist settings passing coverage and sizes on fake data %.3f" % df.passes_12.mean())
    for name, v in (("real A/B silhouette .233", 0.233), ("real K-means silhouette .226", 0.226), ("real best shortlist silhouette .246", 0.246)):
        print(name, "fake datasets at or above, all:", int((mx >= v).sum()), "without trimmed:", int((mx2 >= v).sum()))
    print(mx.round(3).tolist())
    print("DONE", round(time.time() - t0), flush=True)
