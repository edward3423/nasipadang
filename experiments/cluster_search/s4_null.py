"""Selection-adjusted null. Each fake dataset is searched over the same finalist settings. For each dataset we
record the best silhouette among settings that pass coverage and sizes, and how many of them also reach
screen stability >= 0.6. This answers: on data with no joint structure, how good would the winner of this
search look, and would the checks still let settings through?"""
import json, os, sys, time, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["PYTHONPATH"] = HERE
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import cs_lib as L


def one(s, fs, with_stab):
    Y = L.fake_data(fs)
    lab = L.run(s, Y, fs)
    if lab is None or not L.coverage_ok(lab) or not L.sizes_ok(lab, s):
        return (fs, L.key(s), False, np.nan, np.nan)
    st = np.nan
    if with_stab:
        rng = np.random.default_rng(fs + 1)
        out = []
        for r in range(10):
            idx = rng.choice(len(Y), int(0.8 * len(Y)), replace=False)
            l2 = L.run(s, Y[idx], 100 + r)
            out.append(0.0 if l2 is None else L.adjusted_rand_score(lab[idx], l2))
        st = float(np.mean(out))
    return (fs, L.key(s), True, L.sil(Y, lab), st)


if __name__ == "__main__":
    nfake = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    rows = pd.read_pickle(os.path.join(HERE, "full.pkl"))
    fin = [r for r in rows if r["fam"] != "kmeans_ref" and r["cov"] and r["size"] and r["stab"] >= 0.6]
    slow = {"consensus", "ap_k"}
    settings = [r["s"] for r in fin if r["fam"] not in slow]
    print("settings per fake dataset", len(settings), "(of", len(fin), "that pass checks 1-3 on real data)", flush=True)
    t = time.time()
    tasks = [(s, fs) for fs in range(nfake) for s in settings]
    out = Parallel(n_jobs=-1, batch_size=64)(delayed(one)(s, fs, fs < 10) for s, fs in tasks)
    df = pd.DataFrame(out, columns=["fake", "key", "passes_12", "sil", "stab"])
    df.to_pickle(os.path.join(HERE, "null.pkl"))
    mx = df[df.passes_12].groupby("fake").sil.max()
    print("best silhouette per fake dataset: mean", mx.mean(), "p95", mx.quantile(0.95), "max", mx.max(), flush=True)
    print("share of settings passing 1-2 on fake", df.passes_12.mean(), flush=True)
    sub = df[df.fake < 10]
    print("first 10 fake: share passing 1-2 and stability >= .6:", ((sub.passes_12) & (sub.stab >= 0.6)).mean(), flush=True)
    print("DONE", time.time() - t, flush=True)
