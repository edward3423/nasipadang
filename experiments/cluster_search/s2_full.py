"""Stage 2: full checks 1-7 for finalists.
Finalists: every setting that passed coverage and sizes on all screen starts with screen stability >= 0.5,
plus the 3 closest near-misses for any algorithm with no such setting, so every algorithm gets a table row."""
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


def select(df):
    df = df.copy()
    df["s"] = df["key"].map(json.loads)
    df["algo"] = df["s"].map(L.algo_name)
    df["closeness"] = df["min_size"].clip(upper=8) / 8 + (0.5 / df["max_share"].clip(lower=1e-9)).clip(upper=1)
    fin = df[df.cov_ok & df.size_ok & (df.stab >= 0.5)]
    extra = []
    for algo, g in df.groupby("algo"):
        if algo in set(fin.algo):
            continue
        passed = g[g.cov_ok & g.size_ok]
        if len(passed):
            extra.append(passed.sort_values("stab", ascending=False).head(3))
        else:
            extra.append(g.sort_values(["cov_ok", "closeness", "sil"], ascending=False).head(3))
    extra = pd.concat(extra) if extra else df.iloc[:0]
    return fin, extra


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(HERE, "screen.pkl"))
    fin, extra = select(df)
    print("screened", len(df), "finalists", len(fin), "near-miss rows", len(extra), flush=True)
    print(fin.groupby("algo").size().to_string(), flush=True)
    settings = [json.loads(k) for k in pd.concat([fin, extra]).key]
    settings.append(dict(fam="kmeans_ref", ver=8, k=4))
    path = os.path.join(HERE, "full.pkl")
    done = {}
    if os.path.exists(path):
        done = {r["key"]: r for r in pd.read_pickle(path)}
    todo = [s for s in settings if L.key(s) not in done]
    # cheap families first so a partial run is still informative
    cost = {"consensus": 9, "ap_k": 9, "bgmm": 5, "tmix": 5, "trim_gmm": 5, "mppca": 5, "meanshift": 5}
    todo.sort(key=lambda s: cost.get(s["fam"], 1))
    t0 = time.time()
    chunk = 240
    for i in range(0, len(todo), chunk):
        part = todo[i : i + chunk]
        out = Parallel(n_jobs=-1, batch_size=1)(delayed(L.full_check)(s) for s in part)
        for r in out:
            done[r["key"]] = r
        pd.to_pickle(list(done.values()), path)
        print(f"{i + len(part)}/{len(todo)} all5={sum(r['all5'] for r in done.values())} secs={time.time() - t0:.0f}", flush=True)
    # every algorithm's best rows must show all checks, so rerun them without the early stop
    best = {}
    for r in done.values():
        a = L.algo_name(r["s"])
        tier = (r["all5"], bool(r["cov"] and r["size"] and r["fake_pass"]), bool(r["cov"] and r["size"]), bool(r["cov"]))
        margin = (r["sil"] - r["fake_p95"]) if np.isfinite(r["fake_p95"]) else -9.0
        val = (tier, margin if np.isfinite(margin) else -9.0, r["sil"] if np.isfinite(r["sil"]) else -9.0)
        best.setdefault(a, []).append((val, r["key"]))
    redo = []
    for a, lst in best.items():
        lst.sort(key=lambda x: x[0], reverse=True)
        for _v, k in lst[:3]:
            if done[k]["early"] and not done[k]["all5"]:
                redo.append(done[k]["s"])
    print("complete reruns for best rows", len(redo), flush=True)
    out = Parallel(n_jobs=-1, batch_size=1)(delayed(L.full_check)(s, 100, False) for s in redo)
    for r in out:
        done[r["key"]] = r
    pd.to_pickle(list(done.values()), path)
    print("DONE", len(done), "all5", sum(r["all5"] for r in done.values()), time.time() - t0, flush=True)
