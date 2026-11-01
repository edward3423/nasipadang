"""Stage 1: cheap screen of every setting. 5 starts (1 if deterministic), checks 1-2, silhouette and
a 20-run stability estimate for settings that pass 1-2. Saves one row per setting."""
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

if __name__ == "__main__":
    G = L.grid()
    by = {}
    for g in G:
        by.setdefault(g["fam"], []).append(g)
    rows = []
    t0 = time.time()
    for fam, lst in sorted(by.items(), key=lambda kv: len(kv[1])):
        t = time.time()
        out = Parallel(n_jobs=-1, batch_size="auto")(delayed(L.screen_one)(g) for g in lst)
        rows += out
        d = pd.DataFrame(out)
        print(f"{fam:14s} n={len(lst):5d} cov&size={int((d.cov_ok & d.size_ok).sum()):5d} stab>=.5={int((d.stab >= .5).sum()):5d} stab>=.6={int((d.stab >= .6).sum()):5d} secs={time.time() - t:7.1f} total={time.time() - t0:7.1f}", flush=True)
        pd.DataFrame(rows).to_pickle(os.path.join(HERE, "screen.pkl"))
    df = pd.DataFrame(rows)
    df.to_pickle(os.path.join(HERE, "screen.pkl"))
    df.to_csv(os.path.join(HERE, "screen.csv"), index=False)
    print("DONE", len(df), "settings", time.time() - t0, "seconds", flush=True)
