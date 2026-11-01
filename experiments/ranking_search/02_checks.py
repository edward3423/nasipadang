"""Sanity checks before any comparison: does the baseline reproduce the bundle exactly, do the handwritten methods behave?"""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from lab import *

pd.set_option("display.width", 230, "display.max_columns", 40, "display.max_rows", 200)
b = load_bundle()
cols, signs, groups = b["feature_cols"], b["signs"], b["groups"]
g_of = {f: i for i, (k, v) in enumerate(groups.items()) for f in v}
g = np.array([g_of[c] for c in cols])
sgn = np.array([signs[c] for c in cols])


def prep(df, min_pop=1e6):
    r = df[df["population"] >= min_pop]
    return r[cols].values.astype(float) * sgn, r.index, r


# 1. Baseline reproduces the bundle exactly (so clusters play no role in it)
for key, rk in [("features_2030", "rank_2030"), ("features_2030_knn", "rank_2030_knn"), ("features_now", "rank_now")]:
    X, idx, r = prep(b[key])
    s = pd.Series(m_percentile(X, g), index=idx)
    ref = b[rk]
    print(key, "n", len(idx), "same set", set(idx) == set(ref.index), "max abs score diff", float((s - ref["score"].reindex(idx)).abs().max()),
          "rank match", bool((s.rank(ascending=False).astype(int) == ref["rank"].reindex(idx)).all()))

X, idx, r = prep(b["features_2030"])
Xk, idxk, _ = prep(b["features_2030_knn"])
print("SVR-only", sorted(set(idx) - set(idxk)), "KNN-only", sorted(set(idxk) - set(idx)))
print("features tied values share:", {c: round(1 - r[c].nunique() / len(r), 2) for c in cols})

# 2. PROMETHEE-usual is exactly the baseline up to a linear map
w = equal_group_w(g)
u = m_promethee_usual(X, g)
p = m_percentile(X, g)
print("PROMETHEE usual vs percentile: corr", np.corrcoef(u, p)[0, 1], "ratio check", np.ptp(u - (2 * len(X) / (len(X) - 1)) * p), "same order", (order(u) == order(p)).all())

# 3. Timing and first look
t = time.time(); sc = all_scores(X, g); print("all_scores seconds", round(time.time() - t, 2))
for nm in METHODS:
    t = time.time(); METHODS[nm](X, g); print(f"  {nm:26s} {time.time()-t:.3f}s")
A0 = b["backtest_actual_2000"]; ix = A0.index
X3 = np.column_stack([A0["urban"], A0["gni"], -A0["u5"]]); g3 = np.arange(3)
t = time.time(); all_scores(X3, g3); print("all_scores 3-indicator seconds", round(time.time() - t, 2))
print("backtest pop>=1M in 2000:", int((A0["population"] >= np.log(1e6)).sum()), "of", len(A0))

# 4. BoD sanity
e, W = bod(X, g, None, True)
print("BoD pure: share at 1.0", round(float((e > 1 - 1e-6).mean()), 3), "count", int((e > 1 - 1e-6).sum()), "min", round(float(e.min()), 3))
ar = default_ar(g)
e2, W2 = bod(X, g, ar, True)
print("BoD AR: share at 1.0", round(float((e2 > 1 - 1e-6).mean()), 3), "count", int((e2 > 1 - 1e-6).sum()), "min", round(float(e2.min()), 3))
# verify the frontier shortcut gives the same optimum as the full LP on a few countries
Y = pct(X)
for i in [0, 40, 100]:
    full = linprog(-Y[i], A_ub=Y, b_ub=np.ones(len(Y)), bounds=(0, None), method="highs")
    print("  frontier check", i, round(-full.fun, 6), round(float(e[i]), 6))

# 5. Pareto and PCA details
l, _ = pareto_layers(X)
print("Pareto 11 features: first layer", int((l == 0).sum()), "layers", int(l.max()) + 1)
S, _ = group_scores(pct(X), g)
l2, _ = pareto_layers(S)
print("Pareto 4 groups: first layer", int((l2 == 0).sum()), "layers", int(l2.max()) + 1)
_, v, ev = pc1(X)
print("PC1 variance explained", round(float(ev), 3)); print(pd.Series(v, index=cols).round(2).to_dict())
print("entropy weights", pd.Series(entropy_weights(X), index=cols).round(3).to_dict())
print("critic weights", pd.Series(critic_weights(X), index=cols).round(3).to_dict())
print("equal-group weights", pd.Series(equal_group_w(g), index=cols).round(3).to_dict())

# 6. Top 10 by every method
names = r["name"]
print(pd.DataFrame({k: names.iloc[order(v)[:10]].values for k, v in sc.items()}).T.to_string())
