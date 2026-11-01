"""Where does the rank-change skill come from? Forecast one indicator at a time, the others held at 2000 values. Baseline method only.
The hybrid row (forecast urban and u5, freeze income) is post hoc: the idea came from seeing this same test decade, so it is not fair evidence."""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import *

pd.set_option("display.width", 200, "display.max_columns", 30)
b = load_bundle()
A0, A1, P = b["backtest_actual_2000"], b["backtest_actual_2010"], b["backtest_svr_forecast_2010"]
ix = A0.index[A0["population"] >= np.log(1e6)]
mat = lambda d: np.column_stack([d["urban"], d["gni"], -d["u5"]]).astype(float)
X0, X1 = mat(A0.loc[ix]), mat(A1.loc[ix])
t, nc = pct(X1).mean(1), pct(X0).mean(1)
n = len(ix)
r0 = ranks(nc)
d_act = (r0 - ranks(t)).astype(float)
rng = np.random.default_rng(42)
rows = []
for label, use in [("SVR on urban only", ["urban"]), ("SVR on income only", ["gni"]), ("SVR on child mortality only", ["u5"]), ("SVR on urban + child mortality, income frozen (post hoc)", ["urban", "u5"]), ("SVR on all three", ["urban", "gni", "u5"]), ("Actual income change, others frozen (oracle)", "oracle_gni"), ("Actual urban change, others frozen (oracle)", "oracle_urban"), ("Actual u5 change, others frozen (oracle)", "oracle_u5")]:
    F = A0.loc[ix, ["urban", "gni", "u5"]].copy()
    if isinstance(use, str):
        k = use.split("_")[1]
        F[k] = A1.loc[ix, k]
    else:
        for k in use:
            F[k] = P.loc[ix, k]
    X = mat(F)
    s = pct(X).mean(1)
    d = (r0 - ranks(s)).astype(float)
    bs, gaps = [], []
    for _ in range(2000):
        r = rng.integers(0, n, n)
        tb, nb, sb = pct(X1[r]).mean(1), pct(X0[r]).mean(1), pct(X[r]).mean(1)
        rb = ranks(nb)
        da = (rb - ranks(tb)).astype(float)
        bs.append(spear((rb - ranks(sb)).astype(float), da))
        gaps.append(spear(sb, tb) - spear(nb, tb))
    rows.append({"forecast": label, "picks_rho": spear(s, t), "top10": overlap(s, t, 10), "top20": overlap(s, t, 20), "gap_rho_lo": np.nanquantile(gaps, .025), "gap_rho_hi": np.nanquantile(gaps, .975), "mover_rho": spear(d, d_act), "lo": np.nanquantile(bs, .025), "hi": np.nanquantile(bs, .975)})
print("No change: picks_rho", round(spear(nc, t), 4), "top10", overlap(nc, t, 10), "top20", overlap(nc, t, 20))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nShare of the actual rank change explained by each indicator's actual change (oracle rows above). If income alone explains most of it, unpredictable income caps the skill.")
