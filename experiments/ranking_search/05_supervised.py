"""Supervised ranking, tested honestly.

A. Negative control: train a model to reproduce the percentile score from the same features. Any high number here means nothing.
B. Valid framing: predict the composite rank ten years ahead from values known today.
   Train on 1990 inputs -> 2000 composite. Test on 2000 inputs -> 2010 composite, on the same 132 countries as criterion 3.
   The target is still a formula of three indicators, so this tests predictability, not whether the formula is right.
"""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import *
from panels import *
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 300)
b = load_bundle()
cols, signs, groups = b["feature_cols"], b["signs"], b["groups"]
g_of = {f: i for i, v in enumerate(groups.values()) for f in v}
g = np.array([g_of[c] for c in cols])
sgn = np.array([signs[c] for c in cols])
B = 2000


def mimic():
    F = b["features_2030"]
    r = F[F["population"] >= 1e6]
    X = r[cols].values.astype(float) * sgn
    P = pct(X)
    y = m_percentile(X, g)
    kf = KFold(5, shuffle=True, random_state=42)
    M = {"LinearRegression on percentile features": (LinearRegression(), P),
         "LinearRegression on raw features": (LinearRegression(), zs(X)),
         "SVR on raw features": (make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.005)), X),
         "RandomForest on raw features": (RandomForestRegressor(500, min_samples_leaf=2, random_state=42, n_jobs=-1), X),
         "GradBoost on raw features": (HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=5, random_state=42), X)}
    rows = []
    for nm, (m, Z) in M.items():
        oof = np.zeros(len(y))
        for a, c in kf.split(Z):
            oof[c] = m.fit(Z[a], y[a]).predict(Z[c])
        rows.append({"model": nm, "oof_r2": 1 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum(), "oof_spearman": spear(oof, y), "top10": overlap(oof, y, 10), "top20": overlap(oof, y, 20)})
    lab_y = (y >= np.quantile(y, 0.8)).astype(int)
    oof = np.zeros(len(y))
    for a, c in StratifiedKFold(5, shuffle=True, random_state=42).split(P, lab_y):
        oof[c] = LogisticRegression(C=1.0, max_iter=2000).fit(P[a], lab_y[a]).predict_proba(P[c])[:, 1]
    rows.append({"model": "Pseudo-label: logistic on top 20% label (AUC in r2 column)", "oof_r2": roc_auc_score(lab_y, oof), "oof_spearman": spear(oof, y), "top10": overlap(oof, y, 10), "top20": overlap(oof, y, 20)})
    print("== A. Negative control: reproducing the formula from its own inputs, out of fold ==")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    co = LinearRegression().fit(P, y)
    print("LinearRegression coefficients vs the formula weights:")
    print(pd.DataFrame({"coef": co.coef_, "formula": equal_group_w(g)}, index=cols).round(4).T.to_string())


def composite(M):
    return pct(M).mean(1)


class Pairwise:
    """Handwritten pairwise transform, then logistic regression on feature differences (linear RankNet/RankSVM style)."""

    def fit(self, X, y):
        rng = np.random.default_rng(42)
        i, j = np.triu_indices(len(X), 1)
        keep = y[i] != y[j]
        i, j = i[keep], j[keep]
        flip = rng.random(len(i)) < 0.5
        a, c = np.where(flip, j, i), np.where(flip, i, j)
        D = X[a] - X[c]
        lab = (y[a] > y[c]).astype(int)
        self.m = LogisticRegression(C=1.0, fit_intercept=False, max_iter=5000).fit(D, lab)
        return self

    def predict(self, X):
        return X @ self.m.coef_[0]


def models():
    return {"Ridge": lambda: Ridge(1.0), "SVR": lambda: SVR(C=1.0, epsilon=0.05),
            "GradBoost": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=10, random_state=42),
            "RandomForest": lambda: RandomForestRegressor(500, min_samples_leaf=3, random_state=42, n_jobs=1),
            "PairwiseLogistic": lambda: Pairwise()}


def make_x(pan, t, ids, rich, fill=True):
    L = pan["L"]
    base = [L["urban"][t].loc[ids].values, L["gni"][t].loc[ids].values, -L["u5"][t].loc[ids].values]
    Z = pct(np.column_stack(base))
    if not rich:
        return Z
    ex = []
    for k in ["co2", "yield"]:
        v = L[k][t].loc[ids]
        v = v.fillna(v.median()).values
        ex.append(rankdata(v) / len(v))
    reg = pd.get_dummies(pan["region"].loc[ids]).reindex(columns=sorted(pan["region"].unique()), fill_value=0).values.astype(float)
    return np.column_stack([Z, *ex, reg])


def main():
    t0 = time.time()
    mimic()
    pan = load_panels()
    L = pan["L"]
    A0, A1 = b["backtest_actual_2000"], b["backtest_actual_2010"]
    ix = A0.index[A0["population"] >= np.log(1e6)]
    ok_tr = (L["urban"][1990].notna() & L["gni"][1990].notna() & L["u5"][1990].notna() & L["urban"][2000].notna() & L["gni"][2000].notna() & L["u5"][2000].notna() & (L["population"][1990] >= np.log(1e6)))
    tr = ok_tr.index[ok_tr]
    print(f"\nTrain countries {len(tr)} (1990 -> 2000). Test countries {len(ix)} (2000 -> 2010). Unseen-in-training test countries: {len(set(ix) - set(tr))}")
    ytr = composite(np.column_stack([L["urban"][2000].loc[tr], L["gni"][2000].loc[tr], -L["u5"][2000].loc[tr]]))
    M1 = np.column_stack([A1.loc[ix, "urban"], A1.loc[ix, "gni"], -A1.loc[ix, "u5"]]).astype(float)
    M0 = np.column_stack([A0.loc[ix, "urban"], A0.loc[ix, "gni"], -A0.loc[ix, "u5"]]).astype(float)
    y1, y0 = composite(M1), composite(M0)
    n = len(ix)
    rows, mv, boot_rows, mv_boot = [], [], [], []
    rng = np.random.default_rng(42)
    res_rows = rng.integers(0, n, (B, n))
    rich_extra = {}
    for rich in [False, True]:
        Xtr = make_x(pan, 1990, tr, rich)
        Xte = make_x(pan, 2000, ix, rich)
        for nm, mk in models().items():
            m = mk().fit(Xtr, ytr)
            p = m.predict(Xte)
            r0 = ranks(y0)
            d_act = (r0 - ranks(y1)).astype(float)
            d_p = (r0 - ranks(p)).astype(float)
            # bootstrap the test countries; inputs are percentiles so they are recomputed on each resample
            gaps = {k: [] for k in ["spearman", "top10", "top20", "mover", "mover_minus_mr"]}
            for s in range(B):
                rr = res_rows[s]
                if rich:
                    Xb = Xte[rr].copy()
                    Xb[:, :5] = pct(Xte[rr][:, :5])
                    Xb[:, :3] = pct(M0[rr])
                else:
                    Xb = pct(M0[rr])
                pb = m.predict(Xb)
                yb1, yb0 = composite(M1[rr]), composite(M0[rr])
                gaps["spearman"].append(spear(pb, yb1) - spear(yb0, yb1))
                gaps["top10"].append(overlap(pb, yb1, 10) - overlap(yb0, yb1, 10))
                gaps["top20"].append(overlap(pb, yb1, 20) - overlap(yb0, yb1, 20))
                rb0 = ranks(yb0)
                db = (rb0 - ranks(yb1)).astype(float)
                sp_m = spear((rb0 - ranks(pb)).astype(float), db)
                gaps["mover"].append(sp_m)
                gaps["mover_minus_mr"].append(sp_m - spear(rb0 - (n + 1) / 2, db))
            q = {k: np.nanquantile(v, [0.025, 0.975]) for k, v in gaps.items()}
            rows.append({"inputs": "6 indicators + region" if rich else "3 indicators", "model": nm, "spearman": spear(p, y1), "nochange": spear(y0, y1), "gap_lo": q["spearman"][0], "gap_hi": q["spearman"][1],
                         "top10": overlap(p, y1, 10), "nc_top10": overlap(y0, y1, 10), "t10_lo": q["top10"][0], "t10_hi": q["top10"][1], "top20": overlap(p, y1, 20), "nc_top20": overlap(y0, y1, 20), "t20_lo": q["top20"][0], "t20_hi": q["top20"][1],
                         "train_fit_spearman": spear(m.predict(Xtr), ytr)})
            mv.append({"inputs": "6 indicators + region" if rich else "3 indicators", "model": nm, "mover_rho": spear(d_p, d_act), "lo": q["mover"][0], "hi": q["mover"][1], "mean_reversion_rho": spear(r0 - (n + 1) / 2, d_act), "gap_lo": q["mover_minus_mr"][0], "gap_hi": q["mover_minus_mr"][1]})
            print(nm, "rich" if rich else "base", f"{time.time()-t0:.0f}s", flush=True)
    T = pd.DataFrame(rows)
    print("\n== B. Direct model of the composite rank ten years ahead, trained on 1990 -> 2000, tested on 2000 -> 2010 ==")
    print(T.round(3).to_string(index=False))
    MVt = pd.DataFrame(mv)
    print("\nRank-change skill on the same test:"); print(MVt.round(3).to_string(index=False))
    T.to_csv(HERE / "supervised_table.csv", index=False)
    MVt.to_csv(HERE / "supervised_movers.csv", index=False)
    print("done", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
