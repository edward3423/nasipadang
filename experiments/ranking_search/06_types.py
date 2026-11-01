"""A non-clustered market-type story: dominant strength group and biggest weakness group. Compared with the six-type GMM on the same checks."""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import pickle, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import *
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.mixture import GaussianMixture

pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 300, "display.max_colwidth", 90)
b = load_bundle()
cols, signs, groups = b["feature_cols"], b["signs"], b["groups"]
GN = list(groups)
g_of = {f: i for i, v in enumerate(groups.values()) for f in v}
g = np.array([g_of[c] for c in cols])
sgn = np.array([signs[c] for c in cols])
SHORT = {"Demand": "Demand", "Rice and beef supply": "Supply", "Climate and policy": "Climate", "Kitchen and operations": "Kitchen"}


def labels_rule(df):
    r = df[df["population"] >= 1e6]
    X = r[cols].values.astype(float) * sgn
    S, _ = group_scores(pct(X), g)
    # compare each group with its own cross-country mean, so a group with a wide spread does not always win
    Z = S - S.mean(0)
    dom = pd.Series(Z.argmax(1), index=r.index)
    weak = pd.Series(Z.argmin(1), index=r.index)
    return dom, weak, pd.DataFrame(S, index=r.index, columns=GN)


def gmm_labels(C, seed=42):
    Z = PCA(8, random_state=42).fit_transform(C)
    return GaussianMixture(6, covariance_type="spherical", reg_covar=1e-6, init_params="random", n_init=5, max_iter=500, random_state=seed).fit(Z).predict(Z)


def gmm_sub(C, idx, seed):
    return gmm_labels(C[idx], seed)


def eta2(Y, lab):
    """Share of the variance of the matrix Y explained by the labels."""
    tot = ((Y - Y.mean(0)) ** 2).sum()
    wit = 0.0
    for l in np.unique(lab):
        m = lab == l
        wit += ((Y[m] - Y[m].mean(0)) ** 2).sum()
    return 1 - wit / tot


def main():
    all_idx = list(b["index"])
    pos = {c: i for i, c in enumerate(all_idx)}
    d_s, w_s, S_s = labels_rule(b["features_2030"])
    d_k, w_k, S_k = labels_rule(b["features_2030_knn"])
    shared = d_s.index.intersection(d_k.index)
    p = (d_s * 4 + w_s).rename("pair")
    pk = (d_k * 4 + w_k)
    print("== Rule labels on 2030 (SVR) ==")
    print("Dominant strength counts:", d_s.map(lambda i: SHORT[GN[i]]).value_counts().to_dict())
    print("Biggest weakness counts:", w_s.map(lambda i: SHORT[GN[i]]).value_counts().to_dict())
    print("Distinct strength-weakness pairs:", p.nunique(), "of 12; sizes", sorted(p.value_counts().tolist(), reverse=True))

    C = b["C_2030"]; Ck = b["C_2030_knn"]
    rows = [pos[c] for c in shared]
    gm_s = gmm_labels(C); gm_k = gmm_labels(Ck)
    print("\n== Agreement between the SVR and KNN forecast tables, 159 shared ranked countries ==")
    print("Dominant group: same label", round(float((d_s[shared] == d_k[shared]).mean()), 3), "ARI", round(ari(d_s[shared], d_k[shared]), 3))
    print("Weakest group: same label", round(float((w_s[shared] == w_k[shared]).mean()), 3), "ARI", round(ari(w_s[shared], w_k[shared]), 3))
    print("Strength-weakness pair: same label", round(float((p[shared] == pk[shared]).mean()), 3), "ARI", round(ari(p[shared], pk[shared]), 3))
    print("GMM k=6 spherical, re-fit on each table: ARI", round(ari(gm_s[rows], gm_k[rows]), 3))

    # stability when 20% of countries are removed
    rng = np.random.default_rng(42)
    F = b["features_2030"]
    r = F[F["population"] >= 1e6]
    par = Parallel(n_jobs=-1)
    subs = [np.sort(rng.choice(len(C), int(0.8 * len(C)), replace=False)) for _ in range(50)]
    gm_runs = par(delayed(gmm_sub)(C, s, 1000 + i) for i, s in enumerate(subs))
    gmm_stab = np.mean([ari(gm_s[s], l) for s, l in zip(subs, gm_runs)])
    a_dom, a_pair = [], []
    for s in range(50):
        keep = np.sort(rng.choice(len(r), int(0.8 * len(r)), replace=False))
        ds, ws, _ = labels_rule(r.iloc[keep])
        a_dom.append(ari(d_s[ds.index], ds))
        a_pair.append(ari(p[ds.index], ds * 4 + ws))
    print("\n== Refit on random 80% of countries, 50 repeats, mean ARI ==")
    print("Dominant group", round(float(np.mean(a_dom)), 3), "| strength-weakness pair", round(float(np.mean(a_pair)), 3), "| GMM (with a new seed each repeat)", round(float(gmm_stab), 3))
    seeds = par(delayed(gmm_labels)(C, s) for s in range(30))
    print("GMM across 30 seeds on the full table, mean ARI vs seed 42:", round(float(np.mean([ari(gm_s, l) for l in seeds])), 3), "| rule labels have no seed")

    print("\n== How much of the variation in the 11 features do the labels carry? (between-label share of variance, ranked countries) ==")
    Xz = zs(r[cols].values.astype(float) * sgn)
    print("Dominant group (4 labels)", round(eta2(Xz, d_s.values), 3), "| weakest group (4)", round(eta2(Xz, w_s.values), 3), "| pair (up to 12)", round(eta2(Xz, p.values), 3))
    print("GMM (6 labels, of which ranked countries use", len(np.unique(gm_s[[pos[c] for c in r.index]])), ")", round(eta2(Xz, gm_s[[pos[c] for c in r.index]]), 3))
    n_perm = 200
    rg = np.random.default_rng(1)
    print("Same, labels shuffled (null), pair:", round(float(np.mean([eta2(Xz, rg.permutation(p.values)) for _ in range(n_perm)])), 3), "GMM:", round(float(np.mean([eta2(Xz, rg.permutation(gm_s[[pos[c] for c in r.index]])) for _ in range(n_perm)])), 3))

    # relation with the current labels on today's table
    now = pd.read_csv(Path(r"C:\Users\Edward\OneDrive - Nanyang Technological University\Materials\MA0218\miniproject\nasipadang\nasipadang_ranking.csv"), index_col=0)
    dn, wn, _ = labels_rule(b["features_now"])
    sh = dn.index.intersection(now.index)
    print("\n== Rule labels vs today's six GMM types, on today's table ==")
    print(pd.crosstab(now.loc[sh, "market_type"], dn[sh].map(lambda i: SHORT[GN[i]]), margins=True).to_string())
    print(pd.crosstab(now.loc[sh, "market_type"], wn[sh].map(lambda i: SHORT[GN[i]]), margins=True).to_string())
    print("ARI, GMM vs dominant:", round(ari(now.loc[sh, "market_type"], dn[sh]), 3), "vs pair:", round(ari(now.loc[sh, "market_type"], dn[sh] * 4 + wn[sh]), 3))

    # top 20 story
    R = b["rank_2030"].sort_values("rank")
    t = pd.DataFrame({"name": R["name"], "rank": R["rank"], "strength": d_s.map(lambda i: SHORT[GN[i]]), "weakness": w_s.map(lambda i: SHORT[GN[i]])}).loc[R.index].head(20)
    t = pd.concat([t, S_s.loc[t.index].round(2).rename(columns=SHORT)], axis=1)
    print("\n== Top 20 of the baseline with rule labels ==")
    print(t.to_string())
    print("Top 20 strength mix:", t["strength"].value_counts().to_dict(), "weakness mix:", t["weakness"].value_counts().to_dict())
    print("\nAverage score by strength label:")
    print(pd.DataFrame({"score": b["rank_2030"]["score"], "strength": d_s.map(lambda i: SHORT[GN[i]])}).groupby("strength")["score"].agg(["count", "mean"]).round(3).to_string())
    print("Average score by weakness label:")
    print(pd.DataFrame({"score": b["rank_2030"]["score"], "weakness": w_s.map(lambda i: SHORT[GN[i]])}).groupby("weakness")["score"].agg(["count", "mean"]).round(3).to_string())
    print("IDN, MYS, SGP:", {c: (SHORT[GN[d_s[c]]], SHORT[GN[w_s[c]]]) for c in ["IDN", "MYS", "SGP"]})


if __name__ == "__main__":
    main()
