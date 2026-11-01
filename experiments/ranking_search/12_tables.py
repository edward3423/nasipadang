"""Build the report tables from the saved results, so no number is typed by hand. Also a few last checks used in the text."""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import *

b = load_bundle()
cols, signs, groups = b["feature_cols"], b["signs"], b["groups"]
g_of = {f: i for i, v in enumerate(groups.values()) for f in v}
g = np.array([g_of[c] for c in cols])
sgn = np.array([signs[c] for c in cols])
extra = pickle.load(open(HERE / "extra_2030.pkl", "rb"))
FX = extra["features"]

NAMES = {
    "percentile": "Percentile mean (baseline)", "percentile_equal_feature": "Percentile, equal feature weights", "promethee_usual": "PROMETHEE II, usual (identical to baseline)",
    "copeland": "Copeland head-to-head", "smaa_meanrank": "SMAA mean rank", "smaa_top10": "SMAA P(top 10)", "promethee_v": "PROMETHEE II, V-shape 2 SD",
    "zscore_mean": "Z-score mean", "minmax_mean": "Min-max mean", "topsis_minmax": "TOPSIS, min-max", "topsis_pct": "TOPSIS, percentile",
    "kemeny": "Kemeny, 7 methods", "schulze": "Schulze, 7 methods", "geomean_groups": "Geometric mean of groups", "weakest_group": "Weakest group", "critic": "CRITIC weights",
    "vikor_pct": "VIKOR, percentile", "vikor_minmax": "VIKOR, min-max", "entropy": "Entropy weights", "pca": "PCA first component",
    "bod_ar": "DEA BoD, group bounds", "bod_pure": "DEA BoD, free weights", "pareto_groups": "Pareto, 4 groups", "pareto_features": "Pareto, 11 features",
}
ORDER = list(NAMES)
EXPLAIN = {"percentile": 5, "percentile_equal_feature": 5, "promethee_usual": 3, "copeland": 4, "smaa_meanrank": 4, "smaa_top10": 4, "promethee_v": 3, "zscore_mean": 4, "minmax_mean": 4,
           "topsis_minmax": 3, "topsis_pct": 3, "kemeny": 2, "schulze": 2, "geomean_groups": 4, "weakest_group": 5, "critic": 2, "vikor_pct": 2, "vikor_minmax": 2, "entropy": 3, "pca": 4,
           "bod_ar": 2, "bod_pure": 3, "pareto_groups": 4, "pareto_features": 4}
c = pd.read_csv(HERE / "criteria_2030.csv", index_col=0)
T = pd.read_csv(HERE / "backtest_table.csv", index_col=0)
M = pd.read_csv(HERE / "movers_table.csv", index_col=0)


def md(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def na(v, f="{:.2f}"):
    return "n/a" if pd.isna(v) else f.format(v)


A = []
for m in ORDER:
    r = c.loc[m]
    A.append([NAMES[m], f"{r.svr_knn_spearman:.3f}", int(r.svr_knn_top10), int(r.svr_knn_top20), f"{r.models7_top10_mean:.1f}", f"{na(r.grp_top10_mean)} ({na(r.grp_top10_p5, '{:.0f}')})", na(r.feat_top10_mean), f"{r.dropfeat_top10_mean:.1f} ({r.dropfeat_top10_min:.0f})", f"{r.dropgroup_top10_mean:.1f}", f"{r.noise25_top10_mean:.1f}", f"{r.drop10pct_top10_mean:.1f}"])
ta = md(A, ["Method", "SVR-KNN rho", "top 10", "top 20", "7-model pairs top 10", "Dirichlet groups: top 10 kept (p5)", "Dirichlet features: top 10 kept", "Drop 1 feature: top 10 (min)", "Drop 1 group: top 10", "Noise 0.25 SD: top 10", "Drop 10% of countries: top 10"])

B_ = []
for m in ORDER:
    r, v = T.loc[m], M.loc[m]
    B_.append([NAMES[m], f"{r.svr_spearman:.3f} / {r.nc_spearman:.3f}", f"{r.gap_spearman:+.4f} [{r.gap_spearman_lo:+.4f}, {r.gap_spearman_hi:+.4f}]", f"{int(r.svr_top10)} / {int(r.nc_top10)}", f"[{r.gap_top10_lo:+.0f}, {r.gap_top10_hi:+.0f}]", f"{int(r.svr_top20)} / {int(r.nc_top20)}", f"[{r.gap_top20_lo:+.0f}, {r.gap_top20_hi:+.0f}]",
               f"{v.svr_rho:.2f} [{v.svr_rho_lo:.2f}, {v.svr_rho_hi:.2f}]", f"{v.trend_rho:.2f}", f"{v.mean_reversion_rho:.2f}", f"[{v.svr_minus_mr_lo:+.2f}, {v.svr_minus_mr_hi:+.2f}]", f"{int(v.svr_hits)} [{v.svr_hits_lo:.0f}, {v.svr_hits_hi:.0f}]"])
tb = md(B_, ["Method", "rho SVR / no-change", "gap and 95% CI", "top 10 SVR / no-change", "gap CI", "top 20 SVR / no-change", "gap CI", "movers rho, SVR [CI]", "trend line", "mean reversion", "SVR minus mean reversion CI", "top-20 risers hit by SVR [CI]"])

C_ = []
for m in ORDER:
    r = c.loc[m]
    C_.append([NAMES[m], f"{r.spearman_base:.3f}", int(r.top10_base), int(r.top20_base), int(r.IDN), int(r.MYS), int(r.SGP), int(r.PRI), int(r.HKG), int(r.CUB), int(r.MMR), int(r.USA), EXPLAIN[m]])
tc = md(C_, ["Method", "rho vs baseline", "top 10 shared", "top 20 shared", "IDN", "MYS", "SGP", "PRI", "HKG", "CUB", "MMR", "USA", "Explainability 1-5 (my judgement)"])

out = pd.read_csv(HERE / "final_ranking_2030.csv", index_col=0)
ag = pd.read_csv(HERE / "method_agreement.csv", index_col=0)
ids = out.index
per = {}
for k, df in FX.items():
    s = pct(df.loc[ids, cols].values.astype(float) * sgn) @ equal_group_w(g)
    per[k] = pd.Series(ranks(s), index=ids)
per = pd.DataFrame(per)
in10 = (per <= 10).sum(axis=1)
in20 = (per <= 20).sum(axis=1)
D = []
ix132 = set(b["backtest_actual_2000"].index[b["backtest_actual_2000"]["population"] >= np.log(1e6)])
for code, r in out.head(20).iterrows():
    flags = []
    if r.imputed_inputs >= 4:
        flags.append(f"{int(r.imputed_inputs)} of 16 inputs imputed")
    if ag.loc[code, "in_top20_of_6"] <= 3:
        flags.append(f"top 20 in only {int(ag.loc[code, 'in_top20_of_6'])} of 6 methods")
    D.append([int(r["rank"]), r["name"], f"{r.score:.3f}", int(r.baseline_rank), int(r.knn_rank), f"{r.p_top10*100:.0f}% ({r.p10_min_model*100:.0f}-{r.p10_max_model*100:.0f})", f"{r.p_top20*100:.0f}%", f"{r.p_top20_tight*100:.0f}%", f"{int(in10[code])}/7", f"{int(ag.loc[code, 'in_top20_of_6'])}/6", r.strength, r.weakness, "; ".join(flags)])
td = md(D, ["Rank", "Country", "Score", "Baseline rank (SVR only)", "KNN only rank", "P(top 10), range across forecasts", "P(top 20)", "P(top 20), tight weights", "Top 10 in forecast models", "Top 20 in method families", "Strength", "Weakness", "Flag"])

base = out.sort_values("baseline_rank").head(20)
E = []
for code, r in base.iterrows():
    E.append([int(r.baseline_rank), r["name"], int(r["rank"])])
te = md(E, ["Baseline rank", "Country", "Recommended rank"])

Path(HERE / "tables.md").write_text("\n\n".join(["## A", ta, "## B", tb, "## C", tc, "## D", td, "## E", te]), encoding="utf-8")
print("tables written")
print("Top-20 (recommended) not in the 132 backtest countries:", out.head(20).index[~out.head(20).index.isin(ix132)].map(out["name"]).tolist())
print("Baseline top-20 not in backtest:", base.index[~base.index.isin(ix132)].map(out["name"]).tolist())
print("Top 10 membership counts across 7 forecasts:", {out.loc[c_, "name"]: int(in10[c_]) for c_ in in10[in10 > 0].sort_values(ascending=False).index})
print("Top 20 membership counts = 7:", out.loc[in20[in20 == 7].index, "name"].tolist())
print("Scores at recommended ranks 1, 5, 7, 8, 11, 14, 20, 21:", out["score"].iloc[[0, 4, 6, 7, 10, 13, 19, 20]].round(3).tolist(), "SD", round(float(out["score"].std()), 3))
print("Enter top 10 / top 20 summary: top10 shared with baseline", len(set(out.head(10).index) & set(base.head(10).index)), "top20 shared", len(set(out.head(20).index) & set(base.head(20).index)))
print("Spearman recommended vs baseline over all:", round(spear(out["score"], -out["baseline_rank"]), 3))
print("Average imputed inputs: top 20", out.head(20)["imputed_inputs"].mean(), "all", out["imputed_inputs"].mean())
print("Excluding >=4 imputed, top 10:", out[out.imputed_inputs < 4].head(10)["name"].tolist())
print(out.loc[["IDN","MYS","SGP","PRI","HKG","CUB","MMR"], ["name","rank","baseline_rank","p_top10","p_top20","p_top20_tight","rank_p5","rank_p95","rank_p5_tight","rank_p95_tight","strength","weakness","imputed_inputs"]].round(2).to_string())
