"""Small honesty checks: exact ties in the baseline, PROMETHEE-usual identity, whether the Kemeny local search is stable, SMAA tilt fix."""
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab import *

b = load_bundle()
cols, signs, groups = b["feature_cols"], b["signs"], b["groups"]
g_of = {f: i for i, v in enumerate(groups.values()) for f in v}
g = np.array([g_of[c] for c in cols])
sgn = np.array([signs[c] for c in cols])
F = b["features_2030"]
r = F[F["population"] >= 1e6]
X = r[cols].values.astype(float) * sgn

s = m_percentile(X, g)
rk = pd.Series(s, index=r.index).round(10).rank(ascending=False).astype(int)
ref = b["rank_2030"]["rank"].reindex(r.index)
print("Rank mismatches vs bundle after removing floating noise:", int((rk != ref).sum()))
vals, cnt = np.unique(np.round(s, 10), return_counts=True)
print("Exact score ties in the baseline:", int((cnt > 1).sum()), "pairs or larger")
tied = pd.Series(np.round(s, 10), index=r.index)
print("Tied in the top 40?", tied[tied.duplicated(keep=False)].sort_values(ascending=False).head(6).to_dict())

u = m_promethee_usual(X, g)
n = len(X)
print("PROMETHEE-usual minus linear map of percentile, max abs:", float(np.abs(u - (2 * n / (n - 1)) * (s - (n + 1) / (2 * n))).max()))
print("Same order after rounding:", bool((order(u) == order(s)).all()), "rho", spear(u, s))

sc = {v: METHODS[v](X, g) for v in VOTERS}
rm = np.column_stack([ranks(sc[v]) for v in VOTERS])
k0 = kemeny_local(rm)
k20 = kemeny_local(rm, restarts=20, seed=1)
print("Kemeny: local search from Borda vs best of 20 random restarts, same order:", bool((order(k0) == order(k20)).all()), "rho", spear(k0, k20), "top10", overlap(k0, k20, 10), "top20", overlap(k0, k20, 20))
print("Kemeny vs plain Borda of the 7 voters: rho", spear(k0, -rm.mean(1)), "top10", overlap(k0, -rm.mean(1), 10), "top20", overlap(k0, -rm.mean(1), 20))
print("Voter agreement, mean pairwise Spearman:", np.round(np.mean([spear(rm[:, i], rm[:, j]) for i in range(7) for j in range(i + 1, 7)]), 3))

rng = np.random.default_rng(42)
Wg = rng.dirichlet(np.ones(4), size=300)
sizes = np.array([(g == t).sum() for t in range(4)])
Wf = Wg[:, g] / sizes[g]
base = m_smaa_meanrank(X, g)
t = [overlap(m_smaa_meanrank(X, g, w, draws=500), base, 10) for w in Wf]
t2 = [overlap(m_smaa_meanrank(X, g, w, draws=500), base, 20) for w in Wf]
sp = [spear(m_smaa_meanrank(X, g, w, draws=500), base) for w in Wf]
print("SMAA mean rank under tilted priors (group draws, 300 outer, 500 inner): top10", round(np.mean(t), 2), "p5", np.quantile(t, .05), "top20", round(np.mean(t2), 2), "rho", round(np.mean(sp), 3))
print("Expected score identity: mean over Dirichlet(1) of the weighted score equals the equal-weight score. Max abs diff:", float(np.abs(group_scores(pct(X), g)[0] @ rng.dirichlet(np.ones(4), size=200000).mean(0) - s).max()))
