"""Rank-construction methods, all on one interface.

X is an n x m array oriented so that higher = better (raw value * sign).
g is an int array of length m giving each feature's group.
w is an optional feature-weight vector summing to 1. None means the method's own default.
Every method returns a score vector with higher = better.

The normalisations and settings below were fixed before any result was seen:
 - min-max is the textbook normalisation for TOPSIS, VIKOR, entropy and CRITIC; percentile variants are also reported.
 - PROMETHEE V-shape threshold p = 2 std dev per criterion.
 - VIKOR v = 0.5 (textbook compromise).
 - BoD assurance region: every group's share of the country's own score is in [0.5 / G, 2 / G].
"""
import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")  # parallelism comes from joblib; avoid oversubscribing the CPU
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent / "run3" / "bundle_2030.pkl"
XLS = Path(r"C:\Users\Edward\OneDrive - Nanyang Technological University\Materials\MA0218\miniproject\nasipadang\data\climate_change.xls")


def load_bundle():
    with open(BUNDLE, "rb") as fh:
        return pickle.load(fh)


# ---------- helpers ----------

def pct(X):
    """Column percentile, identical to pandas rank(pct=True) with average ties."""
    return np.apply_along_axis(rankdata, 0, X) / X.shape[0]


def minmax(X):
    lo, hi = X.min(0), X.max(0)
    return (X - lo) / np.where(hi > lo, hi - lo, 1.0)


def zs(X):
    s = X.std(0, ddof=0)
    return (X - X.mean(0)) / np.where(s > 0, s, 1.0)


def equal_group_w(g):
    g = np.asarray(g)
    labels, counts = np.unique(g, return_counts=True)
    size = dict(zip(labels, counts))
    return np.array([1 / (len(labels) * size[k]) for k in g])


def group_scores(P, g, w=None):
    """Group score = within-group weighted mean; returns (n x G scores, G weights)."""
    g = np.asarray(g)
    if w is None:
        w = equal_group_w(g)
    labels = np.unique(g)
    S, W = [], []
    for l in labels:
        m = g == l
        tot = w[m].sum()
        W.append(tot)
        S.append(P[:, m] @ (w[m] / tot if tot > 0 else np.full(m.sum(), 1 / m.sum())))
    return np.column_stack(S), np.array(W)


def order(s):
    """Row indices best first. Rounding removes floating-point noise; ties keep row order, so identical rows tie the same way everywhere."""
    return np.argsort(-np.round(np.asarray(s, dtype=float), 10), kind="stable")


def ranks(s):
    r = np.empty(len(s), dtype=int)
    r[order(s)] = np.arange(1, len(s) + 1)
    return r


def spear(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def topk(s, k):
    return set(order(s)[:k].tolist())


def overlap(a, b, k):
    return len(topk(a, k) & topk(b, k))


# ---------- additive methods ----------

def m_percentile(X, g, w=None):
    return pct(X) @ (equal_group_w(g) if w is None else w)


def m_percentile_featw(X, g, w=None):
    return pct(X) @ (np.full(X.shape[1], 1 / X.shape[1]) if w is None else w)


def m_zmean(X, g, w=None):
    return zs(X) @ (equal_group_w(g) if w is None else w)


def m_minmaxmean(X, g, w=None):
    return minmax(X) @ (equal_group_w(g) if w is None else w)


def m_geomean(X, g, w=None):
    S, W = group_scores(pct(X), g, w)
    return np.exp(np.log(np.maximum(S, 1e-9)) @ (W / W.sum()))


def m_weakest(X, g, w=None):
    """Weakest link: the lowest group percentile; ties broken by the mean. Weighted form: S ** (G * W) (Yager)."""
    S, W = group_scores(pct(X), g, w)
    G = len(W)
    T = S ** (G * W)
    return T.min(1) + 1e-6 * T.mean(1)


def _topsis(R, w):
    V = R * w
    dp = np.sqrt(((V.max(0) - V) ** 2).sum(1))
    dn = np.sqrt(((V - V.min(0)) ** 2).sum(1))
    return dn / np.maximum(dp + dn, 1e-12)


def m_topsis_minmax(X, g, w=None):
    return _topsis(minmax(X), equal_group_w(g) if w is None else w)


def m_topsis_pct(X, g, w=None):
    return _topsis(pct(X), equal_group_w(g) if w is None else w)


def promethee_flows(X, kind="v"):
    """Per-criterion net flow, n x m. Weighted sum of these columns is the PROMETHEE II net flow."""
    n, m = X.shape
    F = np.empty((n, m))
    for k in range(m):
        d = X[:, k][:, None] - X[:, k][None, :]
        if kind == "usual":
            P = (d > 0).astype(float)
        else:
            p = 2 * X[:, k].std()
            P = np.clip(d / p, 0, 1) if p > 0 else np.zeros_like(d)
        F[:, k] = (P - P.T).sum(1) / (n - 1)
    return F


def m_promethee_usual(X, g, w=None):
    return promethee_flows(X, "usual") @ (equal_group_w(g) if w is None else w)


def m_promethee_v(X, g, w=None):
    return promethee_flows(X, "v") @ (equal_group_w(g) if w is None else w)


def _vikor(R, w, v=0.5):
    Rw = (1 - R) * w
    S = Rw.sum(1)
    Q2 = Rw.max(1)
    a = (S - S.min()) / max(S.max() - S.min(), 1e-12)
    b = (Q2 - Q2.min()) / max(Q2.max() - Q2.min(), 1e-12)
    return -(v * a + (1 - v) * b)


def m_vikor_minmax(X, g, w=None):
    return _vikor(minmax(X), equal_group_w(g) if w is None else w)


def m_vikor_pct(X, g, w=None):
    return _vikor(pct(X), equal_group_w(g) if w is None else w)


def m_copeland(X, g, w=None):
    """Each feature votes with its weight; A beats B if it wins the weighted majority of features. Score = wins - losses."""
    w = equal_group_w(g) if w is None else w
    n = X.shape[0]
    C = np.zeros((n, n))
    for k in range(X.shape[1]):
        C += w[k] * np.sign(X[:, k][:, None] - X[:, k][None, :])
    s = np.sign(np.round(C, 12)).sum(1)
    # a small tie-break by the margin so that a score is defined everywhere
    return s + 1e-6 * C.sum(1)


# ---------- data-driven weights ----------

def entropy_weights(X):
    R = minmax(X)
    p = R / np.maximum(R.sum(0), 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -np.where(p > 0, p * np.log(p), 0).sum(0) / np.log(X.shape[0])
    d = 1 - e
    return d / d.sum()


def critic_weights(X):
    R = minmax(X)
    s = R.std(0, ddof=1)
    c = np.corrcoef(R, rowvar=False)
    C = s * (1 - c).sum(0)
    return C / C.sum()


def _reweight(base, w, g):
    """Analog of a weight draw for data-driven weights: multiply by draw (scaled so equal-group is neutral)."""
    if w is None:
        return base
    tilt = w / equal_group_w(g)
    out = base * tilt
    return out / out.sum()


def m_entropy(X, g, w=None):
    return minmax(X) @ _reweight(entropy_weights(X), w, g)


def m_critic(X, g, w=None):
    return minmax(X) @ _reweight(critic_weights(X), w, g)


# ---------- PCA ----------

def pc1(X, g=None, w=None):
    Z = zs(X)
    if w is not None:
        Z = Z * np.sqrt(w * X.shape[1])
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    v = Vt[0]
    if v.sum() < 0:  # orientation rule: positive mean loading, never decided by agreement with the baseline
        v = -v
    return Z @ v, v, S[0] ** 2 / (S ** 2).sum()


def m_pca(X, g, w=None):
    return pc1(X, g, w)[0]


# ---------- Pareto ----------

def _dominance(Y):
    ge = (Y[:, None, :] >= Y[None, :, :]).all(2)
    gt = (Y[:, None, :] > Y[None, :, :]).any(2)
    return ge & gt


def pareto_layers(Y):
    D = _dominance(Y)
    n = len(Y)
    layer = np.zeros(n, dtype=int)
    alive = np.ones(n, dtype=bool)
    cur = 0
    while alive.any():
        dominated = D[alive][:, alive].any(0)
        idx = np.where(alive)[0]
        front = idx[~dominated]
        layer[front] = cur
        alive[front] = False
        cur += 1
    return layer, D.sum(1)


def _pareto_score(Y):
    layer, domcount = pareto_layers(Y)
    return -layer + domcount / (len(Y) + 1)


def m_pareto_feat(X, g, w=None):
    return _pareto_score(X)


def m_pareto_group(X, g, w=None):
    S, _ = group_scores(pct(X), g, None)
    return _pareto_score(S)


# ---------- DEA benefit of the doubt ----------

def _frontier(Y):
    D = _dominance(Y)
    return np.where(~D.any(0))[0]


def bod(X, g, bounds=None, return_weights=False):
    """Benefit of the doubt on percentile inputs.

    max sum(w_k y_ik) s.t. sum(w_k y_jk) <= 1 for all j, w >= 0, and optional per-group virtual-share bounds.
    Only frontier countries can bind the sum <= 1 constraints (w >= 0), so only they are included: same optimum, much smaller LP.
    """
    Y = pct(X)
    n, m = Y.shape
    g = np.asarray(g)
    labels = np.unique(g)
    front = _frontier(Y)
    A0 = Y[front]
    out = np.zeros(n)
    W = np.zeros((n, m))
    for i in range(n):
        y = Y[i]
        A, b = [A0], [np.ones(len(front))]
        if bounds is not None:
            lo, hi = bounds
            for t, l in enumerate(labels):
                mask = (g == l).astype(float)
                gy = mask * y
                A.append((lo[t] * y - gy)[None, :])
                b.append(np.zeros(1))
                A.append((gy - hi[t] * y)[None, :])
                b.append(np.zeros(1))
        r = linprog(-y, A_ub=np.vstack(A), b_ub=np.concatenate(b), bounds=(0, None), method="highs")
        if r.status == 0:
            out[i] = -r.fun
            W[i] = r.x
    return (out, W) if return_weights else out


def _bod_tiebreak(Y, W):
    """Cross-efficiency: the average score a country gets under every other country's chosen weights."""
    V = W @ Y.T
    mx = np.maximum(V.max(1, keepdims=True), 1e-12)
    return (V / mx).mean(0)


def m_bod_pure(X, g, w=None):
    e, W = bod(X, g, None, True)
    return e + 1e-3 * _bod_tiebreak(pct(X), W)


def default_ar(g):
    G = len(np.unique(g))
    return np.full(G, 0.5 / G), np.full(G, 2.0 / G)


def m_bod_ar(X, g, w=None):
    if w is None:
        bounds = default_ar(g)
    else:
        labels = np.unique(g)
        W = np.array([w[np.asarray(g) == l].sum() for l in labels])
        bounds = (W / 2, np.minimum(1.0, 2 * W))
    e, Wt = bod(X, g, bounds, True)
    return e + 1e-3 * _bod_tiebreak(pct(X), Wt)


# ---------- SMAA ----------

def smaa(S, draws=5000, seed=42, alpha=None, k=10):
    """S = n x G group scores. Returns mean rank and P(top k) over Dirichlet weights."""
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(S.shape[1]) if alpha is None else alpha, size=draws)
    sc = S @ W.T
    ord_ = np.argsort(-sc, axis=0, kind="stable")
    rk = np.empty_like(ord_)
    np.put_along_axis(rk, ord_, np.arange(1, S.shape[0] + 1)[:, None].repeat(draws, 1), axis=0)
    return rk.mean(1), (rk <= k).mean(1), rk


def _smaa_alpha(g, w):
    """Weight analog for SMAA: tilt the Dirichlet prior so its mean is the draw, keeping the total concentration of Dirichlet(1,...,1)."""
    if w is None:
        return None
    g = np.asarray(g)
    labels = np.unique(g)
    W = np.array([w[g == l].sum() for l in labels])
    return np.maximum(W * len(labels), 1e-3)


def m_smaa_meanrank(X, g, w=None, draws=1000):
    S, _ = group_scores(pct(X), g, None)
    return -smaa(S, draws, alpha=_smaa_alpha(g, w))[0]


def m_smaa_top10(X, g, w=None, draws=1000):
    S, _ = group_scores(pct(X), g, None)
    mr, p, _ = smaa(S, draws, alpha=_smaa_alpha(g, w))
    return p - 1e-6 * mr


# ---------- Rank aggregation across methods ----------

VOTERS = ["percentile", "topsis_minmax", "promethee_v", "vikor_minmax", "pca", "critic", "bod_ar"]


def pair_matrix(rank_mat):
    """rank_mat: n x v, lower = better. P[i,j] = number of voters with i ahead of j (ties count half)."""
    n, v = rank_mat.shape
    P = np.zeros((n, n))
    for c in range(v):
        r = rank_mat[:, c]
        P += (r[:, None] < r[None, :]).astype(float) + 0.5 * (r[:, None] == r[None, :]).astype(float)
    np.fill_diagonal(P, 0)
    return P


def kemeny_local(rank_mat, restarts=0, seed=0):
    """Kemeny-Young by insertion local search from the Borda order. A local optimum, not a proof of the exact optimum."""
    P = pair_matrix(rank_mat)
    M = P - P.T
    n = len(P)

    def run(start):
        o = list(start)
        improved = True
        while improved:
            improved = False
            for x in list(o):
                a = o.index(x)
                arr = np.array(o)
                mv = M[x, arr]
                # moving later: lose M[x,y] for each y passed; moving earlier: gain M[x,y]
                later = -np.cumsum(mv[a + 1:])
                earlier = np.cumsum(mv[:a][::-1])
                best_gain, target = 0.0, a
                if later.size and later.max() > best_gain + 1e-12:
                    best_gain, target = later.max(), a + 1 + int(later.argmax())
                if earlier.size and earlier.max() > best_gain + 1e-12:
                    best_gain, target = earlier.max(), a - 1 - int(earlier.argmax())
                if target != a:
                    o.pop(a)
                    o.insert(target, x)
                    improved = True
        score = sum(P[o[i], o[j]] for i in range(n) for j in range(i + 1, n)) if restarts else 0.0
        return o, score

    borda = np.argsort(rank_mat.mean(1), kind="stable")
    best, best_s = run(borda)
    if restarts:
        rng = np.random.default_rng(seed)
        for _ in range(restarts):
            o, s = run(rng.permutation(n))
            if s > best_s + 1e-9:
                best, best_s = o, s
    out = np.empty(n)
    out[best] = -np.arange(n, dtype=float)
    return out


def schulze(rank_mat):
    P = pair_matrix(rank_mat)
    n = len(P)
    p = np.where(P > P.T, P, 0.0)
    for k in range(n):
        p = np.maximum(p, np.minimum(p[:, k, None], p[k, None, :]))
    np.fill_diagonal(p, 0)
    wins = (p > p.T).sum(1)
    borda = rank_mat.mean(1)
    return wins - borda / (10 * n)


METHODS = {
    "percentile": m_percentile,
    "percentile_equal_feature": m_percentile_featw,
    "zscore_mean": m_zmean,
    "minmax_mean": m_minmaxmean,
    "geomean_groups": m_geomean,
    "weakest_group": m_weakest,
    "topsis_minmax": m_topsis_minmax,
    "topsis_pct": m_topsis_pct,
    "promethee_usual": m_promethee_usual,
    "promethee_v": m_promethee_v,
    "vikor_minmax": m_vikor_minmax,
    "vikor_pct": m_vikor_pct,
    "copeland": m_copeland,
    "entropy": m_entropy,
    "critic": m_critic,
    "pca": m_pca,
    "pareto_features": m_pareto_feat,
    "pareto_groups": m_pareto_group,
    "bod_pure": m_bod_pure,
    "bod_ar": m_bod_ar,
    "smaa_meanrank": m_smaa_meanrank,
    "smaa_top10": m_smaa_top10,
}


def all_scores(X, g, w=None, names=None):
    """All methods plus the aggregators. Voter results are shared, so LPs run only once."""
    names = list(METHODS) if names is None else names
    out = {}
    for nm in names:
        out[nm] = METHODS[nm](X, g, w)
    need = {v for v in VOTERS if v not in out}
    voters = {**out, **{v: METHODS[v](X, g, w) for v in need}}
    rm = np.column_stack([ranks(voters[v]) for v in VOTERS])
    out["kemeny"] = kemeny_local(rm)
    out["schulze"] = schulze(rm)
    return out
