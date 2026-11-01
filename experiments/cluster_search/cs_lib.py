"""Shared library for the clustering search: algorithms, settings grid and the checks.

A setting is a plain dict: {"fam": family, "ver": "raw" | 3 | 5 | 8, ...params}.
A procedure is a function of (data, setting, seed) only. Any data-dependent scale
(DBSCAN eps, RBF gamma, mean-shift bandwidth, AP preference) is recomputed from the
data being fitted, so a refit on a sample or on fake data applies the same rule.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings

warnings.filterwarnings("ignore")

import json
import pickle

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage as sp_linkage
from scipy.linalg import solve_triangular
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gammaln, logsumexp
from sklearn.cluster import (
    DBSCAN,
    HDBSCAN,
    OPTICS,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MeanShift,
    SpectralClustering,
    estimate_bandwidth,
)
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

BUNDLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run3", "bundle_2030.pkl")
OUT = os.path.dirname(os.path.abspath(__file__))
_B = None


def B():
    global _B
    if _B is None:
        with open(BUNDLE, "rb") as f:
            _B = pickle.load(f)
    return _B


# ---------------------------------------------------------------- hand-written algorithms
def _cd(metric):
    return {"manhattan": "cityblock"}.get(metric, metric)


def _nearest(Z, C, metric="euclidean"):
    return cdist(Z, C, _cd(metric)).argmin(1)


def centroids(Z, lab):
    ids = np.unique(lab[lab >= 0])
    return ids, np.vstack([Z[lab == i].mean(0) for i in ids])


def fuzzy_cmeans(Z, k, m, rng, n_init=5, max_iter=300, tol=1e-6):
    n = len(Z)
    best = None
    for _ in range(n_init):
        U = rng.dirichlet(np.ones(k), size=n)
        for _it in range(max_iter):
            Um = U**m
            C = (Um.T @ Z) / Um.sum(0)[:, None]
            d2 = cdist(Z, C, "sqeuclidean") + 1e-12
            ratio = d2 / d2.min(1, keepdims=True)
            inv = ratio ** (-1.0 / (m - 1.0))
            new = inv / inv.sum(1, keepdims=True)
            done = np.abs(new - U).max() < tol
            U = new
            if done:
                break
        Um = U**m
        obj = (Um * cdist(Z, C, "sqeuclidean")).sum()
        if best is None or obj < best[0]:
            best = (obj, U.copy(), C.copy())
    return best[1].argmax(1), best[2]


def pam(D, k, rng=None, init="build", max_iter=200):
    n = len(D)
    if init == "build":
        med = [int(D.sum(1).argmin())]
        while len(med) < k:
            d1 = D[:, med].min(1)
            gain = np.maximum(d1[:, None] - D, 0).sum(0)
            gain[med] = -1
            med.append(int(gain.argmax()))
    else:
        med = list(rng.choice(n, k, replace=False))
    med = np.array(med)
    for _ in range(max_iter):
        dm = D[:, med]
        order = np.argsort(dm, axis=1)
        near = order[:, 0]
        d1 = dm[np.arange(n), near]
        d2 = dm[np.arange(n), order[:, 1]] if k > 1 else np.full(n, np.inf)
        cur = d1.sum()
        best = (cur - 1e-10, None, None)
        for i in range(k):
            base = np.where(near == i, d2, d1)
            cost = np.minimum(D, base[:, None]).sum(0)
            cost[med] = np.inf
            h = int(cost.argmin())
            if cost[h] < best[0]:
                best = (cost[h], i, h)
        if best[1] is None:
            break
        med[best[1]] = best[2]
    return D[:, med].argmin(1), med


def som_train(Z, rows, cols, rng, init="random", epochs=50, sigma_end=0.5):
    grid = np.array([[r, c] for r in range(rows) for c in range(cols)], float)
    g2 = cdist(grid, grid, "sqeuclidean")
    m = rows * cols
    if init == "pca":
        pca = PCA(2).fit(Z)
        sc = pca.transform(Z)
        span = [np.linspace(-1, 1, max(cols, rows)) * 2 * sc[:, 0].std(), np.linspace(-1, 1, min(cols, rows)) * 2 * sc[:, 1].std()]
        W = []
        for r in range(rows):
            for c in range(cols):
                a = span[0][c if cols >= rows else r]
                b = span[1][r if cols >= rows else c]
                W.append(Z.mean(0) + a * pca.components_[0] + b * pca.components_[1])
        W = np.array(W)
    else:
        W = Z[rng.choice(len(Z), m, replace=False)].astype(float).copy()
    sigma0 = max(max(rows, cols) / 2.0, sigma_end)
    for e in range(epochs):
        sig = sigma0 * (sigma_end / sigma0) ** (e / max(epochs - 1, 1))
        bmu = cdist(Z, W, "sqeuclidean").argmin(1)
        h = np.exp(-g2 / (2 * sig * sig))
        H = h[bmu]  # n x m
        den = H.sum(0)
        keep = den > 1e-12
        W[keep] = (H.T @ Z)[keep] / den[keep, None]
    return W


def som_labels(Z, rows, cols, rng, init, sigma_end, merge_k=None):
    W = som_train(Z, rows, cols, rng, init=init, sigma_end=sigma_end)
    bmu = cdist(Z, W, "sqeuclidean").argmin(1)
    hit = np.unique(bmu)
    if merge_k is None:
        return bmu, W, np.arange(len(W))
    if len(hit) < merge_k:
        raise ValueError("too few nodes hit")
    node_cl = np.full(len(W), -1)
    node_cl[hit] = AgglomerativeClustering(merge_k, linkage="ward").fit_predict(W[hit])
    return node_cl[bmu], W[hit], node_cl[hit]


def _kpp_means(Z, k, rng):
    idx = [int(rng.integers(len(Z)))]
    d2 = ((Z - Z[idx[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        tot = d2.sum()
        j = int(rng.choice(len(Z), p=d2 / tot)) if tot > 0 else int(rng.integers(len(Z)))
        idx.append(j)
        d2 = np.minimum(d2, ((Z - Z[j]) ** 2).sum(1))
    return Z[idx].astype(float).copy()


def _constrain_cov(S, cov, reg, d):
    if cov == "diag":
        return np.diag(np.diag(S) + reg)
    if cov == "spherical":
        return np.eye(d) * (np.trace(S) / d + reg)
    return S + reg * np.eye(d)


class MixModel:
    def __init__(self, w, mu, Sig, kind, df=None):
        self.w, self.mu, self.Sig, self.kind, self.df = w, mu, Sig, kind, df

    def logjoint(self, Z):
        n, d = Z.shape
        k = len(self.w)
        lp = np.zeros((n, k))
        ma = np.zeros((n, k))
        for j in range(k):
            L = np.linalg.cholesky(self.Sig[j])
            y = solve_triangular(L, (Z - self.mu[j]).T, lower=True)
            maha = (y**2).sum(0)
            ma[:, j] = maha
            ld = 2 * np.log(np.diag(L)).sum()
            if self.kind == "t":
                df = self.df
                lp[:, j] = gammaln((df + d) / 2) - gammaln(df / 2) - 0.5 * d * np.log(df * np.pi) - 0.5 * ld - 0.5 * (df + d) * np.log1p(maha / df)
            else:
                lp[:, j] = -0.5 * d * np.log(2 * np.pi) - 0.5 * ld - 0.5 * maha
        return lp + np.log(self.w + 1e-300), ma

    def predict(self, Z):
        return self.logjoint(Z)[0].argmax(1)


def t_mixture(Z, k, df, cov, reg, rng, n_init=3, max_iter=150, tol=1e-5):
    n, d = Z.shape
    best = None
    for _ in range(n_init):
        try:
            mu = _kpp_means(Z, k, rng)
            lab = _nearest(Z, mu)
            w = np.array([(lab == j).mean() for j in range(k)]) + 1e-6
            w /= w.sum()
            G = np.cov(Z.T).reshape(d, d)
            Sig = [_constrain_cov(G, cov, reg, d) for _ in range(k)]
            model = MixModel(w, mu, Sig, "t", df)
            prev = -np.inf
            for _it in range(max_iter):
                lj, ma = model.logjoint(Z)
                ll = logsumexp(lj, axis=1)
                r = np.exp(lj - ll[:, None])
                u = (df + d) / (df + ma)
                tot = ll.sum()
                if abs(tot - prev) < tol * (1 + abs(tot)):
                    break
                prev = tot
                Nk = r.sum(0) + 1e-10
                model.w = Nk / n
                pooled = np.zeros((d, d))
                new = []
                for j in range(k):
                    ru = r[:, j] * u[:, j]
                    model.mu[j] = (ru[:, None] * Z).sum(0) / (ru.sum() + 1e-10)
                    diff = Z - model.mu[j]
                    S = (ru[:, None] * diff).T @ diff
                    pooled += S
                    new.append(_constrain_cov(S / Nk[j], cov, reg, d))
                if cov == "tied":
                    t = pooled / n + reg * np.eye(d)
                    new = [t for _ in range(k)]
                model.Sig = new
            if best is None or tot > best[0]:
                best = (tot, model)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
    if best is None:
        raise ValueError("t mixture failed")
    return best[1]


def mppca(Z, k, q, floor, rng, n_init=3, max_iter=200, tol=1e-5):
    n, d = Z.shape
    best = None
    for _ in range(n_init):
        try:
            mu = _kpp_means(Z, k, rng)
            lab = _nearest(Z, mu)
            r = np.zeros((n, k))
            r[np.arange(n), lab] = 1.0
            r = 0.9 * r + 0.1 / k
            prev = -np.inf
            model = None
            for _it in range(max_iter):
                Nk = r.sum(0) + 1e-10
                w = Nk / n
                mus = (r.T @ Z) / Nk[:, None]
                Sig = []
                for j in range(k):
                    diff = Z - mus[j]
                    S = (r[:, j, None] * diff).T @ diff / Nk[j]
                    val, vec = np.linalg.eigh(S)
                    val, vec = val[::-1], vec[:, ::-1]
                    s2 = max(val[q:].mean(), floor)
                    lam = np.maximum(val[:q] - s2, 0)
                    W = vec[:, :q] * np.sqrt(lam)
                    Sig.append(W @ W.T + s2 * np.eye(d))
                model = MixModel(w, mus, Sig, "g")
                lj, _ma = model.logjoint(Z)
                ll = logsumexp(lj, axis=1)
                r = np.exp(lj - ll[:, None])
                tot = ll.sum()
                if abs(tot - prev) < tol * (1 + abs(tot)):
                    break
                prev = tot
            if best is None or tot > best[0]:
                best = (tot, model)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
    if best is None:
        raise ValueError("mppca failed")
    return best[1]


def trimmed_gmm(Z, k, cov, reg, alpha, seed, n_init=5, n_conc=10):
    n = len(Z)
    keep = n - int(alpha * n)
    best = None
    for r in range(n_init):
        try:
            gm = GaussianMixture(k, covariance_type=cov, reg_covar=reg, init_params="k-means++", n_init=1, max_iter=200, warm_start=True, random_state=seed * 1000 + r)
            idx = np.arange(n)
            for _ in range(n_conc):
                gm.fit(Z[idx])
                ll = gm.score_samples(Z)
                new = np.sort(np.argsort(-ll)[:keep])
                same = len(new) == len(idx) and (new == idx).all()
                idx = new
                if same:
                    break
            gm.fit(Z[idx])
            ll = gm.score_samples(Z)
            val = ll[idx].sum()
            if best is None or val > best[0]:
                import copy

                best = (val, copy.deepcopy(gm), idx.copy())
        except (ValueError, np.linalg.LinAlgError):
            continue
    if best is None:
        raise ValueError("trimmed gmm failed")
    lab = best[1].predict(Z)
    out = np.full(n, -1)
    out[best[2]] = lab[best[2]]
    return out, best[1]


def ward_init_gmm(Z, k, cov, reg):
    n, d = Z.shape
    lab = AgglomerativeClustering(k, linkage="ward").fit_predict(Z)
    wts = np.bincount(lab, minlength=k) / n
    means = np.vstack([Z[lab == j].mean(0) for j in range(k)])
    if cov == "full":
        prec = np.stack([np.linalg.inv(np.cov(Z[lab == j].T, bias=True).reshape(d, d) + reg * np.eye(d)) for j in range(k)])
    elif cov == "tied":
        S = sum(((Z[lab == j] - means[j]).T @ (Z[lab == j] - means[j])) for j in range(k)) / n
        prec = np.linalg.inv(S + reg * np.eye(d))
    elif cov == "diag":
        prec = np.vstack([1.0 / (Z[lab == j].var(0) + reg) for j in range(k)])
    else:
        prec = np.array([1.0 / (Z[lab == j].var(0).mean() + reg) for j in range(k)])
    gm = GaussianMixture(k, covariance_type=cov, reg_covar=reg, weights_init=wts, means_init=means, precisions_init=prec, n_init=1, max_iter=500, random_state=0).fit(Z)
    return gm


def _knn_graph(Z, nn, weight):
    n = len(Z)
    nn = min(nn, n - 1)
    idx = NearestNeighbors(n_neighbors=nn + 1).fit(Z).kneighbors(Z)[1][:, 1:]
    A = np.zeros((n, n))
    A[np.repeat(np.arange(n), nn), idx.ravel()] = 1
    if weight == "snn":
        M = A.copy()
        M[np.arange(n), np.arange(n)] = 1
        inter = M @ M.T
        size = M.sum(1)
        union = size[:, None] + size[None, :] - inter
        J = inter / np.maximum(union, 1)
        A = np.maximum(A, A.T) * J
    else:
        A = np.maximum(A, A.T)
    A[np.arange(n), np.arange(n)] = 0
    return A


def louvain(A, gamma, rng, max_levels=10):
    n0 = len(A)
    node2c = np.arange(n0)
    cur = A.copy()
    for _lvl in range(max_levels):
        N = len(cur)
        k = cur.sum(1)
        m2 = cur.sum()
        if m2 <= 0:
            break
        comm = np.arange(N)
        sigma = k.copy()
        any_move = False
        for _sweep in range(50):
            moved = False
            for i in rng.permutation(N):
                ci = comm[i]
                ki = k[i]
                w = np.bincount(comm, weights=cur[i], minlength=N)
                w[ci] -= cur[i, i]
                sigma[ci] -= ki
                gain = w - gamma * sigma * ki / m2
                b = int(gain.argmax())
                if gain[b] <= gain[ci] + 1e-12:
                    b = ci
                comm[i] = b
                sigma[b] += ki
                if b != ci:
                    moved = True
                    any_move = True
            if not moved:
                break
        if not any_move:
            break
        ids, comm = np.unique(comm, return_inverse=True)
        node2c = comm[node2c]
        M = len(ids)
        P = np.zeros((N, M))
        P[np.arange(N), comm] = 1
        cur = P.T @ cur @ P
        if M == N:
            break
    return np.unique(node2c, return_inverse=True)[1]


def mcl(A, inflation, max_iter=100):
    n = len(A)
    M = A + np.eye(n)
    M = M / M.sum(0, keepdims=True)
    for _ in range(max_iter):
        prev = M
        M = M @ M
        M = M**inflation
        M[M < 1e-6] = 0
        M = M / np.maximum(M.sum(0, keepdims=True), 1e-300)
        if np.abs(M - prev).max() < 1e-6:
            break
    return connected_components(M > 1e-6, directed=False)[1]


def pddp(Z, k, crit):
    groups = [np.arange(len(Z))]
    while len(groups) < k:
        if crit == "sse":
            sc = [((Z[g] - Z[g].mean(0)) ** 2).sum() for g in groups]
        else:
            sc = [len(g) for g in groups]
        i = int(np.argmax(sc))
        g = groups.pop(i)
        X = Z[g] - Z[g].mean(0)
        u = np.linalg.svd(X, full_matrices=False)[2][0]
        u = u * np.sign(u[np.abs(u).argmax()])
        p = X @ u
        a, b = g[p <= 0], g[p > 0]
        if len(a) == 0 or len(b) == 0:
            raise ValueError("pddp degenerate split")
        groups += [a, b]
    lab = np.zeros(len(Z), int)
    for i, g in enumerate(groups):
        lab[g] = i
    return lab


def diana(D, k):
    groups = [np.arange(len(D))]
    while len(groups) < k:
        diam = [D[np.ix_(g, g)].max() if len(g) > 1 else -1 for g in groups]
        i = int(np.argmax(diam))
        g = groups.pop(i)
        sub = D[np.ix_(g, g)]
        n = len(g)
        in_spl = np.zeros(n, bool)
        in_spl[(sub.sum(1) / max(n - 1, 1)).argmax()] = True
        while (~in_spl).sum() > 1:
            rest = np.where(~in_spl)[0]
            d_old = (sub[np.ix_(rest, rest)].sum(1)) / (len(rest) - 1)
            d_spl = sub[np.ix_(rest, np.where(in_spl)[0])].mean(1)
            gain = d_old - d_spl
            j = int(gain.argmax())
            if gain[j] <= 0:
                break
            in_spl[rest[j]] = True
        groups += [g[in_spl], g[~in_spl]]
    lab = np.zeros(len(D), int)
    for i, g in enumerate(groups):
        lab[g] = i
    return lab


def consensus(Z, k, base, final, seed, H=50):
    rng = np.random.default_rng(seed + 7919)
    n = len(Z)
    Cc = np.zeros((n, n))
    Nn = np.zeros((n, n))
    for h in range(H):
        idx = rng.choice(n, int(0.8 * n), replace=False)
        S = Z[idx]
        if base == "ward":
            lab = AgglomerativeClustering(k, linkage="ward").fit_predict(S)
        elif base == "average_manhattan":
            lab = AgglomerativeClustering(k, linkage="average", metric="manhattan").fit_predict(S)
        else:
            cov = base.split("_")[1]
            reg = 0.1 if cov == "full" else 1e-3
            lab = GaussianMixture(k, covariance_type=cov, reg_covar=reg, init_params="k-means++", n_init=1, max_iter=200, random_state=seed * 1000 + h).fit(S).predict(S)
        same = (lab[:, None] == lab[None, :]).astype(float)
        ii = np.ix_(idx, idx)
        Cc[ii] += same
        Nn[ii] += 1
    Sim = Cc / np.maximum(Nn, 1)
    np.fill_diagonal(Sim, 1)
    if final == "spectral":
        return SpectralClustering(k, affinity="precomputed", assign_labels="cluster_qr", random_state=seed).fit_predict(Sim)
    return AgglomerativeClustering(k, metric="precomputed", linkage=final).fit_predict(1 - Sim)


# ---------------------------------------------------------------- dispatcher
DET_FAMS = {"agglo", "agglo_conn", "dbscan", "hdbscan", "optics", "meanshift", "pddp", "diana", "mcl", "gmm_ward"}


def is_det(s):
    if s["fam"] in DET_FAMS:
        return True
    if s["fam"] == "pam" and s["init"] == "build":
        return True
    if s["fam"] == "som" and s["init"] == "pca":
        return True
    return False


def embed(X, ver):
    if ver == "raw":
        return X, None
    n = min(ver, X.shape[1])
    pca = PCA(n, random_state=0).fit(X)
    return pca.transform(X), pca


def _kdist(Z, ms, metric):
    return NearestNeighbors(n_neighbors=ms, metric=metric).fit(Z).kneighbors(Z)[0][:, -1]


def _assign_noise(Z, lab):
    if (lab >= 0).sum() == 0 or (lab < 0).sum() == 0:
        return lab
    ids, C = centroids(Z, lab)
    out = lab.copy()
    miss = lab < 0
    out[miss] = ids[_nearest(Z[miss], C)]
    return out


def fit_method(s, Z, seed):
    """Return (labels, assign_fn or None). assign_fn None means no native predict."""
    f = s["fam"]
    rng = np.random.default_rng(seed + 12345)
    k = s.get("k")
    if f == "kmeans_ref":
        m = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
        return m.labels_, m.predict
    if f == "agglo":
        return AgglomerativeClustering(k, linkage=s["linkage"], metric=s["metric"]).fit_predict(Z), None
    if f == "agglo_conn":
        g = kneighbors_graph(Z, s["nn"], include_self=False)
        return AgglomerativeClustering(k, linkage="ward", connectivity=g).fit_predict(Z), None
    if f == "spectral_nn":
        return SpectralClustering(k, affinity="nearest_neighbors", n_neighbors=s["nn"], assign_labels=s["assign"], random_state=seed).fit_predict(Z), None
    if f == "spectral_rbf":
        med = np.median(pdist(Z, "sqeuclidean"))
        return SpectralClustering(k, affinity="rbf", gamma=s["mult"] / med, assign_labels=s["assign"], random_state=seed).fit_predict(Z), None
    if f == "spectral_st":
        D = squareform(pdist(Z))
        sig = np.sort(D, axis=1)[:, min(s["nn"], len(Z) - 1)]
        Aff = np.exp(-(D**2) / (sig[:, None] * sig[None, :] + 1e-12))
        return SpectralClustering(k, affinity="precomputed", assign_labels=s["assign"], random_state=seed).fit_predict(Aff), None
    if f == "gmm":
        m = GaussianMixture(k, covariance_type=s["cov"], reg_covar=s["reg"], init_params=s["init"], n_init=s.get("n_init", 5), max_iter=500, random_state=seed).fit(Z)
        return m.predict(Z), m.predict
    if f == "gmm_ward":
        m = ward_init_gmm(Z, k, s["cov"], s["reg"])
        return m.predict(Z), m.predict
    if f == "bgmm":
        m = BayesianGaussianMixture(
            n_components=k,
            covariance_type=s["cov"],
            weight_concentration_prior_type=s["wtype"],
            weight_concentration_prior=s["wcp"],
            reg_covar=s["reg"],
            init_params="k-means++",
            n_init=3,
            max_iter=500,
            random_state=seed,
        ).fit(Z)
        return m.predict(Z), m.predict
    if f == "fcm":
        lab, C = fuzzy_cmeans(Z, k, s["m"], rng)
        return lab, lambda Y: _nearest(Y, C)
    if f == "pam":
        D = cdist(Z, Z, _cd(s["metric"]))
        lab, med = pam(D, k, rng, s["init"])
        M = Z[med]
        return lab, lambda Y: _nearest(Y, M, s["metric"])
    if f == "som":
        lab, W, ncl = som_labels(Z, s["rows"], s["cols"], rng, s["init"], s["sigma_end"], s.get("merge_k"))
        return lab, lambda Y: ncl[cdist(Y, W, "sqeuclidean").argmin(1)]
    if f == "birch":
        perm = rng.permutation(len(Z))
        m = Birch(threshold=s["thr"], branching_factor=s["bf"], n_clusters=k).fit(Z[perm])
        lab = np.zeros(len(Z), int)
        lab[perm] = m.labels_
        return lab, m.predict
    if f == "ap":
        Sim = -cdist(Z, Z, "sqeuclidean")
        off = Sim[np.triu_indices(len(Z), 1)]
        if s["pmode"] == "quantile":
            pref = np.quantile(off, s["pval"])
        else:
            pref = s["pval"] * off.min()
        m = AffinityPropagation(damping=s["damp"], preference=pref, max_iter=1000, convergence_iter=30, random_state=seed).fit(Z)
        if len(m.cluster_centers_) == 0:
            raise ValueError("AP did not converge")
        return m.labels_, m.predict
    if f == "ap_k":
        Sim = -cdist(Z, Z, "sqeuclidean")
        off = Sim[np.triu_indices(len(Z), 1)]
        lo, hi = 100 * off.min(), np.median(off)
        chosen = None
        for _ in range(25):
            mid = (lo + hi) / 2
            m = AffinityPropagation(damping=s["damp"], preference=mid, max_iter=1000, convergence_iter=30, random_state=seed).fit(Z)
            c = len(m.cluster_centers_)
            if c == k:
                chosen = m
                break
            if c == 0 or c < k:
                lo = mid
            else:
                hi = mid
        if chosen is None:
            raise ValueError("AP target k not found")
        return chosen.labels_, chosen.predict
    if f == "meanshift":
        bw = estimate_bandwidth(Z, quantile=s["q"])
        m = MeanShift(bandwidth=bw, bin_seeding=s["bin"], cluster_all=s["all"], n_jobs=1).fit(Z)
        return m.labels_, m.predict
    if f == "dbscan":
        eps = np.quantile(_kdist(Z, s["ms"], s["metric"]), s["q"])
        return DBSCAN(eps=eps, min_samples=s["ms"], metric=s["metric"]).fit_predict(Z), None
    if f == "optics":
        return OPTICS(min_samples=s["ms"], xi=s["xi"], min_cluster_size=s["mcs"], metric=s["metric"], n_jobs=1).fit_predict(Z), None
    if f == "hdbscan":
        lab = HDBSCAN(min_cluster_size=s["mcs"], min_samples=s["ms"], cluster_selection_method=s["method"], cluster_selection_epsilon=s["cse"], metric=s["metric"], copy=True).fit_predict(Z.copy())
        if s.get("assign_noise"):
            lab = _assign_noise(Z, lab)
        return lab, None
    if f == "louvain":
        return louvain(_knn_graph(Z, s["nn"], s["weight"]), s["gamma"], rng), None
    if f == "mcl":
        return mcl(_knn_graph(Z, s["nn"], "binary"), s["infl"]), None
    if f == "tmix":
        m = t_mixture(Z, k, s["df"], s["cov"], s["reg"], rng)
        return m.predict(Z), m.predict
    if f == "mppca":
        m = mppca(Z, k, s["q"], s["floor"], rng)
        return m.predict(Z), m.predict
    if f == "trim_gmm":
        lab, gm = trimmed_gmm(Z, k, s["cov"], s["reg"], s["alpha"], seed)
        return lab, gm.predict
    if f == "consensus":
        return consensus(Z, k, s["base"], s["final"], seed), None
    if f == "pddp":
        return pddp(Z, k, s["crit"]), None
    if f == "diana":
        return diana(cdist(Z, Z, _cd(s["metric"])), k), None
    raise KeyError(f)


def run(s, X, seed):
    """Labels for procedure s on matrix X, or None if the fit failed."""
    try:
        Z, _ = embed(X, s["ver"])
        return np.asarray(fit_method(s, Z, seed)[0])
    except Exception:
        return None


# ---------------------------------------------------------------- checks
def n_groups(lab):
    return len(np.unique(lab[lab >= 0]))


def sizes(lab):
    c = np.bincount(lab[lab >= 0])
    return sorted(c[c > 0].tolist(), reverse=True)


def coverage_ok(lab):
    return (lab < 0).mean() <= 0.10


def sizes_ok(lab, s):
    """Check 2. Fixed-k methods must also realise k groups. Every method needs at least 3 groups."""
    c = sizes(lab)
    if len(c) < 3:
        return False
    if "k" in s and s["fam"] != "bgmm" and len(c) != s["k"]:
        return False
    return c[-1] >= 8 and c[0] <= 0.5 * len(lab)


def sil(X, lab):
    if lab is None:
        return np.nan
    mask = lab >= 0
    if len(np.unique(lab[mask])) < 2 or mask.sum() < 3 or len(np.unique(lab[mask])) >= mask.sum():
        return np.nan
    return float(silhouette_score(X[mask], lab[mask]))


def stability_ref(s, X, base, runs=25, frac=0.8):
    """Project protocol: reference on all countries, refit on a random 80%, ARI on shared countries.
    Noise labels (-1) count as one extra label. A failed refit scores 0."""
    ref = run(s, X, base)
    if ref is None:
        return 0.0
    rng = np.random.default_rng(base)
    out = []
    for r in range(runs):
        idx = rng.choice(len(X), int(frac * len(X)), replace=False)
        lab = run(s, X[idx], base * 100 + r)
        out.append(0.0 if lab is None else adjusted_rand_score(ref[idx], lab))
    return float(np.mean(out))


_FAKE = {}


def fake_data(seed, C=None):
    """Each principal component of C_2030 shuffled separately, rotated back to feature space."""
    C = B()["C_2030"] if C is None else C
    pca = PCA().fit(C)
    P = pca.transform(C)
    rng = np.random.default_rng(seed)
    Y = np.column_stack([rng.permutation(P[:, j]) for j in range(P.shape[1])])
    return pca.inverse_transform(Y)


def fake_sil(s, seed):
    Y = fake_data(seed)
    return sil(Y, run(s, Y, seed))


def backtest_one(s, seed):
    b = B()
    a10 = b["backtest_actual_2010"].values
    f10 = b["backtest_svr_forecast_2010"].values
    a00 = b["backtest_actual_2000"].values
    mu, sd = a10.mean(0), a10.std(0)
    za, zf, z0 = (a10 - mu) / sd, (f10 - mu) / sd, (a00 - mu) / sd
    try:
        if s["ver"] == "raw":
            Za, Zf, Z0 = za, zf, z0
        else:
            pca = PCA(min(s["ver"], 4), random_state=0).fit(za)
            Za, Zf, Z0 = pca.transform(za), pca.transform(zf), pca.transform(z0)
        lab, assign = fit_method(s, Za, seed)
        lab = np.asarray(lab)
        native = assign is not None
        if assign is None:
            ids, C = centroids(Za, lab)
            assign = lambda Y: ids[_nearest(Y, C)]
        m = lab >= 0
        return adjusted_rand_score(lab[m], assign(Zf)[m]), adjusted_rand_score(lab[m], assign(Z0)[m]), n_groups(lab), native, float((~m).mean())
    except Exception:
        return np.nan, np.nan, 0, False, np.nan


def key(s):
    return json.dumps(s, sort_keys=True, default=str)


def screen_one(s, X=None, nstart=5, nstab=20):
    X = B()["C_2030"] if X is None else X
    seeds = [0] if is_det(s) else list(range(nstart))
    labs, ok_cov, ok_size, sils, grp = [], True, True, [], []
    for sd in seeds:
        lab = run(s, X, sd)
        if lab is None:
            ok_cov, ok_size = False, False
            break
        labs.append(lab)
        ok_cov &= coverage_ok(lab)
        ok_size &= sizes_ok(lab, s)
        sils.append(sil(X, lab))
        grp.append(n_groups(lab))
        if not (ok_cov and ok_size):
            break
    row = dict(key=key(s), fam=s["fam"], ver=str(s["ver"]), det=is_det(s), cov_ok=ok_cov, size_ok=ok_size, sil=float(np.nanmean(sils)) if len(sils) and np.isfinite(sils).any() else np.nan, groups=grp[0] if grp else 0, stab=np.nan)
    if labs:
        c = sizes(labs[-1])
        row.update(min_size=c[-1] if c else 0, max_share=(c[0] / len(X)) if c else 1.0, unassigned=float((labs[-1] < 0).mean()))
    else:
        row.update(min_size=0, max_share=1.0, unassigned=1.0)
    if ok_cov and ok_size:
        ref = labs[0]
        rng = np.random.default_rng(1)
        out = []
        for r in range(nstab):
            idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
            lab = run(s, X[idx], 100 + r)
            out.append(0.0 if lab is None else adjusted_rand_score(ref[idx], lab))
        row["stab"] = float(np.mean(out))
    return row


def full_check(s, nfake=100, early=True):
    """Checks 1-7 with the exact protocol. Sequential inside one worker.
    With early=True it stops at the first check that fails, in the order sizes, fake baseline, stability.
    Skipped values stay NaN. The stop rule cannot change a verdict: a setting only stops once it has failed.
    The one shortcut is that a real silhouette at or below the median of the first 25 fake datasets is
    recorded as a fake-baseline failure without the other 75."""
    b = B()
    C, Ck = b["C_2030"], b["C_2030_knn"]
    seeds = [0] if is_det(s) else list(range(30))
    labs = [run(s, C, sd) for sd in seeds]
    valid = [l for l in labs if l is not None]
    res = dict(key=key(s), fam=s["fam"], ver=str(s["ver"]), det=is_det(s), s=s, n_fail_fit=len(labs) - len(valid), early=early)
    res.update(cov=False, size=False, sil=np.nan, stab=np.nan, stab_by_base=None, fake_mean=np.nan, fake_p95=np.nan, fake_n=0, fake_pass=False, fake_degen=0, knn_ari=np.nan, bt_svr=np.nan, bt_nochange=np.nan, bt_groups=0, bt_native=False, bt_unassigned=np.nan, labels=None, sizes=[], unassigned=1.0, groups=0, groups_range=(0, 0), sizes_worst_min=0, sizes_worst_max=0, start_agreement=np.nan, all5=False)
    if not valid:
        return res
    res["cov"] = all(coverage_ok(l) for l in valid) and not res["n_fail_fit"]
    res["unassigned"] = float(np.mean([(l < 0).mean() for l in valid]))
    res["size"] = all(sizes_ok(l, s) for l in valid) and not res["n_fail_fit"]
    res["sizes_worst_min"] = min(sizes(l)[-1] if sizes(l) else 0 for l in valid)
    res["sizes_worst_max"] = max(sizes(l)[0] if sizes(l) else 0 for l in valid)
    res["groups"] = n_groups(valid[0])
    res["groups_range"] = (min(n_groups(l) for l in valid), max(n_groups(l) for l in valid))
    res["sizes"] = sizes(valid[0])
    res["labels"] = valid[0]
    sl = [sil(C, l) for l in valid]
    res["sil"] = float(np.nanmean(sl)) if np.isfinite(sl).any() else np.nan
    res["start_agreement"] = float(np.mean([adjusted_rand_score(valid[0], l) for l in valid[1:]])) if len(valid) > 1 else 1.0
    if early and not (res["cov"] and res["size"]):
        return res

    def summarise(fake):
        res["fake_n"] = len(fake)
        res["fake_degen"] = int(np.isnan(fake).sum())
        if np.isfinite(fake).any():
            res["fake_mean"] = float(np.nanmean(fake))
            res["fake_p95"] = float(np.nanquantile(fake, 0.95))
        res["fake_pass"] = bool(np.isfinite(res["sil"]) and np.isfinite(res["fake_p95"]) and res["sil"] > res["fake_p95"])

    quick = min(25, nfake)
    fake = np.array([fake_sil(s, fs) for fs in range(quick)], float)
    if early and np.isfinite(fake).any() and res["sil"] <= np.nanmedian(fake):
        summarise(fake)
        res["fake_pass"] = False
        return res
    fake = np.concatenate([fake, np.array([fake_sil(s, fs) for fs in range(quick, nfake)], float)])
    summarise(fake)
    if early and not res["fake_pass"]:
        return res
    res["stab_by_base"] = [stability_ref(s, C, base) for base in range(1, 6)]
    res["stab"] = float(np.mean(res["stab_by_base"]))
    res["all5"] = bool(res["cov"] and res["size"] and res["stab"] >= 0.6 and res["fake_pass"])
    if early and not res["all5"]:
        return res
    kn = []
    for sd in seeds:
        a, c = run(s, C, sd), run(s, Ck, sd)
        kn.append(0.0 if a is None or c is None else adjusted_rand_score(a, c))
    res["knn_ari"] = float(np.mean(kn))
    bt = [backtest_one(s, sd) for sd in seeds]
    res["bt_svr"] = float(np.nanmean([x[0] for x in bt])) if any(np.isfinite(x[0]) for x in bt) else np.nan
    res["bt_nochange"] = float(np.nanmean([x[1] for x in bt])) if any(np.isfinite(x[1]) for x in bt) else np.nan
    res["bt_groups"] = int(bt[0][2])
    res["bt_native"] = bool(bt[0][3])
    res["bt_unassigned"] = bt[0][4]
    return res


# ---------------------------------------------------------------- settings grid
VERS = ["raw", 3, 5, 8]
KS = [3, 4, 5, 6, 7, 8]


def grid():
    S = []

    def add(fam, **p):
        S.append(dict(fam=fam, **p))

    for v in VERS:
        for k in KS:
            add("agglo", ver=v, k=k, linkage="ward", metric="euclidean")
            for link in ("average", "complete", "single"):
                for met in ("euclidean", "manhattan", "cosine"):
                    add("agglo", ver=v, k=k, linkage=link, metric=met)
            for nn in (5, 10, 15, 20, 30):
                add("agglo_conn", ver=v, k=k, nn=nn)
            for al in ("kmeans", "discretize", "cluster_qr"):
                for nn in (5, 7, 10, 15, 20, 30, 40):
                    add("spectral_nn", ver=v, k=k, nn=nn, assign=al)
                    add("spectral_st", ver=v, k=k, nn=nn, assign=al)
                for mult in (0.1, 0.25, 0.5, 1, 2, 4, 8, 16):
                    add("spectral_rbf", ver=v, k=k, mult=mult, assign=al)
            for cov in ("full", "tied", "diag", "spherical"):
                for reg in (1e-6, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0):
                    for init in ("kmeans", "k-means++", "random", "random_from_data"):
                        add("gmm", ver=v, k=k, cov=cov, reg=reg, init=init)
                    add("gmm_ward", ver=v, k=k, cov=cov, reg=reg)
                for wtype in ("dirichlet_process", "dirichlet_distribution"):
                    for wcp in (None, 1e-3, 1.0, 100.0):
                        for reg in (1e-6, 1e-2, 0.1):
                            add("bgmm", ver=v, k=k, cov=cov, wtype=wtype, wcp=wcp, reg=reg)
                for reg in (1e-3, 0.01, 0.1, 0.5):
                    for alpha in (0.05, 0.10):
                        add("trim_gmm", ver=v, k=k, cov=cov, reg=reg, alpha=alpha)
                    for df in (3, 5, 10, 30):
                        add("tmix", ver=v, k=k, cov=cov, reg=reg, df=df)
            for m in (1.1, 1.25, 1.5, 1.75, 2.0, 2.5):
                add("fcm", ver=v, k=k, m=m)
            for met in ("euclidean", "manhattan", "cosine"):
                for init in ("build", "random"):
                    add("pam", ver=v, k=k, metric=met, init=init)
            for gr in {3: [(1, 3)], 4: [(1, 4), (2, 2)], 5: [(1, 5)], 6: [(1, 6), (2, 3)], 7: [(1, 7)], 8: [(1, 8), (2, 4)]}[k]:
                for se in (0.3, 0.5, 1.0):
                    for init in ("random", "pca"):
                        add("som", ver=v, k=k, rows=gr[0], cols=gr[1], sigma_end=se, init=init)
            for side in (4, 5, 6, 8, 10):
                for se in (0.5, 1.0):
                    for init in ("random", "pca"):
                        add("som", ver=v, k=k, rows=side, cols=side, sigma_end=se, init=init, merge_k=k)
            for thr in (0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
                for bf in (10, 25, 50):
                    add("birch", ver=v, k=k, thr=thr, bf=bf)
            for damp in (0.5, 0.9):
                add("ap_k", ver=v, k=k, damp=damp)
            for q in (1, 2, 3):
                if v == "raw" or q < v:
                    for fl in (1e-3, 0.05, 0.2):
                        add("mppca", ver=v, k=k, q=q, floor=fl)
            for base in ("ward", "average_manhattan", "gmm_diag", "gmm_spherical", "gmm_tied", "gmm_full"):
                for fin in ("average", "complete", "spectral"):
                    add("consensus", ver=v, k=k, base=base, final=fin)
            for crit in ("sse", "size"):
                add("pddp", ver=v, k=k, crit=crit)
            for met in ("euclidean", "manhattan"):
                add("diana", ver=v, k=k, metric=met)
        # k-free methods and upper-bound Bayesian mixtures
        for k in (10, 15):
            for cov in ("full", "tied", "diag", "spherical"):
                for wtype in ("dirichlet_process", "dirichlet_distribution"):
                    for wcp in (None, 1e-3, 1.0, 100.0):
                        for reg in (1e-6, 1e-2, 0.1):
                            add("bgmm", ver=v, k=k, cov=cov, wtype=wtype, wcp=wcp, reg=reg)
        for damp in (0.5, 0.75, 0.9):
            for q in (0.0, 0.05, 0.1, 0.25, 0.5):
                add("ap", ver=v, damp=damp, pmode="quantile", pval=q)
            for mult in (2, 3, 5, 10, 20, 50):
                add("ap", ver=v, damp=damp, pmode="minmult", pval=mult)
        for q in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8):
            for bs in (True, False):
                for ca in (True, False):
                    add("meanshift", ver=v, q=q, bin=bs, all=ca)
        for met in ("euclidean", "manhattan"):
            for ms in (3, 4, 5, 6, 8, 10, 12, 15, 20):
                for q in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99):
                    add("dbscan", ver=v, ms=ms, q=q, metric=met)
            for ms in (3, 5, 8, 10, 15, 20):
                for xi in (0.005, 0.01, 0.03, 0.05, 0.1, 0.2):
                    for mcs in (None, 8, 15):
                        add("optics", ver=v, ms=ms, xi=xi, mcs=mcs, metric=met)
            for mcs in (5, 8, 10, 12, 15, 20, 25, 30):
                for ms in (1, 2, 3, 5, 8, 10, 15):
                    for method in ("eom", "leaf"):
                        for cse in (0.0, 0.25, 0.5, 1.0):
                            add("hdbscan", ver=v, mcs=mcs, ms=ms, method=method, cse=cse, metric=met)
                            add("hdbscan", ver=v, mcs=mcs, ms=ms, method=method, cse=cse, metric=met, assign_noise=True)
        for w in ("binary", "snn"):
            for nn in (5, 8, 10, 15, 20, 30, 40):
                for g in (0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
                    add("louvain", ver=v, nn=nn, gamma=g, weight=w)
        for nn in (5, 10, 15, 20, 30):
            for infl in (1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0):
                add("mcl", ver=v, nn=nn, infl=infl)
    return S


def algo_name(s):
    f = s["fam"]
    if f == "agglo":
        return "Agglomerative " + s["linkage"]
    if f == "som":
        return "SOM + Ward merge" if "merge_k" in s else "SOM (node = type)"
    if f == "hdbscan":
        return "HDBSCAN + noise reassigned" if s.get("assign_noise") else "HDBSCAN"
    return {
        "agglo_conn": "Agglomerative ward, kNN-constrained",
        "spectral_nn": "Spectral nearest neighbours",
        "spectral_rbf": "Spectral RBF",
        "spectral_st": "Spectral self-tuning",
        "gmm": "GMM",
        "gmm_ward": "GMM, Ward-initialised",
        "bgmm": "Bayesian GMM",
        "trim_gmm": "Trimmed GMM",
        "tmix": "Student-t mixture",
        "mppca": "Mixture of PPCA",
        "fcm": "Fuzzy c-means",
        "pam": "k-medoids (PAM)",
        "birch": "BIRCH",
        "ap": "Affinity propagation",
        "ap_k": "Affinity propagation, k-targeted",
        "meanshift": "Mean shift",
        "dbscan": "DBSCAN",
        "optics": "OPTICS",
        "louvain": "Louvain on kNN graph",
        "mcl": "Markov clustering",
        "consensus": "Consensus clustering",
        "pddp": "PDDP divisive",
        "diana": "DIANA divisive",
        "kmeans_ref": "K-means (reference only)",
    }[f]
