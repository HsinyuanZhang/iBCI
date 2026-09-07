"""UNSEALED exploratory probe: common-mode structure of Cell-D output error.

STATUS: NON-GOVERNING, NON-RECEIPTED, NOT A WORK-ORDER CELL.
    Reads only the already-materialized source-side body
    results/ac3_action_continuity_v0/trajectories.npz
    (6 sub-C within source sessions, 1206 completed trials, 4 complementary
    unit-group views, M4 activity-only never-commit parent line).

    It opens NO external roster, creates NO result root, writes NO artifact,
    performs NO parameter update on any target session, and does NOT modify any
    dispatched experiment.  Every number it prints is an exploratory diagnostic
    that must be re-derived under a sealed work order before it may be cited.

    Authority for the numbers quoted in
    docs/DESIGN_COMMON_MODE_BOUNDED_CARRIER_SUCCESSOR_20260829.md section 4.

Probes:
    A  group-vs-group error correlation of the velocity output
    B  4-group output ensemble vs single-group and best-group
    C  per-bin credibility weighting of the direction integral
    D  R0 (single-group direction) vs R-GE (zero-parameter group circular mean)
    E  trial-level credibility ranking: rho_GE vs learned logistic vs oracle
    F  common-mode rotation: label-free grid alignment and balanced assignment

Run:  python3 tfpd_exploration/scripts/probe_common_mode_unsealed_20260829.py
"""

from __future__ import annotations

import numpy as np

NPZ = "tfpd_exploration/results/ac3_action_continuity_v0/trajectories.npz"
NPZ_SHA256 = "e3512c9760b72af015ce6ae2ddd79fce79f2aa51e8c93b2240d438b67a5f3eb8"


def wrap(a):
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def snap8(a):
    """Snap a direction to the 8-target center-out grid."""
    return np.round(a / (np.pi / 4)).astype(int) % 8


def vwr2(pred, true):
    """Variance-weighted multi-output R2."""
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0)
    w = ss_tot / ss_tot.sum()
    return float((w * (1 - ss_res / ss_tot)).sum())


def concentration(V):
    """Per-bin circular concentration of the four group direction estimates."""
    n = np.linalg.norm(V, axis=1, keepdims=True)
    u = np.divide(V, np.where(n > 0, n, 1.0))
    return np.linalg.norm(u.mean(2), axis=1)


def fit_logistic(X, y, l2=1.0, iters=400):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        W = np.maximum(p * (1 - p), 1e-8)
        R = np.eye(X.shape[1]) * l2
        R[0, 0] = 0.0
        try:
            w = w + np.linalg.solve(X.T @ (X * W[:, None]) + R, X.T @ (y - p) - R @ w)
        except np.linalg.LinAlgError:
            break
    return w


def predict_logistic(w, X):
    return 1 / (1 + np.exp(-np.column_stack([np.ones(len(X)), X]) @ w))


def auc(y, s):
    order = np.argsort(s)
    r = np.empty(len(s), float)
    r[order] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load():
    import hashlib
    import pathlib

    digest = hashlib.sha256(pathlib.Path(NPZ).read_bytes()).hexdigest()
    if digest != NPZ_SHA256:
        raise SystemExit(f"body SHA drift: expected {NPZ_SHA256}, got {digest}")
    return np.load(NPZ)


def main():
    z = load()
    v, y, ok = z["velocity_flat"], z["true_flat"], z["valid_flat"]
    rc, rs, tsess = z["row_counts"], z["row_starts"], z["trial_session"]
    th = z["pseudo_raw_theta_raw_rad"]
    tt = z["true_direction_theta_raw_rad"]
    dn = z["pseudo_raw_displacement_norm"]
    ms = z["pseudo_raw_mean_speed"]
    mb = z["pseudo_raw_movement_bins"].astype(float)

    row_sess = np.zeros(v.shape[0], dtype=np.int32)
    for i in range(len(tsess)):
        row_sess[rs[i] : rs[i] + rc[i]] = tsess[i]
    sessions = np.unique(row_sess)

    print("== A/B: group output ensemble and group error correlation ==")
    per_group, ensemble = [], []
    for s in sessions:
        m = (row_sess == s) & ok
        P, T = v[m], y[m]
        per_group.append([vwr2(P[:, :, g], T) for g in range(4)])
        ensemble.append(vwr2(P.mean(2), T))
    per_group, ensemble = np.array(per_group), np.array(ensemble)
    print(f"  single-group mean R2      = {per_group.mean():.6f}")
    print(f"  best-single-group  R2     = {per_group.max(1).mean():.6f}")
    print(f"  4-group ensemble   R2     = {ensemble.mean():.6f}")
    print(
        f"  ensemble - mean group     = {ensemble.mean() - per_group.mean():+.6f} "
        f"({int((ensemble - per_group.mean(1) > 0).sum())}/{len(sessions)} sessions)"
    )
    print(
        f"  ensemble - BEST group     = {ensemble.mean() - per_group.max(1).mean():+.6f} "
        f"({int((ensemble - per_group.max(1) > 0).sum())}/{len(sessions)} sessions)"
    )
    E = v[ok] - y[ok][:, :, None]
    for dim, nm in ((0, "x"), (1, "y")):
        C = np.corrcoef(E[:, dim, :].T)[np.triu_indices(4, 1)]
        print(f"  dim {nm} pairwise error corr  = {C.mean():.4f} (min {C.min():.4f} max {C.max():.4f})")
    rho_group = np.corrcoef(
        np.concatenate([E[:, 0, :], E[:, 1, :]], 0).T
    )[np.triu_indices(4, 1)].mean()
    print(f"  POOLED rho_group          = {rho_group:.4f}   (usable independent fraction {1 - rho_group:.4f})")

    print()
    print("== C: per-bin credibility weighting of the direction integral ==")
    sl = [slice(rs[i], rs[i] + rc[i]) for i in range(len(rc))]

    def direction(wfn):
        out = np.zeros(len(rc))
        for i in range(len(rc)):
            V = v[sl[i]]
            d = (wfn(V)[:, None] * V.mean(2)).sum(0)
            out[i] = np.arctan2(d[1], d[0])
        return out

    arms = {"uniform (plain integral)": lambda V: np.ones(V.shape[0])}
    for p in (1, 2, 4, 8):
        arms[f"concentration^{p}"] = lambda V, p=p: concentration(V) ** p
    for t in (0.5, 0.8, 0.9, 0.95):
        def hard(V, t=t):
            w = (concentration(V) >= t).astype(float)
            return w if w.sum() > 0 else np.ones(V.shape[0])
        arms[f"concentration>={t:.2f} hard"] = hard
    for q in (0.5, 0.75):
        def spd(V, q=q):
            s = np.linalg.norm(V.mean(2), axis=1)
            w = (s >= np.quantile(s, q)).astype(float)
            return w if w.sum() > 0 else np.ones(V.shape[0])
        arms[f"speed top {100 * (1 - q):.0f}% bins"] = spd
    for nm, fn in arms.items():
        est = direction(fn)
        print(
            f"  {nm:<28s} mean|err| = {np.abs(wrap(est - tt)).mean():.4f} rad   "
            f"snap_mm = {100 * (snap8(est) != snap8(tt)).mean():.2f}%"
        )

    print()
    print("== D/E: direction baselines and trial-level credibility ranking ==")
    use = np.isfinite(th).all(1) & np.isfinite(tt) & np.isfinite(dn).all(1) & np.isfinite(ms).all(1)
    th, tt2, sess, dn, ms, mb = th[use], tt[use], tsess[use], dn[use], ms[use], mb[use]
    C, S = np.cos(th).mean(1), np.sin(th).mean(1)
    ge, rho = np.arctan2(S, C), np.hypot(C, S)
    err = np.abs(wrap(ge - tt2))
    correct = (snap8(ge) == snap8(tt2)).astype(float)
    eg = np.abs(wrap(th - tt2[:, None]))
    print(f"  n usable trials           = {use.sum()}")
    print(f"  R0  pooled single group   = {eg.mean():.4f} rad   snap_mm = "
          f"{100 * np.mean([(snap8(th[:, g]) != snap8(tt2)).mean() for g in range(4)]):.2f}%")
    print(f"  R0  best single group     = {eg.mean(0).min():.4f} rad")
    print(f"  R-GE group circular mean  = {err.mean():.4f} rad   snap_mm = {100 * (1 - correct.mean()):.2f}%")
    print(f"  R-GE - R0                 = {eg.mean() - err.mean():+.4f} rad")

    spread = np.concatenate(
        [np.abs(wrap(th[:, a] - th[:, b]))[:, None] for a in range(4) for b in range(a + 1, 4)], 1
    )
    F = np.column_stack([
        rho, spread.mean(1), spread.max(1), spread.min(1),
        dn.mean(1), dn.std(1), dn.min(1), dn.max(1),
        ms.mean(1), ms.std(1), mb.mean(1), mb.std(1),
        dn.std(1) / (dn.mean(1) + 1e-9), ms.std(1) / (ms.mean(1) + 1e-9),
    ])
    oof = np.zeros(len(tt2))
    for s in np.unique(sess):
        tr, te = sess != s, sess == s
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        oof[te] = predict_logistic(fit_logistic((F[tr] - mu) / sd, correct[tr]), (F[te] - mu) / sd)

    def topk(score, frac):
        k = max(1, int(len(score) * frac))
        idx = np.argsort(-score)[:k]
        return err[idx].mean(), 1 - correct[idx].mean()

    for nm, sc in (("rho_GE (0 param)", rho), ("logistic 14f LOSO", oof), ("ORACLE (leakage)", -err)):
        e50, _ = topk(sc, 0.5)
        e25, m25 = topk(sc, 0.25)
        print(f"  {nm:<20s} AUC = {auc(correct, sc):.4f}   err@50% = {e50:.4f}   "
              f"err@25% = {e25:.4f}   snap_mm@25% = {100 * m25:.2f}%")

    print()
    print("== F: common-mode rotation and cross-trial balanced assignment ==")
    resid = wrap(ge - tt2)
    pooled = np.arctan2(np.sin(resid).mean(), np.cos(resid).mean())
    print(f"  pooled circular mean error = {pooled:+.4f} rad ({np.degrees(pooled):.1f} deg)")

    def best_phi(a):
        g = np.linspace(-np.pi / 8, np.pi / 8, 721)
        return g[int(np.argmax([np.cos(8 * (a - p)).mean() for p in g]))]

    def balanced(a, K=8, eps=0.30, iters=500):
        cost = np.abs(wrap(a[:, None] - (np.arange(K) * np.pi / 4)[None, :]))
        P = np.exp(-cost / eps)
        u, vv = np.ones(len(a)) / len(a), np.ones(K) / K
        for _ in range(iters):
            P *= (u / (P.sum(1) + 1e-12))[:, None]
            P *= (vv / (P.sum(0) + 1e-12))[None, :]
        return P.argmax(1)

    m1, m2, m3 = [], [], []
    for s in np.unique(sess):
        m = sess == s
        a = ge[m] - best_phi(ge[m])
        t = snap8(tt2[m])
        m1.append((snap8(ge[m]) != t).mean())
        m2.append((snap8(a) != t).mean())
        m3.append((balanced(a) != t).mean())
    print(f"  equal-session snap_mm: argmax {100 * np.mean(m1):.2f}%  -> +phi "
          f"{100 * np.mean(m2):.2f}%  -> +balanced {100 * np.mean(m3):.2f}%")
    print(f"  balanced - argmax = {100 * (np.mean(m1) - np.mean(m3)):+.2f} pp  "
          f"({int(np.sum(np.array(m3) < np.array(m1)))}/{len(m1)} sessions)")


if __name__ == "__main__":
    main()
