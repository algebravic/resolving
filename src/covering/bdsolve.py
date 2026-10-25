#!/usr/bin/env python3
"""
bdsolve.py -- minimum S subset of B(n) covering all m x n 0/1 matrices (see DEEP.md).

Pipeline: projectivize (C(v) = C(-v)) -> forced lines -> Kramer-Mesner covers over Young
subgroups (upper bounds) -> residual ILP via HiGHS (scipy.optimize.milp) -> Hoffman bound
for the size-2 part of the residual (exactly certified) -> verify_rowsets on the answer.

Usage: python bdsolve.py N M [--time-limit SEC] [--km-max-parts K]
"""
import argparse
import itertools
import sys
import time
from collections import Counter

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from .bdcover import balanced_vectors, build_constraints, bits, verify_rowsets, NotAllDetected


def neg(x):
    return tuple(-a for a in x)


def partitions(n, mx=None):
    mx = mx or n
    if n == 0:
        yield ()
        return
    for k in range(min(n, mx), 0, -1):
        for p in partitions(n - k, k):
            yield (k,) + p


def solve_cover(cons, npts, weights=None, time_limit=None, disp=False):
    """min sum w_j x_j s.t. every constraint (set of point ids) is hit; returns (x or None, res)."""
    A = np.zeros((len(cons), npts))
    for r, c in enumerate(cons):
        for j in c:
            A[r, j] = 1
    w = np.ones(npts) if weights is None else np.asarray(weights, float)
    opts = {"disp": disp}
    if time_limit:
        opts["time_limit"] = time_limit
    res = milp(w, constraints=LinearConstraint(A, lb=1), integrality=np.ones(npts),
               bounds=Bounds(0, 1), options=opts)
    return res


def hoffman_certified(edges, verts):
    """Hoffman bound for alpha of the graph, with eigenvalues certified by an exact integer identity."""
    ix = {x: k for k, x in enumerate(verts)}
    nV = len(verts)
    A = np.zeros((nV, nV), dtype=np.int64)
    for a, b in edges:
        A[ix[a], ix[b]] = A[ix[b], ix[a]] = 1
    deg = set(A.sum(1).tolist())
    ev = np.linalg.eigvalsh(A.astype(float))
    cand = sorted({int(round(e)) for e in ev})
    if len(deg) != 1 or any(abs(e - round(e)) > 1e-8 for e in ev):
        return None, "graph not regular or spectrum not integral; no certified bound"
    Z = np.eye(nV, dtype=object)
    Ao = A.astype(object)
    for lam in cand:
        Z = Z.dot(Ao - lam * np.eye(nV, dtype=object))
    if any(Z.flatten()):
        return None, "integer identity failed"
    k, tau = deg.pop(), min(cand)
    return nV * (-tau) / (k - tau), f"spectrum {sorted(Counter(np.round(ev).astype(int)).items())}, degree {k}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("m", type=int)
    ap.add_argument("--time-limit", type=float, default=600)
    ap.add_argument("--km-max-parts", type=int, default=99)
    ap.add_argument("--W", action="store_true", help="cover W = union of C(v) (allows B != M)")
    a = ap.parse_args()
    n, m = a.n, a.m
    log = lambda s: print(s, flush=True)

    vecs = balanced_vectors(n)
    try:
        Ks, nmin, wit = build_constraints(n, m, vecs, log if a.W else (lambda s: None), allow_empty=a.W)
    except NotAllDetected as e:
        log(f"B != M; witness rows: {e.rows}")
        return
    # --- A. projectivize
    idx = {x: i for i, x in enumerate(vecs)}
    P = [i for i, x in enumerate(vecs) if x > neg(x)]
    pid = {}
    for j, i in enumerate(P):
        pid[i] = j
        pid[idx[neg(vecs[i])]] = j
    W = {}
    for K, w in zip(Ks[:nmin], wit[:nmin]):
        W.setdefault(frozenset(pid[i] for i in bits(K)), w)
    cons = list(W)
    typ = lambda j: vecs[P[j]].count(1)
    log(f"n={n} m={m}: {len(P)} lines, {len(cons)} minimal constraints, sizes "
        f"{sorted(Counter(map(len, cons)).items())}")

    # --- B. forced lines
    forced = {next(iter(c)) for c in cons if len(c) == 1}
    rest = [c for c in cons if not (c & forced)]
    log(f"forced lines: {len(forced)} (types {dict(Counter(typ(j) for j in forced))}); "
        f"residual: {len(rest)} constraints on {len(set().union(*rest)) if rest else 0} lines")

    # --- D. Kramer-Mesner upper bounds
    best = None
    for lam in partitions(n):
        if len(lam) > a.km_max_parts or len(lam) == n:
            continue
        blocks, s = [], 0
        for b in lam:
            blocks.append(range(s, s + b))
            s += b
        key = lambda x: tuple(tuple(sorted(x[i] for i in B)) for B in blocks)
        orb = {}
        for j, i in enumerate(P):
            orb.setdefault(min(key(vecs[i]), key(neg(vecs[i]))), []).append(j)
        O = list(orb.values())
        where = {j: o for o, js in enumerate(O) for j in js}
        rows = list({frozenset(where[j] for j in c) for c in cons})
        res = solve_cover(rows, len(O), [len(o) for o in O], time_limit=min(60, a.time_limit))
        if res.x is None:
            continue
        S = sorted(j for o in range(len(O)) if res.x[o] > 0.5 for j in O[o])
        if best is None or len(S) < len(best[0]):
            best = (S, lam)
        log(f"  KM  H = S_{lam}: invariant cover of size {len(S)}")
    ub = len(best[0]) if best else len(P)
    log(f"upper bound {ub} (from H = S_{best[1] if best else '?'})")

    # --- C. Hoffman bound on the size-2 residual graph
    edges = [tuple(c) for c in rest if len(c) == 2]
    lb_certified = len(forced)
    if edges:
        verts = sorted({j for e in edges for j in e})
        h, info = hoffman_certified(edges, verts)
        log(f"residual graph: {len(verts)} vertices, {len(edges)} edges; {info}")
        if h is not None:
            lb_certified = len(forced) + len(verts) - int(np.floor(h + 1e-9))
            log(f"certified lower bound (forced + Hoffman): {lb_certified}")

    # --- E. residual ILP
    t0 = time.time()
    if rest:
        pts = sorted(set().union(*rest))
        loc = {j: k for k, j in enumerate(pts)}
        res = solve_cover([{loc[j] for j in c} for c in rest], len(pts), time_limit=a.time_limit)
        opt = res.status == 0
        val = None if res.x is None else len(forced) + int(round(res.fun))
        dual = len(forced) + (res.mip_dual_bound if getattr(res, "mip_dual_bound", None) is not None else 0)
        log(f"ILP ({time.time() - t0:.1f}s): {'optimal' if opt else res.message}; "
            f"value {val}, dual bound {dual:.2f}")
        if res.x is not None:
            Sx = sorted(forced | {pts[k] for k in range(len(pts)) if res.x[k] > 0.5})
            if best is None or len(Sx) < ub:
                best, ub = (Sx, "ILP"), len(Sx)
    else:
        opt = True
        best, ub = (sorted(forced), "forced"), len(forced)

    # --- verification of the reported cover (independent of the constraint family)
    S = [P[j] for j in best[0]]
    ok = verify_rowsets(S, vecs, n, m, W=a.W) is None
    log(f"cover of size {len(S)} verified by verify_rowsets: {ok}")
    if opt and ok:
        cert = " (matches certified lower bound)" if lb_certified == len(S) else " (optimality: solver claim)"
        log(f"MINIMUM = {len(S)}{cert}")
    for i in S:
        log("   " + " ".join(f"{x:+d}" if x else " 0" for x in vecs[i]))


if __name__ == "__main__":
    main()
