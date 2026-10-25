#!/usr/bin/env python3
"""
bdorbits.py -- classify the S_n-orbits of minimal constraints K(X) (as sets of lines), see CONSTRAINT_ORBITS.md.

  python bdorbits.py N M [--max-lines L] [--W]
"""
import argparse
import itertools
from collections import Counter

import numpy as np

from .bdcover import balanced_vectors, build_constraints, bits


def neg(x):
    return tuple(-a for a in x)


def rank(rows):
    return int(np.linalg.matrix_rank(np.array(rows, dtype=float))) if rows else 0


def components(L):
    L = list(L)
    comp, seen = [], set()
    for i in range(len(L)):
        if i in seen:
            continue
        stack, cur = [i], []
        seen.add(i)
        while stack:
            a = stack.pop()
            cur.append(L[a])
            for b in range(len(L)):
                if b not in seen and sum(p * q for p, q in zip(L[a], L[b])) != 0:
                    seen.add(b)
                    stack.append(b)
        comp.append(cur)
    return comp


def comp_label(c):
    r = rank(c)
    norms = sorted(sum(a * a for a in x) for x in c)
    if len(c) == 1:
        return f"A1[{norms[0]}]"
    if r == 2 and len(c) == 3:
        return "tri" + str(norms)
    if r == 2 and len(c) == 4:
        return "B2" + str(norms)
    if all(nm == 2 for nm in norms):
        k = r + 1
        if len(c) == k * (k - 1) // 2:
            return f"A{r}(type-1 on {k} pts)"
    return f"{len(c)}L/r{r}" + str(dict(Counter(norms)))


def classify(n, m, max_lines=None, W=False, log=print):
    vecs = balanced_vectors(n)
    Ks, nmin, wit = build_constraints(n, m, vecs, lambda s: None, allow_empty=W,
                                      minimal_limit=10 ** 9 if n <= 7 else 0)
    lines = [x for x in vecs if x > neg(x)]
    lid = {}
    for j, x in enumerate(lines):
        lid[x] = j
        lid[neg(x)] = j
    cons = {}
    for K, w in zip(Ks[:nmin], wit[:nmin]):
        c = frozenset(lid[vecs[i]] for i in bits(K))
        if max_lines is None or len(c) <= max_lines:
            cons.setdefault(c, w)
    if n > 7:
        # minimality filter was skipped in build_constraints; decide it exactly for the retained sets:
        # a proper sub-constraint has fewer lines, hence is itself retained, so subset lookup is exact.
        allc = set(cons)
        def minimal(c):
            cl = sorted(c)
            for k in range(1, len(cl)):
                for sub in itertools.combinations(cl, k):
                    if frozenset(sub) in allc:
                        return False
            return True
        cons = {c: w for c, w in cons.items() if minimal(c)}
    # generators on lines
    gens = []
    for perm in [(1, 0) + tuple(range(2, n)), tuple(range(1, n)) + (0,)]:
        gens.append([lid[tuple(x[perm[i]] for i in range(n))] for x in lines])
    # union-find orbits
    orbit_of, orbits = {}, []
    for c in cons:
        if c in orbit_of:
            continue
        k = len(orbits)
        orb = [c]
        orbit_of[c] = k
        i = 0
        while i < len(orb):
            for g in gens:
                d = frozenset(g[j] for j in orb[i])
                if d not in orbit_of:
                    orbit_of[d] = k
                    orb.append(d)
            i += 1
        orbits.append(orb)
    fact = 1
    for i in range(2, n + 1):
        fact *= i
    B = [x for x in lines]
    rows_out = []
    for orb in orbits:
        rep = min(orb, key=lambda c: sorted(c))
        L = [lines[j] for j in sorted(rep)]
        X = cons.get(rep) or cons[next(c for c in orb if c in cons)]
        r = rank(L)
        dX = n - rank(list(X) + [tuple([1] * n)])
        span_lines = [x for x in B if rank(L + [x]) == r]
        closed = len(span_lines) == len(L)
        comps = components(L)
        types = dict(sorted(Counter(x.count(1) for x in L).items()))
        supp = len({i for x in L for i, a in enumerate(x) if a})
        label = " + ".join(sorted(comp_label(c) for c in comps))
        rows_out.append(dict(lines=len(L), orbit=len(orb), stab=fact // len(orb), types=types, supp=supp,
                             rank=r, dX=dX, closed=closed, label=label, rep=L, witness=X))
    rows_out.sort(key=lambda d: (d["lines"], -d["orbit"]))
    return rows_out, len(cons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("m", type=int)
    ap.add_argument("--max-lines", type=int)
    ap.add_argument("--W", action="store_true")
    ap.add_argument("--reps", action="store_true", help="print representatives and witnesses")
    a = ap.parse_args()
    rows, total = classify(a.n, a.m, a.max_lines, a.W)
    print(f"### (n,m) = ({a.n},{a.m}), d = {a.n - 1 - a.m}: {total} minimal constraints"
          f"{' (<= %d lines)' % a.max_lines if a.max_lines else ''}, {len(rows)} S_n-orbits\n")
    print("| lines | orbit | |stab| | types | supp | rank | dim ker | closed | components |")
    print("|---|---|---|---|---|---|---|---|---|")
    for d in rows:
        print(f"| {d['lines']} | {d['orbit']} | {d['stab']} | {d['types']} | {d['supp']} | {d['rank']} | "
              f"{d['dX']} | {'y' if d['closed'] else 'n'} | {d['label']} |")
    if a.reps:
        print()
        for k, d in enumerate(rows):
            print(f"- orbit {k}: rep {d['rep']}; witness rows {d['witness']}")


if __name__ == "__main__":
    main()
