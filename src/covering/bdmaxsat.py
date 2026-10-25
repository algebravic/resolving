#!/usr/bin/env python3
"""
bdmaxsat.py -- SAT / MaxSAT encodings of the minimum balanced-cover problem (see MAXSAT.md).

  python bdmaxsat.py N M                  -> bd_N_M.wcnf              (partial MaxSAT, MSE-2022 format)
  python bdmaxsat.py N M --k K            -> bd_N_M_leK.cnf           (decision: cover with <= K lines)
  options: --symbreak      lex-leader clauses for the adjacent transpositions of S_n
           --solve         run PySAT RC2 (MaxSAT) or a SAT solver (decision) in-process, if PySAT is installed
           --map FILE      write the variable -> vector map (default bd_N_M.map)
Variables 1..L are the lines (+-pairs) of B(n); auxiliaries follow.
"""
import argparse
import itertools
import sys

from .bdcover import balanced_vectors, build_constraints, bits, verify_rowsets, NotAllDetected


def neg(x):
    return tuple(-a for a in x)


class CNF:
    def __init__(self, nvars):
        self.n = nvars
        self.hard = []

    def new(self):
        self.n += 1
        return self.n

    def add(self, cl):
        self.hard.append(list(cl))


def instance(n, m, W=False):
    vecs = balanced_vectors(n)
    Ks, nmin, _ = build_constraints(n, m, vecs, lambda s: None, allow_empty=W)
    lines = [x for x in vecs if x > neg(x)]            # representative of each +- pair
    lid = {}
    for j, x in enumerate(lines):
        lid[x] = j
        lid[neg(x)] = j
    cons = sorted({tuple(sorted({lid[vecs[i]] for i in bits(K)})) for K in Ks[:nmin]},
                  key=lambda c: (len(c), c))
    return vecs, lines, lid, cons


def lex_leader(cnf, perm):
    """Clauses for x <=_lex (x o perm), i.e. (x_1..x_L) <=_lex (x_perm(1)..x_perm(L)); vars 1-based."""
    L = len(perm)
    e_prev = None                                   # None == "true"
    for i in range(L):
        a, b = i + 1, perm[i] + 1
        if a == b:
            continue
        pre = [] if e_prev is None else [-e_prev]
        cnf.add(pre + [-a, b])                      # equal so far -> x_a <= x_b
        e = cnf.new()                               # e -> not needed; only (equal so far & x_a == x_b) -> e
        cnf.add(pre + [a, b, e])
        cnf.add(pre + [-a, -b, e])
        e_prev = e


def symbreak(cnf, lines, lid, n):
    for i in range(n - 1):
        perm = []
        for x in lines:
            y = list(x)
            y[i], y[i + 1] = y[i + 1], y[i]
            perm.append(lid[tuple(y)])
        lex_leader(cnf, perm)


def totalizer_atmost(cnf, xs, k):
    """Totalizer (Bailleux-Boufkhad) for sum xs <= k; unary outputs truncated at k+1."""
    def build(lo, hi):
        if hi - lo == 1:
            return [xs[lo]]
        mid = (lo + hi) // 2
        A, B = build(lo, mid), build(mid, hi)
        size = min(len(A) + len(B), k + 1)
        out = [cnf.new() for _ in range(size)]
        for i in range(len(A) + 1):
            for j in range(len(B) + 1):
                s = i + j
                if s == 0 or s > size:
                    continue
                cl = [out[s - 1]]
                if i:
                    cl.append(-A[i - 1])
                if j:
                    cl.append(-B[j - 1])
                cnf.add(cl)                         # a_i & b_j -> out_{i+j}
        return out
    out = build(0, len(xs))
    if len(out) > k:
        cnf.add([-out[k]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("m", type=int)
    ap.add_argument("--k", type=int)
    ap.add_argument("--symbreak", action="store_true")
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--W", action="store_true", help="cover W = union of C(v) (allows B != M)")
    ap.add_argument("--solver", type=str, default="g4",
                    help="SAT solver to use for MaxSAT")
    a = ap.parse_args()
    try:
        vecs, lines, lid, cons = instance(a.n, a.m, a.W)
    except NotAllDetected as e:
        sys.exit(f"B != M, witness rows {e.rows}")
    L = len(lines)
    cnf = CNF(L)
    for c in cons:
        cnf.add([j + 1 for j in c])
    if a.symbreak:
        symbreak(cnf, lines, lid, a.n)
    if a.k is not None:
        totalizer_atmost(cnf, list(range(1, L + 1)), a.k)
    stem = a.out or (f"bd{'W' if a.W else ''}_{a.n}_{a.m}" + (f"_le{a.k}" if a.k is not None else "") + ("_sb" if a.symbreak else ""))
    with open(f"bd_{a.n}_{a.m}.map", "w") as f:
        for j, x in enumerate(lines):
            f.write(f"{j + 1} {' '.join(map(str, x))}\n")
    if a.k is None:
        path = stem + ".wcnf"
        with open(path, "w") as f:
            f.write(f"c balanced cover n={a.n} m={a.m}; vars 1..{L} are lines, see bd_{a.n}_{a.m}.map\n")
            for cl in cnf.hard:
                f.write("h " + " ".join(map(str, cl)) + " 0\n")
            for j in range(1, L + 1):
                f.write(f"1 -{j} 0\n")
    else:
        path = stem + ".cnf"
        with open(path, "w") as f:
            f.write(f"p cnf {cnf.n} {len(cnf.hard)}\n")
            for cl in cnf.hard:
                f.write(" ".join(map(str, cl)) + " 0\n")
    print(f"n={a.n} m={a.m}: {L} line vars, {cnf.n} vars total, {len(cnf.hard)} hard clauses "
          f"({len(cons)} covering) -> {path}")

    if not a.solve:
        return
    try:
        from pysat.formula import WCNF, CNF as PCNF
        from pysat.examples.rc2 import RC2
        from pysat.solvers import Solver
    except ImportError:
        sys.exit("PySAT not installed (uv pip install python-sat); files written for external solvers")
    if a.k is None:
        w = WCNF()
        for cl in cnf.hard:
            w.append(cl)
        for j in range(1, L + 1):
            w.append([-j], weight=1)
        with RC2(w, solver=a.solver, adapt=True, exhaust=True, minz=True, verbose=10) as rc2:
            model = rc2.compute()
            cost = rc2.cost
        print(f"RC2 optimum cost = {cost}")
    else:
        with Solver(name="cd19", bootstrap_with=cnf.hard) as s:
            sat = s.solve()
            model = s.get_model() if sat else None
        print("SAT" if model else f"UNSAT: no cover with <= {a.k} lines" +
              (" (under symmetry breaking, which is sound)" if a.symbreak else ""))
    if model:
        S = [vecs.index(lines[l - 1]) for l in model if 0 < l <= L]
        print(f"cover size {len(S)}, verify_rowsets: {verify_rowsets(S, vecs, a.n, a.m, W=a.W) is None}")


if __name__ == "__main__":
    main()
