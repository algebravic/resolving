#!/usr/bin/env python3
"""
bdcover.py -- minimum S subset of B(n) with  U_{v in S} C(v) = M  (all m x n 0/1 matrices),
by symmetry-reduced tree search with G = S_n x {+-1}; group work done in GAP via pexpect.

See DESIGN.md for the reformulation (L1-L3) and soundness arguments.

Usage:
    python bdcover.py N M [--mode spec|constraint] [--backend gap|python]
                          [--gap PATH] [--no-memo] [--no-lb] [--verify-brute] [--verify-naive]
                          [--check-backends R]
"""
import argparse
import itertools
import random
import re
import sys
import time


# ----------------------------------------------------------------------------- utilities

def bits(x):
    """Indices of set bits of a nonnegative int, increasing."""
    s = bin(x)[:1:-1]
    out = []
    i = s.find('1')
    while i != -1:
        out.append(i)
        i = s.find('1', i + 1)
    return out


def popcount(x):
    return x.bit_count()


def mask_from_indices(idx, size):
    ba = bytearray((size + 7) // 8)
    for i in idx:
        ba[i >> 3] |= 1 << (i & 7)
    return int.from_bytes(ba, 'little')


# ----------------------------------------------------------------------------- B(n) and G

def balanced_vectors(n):
    """B(n) in lexicographic order (-1 < 0 < 1); point i <-> vecs[i] (GAP point i+1)."""
    return [v for v in itertools.product((-1, 0, 1), repeat=n) if any(v) and sum(v) == 0]


def group_generators(vecs):
    """Generators of G = S_n x {+-1} as 0-based image lists on the points of B(n)."""
    n = len(vecs[0])
    index = {v: i for i, v in enumerate(vecs)}
    maps = [lambda v: (v[1], v[0]) + v[2:],          # transposition (1 2)
            lambda v: v[1:] + v[:1],                  # n-cycle
            lambda v: tuple(-x for x in v)]           # sign
    gens = []
    ident = list(range(len(vecs)))
    for f in maps:
        p = [index[f(v)] for v in vecs]
        if p != ident and p not in gens:
            gens.append(p)
    return gens


# ----------------------------------------------------------------------------- constraints

class NotAllDetected(Exception):
    """Raised when B != M; carries the rows of a witness matrix."""
    def __init__(self, rows):
        super().__init__(rows)
        self.rows = rows


def build_constraints(n, m, vecs, log, allow_empty=False, minimal_limit=30000):
    """
    Returns (Ks, nmin, witnesses):
      Ks        list of bitsets over B(n): all distinct K(X), X subset W', |X| <= m  (= script-K),
                ordered: the nmin inclusion-minimal ones first (by popcount), then the rest.
      witnesses parallel list of row tuples X realizing each K.
    Raises NotAllDetected if some K(X) is empty, unless allow_empty (the W-cover problem:
    empty kernels are matrices outside W and are simply dropped, with their supersets).
    """
    N = len(vecs)
    W = [a for a in itertools.product((0, 1), repeat=n) if a[0] == 0 and any(a)]
    col = []
    for a in W:
        col.append(mask_from_indices(
            [i for i, v in enumerate(vecs) if sum(x * y for x, y in zip(a, v)) == 0], N))
    full = (1 << N) - 1
    allK = {full: ()}
    frontier = {full: ()}
    empty_witness = None
    for d in range(1, m + 1):
        new = {}
        for K, rows in frontier.items():
            for a, c in zip(W, col):
                K2 = K & c
                if K2 == K:
                    continue
                if K2 == 0:
                    if not allow_empty:
                        raise NotAllDetected(rows + (a,))
                    if empty_witness is None:
                        empty_witness = rows + (a,)
                    continue
                if K2 not in allK and K2 not in new:
                    new[K2] = rows + (a,)
        allK.update(new)
        frontier = new
        log(f"  level {d}: {len(new)} new kernel sets, {len(allK)} total")
        if not new:
            break
    # inclusion-minimal members (quadratic; skipped for large families -- all members are then
    # returned as 'minimal', which is sound for every downstream use: dominated constraints are
    # redundant, never wrong)
    items = sorted(allK.items(), key=lambda kv: popcount(kv[0]))
    if len(items) > minimal_limit:
        log(f"  |K| = {len(allK)} > {minimal_limit}: minimality filter skipped" +
            (f"; B != M (e.g. rows {empty_witness}), covering W only" if empty_witness else ""))
        return [k for k, _ in items], len(items), [r for _, r in items]
    minimal, rest = [], []
    for K, rows in items:
        if any((M & K) == M for M, _ in minimal):
            rest.append((K, rows))
        else:
            minimal.append((K, rows))
    log(f"  |K| = {len(allK)}, minimal = {len(minimal)}" +
        (f"; B != M (e.g. rows {empty_witness} have no balanced kernel), covering W only"
         if empty_witness else ""))
    ordered = minimal + rest
    return [k for k, _ in ordered], len(minimal), [r for _, r in ordered]


def incidence(Ks, N):
    """inc[v] = bitset over constraint indices i with v in Ks[i]."""
    lists = [[] for _ in range(N)]
    for i, K in enumerate(Ks):
        for v in bits(K):
            lists[v].append(i)
    nc = len(Ks)
    return [mask_from_indices(l, nc) for l in lists]


# ----------------------------------------------------------------------------- backends

class PythonBackend:
    """Enumerates G explicitly. For testing / cross-checking; practical for n <= 7."""

    def __init__(self, gens, N):
        self.N = N
        ident = tuple(range(N))
        elems = {ident}
        todo = [ident]
        while todo:
            g = todo.pop()
            for s in gens:
                h = tuple(s[g[i]] for i in range(N))   # s o g
                if h not in elems:
                    elems.add(h)
                    todo.append(h)
        self.elems = list(elems)
        self.order = len(self.elems)
        self._cache = {}

    def orbit_min_map(self, U):
        key = frozenset(U)
        if key in self._cache:
            return self._cache[key]
        H = [g for g in self.elems if frozenset(g[u] for u in U) == key]
        mins = list(range(self.N))
        for p in range(self.N):
            mins[p] = min(g[p] for g in H)
        self._cache[key] = (mins, len(H))
        return mins, len(H)

    def canon(self, U):
        return min(tuple(sorted(g[u] for u in U)) for g in self.elems)

    def close(self):
        pass


class GapBackend:
    """GAP over pexpect. Points are 1-based in GAP, 0-based in Python."""

    SENT = "@@DONE@@"

    def __init__(self, gens, N, gap="gap", log=print):
        import pexpect
        self.N = N
        self.log = log
        self.p = pexpect.spawn(gap, ["-q", "-b"], encoding="utf-8", echo=False,
                               timeout=None, maxread=1 << 20)
        self.run("BreakOnError := false;;")
        self.run("SizeScreen([4096, 1000]);;")
        gl = ",".join("PermList([" + ",".join(str(x + 1) for x in g) + "])" for g in gens)
        self.run(f"NPTS := {N};;")
        self.run(f"G := Group({gl});;")
        self.order = int(self.eval_ints("Size(G)")[0])
        ok = self.run('Print(LoadPackage("images", false) = true);')
        self.have_images = "true" in ok
        self.run("""
BDOrbMins := function(U)
  local H, mins, o, m, p;
  if U = [] then H := G; else H := Stabilizer(G, U, OnSets); fi;
  mins := [1..NPTS];
  for o in Orbits(H, [1..NPTS]) do
    m := Minimum(o);
    for p in o do mins[p] := m; od;
  od;
  return [Size(H), mins];
end;;""")
        self._cache = {}
        log(f"  GAP: |G| = {self.order}, images package: {self.have_images}")

    @staticmethod
    def _chunk(cmd, width=1000):
        """Split a long command into lines <= width at commas (pty line limit ~4 KB)."""
        lines = []
        for line in cmd.split("\n"):
            while len(line) > width:
                cut = line.rfind(",", 0, width)
                if cut <= 0:
                    raise ValueError("cannot split GAP command line")
                lines.append(line[:cut + 1])
                line = line[cut + 1:]
            lines.append(line)
        return lines

    def run(self, cmd):
        for line in self._chunk(cmd):
            self.p.sendline(line)
        # sentinel is assembled by GAP so it never appears literally in echoed input
        self.p.sendline('Print("@@", "DONE", "@@\\n");')
        self.p.expect_exact(self.SENT)
        out = self.p.before
        if "Error" in out:
            raise RuntimeError(f"GAP error on:\n{cmd[:500]}\n---\n{out}")
        return out

    def eval_ints(self, expr):
        out = self.run(f"Print({expr}, \"\\n\");")
        out = out.replace("\\\n", "")
        return [int(t) for t in re.findall(r"-?\d+", out)]

    @staticmethod
    def _gset(U):
        return "[" + ",".join(str(u + 1) for u in sorted(U)) + "]"

    def orbit_min_map(self, U):
        key = frozenset(U)
        if key in self._cache:
            return self._cache[key]
        vals = self.eval_ints(f"BDOrbMins({self._gset(U)})")
        size, mins = vals[0], [x - 1 for x in vals[1:]]
        assert len(mins) == self.N, "unexpected GAP output length"
        self._cache[key] = (mins, size)
        return mins, size

    def canon(self, U):
        if not self.have_images:
            return tuple(sorted(U))            # sound, just weaker
        if not U:
            return ()
        return tuple(x - 1 for x in self.eval_ints(f"MinimalImage(G, {self._gset(U)}, OnSets)"))

    def close(self):
        try:
            self.p.sendline("quit;")
            self.p.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------- search

class Search:
    def __init__(self, Ks, nmin, inc, backend, mode="spec", use_memo=True, use_lb=True,
                 log=print):
        self.Ks, self.nmin, self.inc = Ks, nmin, inc
        self.minmask = (1 << nmin) - 1
        self.allmask = (1 << len(Ks)) - 1
        self.B = backend
        self.mode = mode
        self.use_memo, self.use_lb = use_memo, use_lb
        self.memo = {}
        self.nodes = 0
        self.log = log

    def packing_lb(self, unhit):
        used, c = 0, 0
        for i in bits(unhit & self.minmask):          # minimal ones, by increasing size
            K = self.Ks[i]
            if K & used == 0:
                used |= K
                c += 1
        return c

    def greedy(self):
        unhit, S = self.allmask, []
        while unhit:
            um = unhit & self.minmask
            v = max(range(len(self.inc)), key=lambda x: popcount(um & self.inc[x]))
            S.append(v)
            unhit &= ~self.inc[v]
        return S

    def candidates(self, U, unhit):
        mins, hsize = self.B.orbit_min_map(U)
        if self.mode == "spec":
            T = 0
            for i in bits(unhit):
                T |= self.Ks[i]
            cands = {mins[t] for t in bits(T)}
        else:
            best = None
            for i in bits(unhit & self.minmask):
                r = {mins[v] for v in bits(self.Ks[i])}
                if best is None or len(r) < len(best):
                    best = r
                    if len(r) == 1:
                        break
            cands = best
        um = unhit & self.minmask
        return sorted(cands, key=lambda v: (-popcount(um & self.inc[v]), v)), hsize

    def extend(self, U, unhit, k):
        self.nodes += 1
        if unhit == 0:
            return list(U)
        if len(U) >= k:
            return None
        key = self.B.canon(U) if self.use_memo else None
        if self.use_memo and self.memo.get(key, -1) >= k:
            return None
        if self.use_lb and len(U) + self.packing_lb(unhit) > k:
            if self.use_memo:
                self.memo[key] = k
            return None
        cands, _ = self.candidates(U, unhit)
        for v in cands:
            assert v not in U
            res = self.extend(tuple(sorted(U + (v,))), unhit & ~self.inc[v], k)
            if res is not None:
                return res
        if self.use_memo:
            self.memo[key] = max(k, self.memo.get(key, -1))
        return None

    def solve(self):
        g = self.greedy()
        lb = max(1, self.packing_lb(self.allmask))
        self.log(f"  greedy upper bound {len(g)}, packing lower bound {lb}")
        for k in range(lb, len(g)):
            t0, n0 = time.time(), self.nodes
            res = self.extend((), self.allmask, k)
            self.log(f"  k = {k}: {'FOUND' if res else 'none'}  "
                     f"({self.nodes - n0} nodes, {time.time() - t0:.1f}s, memo {len(self.memo)})")
            if res is not None:
                return sorted(res), k
        return sorted(g), len(g)


# ----------------------------------------------------------------------------- verification

def check_cover(S, Ks):
    Smask = mask_from_indices(S, max(S) + 1 if S else 1)
    return all(K & Smask for K in Ks)


def verify_rowsets(S, vecs, n, m, W=False):
    """
    Check that S covers every m x n 0/1 matrix, using only: rows may be taken distinct and
    nonzero (the 0 row kills everything; repeats change nothing), and only the row SET matters.
    BFS over row sets, deduplicated on c(X) = {v in S : Xv = 0} (bitmask over S).
    Returns None if covered, else an uncovered row set.
    """
    if W:
        return verify_rowsets_W(S, vecs, n, m)
    rows = [a for a in itertools.product((0, 1), repeat=n) if any(a)]
    Sv = [vecs[i] for i in S]
    colS = [sum(1 << j for j, v in enumerate(Sv) if sum(x * y for x, y in zip(a, v)) == 0)
            for a in rows]
    full = (1 << len(S)) - 1
    seen = {full}
    frontier = {full: ()}
    for d in range(1, m + 1):
        new = {}
        for c, X in frontier.items():
            for a, cs in zip(rows, colS):
                c2 = c & cs
                if c2 == 0:
                    return X + (a,)
                if c2 not in seen:
                    seen.add(c2)
                    new[c2] = X + (a,)
        frontier = new
        if not new:
            break
    return None


def verify_rowsets_W(S, vecs, n, m):
    """
    W-version: every row set X (distinct nonzero rows, |X| <= m) whose kernel on B(n) is nonempty
    must kill some element of S. BFS deduplicated on d(X) = {v in B(n) : Xv = 0}; kernels are
    recomputed by direct dot products over the full row alphabet (not the reduced alphabet of K).
    """
    N = len(vecs)
    rows = [a for a in itertools.product((0, 1), repeat=n) if any(a)]
    col = [mask_from_indices([i for i, v in enumerate(vecs) if sum(x * y for x, y in zip(a, v)) == 0], N)
           for a in rows]
    Smask = mask_from_indices(S, N)
    full = (1 << N) - 1
    seen = {full}
    frontier = {full: ()}
    for d in range(1, m + 1):
        new = {}
        for K, X in frontier.items():
            for a, c in zip(rows, col):
                K2 = K & c
                if K2 == 0 or K2 in seen:
                    continue
                if K2 & Smask == 0:
                    return X + (a,)
                seen.add(K2)
                new[K2] = X + (a,)
        frontier = new
        if not new:
            break
    return None


def brute_verify(S, vecs, n, m, W=False):
    """Directly check every m x n 0/1 matrix (independent of L1-L3)."""
    rows = list(itertools.product((0, 1), repeat=n))
    Svecs = [vecs[i] for i in S]
    # R(v) as a frozenset of rows, then A covered iff rows(A) subset R(v) for some v
    R = [frozenset(a for a in rows if sum(x * y for x, y in zip(a, v)) == 0) for v in Svecs]
    Rall = [frozenset(a for a in rows if sum(x * y for x, y in zip(a, v)) == 0) for v in vecs] if W else None
    for A in itertools.product(rows, repeat=m):
        rs = set(A)
        if not any(rs <= r for r in R):
            if W and not any(rs <= r for r in Rall):
                continue                      # A not in W: nothing to cover
            return A
    return None


def check_backends(gens, N, gap, trials, log):
    py = PythonBackend(gens, N)
    gp = GapBackend(gens, N, gap, log)
    assert py.order == gp.order, (py.order, gp.order)
    rng = random.Random(1)
    for t in range(trials):
        U = tuple(sorted(rng.sample(range(N), rng.randint(0, min(5, N)))))
        a, b = py.orbit_min_map(U), gp.orbit_min_map(U)
        assert a == b, f"orbit maps differ for U={U}"
        if gp.have_images:
            assert py.canon(U) == gp.canon(U), f"canonical forms differ for U={U}"
    gp.close()
    log(f"backends agree on {trials} random sets (|G| = {py.order})")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("n", type=int)
    ap.add_argument("m", type=int)
    ap.add_argument("--mode", choices=["spec", "constraint"], default="spec")
    ap.add_argument("--backend", choices=["gap", "python"], default="gap")
    ap.add_argument("--gap", default="gap", help="GAP executable")
    ap.add_argument("--W", action="store_true", help="cover W = union of C(v) (allows B != M)")
    ap.add_argument("--no-memo", action="store_true")
    ap.add_argument("--no-lb", action="store_true")
    ap.add_argument("--verify-brute", action="store_true",
                    help="row-set check of the final S (uses only observations (1),(2))")
    ap.add_argument("--verify-naive", action="store_true",
                    help="also check all 2^(mn) ordered matrices (mn <= 24)")
    ap.add_argument("--check-backends", type=int, default=0, metavar="R")
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)

    if a.n < 2 or a.m < 1:
        sys.exit("need n >= 2, m >= 1")
    vecs = balanced_vectors(a.n)
    N = len(vecs)
    gens = group_generators(vecs)
    log(f"n = {a.n}, m = {a.m}, |B(n)| = {N}")

    if a.check_backends:
        check_backends(gens, N, a.gap, a.check_backends, log)
        return

    t0 = time.time()
    try:
        Ks, nmin, wit = build_constraints(a.n, a.m, vecs, log, allow_empty=a.W)
    except NotAllDetected as e:
        log("B != M: this matrix (rows mod complement; any padding with 0/1-rows) "
            "has no balanced kernel vector:")
        for r in e.rows:
            log("   " + " ".join(map(str, r)))
        return
    inc = incidence(Ks, N)
    log(f"  constraints built in {time.time() - t0:.1f}s")

    B = PythonBackend(gens, N) if a.backend == "python" else GapBackend(gens, N, a.gap, log)
    try:
        s = Search(Ks, nmin, inc, B, a.mode, not a.no_memo, not a.no_lb, log)
        t0 = time.time()
        S, k = s.solve()
    finally:
        B.close()

    assert check_cover(S, Ks), "internal error: result does not hit every K"
    log(f"minimum |S| = {k}   ({s.nodes} nodes, {time.time() - t0:.1f}s)")
    for i in S:
        log("   " + " ".join(f"{x:+d}" if x else " 0" for x in vecs[i]))
    if a.verify_brute:
        t1 = time.time()
        bad = verify_rowsets(S, vecs, a.n, a.m, W=a.W)
        log(f"row-set check (distinct nonzero rows, sets of size <= m): " +
            ("OK" if bad is None else f"FAILED on rows {bad}") + f"  ({time.time() - t1:.1f}s)")
    if a.verify_naive:
        if a.m * a.n > 24:
            log("naive check skipped (mn > 24)")
        else:
            bad = brute_verify(S, vecs, a.n, a.m, W=a.W)
            log("naive check over all 2^(mn) ordered matrices: " +
                ("OK" if bad is None else f"FAILED on {bad}"))


if __name__ == "__main__":
    main()
