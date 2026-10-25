#!/usr/bin/env python3
"""Independent checks for orbit_count.py:
 1. brute-force orbit counts (union-find on all 2^{mn} matrices, generators only)
 2. Schreier-Sims stabilizer chain of G acting on the 2mn points (cell,value):
    group order, and Burnside summed over *every* element via transversal products
 3. part 2: brute-force (b) Stab_G(B)-orbits on B, and (a) G-orbits meeting B
 4. closed-form N(lam,mu) vs. explicit linear-system count
"""
import itertools, sys
from math import factorial
from orbit_count import (count_orbits, count_orbits_rowsums, n_parity_patterns,
                         partitions)


# ------------------------------------------------------------ Schreier-Sims
def compose(p, q):            # apply q first, then p
    return tuple(p[i] for i in q)

def inverse(p):
    r = [0] * len(p)
    for i, x in enumerate(p):
        r[x] = i
    return tuple(r)

def orbit_transversal(beta, gens, deg):
    ident = tuple(range(deg))
    T = {beta: ident}
    queue = [beta]
    for x in queue:
        for g in gens:
            y = g[x]
            if y not in T:
                T[y] = compose(g, T[x])
                queue.append(y)
    return T

class Chain:
    """Deterministic Schreier-Sims: levels[i] = (beta, gens, transversal)."""
    def __init__(self, gens, deg):
        self.deg = deg
        self.ident = tuple(range(deg))
        self.levels = []
        for g in gens:
            self.add(g)
        self.complete()

    def sift(self, g):
        for i, (beta, gens, T) in enumerate(self.levels):
            y = g[beta]
            if y not in T:
                return g, i
            g = compose(inverse(T[y]), g)
        return g, len(self.levels)

    def add(self, g):
        h, i = self.sift(g)
        if h == self.ident:
            return False
        if i == len(self.levels):
            beta = next(x for x in range(self.deg) if h[x] != x)
            self.levels.append((beta, [], {beta: self.ident}))
        for j in range(i + 1):        # h fixes base points before level i, so
            beta, gens, _ = self.levels[j]   # it belongs to every S_j, j<=i
            gens.append(h)
            self.levels[j] = (beta, gens, orbit_transversal(beta, gens, self.deg))
        return True

    def complete(self):
        changed = True
        while changed:
            changed = False
            i = len(self.levels) - 1
            while i >= 0:
                beta, gens, T = self.levels[i]
                for x, u in list(T.items()):
                    for s in list(gens):
                        y = s[x]
                        sg = compose(inverse(T[y]), compose(s, u))
                        # Schreier generator fixes beta; sift through deeper levels
                        h = sg
                        ok = True
                        for j in range(i + 1, len(self.levels)):
                            b2, _, T2 = self.levels[j]
                            z = h[b2]
                            if z not in T2:
                                ok = False
                                break
                            h = compose(inverse(T2[z]), h)
                        if ok and h == self.ident:
                            continue
                        # residue is nontrivial: add to the deeper part of the chain
                        if not ok or h != self.ident:
                            self.add_at(h, i + 1)
                            changed = True
                            beta, gens, T = self.levels[i]
                i -= 1

    def add_at(self, h, start):
        # h fixes base points 0..start-1; sift from `start`
        g = h
        j = start
        while j < len(self.levels):
            b2, _, T2 = self.levels[j]
            z = g[b2]
            if z not in T2:
                break
            g = compose(inverse(T2[z]), g)
            j += 1
        if g == self.ident:
            return
        if j == len(self.levels):
            beta = next(x for x in range(self.deg) if g[x] != x)
            self.levels.append((beta, [], {beta: self.ident}))
        for k in range(start, j + 1):
            beta, gens, _ = self.levels[k]
            gens.append(g)
            self.levels[k] = (beta, gens, orbit_transversal(beta, gens, self.deg))

    def order(self):
        o = 1
        for _, _, T in self.levels:
            o *= len(T)
        return o

    def elements(self):
        trans = [list(T.values()) for _, _, T in self.levels]
        for combo in itertools.product(*trans[::-1]):   # deepest first
            g = self.ident
            for u in combo:
                g = compose(u, g)      # g = u_1 o u_2 o ... o u_k
            yield g


# ------------------------------------------------------------ the group on points
def point(i, j, v, n):
    return 2 * (i * n + j) + v

def group_generators(m, n):
    deg = 2 * m * n
    def build(fn):
        p = [0] * deg
        for i in range(m):
            for j in range(n):
                for v in range(2):
                    (i2, j2, v2_) = fn(i, j, v)
                    p[point(i, j, v, n)] = point(i2, j2, v2_, n)
        return tuple(p)
    gens = []
    if m >= 2:
        gens.append(build(lambda i, j, v: ({0: 1, 1: 0}.get(i, i), j, v)))
        gens.append(build(lambda i, j, v: ((i + 1) % m, j, v)))
    if n >= 2:
        gens.append(build(lambda i, j, v: (i, {0: 1, 1: 0}.get(j, j), v)))
        gens.append(build(lambda i, j, v: (i, (j + 1) % n, v)))
    gens.append(build(lambda i, j, v: (i, j, v ^ 1 if i == 0 else v)))
    gens.append(build(lambda i, j, v: (i, j, v ^ 1 if j == 0 else v)))
    return gens

def fixed_matrices(perm, m, n):
    """Number of 0/1 matrices fixed by a point permutation of the cell-value set."""
    ncell = m * n
    seen = [False] * ncell
    cycles = 0
    for c in range(ncell):
        if seen[c]:
            continue
        x = 2 * c
        y = perm[x]
        seen[c] = True
        while y // 2 != c:
            seen[y // 2] = True
            y = perm[y]
        if y != x:          # returned to the other value: odd flip on this cycle
            return 0
        cycles += 1
    return 1 << cycles

def apply_to_matrix(perm, M):
    """M: tuple of 0/1 per cell.  Returns g.M."""
    out = [0] * len(M)
    for c, v in enumerate(M):
        y = perm[2 * c + v]
        out[y // 2] = y % 2
    return tuple(out)


# ------------------------------------------------------------ checks
def brute_orbits(m, n):
    gens = [g for g in group_generators(m, n)]
    N = 1 << (m * n)
    parent = list(range(N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def bits(x):
        return tuple((x >> c) & 1 for c in range(m * n))
    def toint(M):
        return sum(v << c for c, v in enumerate(M))
    for x in range(N):
        M = bits(x)
        for g in gens:
            y = toint(apply_to_matrix(g, M))
            a, b = find(x), find(y)
            if a != b:
                parent[a] = b
    return len({find(x) for x in range(N)})

def check_part1():
    print("== part 1: formula vs brute force vs Schreier-Sims Burnside")
    for m, n in [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3), (2, 4), (3, 4), (1, 5), (2, 5), (4, 4)]:
        f = count_orbits(m, n)
        b = brute_orbits(m, n) if m * n <= 16 else None
        s = None
        ch = Chain(group_generators(m, n), 2 * m * n)
        order_ok = ch.order() == factorial(m) * factorial(n) * 2 ** (m + n - 1)
        if ch.order() <= 400000:
            tot = sum(fixed_matrices(g, m, n) for g in ch.elements())
            assert tot % ch.order() == 0
            s = tot // ch.order()
        print(f"m={m} n={n}: formula={f} brute={b} SS-Burnside={s} |G|-ok={order_ok}")
        assert order_ok and (b is None or b == f) and (s is None or s == f)

def check_N():
    print("== closed form N(lam,mu) vs explicit F_2 system")
    from math import gcd
    for m in range(1, 8):
        for n in range(1, 8):
            for lam in partitions(m):
                for mu in partitions(n):
                    cnt = 0
                    for A in itertools.product((0, 1), repeat=len(lam)):
                        for B in itertools.product((0, 1), repeat=len(mu)):
                            ok = all(((q // gcd(p, q)) * a + (p // gcd(p, q)) * b) % 2 == 0
                                     for p, a in zip(lam, A) for q, b in zip(mu, B))
                            cnt += ok
                    assert cnt == n_parity_patterns(lam, mu), (lam, mu)
    print("ok")

def check_part2():
    print("== part 2: (b) Stab_G(B)-orbits on B  and  (a) G-orbits meeting B")
    cases = []
    for m, n in [(1, 4), (2, 3), (2, 4), (3, 3), (3, 4)]:
        for r in itertools.product(range(n + 1), repeat=m):
            if list(r) == sorted(r):
                cases.append((m, n, r))
    chains = {}
    bad = 0
    for m, n, r in cases:
        if (m, n) not in chains:
            ch = Chain(group_generators(m, n), 2 * m * n)
            chains[(m, n)] = list(ch.elements())
        G = chains[(m, n)]
        allM = [tuple((x >> c) & 1 for c in range(m * n)) for x in range(1 << (m * n))]
        B = [M for M in allM if all(sum(M[i * n:(i + 1) * n]) == r[i] for i in range(m))]
        Bset = set(B)
        H = [g for g in G if all(apply_to_matrix(g, M) in Bset for M in B)]
        # (b) orbits of H on B (union-find)
        idx = {M: k for k, M in enumerate(B)}
        parent = list(range(len(B)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for M in B:
            for g in H:
                a, b_ = find(idx[M]), find(idx[apply_to_matrix(g, M)])
                if a != b_:
                    parent[a] = b_
        ob = len({find(k) for k in range(len(B))})
        # (a) G-orbits meeting B
        oa = len({min(apply_to_matrix(g, M) for g in G) for M in B})
        f = count_orbits_rowsums(n, r)
        flag = "" if f == ob else "  <-- MISMATCH"
        if f != ob:
            bad += 1
        print(f"m={m} n={n} r={r}: |B|={len(B)} |H|={len(H)} formula={f} brute(b)={ob} meets-B(a)={oa}{flag}")
    assert bad == 0

if __name__ == "__main__":
    check_N()
    check_part1()
    check_part2()
