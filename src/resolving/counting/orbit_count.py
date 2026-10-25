#!/usr/bin/env python3
"""
Exact orbit counts for m x n 0/1 matrices under row/column permutations and
row/column complementation.  See orbit_count_derivation.md for the proofs.

  python orbit_count.py table 16          # part 1, all 1<=m<n<=16
  python orbit_count.py count 5 7         # part 1, single (m,n)
  python orbit_count.py rows 10 5 5 4 6 3 # part 2: n=10, row sums r = 5 5 4 6 3
"""
import sys
from math import gcd, factorial
from functools import lru_cache
from collections import Counter


# ----------------------------------------------------------------- partitions
@lru_cache(maxsize=None)
def partitions(n, maxpart=None):
    """All partitions of n as tuples (non-increasing)."""
    if maxpart is None:
        maxpart = n
    if n == 0:
        return [()]
    out = []
    for k in range(min(n, maxpart), 0, -1):
        for rest in partitions(n - k, k):
            out.append((k,) + rest)
    return out


def z_of(lam):
    """|centralizer| in S_n of an element of cycle type lam."""
    z = 1
    for k, mk in Counter(lam).items():
        z *= k ** mk * factorial(mk)
    return z


def v2(x):
    return (x & -x).bit_length() - 1


# ----------------------------------------------------------------- part 1
def n_parity_patterns(lam, mu):
    """Number of (A,B) in F_2^{cycles(lam)} x F_2^{cycles(mu)} with
    (q/g)A_C + (p/g)B_D = 0 for all cycle pairs.  Closed form by 2-adic level."""
    R, K = Counter(v2(p) for p in lam), Counter(v2(q) for q in mu)
    smin, tmin = min(R), min(K)
    N = 1
    for v in set(R) | set(K):
        r, k = R.get(v, 0), K.get(v, 0)
        if r and not k:
            if tmin >= v:
                N *= 1 << r
        elif k and not r:
            if smin >= v:
                N *= 1 << k
        else:
            if smin >= v and tmin >= v:
                N *= 2
    return N


def count_orbits(m, n):
    """Number of orbits of G on all m x n 0/1 matrices (exact integer)."""
    num = 0
    fm, fn = factorial(m), factorial(n)
    P, Q = partitions(m), partitions(n)
    zP = [fm // z_of(l) for l in P]
    zQ = [fn // z_of(u) for u in Q]
    for l, cl in zip(P, zP):
        for u, cu in zip(Q, zQ):
            gs = sum(gcd(p, q) for p in l for q in u)
            e = gs + m + n - len(l) - len(u)
            num += cl * cu * n_parity_patterns(l, u) * (1 << e)
    den = fm * fn * (1 << (m + n))
    q, r = divmod(num, den)
    assert r == 0, "Burnside sum not divisible: bug"
    return q


# ----------------------------------------------------------------- part 2
def _poly_mul(a, b, deg):
    out = [0] * (deg + 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if i + j > deg:
                    break
                out[i + j] += x * y
    return out


def _phi(mu, p, w, n):
    """Contribution of one row cycle of length p (base row weight w) given the
    column cycle type mu:  sum over parity A of #(a-vectors with sum A) * #x."""
    # A = 0:  [t^w] prod_D (1 + t^{q/g})^g
    poly = [1] + [0] * w
    for q in mu:
        g = gcd(p, q)
        l = q // g
        fac = [0] * (w + 1)
        # (1+t^l)^g
        from math import comb
        for k in range(g + 1):
            if k * l <= w:
                fac[k * l] += comb(g, k)
        poly = _poly_mul(poly, fac, w)
    f0 = poly[w]
    if 2 * w < n:
        return f0                      # a_i = 0 forced
    # 2w == n : both parities of A, each realised by 2^{p-1} vectors a
    f1 = 0
    if all((q // gcd(p, q)) % 2 == 0 for q in mu):
        f1 = 1 << sum(gcd(p, q) for q in mu)
    return (1 << (p - 1)) * (f0 + f1)


def _T(phis, u):
    """T_u = sum_{sigma in S_u} prod_{cycles} phi(len), via cycle of element 1."""
    T = [1] + [0] * u
    for k in range(1, u + 1):
        s, ff = 0, 1                   # ff = (k-1)!/(k-p)!
        for p in range(1, k + 1):
            if p > 1:
                ff *= (k - p + 1)
            s += ff * phis[p] * T[k - p]
        T[k] = s
    return T[u]


def count_orbits_rowsums(n, r):
    """Number of orbits of Stab_G(B) on B, B = {M : row i sums to r[i]}.
    (B itself is not G-invariant; see derivation section 2.)"""
    r = list(r)
    assert all(0 <= x <= n for x in r)
    U = Counter(min(x, n - x) for x in r)          # class sizes u_w
    Hord = factorial(n)
    for w, u in U.items():
        Hord *= factorial(u) * (2 ** u if 2 * w == n else 1)
    fn = factorial(n)
    total = 0
    for mu in partitions(n):
        prod = 1
        for w, u in U.items():
            phis = [0] + [_phi(mu, p, w, n) for p in range(1, u + 1)]
            prod *= _T(phis, u)
            if prod == 0:
                break
        total += (fn // z_of(mu)) * prod
    q, rem = divmod(total, Hord)
    assert rem == 0, "Burnside sum not divisible: bug"
    return q


# ----------------------------------------------------------------- CLI
def main(argv):
    if len(argv) >= 3 and argv[1] == "table":
        N = int(argv[2])
        for n in range(2, N + 1):
            for m in range(1, n):
                print(f"m={m:2d} n={n:2d}  {count_orbits(m, n)}", flush=True)
    elif len(argv) == 4 and argv[1] == "count":
        print(count_orbits(int(argv[2]), int(argv[3])))
    elif len(argv) >= 4 and argv[1] == "rows":
        n = int(argv[2])
        print(count_orbits_rowsums(n, [int(x) for x in argv[3:]]))
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
