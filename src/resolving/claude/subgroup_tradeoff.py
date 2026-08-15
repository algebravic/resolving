"""
Trade-off for a transitive subgroup with all 1-dimensional irreps:
the toroidal C_m x C_n (independent cyclic shift of rows and columns),
compared against the full S_m x S_n, on m-by-n binary matrices.

We report, for each group:
  N   = Burnside orbit count
  p1  = number of independent degree-1 invariants (always 2)
  p2  = number of independent degree-2 invariants
        (for C_m x C_n these are the 2-D autocorrelations -- ~mn/2 of them;
         for S_m x S_n there are only 3)
  d*  = exact least degree at which <=d-edge subgraph counts separate orbits
        (brute force; small cases only)
"""
import itertools
from fractions import Fraction
from math import gcd, lcm
from collections import Counter

from symmetry_ceiling import analyze, _series_mul_binomial, separating_degree


def cyclic_counts(m, n):
    """Burnside N and Boolean degree spectrum p_e for C_m x C_n on cells."""
    mn = m * n
    boolean = [Fraction(0)] * (mn + 1)
    w = Fraction(1, mn)
    for a in range(m):
        for b in range(n):
            L = lcm(m // gcd(a, m), n // gcd(b, n))
            ncyc = mn // L
            bp = [Fraction(0)] * (mn + 1)
            bp[0] = Fraction(1)
            bp = _series_mul_binomial(bp, L, ncyc, mn)
            for d in range(mn + 1):
                boolean[d] += w * bp[d]
    boolean = [int(x) for x in boolean]
    return sum(boolean), boolean


def cyclic_group_perms(m, n):
    def idx(i, j):
        return i * n + j
    perms = []
    for a in range(m):
        for b in range(n):
            perms.append(tuple(idx((i + a) % m, (j + b) % n)
                               for i in range(m) for j in range(n)))
    return perms


def separating_degree_general(m, n, group_perms):
    """Least d s.t. <=d-subset orbit-count signatures separate full-matrix
    orbits, for an arbitrary group given as cell permutations."""
    ncells = m * n
    cache = {}

    def canon(S):
        S = frozenset(S)
        if S in cache:
            return cache[S]
        best = min(tuple(sorted(p[k] for k in S)) for p in group_perms)
        cache[S] = best
        return best

    seen, reps = set(), []
    for r in range(ncells + 1):
        for S in itertools.combinations(range(ncells), r):
            c = canon(S)
            if c not in seen:
                seen.add(c)
                reps.append(frozenset(S))

    def signature(support, d):
        supp = list(support)
        return tuple(tuple(sorted(Counter(canon(T)
                     for T in itertools.combinations(supp, e)).items()))
                     for e in range(d + 1))

    for d in range(ncells + 1):
        if len({signature(R, d) for R in reps}) == len(reps):
            return d
    return ncells


if __name__ == "__main__":
    print("group            m x n |       orbits N |  p1 |   p2 |   d*")
    print("-" * 64)
    for (m, n) in [(3, 3), (4, 4), (5, 5), (6, 6)]:
        Ns, _mol_s, boo_s = analyze(m, n)
        Nc, boo_c = cyclic_counts(m, n)
        # d* only for the cheap-to-brute cases
        dc = (separating_degree_general(m, n, cyclic_group_perms(m, n))
              if m * n <= 16 else None)
        ds = (separating_degree(m, n)[0] if m * n <= 12 else None)
        ds_str = str(ds) if ds is not None else " . "
        dc_str = str(dc) if dc is not None else " . "
        print(f"S_{m} x S_{n}        {m}x{n}  | {Ns:>13,} | {boo_s[1]:>3} | {boo_s[2]:>4} | {ds_str:>4}")
        print(f"C_{m} x C_{n}        {m}x{n}  | {Nc:>13,} | {boo_c[1]:>3} | {boo_c[2]:>4} | {dc_str:>4}")
        print("-" * 64)
    print()
    print("(d* = exact separating degree; '.' = not brute-forced at this size)")
    print("note: C_m x C_n order = mn; S_m x S_n order = m! n!")
    for (m, n) in [(4, 4), (6, 6)]:
        import math
        print(f"  {m}x{n}: |C|={m*n},  |S|={math.factorial(m)*math.factorial(n):,}"
              f"  (ratio {math.factorial(m)*math.factorial(n)//(m*n):,}x)")
