"""Two facts about conjugate-transport of lex-leader clauses.

(A) The order trap: transporting a complete break of the SAME row-swap by a
    column relabeling gives a break in a different reading order; conjoining the
    two empties an orbit -> UNSOUND.

(B) Where conjugate-covering of small-support breaks is/ isn't complete:
    - row-swap conjugates close into "sort rows" = complete for S_m (one axis);
    - row + column transposition conjugates give double-lex = incomplete for
      the full S_m x S_n.
"""
import itertools


def rows(X, m, n):
    return [tuple(X[i * n + j] for j in range(n)) for i in range(m)]


def cols(X, m, n):
    return [tuple(X[i * n + j] for i in range(m)) for j in range(n)]


def nondecr(seq):
    return all(a <= b for a, b in zip(seq, seq[1:]))


# ---------- (A) the order trap on 2x2, H = swap the two rows ----------
def trap():
    m = n = 2

    def rowswap(X):
        r = rows(X, m, n)
        return tuple(r[1] + r[0])

    def read(X, colorder):                       # row-major, columns reordered
        r = rows(X, m, n)
        return tuple(v for row in r for v in (row[c] for c in colorder))

    def keeps(X, colorder):                      # X is lex-min in its H-orbit
        return read(X, colorder) <= read(rowswap(X), colorder)

    order_LR, order_RL = (0, 1), (1, 0)          # two column reading orders
    seen, emptied = set(), []
    for X in itertools.product((0, 1), repeat=4):
        orbit = frozenset({X, rowswap(X)})
        if orbit in seen:
            continue
        seen.add(orbit)
        survivors = [Y for Y in orbit
                     if keeps(Y, order_LR) and keeps(Y, order_RL)]
        if not survivors:
            emptied.append(sorted(orbit))
    return emptied


# ---------- (B) completeness of conjugate-covering, 4x4 ----------
def coverage(m, n):
    sorted_rows = 0          # survivors of "sort rows" (row-swap conjugates)
    double_lex = 0           # survivors of "sort rows AND sort cols"
    for X in itertools.product((0, 1), repeat=m * n):
        r_ok = nondecr(rows(X, m, n))
        if r_ok:
            sorted_rows += 1
            if nondecr(cols(X, m, n)):
                double_lex += 1
    return sorted_rows, double_lex


def s_m_row_orbits(m, n):
    # orbits under row permutations only = multisets of m rows from 2^n options
    from math import comb
    return comb(2 ** n + m - 1, m)


def burnside(m, n):
    from math import gcd
    from collections import Counter
    from fractions import Fraction

    def parts(k, mx=None):
        mx = k if mx is None else mx
        if k == 0:
            yield ()
            return
        for t in range(min(k, mx), 0, -1):
            for rest in parts(k - t, t):
                yield (t,) + rest

    def z(p):
        zz = 1
        for v, mlt in Counter(p).items():
            f = 1
            for i in range(2, mlt + 1):
                f *= i
            zz *= (v ** mlt) * f
        return zz

    tot = Fraction(0)
    for lam in parts(m):
        for mu in parts(n):
            c = sum(gcd(a, b) for a in lam for b in mu)
            tot += Fraction(2 ** c, z(lam) * z(mu))
    return int(tot)


if __name__ == "__main__":
    print("(A) Order trap on 2x2, H = row swap:")
    for orb in trap():
        a, b = orb
        print(f"    orbit {{{a}, {b}}} has ZERO survivors under the two "
              f"column-reading orders -> unsound")
    print()

    print("(B) Conjugate-covering completeness, 4x4 binary matrices:")
    sr, dl = coverage(4, 4)
    print(f"    sort-rows survivors           = {sr:>6}   "
          f"(S_4 row-only orbits = {s_m_row_orbits(4, 4)})  -> complete for one axis")
    print(f"    double-lex survivors          = {dl:>6}   "
          f"(S_4 x S_4 orbits     = {burnside(4, 4)})  -> NOT complete for both")
