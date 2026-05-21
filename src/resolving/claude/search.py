"""
For each n, find:
  m*(n) = min m such that an m x n balanced detecting matrix exists
  m*_full(n) = min m such that an m x n FULL balanced detecting matrix exists
Conjecture says these are equal.
"""
import itertools
from itertools import combinations, product

def balanced_vectors(n):
    out = []
    for v in product((-1, 0, 1), repeat=n):
        if sum(v) == 0 and any(v):
            out.append(v)
    return out

def is_balanced_detecting(rows, balanced_xs):
    for x in balanced_xs:
        ok = False
        for r in rows:
            s = 0
            for i in range(len(x)):
                s += r[i]*x[i]
            if s != 0:
                ok = True
                break
        if not ok:
            return False
    return True

def all_rows(n):
    return list(product((0,1), repeat=n))

def full_rows(n):
    w = n // 2
    return [tuple(1 if i in s else 0 for i in range(n))
            for s in combinations(range(n), w)]

def find_min_m(n, row_pool, balanced_xs, max_m):
    for m in range(1, max_m+1):
        for choice in combinations(row_pool, m):
            if is_balanced_detecting(choice, balanced_xs):
                return m, choice
    return None, None

def search(n_max):
    for n in range(2, n_max+1):
        bx = balanced_vectors(n)
        print(f"n={n}: |B^(n)|={len(bx)}", flush=True)
        fr = full_rows(n)
        # For unrestricted we still drop the all-zero row; complement-normalize.
        ar_norm = [r for r in all_rows(n) if sum(r) <= n // 2 and any(r)]
        m_full, choice_full = find_min_m(n, fr, bx, max_m=n+1)
        m_unr, choice_unr = find_min_m(n, ar_norm, bx, max_m=n+1)
        print(f"  m*_unrestricted={m_unr}, m*_full={m_full}", flush=True)
        if m_unr != m_full:
            print(f"  *** COUNTEREXAMPLE at n={n} ***")
            print(f"  unrestricted matrix at m={m_unr}: {choice_unr}")

search(7)
