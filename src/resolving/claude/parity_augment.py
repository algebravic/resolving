#!/usr/bin/env python3
"""
parity_augment.py -- CEGAR-compatible parity augmentation library.
====================================================================

Purpose: augment an existing CEGAR-based BD search with parity variables.

The idea: rather than emit millions of r_{i,x} variables upfront, the CEGAR
conflictor already introduces r_{i,x} lazily (one per undetected x it finds).
This library plugs into that flow:

  When the conflictor introduces r_{i,x} for a new x, call
      augment_r_variable(encoder, i, x_tuple)
  which:
    1. Identifies the support S = supp(x).
    2. Generates (or reuses) parity variable P_{i,S} via XOR chain over
       {A_{i,j} : j in S}.
    3. Adds the implication clause  P_{i,S} -> r_{i,x}.

The augmentation is INCREMENTAL and SHARED: parity variables and XOR chain
clauses are generated once per (i, S) and reused for all x with support S.

For a set of currently-active x's, the total overhead is bounded by:
  - # parity vars = m * (# distinct supports seen)
  - # XOR chain clauses = 4 * m * sum_S |S| for supports seen
  - # implication clauses = 1 per (i, x) pair introduced

TYPICAL USAGE (pseudo-code integrating with existing CEGAR):

    aug = ParityAugmenter(m, n, var_A_func, new_var_func, add_clause_func)
    
    # In CEGAR loop, when conflictor returns undetected x:
    r_vars = introduce_r_variables_as_usual(x)
    coverage_clause = tuple(r_vars) + (0,)  # your existing coverage clause
    add_clause(coverage_clause)
    
    # Then augment with parity:
    for i in range(m):
        aug.augment(i, x, r_vars[i])
    
    # The augmenter has now added parity vars, XOR clauses, and P -> r implications.

The augmenter is meant to be a small drop-in that doesn't change the CEGAR
architecture, only enriches the SAT solver's atom vocabulary.
"""

from typing import Callable, Iterable


class ParityAugmenter:
    """Adds parity augmentation to an existing SAT encoding.

    Parameters:
      m, n:           matrix dimensions
      var_A:          callable (i, j) -> SAT var id for A_{i,j}
      alloc_var:      callable () -> new SAT var id
      add_clause:     callable (tuple of ints) -> None
      max_support:    ignore supports larger than this (default: n)
    """

    def __init__(self, m, n, var_A: Callable[[int, int], int],
                 alloc_var: Callable[[], int],
                 add_clause: Callable[[Iterable[int]], None],
                 max_support: int = None):
        self.m = m
        self.n = n
        self.var_A = var_A
        self.alloc_var = alloc_var
        self.add_clause = add_clause
        self.max_support = max_support if max_support is not None else n

        # Memoize: (i, tuple(S)) -> parity var
        self._P_cache = {}

        # Stats
        self.n_parity_vars_created = 0
        self.n_xor_clauses = 0
        self.n_implication_clauses = 0

    def _emit_xor(self, a: int, b: int, c: int):
        """Emit c = a XOR b as 4 CNF clauses."""
        self.add_clause((-a, -b, -c))
        self.add_clause((-a, b, c))
        self.add_clause((a, -b, c))
        self.add_clause((a, b, -c))
        self.n_xor_clauses += 4

    def parity_var(self, i: int, S: tuple) -> int:
        """Return the SAT var id for P_{i, S}, creating it if needed.

        P_{i, S} is defined as the XOR of {A_{i, j} : j in S}, encoded via
        a chain of aux vars. This method is idempotent per (i, S).
        """
        key = (i, tuple(sorted(S)))
        if key in self._P_cache:
            return self._P_cache[key]

        S_sorted = list(sorted(S))
        w = len(S_sorted)
        assert w >= 2

        # Chain: q_1 = A_{i, S[0]}, q_{k+1} = q_k XOR A_{i, S[k]}, P = q_w.
        prev = self.var_A(i, S_sorted[0])
        for k in range(1, w):
            A_var = self.var_A(i, S_sorted[k])
            q_new = self.alloc_var()
            self.n_parity_vars_created += 1
            self._emit_xor(prev, A_var, q_new)
            prev = q_new

        self._P_cache[key] = prev
        return prev

    def augment(self, i: int, x: tuple, r_var: int):
        """Add parity augmentation for row i, balanced vector x, given the
        already-allocated r_{i,x} var.

        Effect: adds the clause P_{i, supp(x)} -> r_{i,x}, which is valid
        because if row i's entries on supp(x) have odd sum, A_i . x is odd
        and hence nonzero, so r_{i,x} should be 1.

        Creates parity chain for (i, supp(x)) if not already cached.
        """
        S = tuple(idx for idx, v in enumerate(x) if v != 0)
        if len(S) > self.max_support:
            return                              # skip large supports
        if len(S) < 2:
            return                              # trivial: weight-1 is not balanced anyway
        P = self.parity_var(i, S)
        # Clause: (~P, r_var)  i.e. P -> r
        self.add_clause((-P, r_var))
        self.n_implication_clauses += 1

    def augment_batch(self, i: int, x: tuple, r_vars_by_row: list):
        """Convenience: augment for all rows for a given x.

        r_vars_by_row[i] = SAT var id for r_{i, x}, for each i in [m].
        """
        for i in range(self.m):
            self.augment(i, x, r_vars_by_row[i])

    def stats(self):
        return {
            'parity_vars_created': self.n_parity_vars_created,
            'xor_chain_clauses': self.n_xor_clauses,
            'implication_clauses': self.n_implication_clauses,
            'distinct_parity_atoms': len(self._P_cache),
        }


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Simulate integration with a hypothetical CEGAR loop
    print("Simulating parity augmentation on a synthetic CEGAR trace...")

    m, n = 7, 14
    # A_{i,j} vars: 1 .. m*n
    def var_A(i, j):
        return 1 + i * n + j

    next_var = [1 + m * n]
    clauses = []

    def alloc():
        v = next_var[0]
        next_var[0] += 1
        return v

    def add_clause(cl):
        clauses.append(tuple(cl))

    aug = ParityAugmenter(m, n, var_A, alloc, add_clause, max_support=6)

    # Simulate the CEGAR conflictor introducing r_{i,x} for a few undetected x's.
    # Each x we introduce would come from the conflictor. Here we pick a handful.
    import random
    random.seed(0)

    # Generate a few random balanced x's
    from itertools import combinations
    def gen_random_balanced_x(n, weight):
        S = random.sample(range(n), weight)
        pos = set(random.sample(S, weight // 2))
        x = tuple(1 if j in pos else -1 if j in S else 0 for j in range(n))
        # Normalize first-nonzero-positive
        first_nz = next(v for v in x if v != 0)
        if first_nz < 0:
            x = tuple(-v for v in x)
        return x

    trace = []
    for _ in range(50):
        w = random.choice([2, 4, 4, 6])         # bias toward small
        x = gen_random_balanced_x(n, w)
        # Introduce r vars for this x
        r_vars = [alloc() for _ in range(m)]
        # Coverage
        add_clause(tuple(r_vars))
        # Augment
        aug.augment_batch(0, x, r_vars)         # apparently we're only augmenting row 0
        # Actually augment all rows:
        for i in range(m):
            aug.augment(i, x, r_vars[i])
        trace.append(x)

    stats = aug.stats()
    print(f"\nAfter {len(trace)} x's introduced by conflictor:")
    print(f"  Total SAT variables allocated: {next_var[0] - 1}")
    print(f"    Primary A: {m * n}")
    print(f"    r_{{i,x}}: {len(trace) * m}")
    print(f"    Parity chain + P: {stats['parity_vars_created']}")
    print(f"  Total clauses added: {len(clauses)}")
    print(f"    XOR chain: {stats['xor_chain_clauses']}")
    print(f"    Implication (P -> r): {stats['implication_clauses']}")
    print(f"    Coverage: {len(trace)}")
    print(f"  Distinct parity atoms P_{{i,S}}: {stats['distinct_parity_atoms']}")
    print(f"    (equals m * # distinct supports seen)")

    n_distinct_supports = stats['distinct_parity_atoms'] // m
    print(f"    Distinct supports seen: {n_distinct_supports} / {len(trace)} x's")

    print(f"\nScaling extrapolation for a full CEGAR run with ~10K x's introduced:")
    est_r_vars = 10000 * m
    est_parity_vars = int(stats['parity_vars_created'] * 10000 / len(trace))
    est_clauses = int(len(clauses) * 10000 / len(trace))
    print(f"  Estimated r vars: {est_r_vars}")
    print(f"  Estimated parity aux vars: {est_parity_vars}")
    print(f"  Estimated clauses: {est_clauses}")
