#!/usr/bin/env python3
"""
SAT/PB encoding of the matrix necessary condition for balanced-detecting.
==========================================================================

Encodes v^T (Phi(A) - XX^T) v >= 0 for a family of test vectors v as
pseudo-Boolean constraints over the variables:
  - Primary: A_{i,j} in {0, 1} for i in [m], j in [n]
  - Auxiliary: z_{i,p,q} = A_{i,p} AND A_{i,q} for i in [m], 0 <= p < q < n

Since PHI(A)[j,k] = sum_i sum_{p,q} T4[j,k,p,q] A_{i,p} A_{i,q}, we have
  v^T Phi(A) v = sum_i [sum_p c_{p,p}(v) A_{i,p} + sum_{p<q} 2 c_{p,q}(v) z_{i,p,q}]
where c_{p,q}(v) = sum_{j,k} v_j v_k T4[j,k,p,q].

The constraint  v^T Phi v >= v^T XX^T v = K(v)  becomes a PB inequality
in {A_{i,j}, z_{i,p,q}}, with the auxiliary variables defined by Tseitin
clauses  z_{i,p,q} <-> A_{i,p} AND A_{i,q}.

OUTPUT FORMAT: OPB (Pseudo-Boolean) format, compatible with RoundingSat,
SAT4J, and Naps. Constraint syntax: `<coefs>*<var> ... >= K ;`

Reference: OPB format spec at http://www.cril.univ-artois.fr/PB16/format.pdf

USAGE:
    python3 sat_encode_matrix.py --m 7 --n 14 --output beta14.opb
    python3 sat_encode_matrix.py --m 3 --n 6 --output beta6.opb --test-vectors e_j,e_j-e_k
"""

import argparse
import sys
from itertools import combinations
from collections import Counter
from math import comb
import numpy as np


# ---------------------------------------------------------------------------
# Enumerate B(n) and build T4 tensor
# ---------------------------------------------------------------------------

def enum_balanced(n):
    """One representative per +/- pair of nonzero balanced x in {-1,0,1}^n."""
    out = []
    for k in range(1, n // 2 + 1):
        for support in combinations(range(n), 2 * k):
            first = support[0]
            rest = support[1:]
            for pos_ex in combinations(rest, k - 1):
                x = np.zeros(n, dtype=np.int8)
                x[first] = 1
                for p in pos_ex:
                    x[p] = 1
                for p in rest:
                    if p not in pos_ex:
                        x[p] = -1
                out.append(x)
    return out


def T4_pattern_values(n):
    """Return dict from multiplicity pattern -> value of T4[i,j,k,l]."""
    B = enum_balanced(n)
    Vs = {
        (4,):    sum(int(x[0])**2 for x in B),
        (3, 1):  sum(int(x[0])**3 * int(x[1]) for x in B),
        (2, 2):  sum(int(x[0])**2 * int(x[1])**2 for x in B),
        (2, 1, 1): sum(int(x[0])**2 * int(x[1]) * int(x[2]) for x in B),
        (1, 1, 1, 1): sum(int(x[0]) * int(x[1]) * int(x[2]) * int(x[3]) for x in B)
                       if n >= 4 else 0,
    }
    return Vs, len(B)


def build_T4(n, Vs):
    T4 = np.zeros((n, n, n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    pat = tuple(sorted(Counter((i, j, k, l)).values(), reverse=True))
                    T4[i, j, k, l] = Vs[pat]
    return T4


def build_XXT(n, Vs):
    alpha = Vs[(4,)]
    beta = -alpha // (n - 1) if n > 1 else 0
    XXT = np.full((n, n), beta, dtype=np.int64)
    np.fill_diagonal(XXT, alpha)
    return XXT


# ---------------------------------------------------------------------------
# Test vector families
# ---------------------------------------------------------------------------

def test_vectors_ej(n):
    """Family A: v = e_j."""
    for j in range(n):
        v = np.zeros(n, dtype=np.int64)
        v[j] = 1
        yield f"e_{j}", v


def test_vectors_ej_ek(n):
    """Family C: v = e_j - e_k for j < k."""
    for j, k in combinations(range(n), 2):
        v = np.zeros(n, dtype=np.int64)
        v[j] = 1
        v[k] = -1
        yield f"e_{j}-e_{k}", v


def test_vectors_ej_plus_ek(n):
    """Family B: v = e_j + e_k."""
    for j, k in combinations(range(n), 2):
        v = np.zeros(n, dtype=np.int64)
        v[j] = 1
        v[k] = 1
        yield f"e_{j}+e_{k}", v


def test_vectors_pair_pair(n):
    """Family E: v = e_a + e_b - e_c - e_d for {a,b} disjoint from {c,d}.
    These are balanced weight-4 directions.
    """
    from itertools import combinations
    for ab in combinations(range(n), 2):
        for cd in combinations(range(n), 2):
            if set(ab) & set(cd): continue
            if ab >= cd: continue                 # dedupe (a,b) vs (c,d) swap
            a, b = ab; c, d = cd
            v = np.zeros(n, dtype=np.int64)
            v[a] = 1; v[b] = 1; v[c] = -1; v[d] = -1
            yield f"e_{a}+e_{b}-e_{c}-e_{d}", v


TEST_VECTOR_FAMILIES = {
    "e_j":         test_vectors_ej,
    "e_j+e_k":     test_vectors_ej_plus_ek,
    "e_j-e_k":     test_vectors_ej_ek,
    "pair_pair":   test_vectors_pair_pair,
}


# ---------------------------------------------------------------------------
# Compute coefficient vector for a test vector
# ---------------------------------------------------------------------------

def coefficients_for_v(v, T4):
    """Return c[p,q] = sum_{j,k} v_j v_k T4[j,k,p,q]. Symmetric in (p,q)."""
    return np.einsum('j,k,jkpq->pq', v, v, T4)


def rhs_for_v(v, XXT):
    return int(v @ XXT @ v)


# ---------------------------------------------------------------------------
# OPB encoder
# ---------------------------------------------------------------------------

class OPBEncoder:
    """Emit OPB (Pseudo-Boolean) constraints.

    Variable numbering (1-indexed for OPB):
        A[i,j]   -> x{1 + i*n + j}                              for i in [m], j in [n]
        z[i,p,q] -> x{1 + m*n + i*C(n,2) + pair_index(p,q,n)}   for i in [m], p<q
    """

    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.num_primary = m * n
        self.pair_idx = {}
        idx = 0
        for p in range(n):
            for q in range(p + 1, n):
                self.pair_idx[(p, q)] = idx
                idx += 1
        self.num_pairs = idx                                     # = C(n, 2)
        self.num_aux = m * self.num_pairs
        self.total_vars = self.num_primary + self.num_aux
        self.constraints = []                                    # list of str
        self.num_constraints_definitional = 0
        self.num_constraints_pb = 0

    def var_A(self, i, j):
        return 1 + i * self.n + j

    def var_z(self, i, p, q):
        if p > q:
            p, q = q, p
        pi = self.pair_idx[(p, q)]
        return 1 + self.num_primary + i * self.num_pairs + pi

    def add_definitional_clauses(self):
        """z_{i,p,q} <-> A_{i,p} AND A_{i,q}.

        Encoded as three PB constraints (equivalent to 3 clauses):
          (1) A_{i,p} + A_{i,q} - z_{i,p,q} <= 1     ->   +1 A_ip +1 A_iq -1 z >= -1 (equivalent)
          Actually written cleaner as clauses:
              z -> A_ip:            (-z + A_ip >= 0)    i.e. +1 A_ip -1 z >= 0
              z -> A_iq:            (-z + A_iq >= 0)    i.e. +1 A_iq -1 z >= 0
              A_ip & A_iq -> z:     (A_ip + A_iq -> z)  i.e. -1 A_ip -1 A_iq +1 z >= -1
        """
        for i in range(self.m):
            for p in range(self.n):
                for q in range(p + 1, self.n):
                    Ap = self.var_A(i, p)
                    Aq = self.var_A(i, q)
                    z  = self.var_z(i, p, q)
                    # z -> A_ip: A_ip - z >= 0
                    self.constraints.append(f"+1 x{Ap} -1 x{z} >= 0 ;")
                    # z -> A_iq: A_iq - z >= 0
                    self.constraints.append(f"+1 x{Aq} -1 x{z} >= 0 ;")
                    # A_ip & A_iq -> z: -A_ip - A_iq + z >= -1
                    self.constraints.append(f"-1 x{Ap} -1 x{Aq} +1 x{z} >= -1 ;")
                    self.num_constraints_definitional += 3

    def add_matrix_constraint(self, v, name, T4, XXT):
        """Add PB constraint v^T Phi(A) v >= v^T XX^T v.

        Structure:
          sum_i sum_p c[p,p] A_{i,p} + sum_i sum_{p<q} 2*c[p,q] z_{i,p,q} >= K
        """
        c = coefficients_for_v(v, T4)                              # n x n int matrix
        K = rhs_for_v(v, XXT)                                     # int
        terms = []
        for i in range(self.m):
            for p in range(self.n):
                cpp = int(c[p, p])
                if cpp != 0:
                    terms.append((cpp, self.var_A(i, p)))
            for p in range(self.n):
                for q in range(p + 1, self.n):
                    cpq = int(c[p, q]) + int(c[q, p])              # 2 * c[p,q] by symmetry
                    if cpq != 0:
                        terms.append((cpq, self.var_z(i, p, q)))
        if not terms:
            return
        line = ""
        for coef, var in terms:
            sign = "+" if coef >= 0 else "-"
            line += f" {sign}{abs(coef)} x{var}"
        line += f" >= {K} ;"
        line = f"* test vector: {name}\n" + line.lstrip()
        self.constraints.append(line)
        self.num_constraints_pb += 1

    def add_boolean_constraints(self, row_cap=None, no_zero_col=True):
        """Optional: extra 0/1 side constraints often used in balanced-detecting search."""
        # Row weight cap
        if row_cap is not None:
            for i in range(self.m):
                terms = " ".join(f"+1 x{self.var_A(i,j)}" for j in range(self.n))
                self.constraints.append(f"* row {i} weight cap")
                self.constraints.append(f"{terms} <= {row_cap} ;")
                self.num_constraints_pb += 1
        # No zero column
        if no_zero_col:
            for j in range(self.n):
                terms = " ".join(f"+1 x{self.var_A(i,j)}" for i in range(self.m))
                self.constraints.append(f"* col {j} nonempty")
                self.constraints.append(f"{terms} >= 1 ;")
                self.num_constraints_pb += 1

    def add_perxconstraints(self, B_n):
        """For comparison: encode per-x constraints ||A x||^2 >= 1 as PB.

        Since ||A x||^2 = sum_i (A_i . x)^2, and this is sum_i sum_{p,q} x_p x_q A_{i,p} A_{i,q},
        the per-x constraint is:
          sum_i [sum_p x_p^2 A_{i,p} + sum_{p<q} 2 x_p x_q z_{i,p,q}] >= 1

        This is exactly the v = x case of our test-vector encoding, with a different RHS.
        Alternatively, per-x can be encoded as "at least one row has A_i . x != 0", which
        needs more variables (row-satisfaction indicators).
        """
        for xi, x in enumerate(B_n):
            terms = []
            for i in range(self.m):
                for p in range(self.n):
                    if x[p] != 0:
                        terms.append((x[p] * x[p], self.var_A(i, p)))
                for p in range(self.n):
                    for q in range(p + 1, self.n):
                        if x[p] * x[q] != 0:
                            terms.append((2 * x[p] * x[q], self.var_z(i, p, q)))
            if not terms:
                continue
            line = f"* per-x constraint x_{xi} = {tuple(int(v) for v in x)}\n"
            parts = []
            for coef, var in terms:
                sign = "+" if coef >= 0 else "-"
                parts.append(f"{sign}{abs(coef)} x{var}")
            line += " ".join(parts) + " >= 1 ;"
            self.constraints.append(line)
            self.num_constraints_pb += 1

    def write(self, path):
        with open(path, 'w') as f:
            f.write(f"* Balanced-detecting SAT encoding, m={self.m}, n={self.n}\n")
            f.write(f"* {self.num_primary} primary vars (A_ij), {self.num_aux} aux vars (z_ipq)\n")
            f.write(f"* {self.num_constraints_definitional} definitional constraints (Tseitin)\n")
            f.write(f"* {self.num_constraints_pb} matrix-condition PB constraints\n")
            f.write(f"* Total: {self.total_vars} variables, {len(self.constraints)} constraints\n")
            f.write(f"#variable= {self.total_vars} #constraint= {len(self.constraints)}\n")
            for c in self.constraints:
                f.write(c + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--m', type=int, required=True, help='number of rows')
    ap.add_argument('--n', type=int, required=True, help='number of columns')
    ap.add_argument('--output', type=str, default=None, help='output OPB filepath')
    ap.add_argument('--test-vectors', type=str, default='e_j,e_j-e_k',
                    help='comma-separated families: e_j, e_j+e_k, e_j-e_k, pair_pair')
    ap.add_argument('--row-cap', type=int, default=None, help='row weight cap constraint')
    ap.add_argument('--include-perx', action='store_true',
                    help='also include per-x constraints (redundant baseline)')
    ap.add_argument('--stats-only', action='store_true',
                    help='print size stats and exit')
    args = ap.parse_args()

    print(f"Encoding balanced-detecting matrix condition for m={args.m}, n={args.n}", file=sys.stderr)

    # Build T4 and XXT
    print(f"Computing T4 pattern values...", file=sys.stderr)
    Vs, N_B = T4_pattern_values(args.n)
    print(f"  |B(n)| = {N_B}", file=sys.stderr)
    print(f"  Pattern values: V_4={Vs[(4,)]}, V_31={Vs[(3,1)]}, V_22={Vs[(2,2)]}, "
          f"V_211={Vs[(2,1,1)]}, V_1111={Vs[(1,1,1,1)]}", file=sys.stderr)
    T4 = build_T4(args.n, Vs)
    XXT = build_XXT(args.n, Vs)

    enc = OPBEncoder(args.m, args.n)

    # Test vector families
    families = [f.strip() for f in args.test_vectors.split(',') if f.strip()]
    print(f"Test vector families: {families}", file=sys.stderr)
    n_tv = 0
    for fam in families:
        if fam not in TEST_VECTOR_FAMILIES:
            print(f"ERROR: unknown family '{fam}'", file=sys.stderr)
            sys.exit(1)
        for name, v in TEST_VECTOR_FAMILIES[fam](args.n):
            enc.add_matrix_constraint(v, name, T4, XXT)
            n_tv += 1
    print(f"  Total test vectors: {n_tv}", file=sys.stderr)

    # Definitional Tseitin clauses for aux vars
    print(f"Adding definitional clauses...", file=sys.stderr)
    enc.add_definitional_clauses()

    # Optional row cap
    if args.row_cap is not None:
        enc.add_boolean_constraints(row_cap=args.row_cap)

    # Optional per-x constraints (for comparison against matrix-only)
    if args.include_perx:
        print(f"Adding per-x constraints ({N_B} constraints)...", file=sys.stderr)
        B_n = enum_balanced(args.n)
        enc.add_perxconstraints(B_n)

    print(f"\nEncoding statistics:", file=sys.stderr)
    print(f"  Primary variables: {enc.num_primary}", file=sys.stderr)
    print(f"  Aux variables:     {enc.num_aux}", file=sys.stderr)
    print(f"  Total variables:   {enc.total_vars}", file=sys.stderr)
    print(f"  Definitional constraints: {enc.num_constraints_definitional}", file=sys.stderr)
    print(f"  Matrix + optional PB constraints: {enc.num_constraints_pb}", file=sys.stderr)
    print(f"  Total constraints: {len(enc.constraints)}", file=sys.stderr)

    if args.stats_only:
        return

    if args.output is None:
        args.output = f"bd_m{args.m}_n{args.n}.opb"
    print(f"\nWriting to {args.output}...", file=sys.stderr)
    enc.write(args.output)
    print(f"Done.", file=sys.stderr)


if __name__ == '__main__':
    main()
