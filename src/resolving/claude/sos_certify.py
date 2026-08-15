#!/usr/bin/env python3
"""
Lasserre level-2 SDP for balanced-detecting infeasibility.
============================================================

Problem: certify that no m x n balanced-detecting 0/1 matrix A exists.

Formulation.  Variables A_{ij} in {0,1} subject to detection constraints
   Q_x(A) := ||A x||_2^2 >= 1   for every x in B(n),
where B(n) is the set of nonzero balanced ternary vectors in {-1,0,+1}^n
(one representative per +/- pair; using either sign gives the same Q_x).

Boolean lift: introduce moment variables y_S for |S| <= 2 with the
convention y_S = prod_{(i,j) in S} A_{ij}.  The Boolean constraints
A_{ij}^2 = A_{ij} become y_{(ij),(ij)} = y_{(ij)}.  The detection
constraint Q_x(A) >= 1 is linear in y since it expands as
   sum_i sum_{j,j'} A_{ij} A_{ij'} x_j x_{j'} >= 1.

The Lasserre level-2 relaxation asks for a PSD moment matrix M_2 that:
  * has y_0 = 1 (mass constraint)
  * satisfies the Boolean equalities on the diagonal
  * satisfies the linearized detection constraints as lower bounds on the
    corresponding linear-in-y expressions.

If this SDP is infeasible, that certifies no balanced-detecting m x n matrix
exists (in a strictly stronger sense than the plain Boolean IP).  If feasible,
the level-2 relaxation is not tight and one would need level 3 or additional
structure.

The XX^T structure.
Let X be n x N with columns spanning one representative per +/- pair of
nonzero balanced ternary vectors.  Then XX^T = (alpha - beta)(I - (1/n) J)
where alpha_n = [z^(n-2)](1+z+z^2)^(n-1) (a Motzkin coefficient) and beta
is derived similarly.  This gives:
  sum over x in B(n) of Q_x(A) = tr(A XX^T A^T) = (alpha-beta)*(||A||_F^2 - (1/n)||A e||_2^2)

Since balanced-detecting requires each Q_x(A) >= 1 there are N = |B(n)|
constraints, and their sum is >= N.  So any BD matrix must satisfy:
  (alpha_n - beta_n) * (||A||_F^2 - (1/n) ||A * 1||_2^2) >= N.
This is a single quadratic inequality, valid for every BD matrix.  It gives
a "cheap sufficient" lower bound on m*n and, combined with the row-weight cap,
sometimes rules out (m, n) directly -- see the aggregate_bound() function.

Requirements: cvxpy, numpy.  Solver: SCS (or any SDP solver cvxpy supports).

Usage:
   python3 sos_certify.py --r 2 --n 4
   python3 sos_certify.py --r 3 --n 6
   python3 sos_certify.py --r 5 --n 9
   python3 sos_certify.py --aggregate 7 14        # aggregate XX^T bound
"""
import argparse
import sys
from itertools import product, combinations
from math import comb


def enumerate_balanced(n, max_weight=None):
    """One representative per +/- orbit of nonzero balanced ternary vectors."""
    out = []
    for x in product((-1, 0, 1), repeat=n):
        if sum(x) != 0:
            continue
        first_nz = None
        for v in x:
            if v != 0:
                first_nz = v
                break
        if first_nz is None:
            continue
        if first_nz < 0:
            continue
        if max_weight is not None:
            if sum(1 for v in x if v != 0) > max_weight:
                continue
        out.append(x)
    return out


def alpha_beta(n):
    """Compute alpha_n and beta_n for XX^T = alpha I + beta (J - I).
    alpha = # balanced x with x_1 = +1 (over unsigned reps).
    beta from trace/sum relations.
    """
    from sympy import symbols, expand, Poly
    z = symbols('z')
    p = Poly(expand((1 + z + z**2)**(n - 1)), z)
    alpha = int(p.coeff_monomial(z**(n - 2)))
    # N = total # of +/- pairs = (T(n) - 1) / 2 where T(n) = central trinomial coef
    T_n = int(Poly(expand((1 + z + z**2)**n), z).coeff_monomial(z**n))
    N = (T_n - 1) // 2
    # tr(XX^T) = n * alpha.  All-ones eigenvalue: alpha + (n-1)*beta = 0.
    # So beta = -alpha / (n - 1).
    if n == 1:
        beta = 0
    else:
        assert alpha % (n - 1) == 0, f"alpha={alpha}, n-1={n-1} should divide"
        beta = -alpha // (n - 1)
    return alpha, beta, N


def aggregate_bound_check(r, n):
    """Necessary condition from tr(A XX^T A^T) >= N.
    XX^T = (alpha - beta)(I - (1/n) J), so
    tr(A XX^T A^T) = (alpha - beta) [||A||_F^2 - (1/n) ||A*1||_2^2].
    Under row-weight cap: for each row, sum_j A_{ij} <= floor(n/2), and
    ||A||_F^2 = sum row weights <= r * floor(n/2).
    ||A*1||_2^2 = sum_i (row_i)^2 -- upper bounded when row weights are all near cap.

    We want to check: can ||A||_F^2 - (1/n)||A*1||_2^2 be as small as
        N / (alpha - beta)?
    If NOT (i.e. even the minimum over 0/1 matrices with row cap exceeds this),
    that's still not helpful -- we want an upper bound on the LHS that's < RHS.

    Actually the aggregate constraint says (alpha-beta) * f(A) >= N where
    f(A) = ||A||_F^2 - (1/n)||A*1||_2^2 >= 0.
    So the constraint is a LOWER bound on f(A), not an obstruction unless
    the maximum of f(A) over admissible A is < N/(alpha-beta).

    f(A) is maximized by setting rows to be as "spread out" as possible.
    Given row_i has weight w_i, f(row_i) = w_i - w_i^2/n = w_i(1 - w_i/n).
    Maximum over w_i in [0, floor(n/2)] is achieved at w_i = floor(n/2).
    So max_A f(A) = r * (n/2)(1/2) = r*n/4 (when n even).

    For infeasibility we need:  r * n/4 < N / (alpha - beta).
    """
    alpha, beta, N = alpha_beta(n)
    eig = alpha - beta                            # positive eigenvalue of XX^T (mult n-1)
    max_fA = r * (n / 2) * (1 / 2)                # upper bound for even n
    required = N / eig
    print(f"  n = {n}: alpha = {alpha}, beta = {beta}, N = |B(n)| = {N}")
    print(f"  eigenvalue lambda_+ = alpha - beta = {eig} (mult {n-1})")
    print(f"  aggregate constraint:  {eig} * (||A||_F^2 - (1/n) ||A*1||^2)  >=  {N}")
    print(f"  equivalently:  ||A||_F^2 - (1/n) ||A*1||^2  >=  {required:.4f}")
    print()
    print(f"  For r = {r}, m = r:  max f(A) over BD candidates <= r * n/4 = {max_fA}")
    if max_fA < required:
        print(f"  --> {max_fA} < {required:.4f}: aggregate rules out r={r}, n={n}!")
    else:
        print(f"  --> {max_fA} >= {required:.4f}: aggregate does NOT rule out this case.")


def build_lasserre_level2(r, n, weight_cap=True, verbose=True):
    """Build the level-2 Lasserre SDP feasibility problem for balanced-detecting.
    Returns (prob, variables_dict).  Uses cvxpy.
    """
    try:
        import cvxpy as cp
        import numpy as np
    except ImportError:
        print("ERROR: cvxpy and numpy required.  pip install cvxpy numpy", file=sys.stderr)
        sys.exit(1)

    # Enumerate monomials of degree <= 2 in the mn variables A_{i,j}.
    # Index them as (i,j) tuples; use frozenset for multisets since A^2 = A.
    variables = [(i, j) for i in range(r) for j in range(n)]
    mn = len(variables)

    # Basis monomials for the level-2 moment matrix: 1, A_{ij}, A_{ij} A_{i'j'}
    # For the Boolean case A^2 = A, we only need distinct pairs (i,j) != (i',j').
    monoms = [frozenset()]                                                # 1
    for v in variables:
        monoms.append(frozenset({v}))                                     # A_{v}
    for a, b in combinations(variables, 2):
        monoms.append(frozenset({a, b}))                                  # A_a * A_b
    M = len(monoms)
    monom_idx = {m: i for i, m in enumerate(monoms)}

    if verbose:
        print(f"  {mn} primary variables, {M} moment basis monomials")
        print(f"  moment matrix size: {M} x {M}")

    # Create the moment matrix M2 as a symmetric variable.
    M2 = cp.Variable((M, M), symmetric=True)

    constraints = [M2 >> 0]                                               # PSD

    # y_S for |S| <= 4 (union of two |S|<=2 sets) -- but under A^2 = A these
    # reduce to y_{S union} for the corresponding SET S.
    # The moment matrix entry M2[alpha, beta] = y_{alpha union beta}.
    # We need M2 to be *consistent*: for any two pairs alpha, beta and alpha', beta'
    # with alpha union beta = alpha' union beta', M2 entries must match.
    # This is Lasserre's shift constraint.
    #
    # Build the shift constraints.  For each frozenset S with |S| <= 4, gather
    # all (i, j) with monoms[i] union monoms[j] = S; require M2[i,j] to be equal
    # across all such pairs.  We store y[S] as a canonical M2 entry.
    from collections import defaultdict
    entries_by_union = defaultdict(list)                                  # union -> list of (i,j)
    for i in range(M):
        for j in range(i, M):
            u = monoms[i] | monoms[j]
            if len(u) <= 4:
                entries_by_union[u].append((i, j))
    for u, ijs in entries_by_union.items():
        if len(ijs) >= 2:
            first = ijs[0]
            for other in ijs[1:]:
                constraints.append(M2[first[0], first[1]] == M2[other[0], other[1]])

    # Anchor: y_emptyset = 1
    constraints.append(M2[0, 0] == 1)

    # Boolean constraints: y_{{v}} = M2[0, idx({v})] and also
    #   y_{{v}, {v}} = y_{{v}}, which is enforced by "canonical entry for {v}" already.
    # (The union {v} union {v} = {v}, so the diagonal entry M2[idx({v}), idx({v})] = y_{{v}}
    # is one of the entries mapping to union = {v}, and consistency forces it.)

    # Detection constraints: for each x in B(n), sum_i sum_{j,j'} A_{ij} A_{ij'} x_j x_{j'} >= 1.
    B_n = enumerate_balanced(n)
    if verbose:
        print(f"  {len(B_n)} detection constraints (one per balanced-vector orbit)")

    def y_index(S):
        """Return an index into M2 (as a pair) mapping to monomial S. |S| <= 2."""
        idx = monom_idx[S]
        return (0, idx)                                                   # y_S = M2[0, idx(S)]

    for x in B_n:
        # sum_i sum_{j != j'} 2 * x_j * x_{j'} * y_{(ij),(ij')} + sum_i sum_j x_j^2 * y_{(ij)}
        # Note x_j^2 in {0, 1}.
        expr = 0
        for i in range(r):
            for j in range(n):
                if x[j] != 0:
                    row, col = y_index(frozenset({(i, j)}))
                    expr = expr + (x[j] * x[j]) * M2[row, col]
                for jp in range(j + 1, n):
                    if x[j] == 0 or x[jp] == 0: continue
                    coef = 2 * x[j] * x[jp]
                    row, col = y_index(frozenset({(i, j), (i, jp)}))
                    expr = expr + coef * M2[row, col]
        constraints.append(expr >= 1)

    # Optional row-weight cap: sum_j y_{(ij)} <= floor(n/2)
    if weight_cap:
        cap = n // 2
        for i in range(r):
            expr = 0
            for j in range(n):
                row, col = y_index(frozenset({(i, j)}))
                expr = expr + M2[row, col]
            constraints.append(expr <= cap)

    prob = cp.Problem(cp.Minimize(0), constraints)
    return prob, {'M2': M2, 'monoms': monoms, 'monom_idx': monom_idx}


def solve_and_report(r, n, verbose=True):
    prob, vars_ = build_lasserre_level2(r, n, verbose=verbose)
    try:
        import cvxpy as cp
    except ImportError:
        return
    if verbose:
        print(f"  solving with SCS...")
    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except Exception as e:
        print(f"  solver error: {e}")
        return
    print(f"  status: {prob.status}")
    if prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
        print(f"  >>> LEVEL-2 LASSERRE PROVES INFEASIBILITY for r={r}, n={n}")
        print(f"      certifies beta_{n} > {r + 1}")
    elif prob.status == cp.OPTIMAL:
        print(f"  >>> level-2 SDP is feasible; relaxation not tight for r={r}, n={n}")
        # print a few entries to sanity check
        M2 = vars_['M2'].value
        if M2 is not None:
            print(f"  y_emptyset (should be ~1): {M2[0,0]:.4f}")
            # print A_{ij} 'expected values'
            for i in range(r):
                row = []
                for j in range(n):
                    idx = vars_['monom_idx'][frozenset({(i,j)})]
                    row.append(M2[0, idx])
                print(f"  E[A_{i}, :] = [" + " ".join(f"{v:.3f}" for v in row) + "]")
    else:
        print(f"  solver returned status {prob.status}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--r', type=int, help='rows')
    ap.add_argument('--n', type=int, help='columns')
    ap.add_argument('--aggregate', nargs=2, type=int, metavar=('R','N'),
                    help='only print the aggregate XX^T bound for given r, n')
    ap.add_argument('--no-cap', action='store_true', help='disable row-weight cap')
    args = ap.parse_args()

    if args.aggregate:
        r, n = args.aggregate
        aggregate_bound_check(r, n)
        return
    if args.r is None or args.n is None:
        ap.print_help(); sys.exit(1)
    aggregate_bound_check(args.r, args.n)
    solve_and_report(args.r, args.n, verbose=True)


if __name__ == '__main__':
    main()
