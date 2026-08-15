#!/usr/bin/env python3
"""
Restricted Sherali-Adams LP for certifying balanced-detecting infeasibility.
=============================================================================

STATUS: PROOF OF CONCEPT.  See "Empirical finding" below.

Goal
----
Given (r, n), attempt to prove NO r x n balanced-detecting matrix exists, via a
polynomial-size LP relaxation.  If the LP is infeasible, the solver's Farkas
certificate is a proof: a nonneg combination of the constraints that yields
0 >= 1, cheaply verifiable.

Empirical finding
-----------------
Level-2 same-row SA as formulated below is NOT tight for any known infeasible
(r, n): every LP for (2,4), (2,5), (3,6), (4,8) is feasible with a fractional
solution near A = (1/2) * (some 0/1 matrix).  The root cause is intrinsic to
level-2 LP hierarchies: the diagonal envelope 2A-1 <= d <= A pins d = A^2 only
at A in {0,1} and has slack up to 1/4 at A = 1/2.  The LP exploits this slack
to satisfy detection constraints Q_x >= 1 with fractional A that don't
correspond to any 0/1 matrix.

To close the gap, one needs either:
  (a) Level-3+ SA (adds triple products; adds more envelope constraints; may
      still be insufficient),
  (b) Lasserre / SDP lift (moment SDP replaces the linear d envelope with
      d = A^2 exactly via PSD constraints; likely to work but SDP-sized),
  (c) Cross-row lifts (adds y_{ij, i'j'} for i != i'; more variables but
      captures the r x r structure of A^T A).

This program is kept as a testbed for (a) and (c) if you want to experiment.
The current version implements only "level-2 same-row" and reports feasible
in every tested case.

The relaxation
--------------
Primary variables:
    A_{ij} in [0, 1] for i in [r], j in [n]  (relaxed matrix entries)

Level-2 same-row lift:
    y_{i,j,j'} in [0, 1] for i in [r], j < j' in [n]
    Interpretation: y_{i,j,j'} = A_{ij} * A_{ij'}.
Standard Sherali-Adams linearization of the product:
    y_{i,j,j'} <= A_{ij}                                (S1)
    y_{i,j,j'} <= A_{ij'}                               (S2)
    y_{i,j,j'} >= A_{ij} + A_{ij'} - 1                  (S3)
    y_{i,j,j'} >= 0                                     (S4)

Detection constraints (the substance):
For each nontrivial balanced x in {-1,0,+1}^n (i.e., sum(x) = 0, x != 0),
representative up to +/-, we require Q_x(A) := ||A x||_2^2 >= 1.

Expanded:
    Q_x(A) = sum_i (sum_j A_{ij} x_j)^2
           = sum_i sum_{j,j'} A_{ij} A_{ij'} x_j x_{j'}
           = sum_i sum_j x_j^2 * A_{ij}^2  +  2 sum_i sum_{j<j'} x_j x_{j'} y_{i,j,j'}

Now, in the LP we cannot use A_{ij}^2 directly, but we can lift:
    y_{i,j,j} := A_{ij}^2, but for A in [0,1], A^2 <= A, so we substitute
    A_{ij}^2 -> A_{ij}.  This is a VALID upper bound on A_{ij}^2, hence a valid
    UPPER bound on the "diagonal" part.  For a LOWER bound on Q_x needed for
    infeasibility, we need a lower bound on A_{ij}^2, which is A_{ij}^2 >= 0 or
    A_{ij}^2 >= 2A_{ij} - 1 (tangent at A=1).  We take max(0, 2A_{ij} - 1).

Wait: since we want to PROVE Q_x >= 1 is impossible when combined with other
constraints, we need Q_x itself to be LARGE.  So we lower-bound Q_x.  For the
diagonal part sum_j x_j^2 A_{ij}^2, since x_j^2 in {0, 1}:
    sum_j x_j^2 A_{ij}^2 = sum_{j : x_j != 0} A_{ij}^2  >=  sum_{j: x_j != 0} (2 A_{ij} - 1)_+
The tangent-at-1 bound is exact at integer solutions.  In the LP relaxation
this can be conservative, so we use a variable d_{i,j} >= 0, d_{i,j} >= 2A_{ij} - 1
and lower-bound A_{ij}^2 by d_{i,j}.  Effectively:
    Q_x(A) >= sum_i [ sum_{j : x_j!=0} d_{i,j} + 2 sum_{j<j', x_j x_{j'}!=0} x_j x_{j'} y_{i,j,j'} ]

Row-weight cap (constraint 1, sound canonicalization):
    sum_j A_{ij} <= floor(n/2)  for each i in [r]

Full LP:
    minimize 0     (feasibility LP)
    subject to:
        A_{ij} in [0, 1]                                     (bounds on A)
        y_{i,j,j'} in [0, 1]                                 (bounds on y)
        d_{i,j} >= 0, d_{i,j} >= 2 A_{ij} - 1                (bounds on d)
        S1, S2, S3 (SA multilinear on y)
        sum_j A_{ij} <= floor(n/2)                           (row cap)
        Q_x_lb(A) >= 1  for each x in constraint set X       (detection)

If infeasible: the LP proves NO r x n balanced-detecting matrix exists (in fact
NO fractional A in [0,1]^{rxn} satisfies the relaxation, which is strictly
stronger and yields the certificate).

Two enumeration strategies for X:
    * ALL: enumerate all balanced ternary vectors up to +/- (~3^n / 2)
    * ITER: iterative cutting-plane -- start small, solve, add most-violated x

Usage
-----
    python3 sa_certify.py --r 3 --n 5              # cert beta_5 >= 4
    python3 sa_certify.py --r 4 --n 7 --iter       # iterative
    python3 sa_certify.py --r 5 --n 9              # cert beta_9 >= 7
    python3 sa_certify.py --r 5 --n 9 --solver SCIP

The KNOWN answer for each (r, n):
    (r, n) infeasible IFF beta_n > r+1 (there is no (r) x n bal-det matrix).
    Equivalently: r < beta_n - 1.
    Reference: beta_n = 5,6,6,7,7,8,8,8 for n=6..13.
    So (r, n) that SHOULD be infeasible include:
        (2, 5) - trivially small
        (3, 6): beta_6=5, so need r=4 -> (3,6) should be infeasible
        (3, 7): beta_7=6, need r=5 -> (3,7) infeasible
        (4, 7): beta_7=6, need r=5 -> (4,7) infeasible
        (5, 9): beta_9=7, need r=6 -> (5,9) infeasible  <-- INTERESTING
        (6, 11): beta_11=8, need r=7 -> (6,11) infeasible
        (7, 14): OPEN
"""

import argparse
import time
import sys
from itertools import product

try:
    import pulp
    HAVE_PULP = True
except ImportError:
    HAVE_PULP = False

try:
    from scipy.optimize import linprog
    import numpy as np
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ============================================================================
# Enumerate balanced ternary vectors
# ============================================================================

def enumerate_balanced(n, max_weight=None):
    """Enumerate balanced x in {-1,0,+1}^n with sum(x)=0, x != 0, one per +/- pair.
    Yields tuples.  If max_weight given, restrict to |{j : x_j != 0}| <= max_weight."""
    for x in product((-1, 0, 1), repeat=n):
        if sum(x) != 0:
            continue
        # find first nonzero
        first_nz = None
        for k, v in enumerate(x):
            if v != 0:
                first_nz = v
                break
        if first_nz is None:
            continue  # all zero
        if first_nz < 0:
            continue  # keep the +/- rep whose first nonzero is +1
        if max_weight is not None:
            if sum(1 for v in x if v != 0) > max_weight:
                continue
        yield x


# ============================================================================
# PuLP model
# ============================================================================

def build_model(r, n, X_vectors, solver_name="SCIP", verbose=False):
    """Build the LP feasibility model.  Returns (prob, A_vars, y_vars, d_vars)."""
    if not HAVE_PULP:
        raise RuntimeError("pulp not available; install with 'pip install pulp'")

    prob = pulp.LpProblem("bal_det_SA", pulp.LpMinimize)
    prob += 0  # feasibility

    # A_{ij} in [0, 1]
    A = {(i, j): pulp.LpVariable(f"A_{i}_{j}", lowBound=0, upBound=1)
         for i in range(r) for j in range(n)}
    # y_{i,j,j'} in [0, 1], for j < j'
    y = {(i, j, jp): pulp.LpVariable(f"y_{i}_{j}_{jp}", lowBound=0, upBound=1)
         for i in range(r) for j in range(n) for jp in range(j + 1, n)}
    # d_{i,j} in [0,1] representing A_{ij}^2; constrained by 2A-1 <= d <= A.
    d = {(i, j): pulp.LpVariable(f"d_{i}_{j}", lowBound=0, upBound=1)
         for i in range(r) for j in range(n)}

    # Sherali-Adams multilinear constraints on y
    for (i, j, jp), yv in y.items():
        prob += yv <= A[(i, j)],  f"SA1_{i}_{j}_{jp}"
        prob += yv <= A[(i, jp)], f"SA2_{i}_{j}_{jp}"
        prob += yv >= A[(i, j)] + A[(i, jp)] - 1, f"SA3_{i}_{j}_{jp}"

    # d bounds:  d >= 2A - 1 (lower env of A^2 on [0,1]) and d <= A (upper env A^2 <= A)
    # BOTH bounds are needed; without d <= A, the LP decouples d from A and
    # trivially satisfies the detection constraint.
    for (i, j), dv in d.items():
        prob += dv >= 2 * A[(i, j)] - 1, f"D_lo_{i}_{j}"
        prob += dv <= A[(i, j)],         f"D_hi_{i}_{j}"

    # Row-weight cap:  sum_j A_{ij} <= floor(n/2)
    cap = n // 2
    for i in range(r):
        prob += pulp.lpSum(A[(i, j)] for j in range(n)) <= cap, f"ROWCAP_{i}"

    # Detection constraints for each x in X_vectors:
    #   Q_x(A) >= 1
    # where Q_x(A) is expressed via d (diagonal) + y (off-diagonal).
    for kx, x in enumerate(X_vectors):
        nz = [(j, x[j]) for j in range(n) if x[j] != 0]
        # Q_x lower bound expression
        expr = 0
        for i in range(r):
            # diagonal: sum_{j : x_j != 0} d_{i,j}   (since x_j^2 = 1)
            for (j, _) in nz:
                expr = expr + d[(i, j)]
            # off-diagonal: 2 sum_{j<j'} x_j x_j' y_{i,j,j'}
            for a in range(len(nz)):
                for b in range(a + 1, len(nz)):
                    j, xj = nz[a]
                    jp, xjp = nz[b]
                    # j < jp guaranteed since nz is in increasing j order
                    coef = 2 * xj * xjp
                    if coef != 0:
                        expr = expr + coef * y[(i, j, jp)]
        prob += expr >= 1, f"DET_{kx}"

    if verbose:
        print(f"  LP built: {len(prob.variables())} vars, {len(prob.constraints)} constraints")

    return prob, A, y, d


def solve_lp(prob, solver_name, timelimit=None, verbose=False):
    """Try to solve with the requested solver.  Returns (status, wall_time)."""
    solver_map = {
        "SCIP":   pulp.SCIP_CMD if hasattr(pulp, "SCIP_CMD") else None,
        "HiGHS":  pulp.HiGHS_CMD if hasattr(pulp, "HiGHS_CMD") else None,
        "CBC":    pulp.PULP_CBC_CMD,
        "GUROBI": pulp.GUROBI_CMD if hasattr(pulp, "GUROBI_CMD") else None,
        "CPLEX":  pulp.CPLEX_CMD if hasattr(pulp, "CPLEX_CMD") else None,
    }
    solver_cls = solver_map.get(solver_name)
    if solver_cls is None:
        print(f"  solver {solver_name} not available; falling back to CBC")
        solver_cls = pulp.PULP_CBC_CMD
    kwargs = {"msg": 1 if verbose else 0}
    if timelimit:
        kwargs["timeLimit"] = timelimit
    solver = solver_cls(**kwargs)
    t0 = time.time()
    prob.solve(solver)
    dt = time.time() - t0
    status = pulp.LpStatus[prob.status]
    return status, dt


# ============================================================================
# Iterative cutting-plane version
# ============================================================================

def solve_iterative(r, n, solver_name, max_iters=50, cut_batch=20,
                    max_weight=None, verbose=False, timelimit_per=60):
    """Start with a small set of x's, solve, add violated x's, repeat."""
    print(f"\n=== Iterative SA for r={r}, n={n}, solver={solver_name} ===")
    all_x = list(enumerate_balanced(n, max_weight=max_weight))
    print(f"  Candidate x's (up to +/-, weight<={max_weight or 'any'}): {len(all_x)}")

    # Start with all weight-2 vectors (there are (n choose 2) of them: e_j - e_j')
    active = [x for x in all_x if sum(1 for v in x if v != 0) == 2]
    print(f"  Starting with {len(active)} weight-2 vectors")

    for it in range(max_iters):
        prob, A_vars, y_vars, d_vars = build_model(r, n, active,
                                                    solver_name=solver_name,
                                                    verbose=verbose)
        status, dt = solve_lp(prob, solver_name, timelimit=timelimit_per, verbose=verbose)
        print(f"  iter {it}: |X|={len(active)}, status={status}, {dt:.2f}s")
        if status == "Infeasible":
            print(f"  >>> INFEASIBLE at iter {it} with |X|={len(active)}")
            print(f"  >>> CERTIFIED: no r={r} x n={n} balanced-detecting matrix exists")
            return "Infeasible", active
        if status != "Optimal":
            print(f"  solver returned {status}; aborting")
            return status, active

        # Extract fractional solution
        A_val = {(i, j): pulp.value(A_vars[(i, j)]) or 0.0
                 for i in range(r) for j in range(n)}

        # Find most-violated x's not yet in active (compute true Q_x on A_val)
        active_set = set(active)
        violations = []
        for x in all_x:
            if x in active_set:
                continue
            q = 0.0
            for i in range(r):
                s = sum(A_val[(i, j)] * x[j] for j in range(n))
                q += s * s
            if q < 0.999:
                violations.append((q, x))
        violations.sort()
        if not violations:
            print(f"  >>> LP feasible and no violated x remains; RELAXATION IS FEASIBLE")
            print(f"      This means level-2 same-row SA is NOT tight for (r={r}, n={n})")
            return "Optimal_TIGHT", active

        # Add worst offenders
        new_cuts = [x for (_, x) in violations[:cut_batch]]
        active.extend(new_cuts)
        print(f"    added {len(new_cuts)} cuts; min Q_x on fractional soln = {violations[0][0]:.4f}")

    print(f"  >>> hit max_iters={max_iters} without terminating")
    return "MaxIters", active


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--r", type=int, required=True, help="number of rows")
    ap.add_argument("--n", type=int, required=True, help="number of columns")
    ap.add_argument("--iter", action="store_true",
                    help="iterative cutting-plane (default: emit all balanced x)")
    ap.add_argument("--max-weight", type=int, default=None,
                    help="restrict to x with Hamming weight <= this (default: no limit)")
    ap.add_argument("--solver", default="HiGHS",
                    choices=["SCIP", "HiGHS", "CBC", "GUROBI", "CPLEX"])
    ap.add_argument("--timelimit", type=float, default=300)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not HAVE_PULP:
        print("ERROR: pulp is required.  Install with 'pip install pulp'.")
        sys.exit(1)

    if args.iter:
        solve_iterative(args.r, args.n, args.solver,
                        max_weight=args.max_weight, verbose=args.verbose,
                        timelimit_per=args.timelimit)
    else:
        print(f"\n=== Full SA for r={args.r}, n={args.n}, solver={args.solver} ===")
        X = list(enumerate_balanced(args.n, max_weight=args.max_weight))
        print(f"  Balanced x's (up to +/-, weight<={args.max_weight or 'any'}): {len(X)}")
        prob, A_vars, y_vars, d_vars = build_model(args.r, args.n, X,
                                                    solver_name=args.solver,
                                                    verbose=args.verbose)
        status, dt = solve_lp(prob, args.solver, timelimit=args.timelimit,
                              verbose=args.verbose)
        print(f"  status: {status}  time: {dt:.2f}s")
        if status == "Infeasible":
            print(f"  >>> CERTIFIED: no r={args.r} x n={args.n} balanced-detecting matrix exists")
        elif status == "Optimal":
            print(f"  >>> LP is feasible; level-2 same-row SA is NOT tight for this instance")
            # Print a fractional witness
            print(f"  Fractional A (rounded to 2 decimals):")
            for i in range(args.r):
                row = [pulp.value(A_vars[(i, j)]) or 0.0 for j in range(args.n)]
                print("   [" + " ".join(f"{v:.2f}" for v in row) + "]")


if __name__ == "__main__":
    main()
