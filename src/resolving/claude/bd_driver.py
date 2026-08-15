#!/usr/bin/env python3
"""
bd_driver.py -- Driver for balanced-detecting search with matrix propagator.

If PySAT is available, uses real CaDiCaL 1.9.5 / 3.0.0 with IPASIR-UP.
If not, uses a mock solver that iterates all m*n binary assignments; this is
only useful for tiny cases (n <= 6) but verifies the propagator logic end-to-end.

USAGE:
  python3 bd_driver.py --m 3 --n 4
  python3 bd_driver.py --m 2 --n 4                # known infeasible (beta_4 = 4)
  python3 bd_driver.py --m 7 --n 14               # target scale
"""

import argparse
import sys
import time
from itertools import product
import numpy as np

from bd_theory import BDTheory, enum_balanced
from bd_propagator import BDPropagator, HAVE_PYSAT


# ---------------------------------------------------------------------------
# Base CNF construction
# ---------------------------------------------------------------------------

def build_per_x_cnf(m, n):
    """Build baseline CNF for balanced-detecting: for each x in B(n), require
    that Ax != 0.

    ||Ax||^2 >= 1 (integer) is  equivalent to  exists i: (A_i . x) != 0.

    We encode this via row-indicator variables r_{i,x}: r_{i,x} true iff
    A_i . x != 0. Then require sum_i r_{i,x} >= 1 for each x.

    r_{i,x} definition: A_i . x != 0. For x in B(n), A_i . x = sum_{p} A_{i,p} x_p
    with x_p in {-1, 0, 1}. So A_i . x = (# entries with A_{i,p}=1 and x_p=+1)
    - (# entries with A_{i,p}=1 and x_p=-1).

    Denote S+_x = {p : x_p = +1}, S-_x = {p : x_p = -1}. Then r_{i,x} is true iff
    sum_{p in S+_x} A_{i,p} != sum_{p in S-_x} A_{i,p}.

    Since S+_x and S-_x have the same size (x is balanced), the two sums are both
    integers in [0, |S+|]. The inequality "sums differ" is expressible in CNF
    but blows up. FOR NOW we use a SIMPLIFIED encoding that only uses weight-2
    x (the pairwise column-distinct constraints):
      For j != k in [n], at least one row has A_{i,j} != A_{i,k}
      = OR_{i} (A_{i,j} XOR A_{i,k})
      = OR_{i} ( (A_{i,j} & ~A_{i,k}) | (~A_{i,j} & A_{i,k}) )

    Encoding via Tseitin: introduce a var d_{i,j,k} = A_{i,j} XOR A_{i,k}, then
    require OR_i d_{i,j,k}.

    This is INCOMPLETE (doesn't cover weight-4+ x's) but is sufficient for
    small n (where weight-2 constraints imply weight-4 constraints).
    For a real search you'd add the full per-x encoding or use the matrix
    propagator to catch what pairwise misses.
    """
    clauses = []
    next_var = m * n + 1

    def A_var(i, j):
        return 1 + i * n + j

    # For each column pair (j, k), we need OR_i (A_{i,j} XOR A_{i,k}).
    # Introduce d_{i,j,k} for each i, then a "big-OR" clause.
    # d = A XOR B  <->  (d | ~A | ~B) & (d | A | B) & (~d | A | ~B) & (~d | ~A | B)
    for j in range(n):
        for k in range(j + 1, n):
            or_clause = []
            for i in range(m):
                Aij = A_var(i, j)
                Aik = A_var(i, k)
                dijk = next_var
                next_var += 1
                # d = Aij XOR Aik
                clauses.append([dijk, -Aij, -Aik])
                clauses.append([dijk, Aij, Aik])
                clauses.append([-dijk, Aij, -Aik])
                clauses.append([-dijk, -Aij, Aik])
                or_clause.append(dijk)
            clauses.append(or_clause)

    return clauses, next_var - 1


# ---------------------------------------------------------------------------
# Mock solver: enumerates all 2^(mn) assignments and checks each
# ---------------------------------------------------------------------------

class MockSolver:
    """A mock CaDiCaL that iterates all m*n binary assignments, calling the
    propagator's cb_check_found_model on each and adding lemma clauses.
    Only useful for tiny problems.
    """

    def __init__(self, m, n, propagator, base_clauses=None, verbose=False):
        self.m = m
        self.n = n
        self.prop = propagator
        self.base_clauses = base_clauses or []
        self.verbose = verbose

    def solve(self):
        m, n = self.m, self.n
        blocked_models = set()          # set of frozenset(true primary lits)
        n_models_tried = 0
        for bits in product([0, 1], repeat=m * n):
            n_models_tried += 1
            # Build assignment (only primary vars)
            self.prop.assignment.clear()
            for k in range(m * n):
                self.prop.assignment[self.prop.pv0 + k] = bool(bits[k])
            true_prims = frozenset(v for v in self.prop.observed_variables()
                                    if self.prop.assignment.get(v, False))
            if true_prims in blocked_models:
                continue

            # Check base CNF satisfaction (simplified: just check per-x is satisfiable
            # by this assignment via direct BD check)
            A = self.prop.build_A()
            # Simple pairwise column-distinct check
            all_distinct = True
            for j in range(n):
                for k in range(j + 1, n):
                    if all(A[i, j] == A[i, k] for i in range(m)):
                        all_distinct = False
                        break
                if not all_distinct:
                    break
            if not all_distinct:
                continue                                # violates base

            # Build model list
            model = []
            for v in self.prop.observed_variables():
                model.append(v if self.prop.assignment.get(v, False) else -v)

            # Call propagator
            accept = self.prop.cb_check_found_model(model)
            if accept:
                if self.verbose:
                    print(f"[mock] accepted A =")
                    print(A)
                return True, A, self.prop.stats()

            # Rejected: enqueue clauses
            while self.prop.cb_has_external_clause():
                clause = []
                while True:
                    lit = self.prop.cb_add_external_clause_lit()
                    if lit == 0:
                        break
                    clause.append(lit)
                # Block this model
                blocked_models.add(true_prims)

        return False, None, self.prop.stats()


# ---------------------------------------------------------------------------
# Real PySAT + CaDiCaL driver
# ---------------------------------------------------------------------------

def run_with_pysat(m, n, verbose=False):
    """Real search using PySAT + CaDiCaL 1.9.5/3.0.0 with IPASIR-UP."""
    from pysat.solvers import Cadical195
    from pysat.formula import CNF

    clauses, last_var = build_per_x_cnf(m, n)
    cnf = CNF(from_clauses=clauses)

    prop = BDPropagator(m, n, primary_var_start=1, verbose=verbose)

    with Cadical195(bootstrap_with=cnf) as solver:
        solver.connect_propagator(prop)
        for var in prop.observed_variables():
            solver.observe(var)
        t0 = time.time()
        result = solver.solve()
        elapsed = time.time() - t0

    print(f"\nResult: {'SAT' if result else 'UNSAT'} in {elapsed:.2f} s")
    print(f"Propagator stats: {prop.stats()}")
    if result:
        model = solver.get_model()
        A = np.zeros((m, n), dtype=np.int8)
        for i in range(m):
            for j in range(n):
                v = prop.var_A(i, j)
                A[i, j] = 1 if v in model else 0
        print(f"Model A:\n{A}")
        return A
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m', type=int, required=True)
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--mock', action='store_true', help='use mock solver')
    args = ap.parse_args()

    if not args.mock and HAVE_PYSAT:
        run_with_pysat(args.m, args.n, verbose=args.verbose)
    else:
        if not args.mock:
            print(f"WARNING: PySAT not available; using mock solver", file=sys.stderr)
        # Mock solver (only for tiny cases)
        if args.m * args.n > 20:
            print(f"Mock solver requires m*n <= 20 (2^20 = ~1M candidates)")
            sys.exit(1)
        prop = BDPropagator(args.m, args.n, primary_var_start=1, verbose=args.verbose)
        solver = MockSolver(args.m, args.n, prop, verbose=args.verbose)
        t0 = time.time()
        found, A, stats = solver.solve()
        elapsed = time.time() - t0
        print(f"\nResult: {'SAT' if found else 'UNSAT'} in {elapsed:.2f} s")
        print(f"Propagator stats: {stats}")
        if found:
            print(f"A =")
            print(A)
            # Verify: is A actually balanced-detecting?
            theory = BDTheory(args.m, args.n)
            ok, mineig, _ = theory.check(A)
            print(f"  Matrix filter on found A: {'PASS' if ok else 'FAIL'}, min_eig={mineig:.4f}")
            # Direct BD check
            B = enum_balanced(args.n)
            all_ok = all(np.any(A @ x) for x in B)
            print(f"  Direct BD check: {'PASS' if all_ok else 'FAIL'}")


if __name__ == '__main__':
    main()
