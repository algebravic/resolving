"""
MaxSAT-based SMUS solver for balanced-detecting UNSAT proofs — FIXED VERSION.

Fixes bug in the lex encoding of the SAT verifier: the extraneous clause 
[-eq[k], a, -b, eq[k+1]] incorrectly forced eq[k+1]=TRUE when positions 
differed with a=0, b=1, contradicting the forward implication eq[k+1] → (b → a)
and making the verifier UNSAT for any pair of non-equal rows/cols. 
This trivially collapsed the CEGAR to canSMUS = 1.

Requires: pip install python-sat
"""

import itertools
import numpy as np
from math import comb


# =========================================================================
# Balanced-vector enumeration
# =========================================================================

def enumerate_canonical_balanced(n):
    seen = set()
    for w in range(2, n + 1, 2):
        k = w // 2
        for support in itertools.combinations(range(n), w):
            for plus in itertools.combinations(support, k):
                y = [0] * n
                plus_set = set(plus)
                for j in support:
                    y[j] = 1 if j in plus_set else -1
                y_tuple = tuple(y)
                neg_y = tuple(-x for x in y_tuple)
                seen.add(min(y_tuple, neg_y))
    return sorted(seen, key=lambda y: (sum(abs(x) for x in y), y))


# =========================================================================
# MaxSAT solver for min cover (RC2)
# =========================================================================

def solve_min_cover_maxsat(constraint_masks, n_ys, verbose=False):
    """Solve min |S| such that S ∩ K ≠ ∅ for each K in constraint_masks.
    Returns (min_size, selected_indices)."""
    from pysat.formula import WCNF
    from pysat.examples.rc2 import RC2
    
    wcnf = WCNF()
    for mask in constraint_masks:
        clause = []
        m = mask
        idx = 0
        while m:
            if m & 1:
                clause.append(idx + 1)
            m >>= 1
            idx += 1
        if clause:
            wcnf.append(clause)  # hard
    for j in range(n_ys):
        wcnf.append([-(j + 1)], weight=1)  # soft
    
    with RC2(wcnf) as solver:
        model = solver.compute()
        if model is None:
            return None, None
        model_set = set(model)
        selected = [j for j in range(n_ys) if (j + 1) in model_set]
        return len(selected), selected


# =========================================================================
# Correct lex-leq encoding as a helper
# =========================================================================

def encode_lex_leq(cnf, x_vars, y_vars, new_var_fn):
    """Enforce (x_0, x_1, ..., x_{k-1}) ≤_lex (y_0, y_1, ..., y_{k-1}) 
    as clauses on 'cnf'. All x_vars and y_vars are positive literal ids.
    
    Uses auxiliary eq[k] = "equal in first k positions".
    """
    assert len(x_vars) == len(y_vars)
    n = len(x_vars)
    eq = [None] * (n + 1)
    eq[0] = new_var_fn()
    cnf.append([eq[0]])  # eq[0] is unconditionally true (empty prefix)
    for k in range(n):
        eq[k + 1] = new_var_fn()
        a = x_vars[k]
        b = y_vars[k]
        # Forward: eq[k+1] → eq[k] ∧ (a = b)
        cnf.append([-eq[k + 1], eq[k]])
        cnf.append([-eq[k + 1], -a, b])   # eq[k+1] → (a → b)
        cnf.append([-eq[k + 1], a, -b])   # eq[k+1] → (b → a)
        # Backward: eq[k] ∧ (a = b) → eq[k+1]
        cnf.append([-eq[k], -a, -b, eq[k + 1]])  # eq[k] ∧ a=b=1 → eq[k+1]
        cnf.append([-eq[k], a, b, eq[k + 1]])    # eq[k] ∧ a=b=0 → eq[k+1]
        # Lex condition at position k: eq[k] ∧ a=1 → b=1 (rules out a>b at first diff)
        cnf.append([-eq[k], -a, b])
    return eq


# =========================================================================
# SAT verifier: does an uncovered canonical matrix exist?
# =========================================================================

def find_uncovered_canonical_matrix(S_indices, Y, m, n, verbose=False):
    from pysat.formula import CNF
    from pysat.card import CardEnc, EncType
    from pysat.solvers import Solver
    
    Y_arr = np.array(Y, dtype=np.int8)
    S_ys = [Y_arr[i] for i in S_indices]
    
    # A[i,j] var id
    def A_var(i, j):
        return i * n + j + 1
    
    next_var = [m * n + 1]
    def new_var():
        v = next_var[0]
        next_var[0] += 1
        return v
    
    cnf = CNF()
    
    # ---- Row weight ≤ n/2 ----
    n_over_2 = n // 2
    for i in range(m):
        row_lits = [A_var(i, j) for j in range(n)]
        card = CardEnc.atmost(lits=row_lits, bound=n_over_2, 
                               top_id=next_var[0] - 1, encoding=EncType.seqcounter)
        cnf.extend(card.clauses)
        if card.nv > next_var[0] - 1:
            next_var[0] = card.nv + 1
    
    # ---- Rows lex-sorted ----
    for i in range(m - 1):
        x_vars = [A_var(i, k) for k in range(n)]
        y_vars = [A_var(i + 1, k) for k in range(n)]
        encode_lex_leq(cnf, x_vars, y_vars, new_var)
    
    # ---- Columns lex-sorted ----
    for j in range(n - 1):
        x_vars = [A_var(k, j) for k in range(m)]
        y_vars = [A_var(k, j + 1) for k in range(m)]
        encode_lex_leq(cnf, x_vars, y_vars, new_var)
    
    # ---- For each y in S: some row detects y ----
    for k, y in enumerate(S_ys):
        pos_positions = [j for j in range(n) if y[j] == 1]
        neg_positions = [j for j in range(n) if y[j] == -1]
        row_dets = []
        for i in range(m):
            det = new_var()
            row_dets.append(det)
            pos_gt = new_var()
            neg_gt = new_var()
            # det → pos_gt ∨ neg_gt
            cnf.append([-det, pos_gt, neg_gt])
            
            # pos_gt → sum_pos - sum_neg ≥ 1  ⇔ sum(A[i,pos]) + sum(¬A[i,neg]) ≥ |neg|+1
            lits_side1 = [A_var(i, j) for j in pos_positions] + \
                         [-A_var(i, j) for j in neg_positions]
            bound1 = len(neg_positions) + 1
            atleast1 = CardEnc.atleast(lits=lits_side1, bound=bound1, 
                                        top_id=next_var[0] - 1, encoding=EncType.seqcounter)
            for clause in atleast1.clauses:
                cnf.append(clause + [-pos_gt])
            if atleast1.nv > next_var[0] - 1:
                next_var[0] = atleast1.nv + 1
            
            # neg_gt → sum_neg - sum_pos ≥ 1
            lits_side2 = [A_var(i, j) for j in neg_positions] + \
                         [-A_var(i, j) for j in pos_positions]
            bound2 = len(pos_positions) + 1
            atleast2 = CardEnc.atleast(lits=lits_side2, bound=bound2,
                                        top_id=next_var[0] - 1, encoding=EncType.seqcounter)
            for clause in atleast2.clauses:
                cnf.append(clause + [-neg_gt])
            if atleast2.nv > next_var[0] - 1:
                next_var[0] = atleast2.nv + 1
        
        # some row detects y_k
        cnf.append(row_dets)
    
    if verbose:
        print(f"    SAT verifier: {cnf.nv} vars, {len(cnf.clauses)} clauses")
    
    with Solver(name='cadical195', bootstrap_with=cnf.clauses) as solver:
        sat = solver.solve()
        if not sat:
            return None
        model_set = set(solver.get_model())
        mat = np.zeros((m, n), dtype=np.int8)
        for i in range(m):
            for j in range(n):
                if A_var(i, j) in model_set:
                    mat[i, j] = 1
        return mat


# =========================================================================
# CEGAR loop
# =========================================================================

def cegar_canonical_smus(m, n, seed_matrices=None, max_iter=5000, verbose=True):
    Y = enumerate_canonical_balanced(n)
    n_ys = len(Y)
    Y_arr = np.array(Y, dtype=np.int8)
    if verbose:
        print(f"n={n}, m={m}: |B/±|={n_ys}")
    
    def kernel_of(mat):
        dots = Y_arr @ mat.T
        all_zero = np.all(dots == 0, axis=1)
        mask = 0
        for j in np.where(all_zero)[0]:
            mask |= (1 << int(j))
        return mask
    
    constraints = set()
    if seed_matrices:
        for mat in seed_matrices:
            k = kernel_of(mat)
            if k != 0:
                constraints.add(k)
    
    for it in range(max_iter):
        if not constraints:
            S_indices = []
            size = 0
        else:
            size, S_indices = solve_min_cover_maxsat(constraints, n_ys, verbose=False)
        
        if verbose and (it % 10 == 0 or it < 5):
            print(f"  iter {it}: |S|={size}, |constraints|={len(constraints)}")
        
        uncov_mat = find_uncovered_canonical_matrix(S_indices, Y, m, n, 
                                                      verbose=(it < 3 and verbose))
        
        if uncov_mat is None:
            if verbose:
                print(f"  Converged in {it+1} iterations: canSMUS = {size}")
            S = [Y[i] for i in S_indices]
            return size, S
        
        new_kernel = kernel_of(uncov_mat)
        if new_kernel == 0:
            if verbose:
                print(f"  ERROR: verifier returned a BD matrix (empty kernel).")
            return None, None
        if new_kernel in constraints:
            if verbose:
                print(f"  WARNING: verifier returned already-known constraint. Stopping.")
                print(f"  Matrix returned:\n{uncov_mat}")
            return size, [Y[i] for i in S_indices]
        constraints.add(new_kernel)
    
    if verbose:
        print(f"  Reached max_iter={max_iter} without convergence.")
    return None, None


# =========================================================================
# Demo
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("n=6 sanity check (expected canSMUS=8)")
    print("=" * 60)
    smus, S = cegar_canonical_smus(3, 6, verbose=True)
    if smus is not None:
        print(f"Result: canSMUS(3, 6) = {smus}")
    
    print()
    print("=" * 60)
    print("n=7 sanity check (expected canSMUS=39)")
    print("=" * 60)
    smus, S = cegar_canonical_smus(4, 7, verbose=True)
    if smus is not None:
        print(f"Result: canSMUS(4, 7) = {smus}")
    
    print()
    print("=" * 60)
    print("n=9 target")
    print("=" * 60)
    smus, S = cegar_canonical_smus(5, 9, verbose=True, max_iter=5000)
    if smus is not None:
        print(f"Result: canSMUS(5, 9) = {smus}")
