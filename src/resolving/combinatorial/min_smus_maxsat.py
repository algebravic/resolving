"""
MaxSAT-based SMUS solver for balanced-detecting UNSAT proofs.

Uses RC2 from python-sat (pysat) for the outer min-cover, and a CEGAR loop
with a SAT verifier for the inner "find uncovered canonical matrix" step.

Requires: pip install python-sat

For our problem the MaxSAT encoding is:

  Variables:
    x_y  ∈ {0, 1} for each y ∈ B(n)/±  (one Boolean per canonical balanced vector)

  Hard clauses:
    For each kernel K ⊆ B(n)/± (from an already-known canonical matrix A):
      OR_{y ∈ K} x_y     (cover this constraint)

  Soft clauses (all weight 1):
    ¬x_y   for each y  (we want as many y's OFF as possible)

  Minimizing weight of violated soft clauses = |{y : x_y = 1}| = SMUS size.

RC2 is a very efficient MaxSAT algorithm (Ignatiev, Morgado, Marques-Silva,
"RC2: an Efficient MaxSAT Solver", JSAT 2019). Its core-guided approach
exploits unsat cores from the SAT sub-calls and handles thousands to
millions of soft clauses well.

For CEGAR: after solving min cover on the current constraint set S_cons,
we ask "does there exist a canonical matrix A whose kernel is disjoint 
from the current selected S?" — a SAT question with variables 
{A_{ij}} ∈ {0, 1} and constraints (a) A is canonical (row weight ≤ n/2, 
rows lex-sorted, cols lex-sorted), (b) for every y ∈ S, some row detects
y (r_i · y ≠ 0). If SAT: found uncovered A, add its kernel to constraints
and iterate. If UNSAT: current S is a valid cover, return it.
"""

import itertools
import numpy as np
from math import comb


# =========================================================================
# Balanced-vector enumeration (import from min_smus.py if you prefer)
# =========================================================================

def enumerate_canonical_balanced(n):
    """B(n)/{±}, sorted by weight then lex."""
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
    
    constraint_masks: iterable of Python int bitmasks (bit j set iff y_j ∈ K)
    n_ys: number of y's (= number of variables)
    
    Returns (min_size, selected_indices).
    """
    from pysat.formula import WCNF
    from pysat.examples.rc2 import RC2
    
    wcnf = WCNF()
    # Variable j+1 corresponds to x_{y_j}  (pysat uses 1-indexed positive ints)
    
    # Hard clauses: one per constraint
    for mask in constraint_masks:
        clause = []
        m = mask
        idx = 0
        while m:
            if m & 1:
                clause.append(idx + 1)  # positive literal
            m >>= 1
            idx += 1
        if clause:
            wcnf.append(clause)  # HARD by default when no weight
    
    # Soft clauses: unit ¬x_y with weight 1 for each y
    for j in range(n_ys):
        wcnf.append([-(j + 1)], weight=1)
    
    with RC2(wcnf) as solver:
        model = solver.compute()
        if model is None:
            return None, None
        # model is a list of literals; positive j+1 means x_j = 1
        selected = [j for j in range(n_ys) if (j + 1) in model]
        return len(selected), selected


# =========================================================================
# SAT verifier for CEGAR: does an uncovered canonical matrix exist?
# =========================================================================

def find_uncovered_canonical_matrix(S_indices, Y, m, n, verbose=False):
    """Look for a canonical m×n matrix whose kernel is disjoint from S.
    
    Canonical form (matches Victor's scheme):
      - Each row weight ≤ n/2 (row complement broken)
      - Rows lex-sorted (row permutation broken)
      - Columns lex-sorted (column permutation broken)
    
    Kernel-disjoint condition: for each y ∈ S, some row r_i detects y
    (i.e., r_i · y ≠ 0).
    
    Encoded as pure SAT with auxiliary variables for detection indicators.
    
    Returns:
      - Matrix (as m×n numpy array of 0/1) if one exists (SAT)
      - None if provably no such matrix exists (UNSAT). In this case,
        S covers all canonical matrices and is a valid SMUS.
    """
    from pysat.formula import CNF
    from pysat.card import CardEnc, EncType
    from pysat.solvers import Solver
    
    Y_arr = np.array(Y, dtype=np.int8)
    S_ys = [Y_arr[i] for i in S_indices]
    
    # Variables:
    #   A[i,j] ∈ {0,1} for i in [m], j in [n]   →  var id = i*n + j + 1
    #   det[i,k] = "row i detects y_k for k-th y in S"  →  aux vars
    
    def A_var(i, j):
        return i * n + j + 1
    
    next_var = m * n + 1
    
    def new_var():
        nonlocal next_var
        v = next_var
        next_var += 1
        return v
    
    cnf = CNF()
    
    # ---- Row weight ≤ n/2 ----
    n_over_2 = n // 2
    for i in range(m):
        row_lits = [A_var(i, j) for j in range(n)]
        # Cardinality constraint sum ≤ n_over_2
        card = CardEnc.atmost(lits=row_lits, bound=n_over_2, 
                               top_id=next_var - 1, encoding=EncType.seqcounter)
        cnf.extend(card.clauses)
        if card.nv > next_var - 1:
            next_var = card.nv + 1
    
    # ---- Rows lex-sorted: row_i ≤ row_{i+1} for i = 0..m-2 ----
    # Standard lex-leq encoding: use auxiliary "equal-so-far" variables.
    for i in range(m - 1):
        eq = [None] * (n + 1)  # eq[k] means rows i, i+1 agree on positions 0..k-1
        eq[0] = new_var()
        cnf.append([eq[0]])  # eq[0] is always true
        for k in range(n):
            eq[k+1] = new_var()
            a = A_var(i, k)
            b = A_var(i+1, k)
            # eq[k+1] ↔ (eq[k] ∧ (a ↔ b))
            #   ⇒ ¬eq[k+1] ∨ eq[k]
            #   ⇒ ¬eq[k+1] ∨ (¬a ∨ b)  and  ¬eq[k+1] ∨ (a ∨ ¬b)
            cnf.append([-eq[k+1], eq[k]])
            cnf.append([-eq[k+1], -a, b])
            cnf.append([-eq[k+1], a, -b])
            #   ⇐ (eq[k] ∧ ¬a ∧ b) ∨ (eq[k] ∧ a ∧ ¬b are excluded), etc.
            # Actually easier: eq[k+1] ← eq[k] ∧ (a=b)
            cnf.append([-eq[k], a, -b, eq[k+1]])   # eq[k] ∧ ¬a=b → eq[k+1] (only if a=b)
            # If eq[k] ∧ a ∧ b: eq[k+1] must be true
            cnf.append([-eq[k], -a, -b, eq[k+1]])
            cnf.append([-eq[k], a, b, eq[k+1]])
            # Lex condition at position k: if eq[k] and a=1 and b=0, violated
            #   ⇒ ¬eq[k] ∨ ¬a ∨ b   (i.e., not both eq[k] and a>b)
            cnf.append([-eq[k], -a, b])
    
    # ---- Columns lex-sorted: col_j ≤ col_{j+1} for j = 0..n-2 ----
    # Same lex encoding applied to columns.
    for j in range(n - 1):
        eq = [None] * (m + 1)
        eq[0] = new_var()
        cnf.append([eq[0]])
        for k in range(m):
            eq[k+1] = new_var()
            a = A_var(k, j)
            b = A_var(k, j+1)
            cnf.append([-eq[k+1], eq[k]])
            cnf.append([-eq[k+1], -a, b])
            cnf.append([-eq[k+1], a, -b])
            cnf.append([-eq[k], a, -b, eq[k+1]])
            cnf.append([-eq[k], -a, -b, eq[k+1]])
            cnf.append([-eq[k], a, b, eq[k+1]])
            cnf.append([-eq[k], -a, b])  # lex condition
    
    # ---- For each y in S: some row detects y ----
    # Detection: r_i · y ≠ 0.
    # For balanced y with entries {-1, 0, 1}: r_i · y = sum_{j: y_j=+1} A[i,j] - sum_{j: y_j=-1} A[i,j].
    # This equals #(positions with y=+ and A=1) − #(positions with y=− and A=1).
    # It is 0 iff both counts are equal.
    # Introduce det[i,k] indicator; det[i,k] = 1 → r_i · y_k ≠ 0.
    # We encode: det[i,k] → (dot > 0 OR dot < 0), split by cases via aux vars.
    #
    # For simplicity, we use a pseudo-Boolean-style encoding:
    #   let p = sum over j:y_k[j]=+1  of A[i,j]  ; ranges [0, |pos support|]
    #   let m = sum over j:y_k[j]=-1  of A[i,j]  ; ranges [0, |neg support|]
    #   det[i,k] → (p ≠ m)
    # 
    # Encoded via: det[i,k] ↔ ¬(p = m), and requiring OR_i det[i,k] for each y_k.
    
    # The cleanest general encoding: for each row i and each y_k, introduce 
    # aux vars for p and m as unary counters, then equality-test.
    # This adds |support(y)| aux vars per (row, y_k). Manageable if |S| is 
    # small; can get large for many long-support y's.
    #
    # A simpler but weaker approach that works well in practice:
    #   Enumerate assignment patterns and directly encode detection.
    # For our problem, since m is small (~5) and |support(y)| is at most n,
    # we can use CardEnc to compare two sums:  det[i,k] → (p ≠ m).
    #
    # We do (p ≥ m + 1) ∨ (m ≥ p + 1) using two CardEnc.atleast calls
    # implicated by det.
    
    det_vars = {}  # (i, k) -> det var
    
    for k, y in enumerate(S_ys):
        pos_positions = [j for j in range(n) if y[j] == 1]
        neg_positions = [j for j in range(n) if y[j] == -1]
        
        row_dets = []
        for i in range(m):
            det = new_var()
            det_vars[(i, k)] = det
            row_dets.append(det)
            
            # We want: det → (sum_pos(A[i,·]) ≠ sum_neg(A[i,·]))
            # Equivalent: det → (sum_pos - sum_neg ≥ 1  OR  sum_neg - sum_pos ≥ 1)
            # Introduce two aux Bools: pos_gt (sum_pos > sum_neg), neg_gt (sum_neg > sum_pos)
            # det → pos_gt ∨ neg_gt
            pos_gt = new_var()
            neg_gt = new_var()
            cnf.append([-det, pos_gt, neg_gt])
            
            # pos_gt → sum(A[i,pos]) - sum(A[i,neg]) ≥ 1
            #   equivalently: sum(A[i,pos]) + sum(1 - A[i,neg]) ≥ len(neg) + 1
            #   which is a cardinality constraint on the |pos|+|neg| literals
            #   (positive for A[i,pos_j], negative for A[i,neg_j] then negated → positive again after complement).
            # Simpler: use a big-M-style encoding:
            #   pos_gt → (sum_pos ≥ 1) ∧ (for each t, sum_pos ≥ t → sum_neg ≤ t-1)
            # For our small m (≤ 7 or so), a direct encoding via CardEnc.atleast works:
            #   sum(A[i,pos_j]) - sum(A[i,neg_j]) ≥ 1
            # can be rewritten: sum(A[i,pos_j]) + sum(¬A[i,neg_j]) ≥ len(neg) + 1
            lits_pos = [A_var(i, j) for j in pos_positions]
            lits_neg_neg = [-A_var(i, j) for j in neg_positions]
            lits_side1 = lits_pos + lits_neg_neg
            bound1 = len(neg_positions) + 1
            # We want:  pos_gt → sum(lits_side1) ≥ bound1
            # Standard trick: encode sum ≥ bound1 with CardEnc, then implicate.
            # Since CardEnc doesn't directly support implication, add pos_gt's 
            # negation as a "relaxation literal" to each of the ≥ bound1's clauses,
            # or use a "conditional" encoding.
            #
            # Practical approach: encode the atleast as clauses that are 
            # relaxed by ¬pos_gt (adding ¬pos_gt to each). Since CardEnc.atleast 
            # returns a CNF, we augment.
            atleast1 = CardEnc.atleast(lits=lits_side1, bound=bound1, 
                                        top_id=next_var - 1, encoding=EncType.seqcounter)
            for clause in atleast1.clauses:
                cnf.append(clause + [-pos_gt])
            if atleast1.nv > next_var - 1:
                next_var = atleast1.nv + 1
            
            # neg_gt → sum(A[i,neg_j]) - sum(A[i,pos_j]) ≥ 1
            lits_side2 = [A_var(i, j) for j in neg_positions] + [-A_var(i, j) for j in pos_positions]
            bound2 = len(pos_positions) + 1
            atleast2 = CardEnc.atleast(lits=lits_side2, bound=bound2,
                                        top_id=next_var - 1, encoding=EncType.seqcounter)
            for clause in atleast2.clauses:
                cnf.append(clause + [-neg_gt])
            if atleast2.nv > next_var - 1:
                next_var = atleast2.nv + 1
        
        # OR over rows: some row detects y_k
        cnf.append(row_dets)
    
    if verbose:
        print(f"    SAT verifier: {cnf.nv} vars, {len(cnf.clauses)} clauses")
    
    with Solver(name='cadical195', bootstrap_with=cnf.clauses) as solver:
        sat = solver.solve()
        if not sat:
            return None  # UNSAT: no uncovered canonical matrix exists
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

def cegar_canonical_smus(m, n, seed_matrices=None, max_iter=1000, verbose=True):
    """CEGAR-style computation of canonical-scheme SMUS.
    
    Uses RC2 MaxSAT for the min-cover step and a SAT verifier for 
    "find uncovered canonical matrix".
    """
    Y = enumerate_canonical_balanced(n)
    n_ys = len(Y)
    Y_arr = np.array(Y, dtype=np.int8)
    if verbose:
        print(f"n={n}, m={m}: |B/±|={n_ys}")
    
    # Seed with a few canonical matrices (e.g., random samples), so we have 
    # SOME constraints to start. If no seed provided, start with an empty set 
    # and let the verifier produce them all.
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
        # Solve min-cover on current constraints
        if not constraints:
            # No constraints yet — any singleton (say {y_0}) is a cover
            S_indices = []
            size = 0
        else:
            size, S_indices = solve_min_cover_maxsat(constraints, n_ys, verbose=False)
        
        if verbose and (it % 10 == 0 or it < 5):
            print(f"  iter {it}: |S|={size}, |constraints|={len(constraints)}")
        
        # Verifier: find an uncovered canonical matrix
        uncov_mat = find_uncovered_canonical_matrix(S_indices, Y, m, n, verbose=(it < 3 and verbose))
        
        if uncov_mat is None:
            if verbose:
                print(f"  Converged in {it+1} iterations: canSMUS = {size}")
            S = [Y[i] for i in S_indices]
            return size, S
        
        # Add the new constraint
        new_kernel = kernel_of(uncov_mat)
        if new_kernel == 0:
            # Empty kernel = balanced-detecting matrix; means SAT, not UNSAT
            if verbose:
                print(f"  ERROR: verifier returned a BD matrix (empty kernel). Not at UNSAT boundary.")
            return None, None
        if new_kernel in constraints:
            # Should not happen if verifier is correct
            if verbose:
                print(f"  WARNING: verifier returned already-known constraint. Stopping.")
            return size, [Y[i] for i in S_indices]
        constraints.add(new_kernel)
    
    if verbose:
        print(f"  Reached max_iter={max_iter} without convergence.")
    return None, None


# =========================================================================
# Demo
# =========================================================================

ALPHA = {3: 2, 4: 3, 5: 3, 6: 4, 7: 5, 8: 5, 9: 6, 10: 6, 11: 7, 12: 7, 13: 7}


if __name__ == "__main__":
    # Verify at n=6 first (should give canSMUS = 8)
    print("=" * 60)
    print("n=6 sanity check (expected canSMUS=8)")
    print("=" * 60)
    smus, S = cegar_canonical_smus(3, 6, verbose=True)
    if smus is not None:
        print(f"Result: canSMUS(3, 6) = {smus}")
    
    # Then push toward n=9
    print()
    print("=" * 60)
    print("n=9 target")
    print("=" * 60)
    smus, S = cegar_canonical_smus(5, 9, verbose=True, max_iter=5000)
    if smus is not None:
        print(f"Result: canSMUS(5, 9) = {smus}")
