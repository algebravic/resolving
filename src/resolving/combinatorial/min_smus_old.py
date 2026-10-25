"""
Minimum SMUS calculator for balanced-detecting UNSAT proofs.

For (m, n) at the UNSAT boundary — the largest m such that no m×n
balanced-detecting matrix exists — computes the minimum size of a
subset S ⊆ B(n)/{±} such that every m×n 0/1 matrix has some y ∈ S
in its kernel (i.e., Ay = 0).

Approach:
    1. Direct ILP for small n (via scipy.optimize.milp with HiGHS)
    2. For each matrix A, the constraint is: sum(x_y : Ay = 0) ≥ 1
    3. Minimize sum(x_y).

Deduplication and dominance-removal on constraints keeps the ILP tractable.

For larger n where direct enumeration of matrices is infeasible, a
lazy-constraint-generation (CEGAR-style) version is provided.
"""

import itertools
import time
import numpy as np
from math import comb
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


# α(n) = min m such that an m×n BD matrix exists = β(Q_n) - 1
# β(Q_n) values from OEIS A303735
ALPHA = {1: 0, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 5, 8: 5, 9: 6, 10: 6,
         11: 7, 12: 7, 13: 7, 14: 8}


def unsat_boundary(n):
    """Largest m for which no m×n BD matrix exists (UNSAT)."""
    return ALPHA[n] - 1


def enumerate_canonical_balanced(n):
    """Enumerate canonical balanced vectors: B(n)/{±}, sorted by weight then lex."""
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


def build_constraints(Y, m, n):
    """For each m-subset of distinct rows, compute constraint = bitmask of y's in joint kernel.
    
    Optimization: rows r and (1-r) give the same K(r) on canonical B(n)/±
    (since for balanced y: (1-r)·y = -r·y, so K(1-r) = K(r)).
    So we canonicalize rows to have first '1' set (or all-zeros).
    Then enumerate m-subsets over 2^(n-1) canonical rows.
    Matrices with repeated K-values give subsumed (superset) kernels, so
    it suffices to enumerate distinct-K matrices for finding minimal constraints.
    
    Returns:
        (constraints_set, num_kernels_enumerated, num_empty_kernel)
    """
    n_ys = len(Y)
    Y_arr = np.array(Y, dtype=np.int8)  # shape (n_ys, n)
    
    # Enumerate rows in {0, 1}^n, canonicalize to representatives under complement.
    # Choose the row with r[0] == 0 as canonical (or, all-zero row).
    canon_rows = []
    for r_int in range(2 ** (n - 1)):  # r[0] = 0
        r = [(r_int >> j) & 1 for j in range(n - 1)]
        canon_rows.append(tuple([0] + r))
    # (n_canon = 2^(n-1))
    n_canon = len(canon_rows)
    
    # Compute K-masks for canonical rows
    row_masks = [0] * n_canon
    for i, r in enumerate(canon_rows):
        r_arr = np.array(r, dtype=np.int8)
        dots = Y_arr @ r_arr
        mask = 0
        for j in range(n_ys):
            if int(dots[j]) == 0:
                mask |= (1 << j)
        row_masks[i] = mask
    
    # For each m-subset of canonical rows, joint kernel = AND
    constraints_set = set()
    num_kernels = 0
    num_empty = 0
    for m_tuple in itertools.combinations(range(n_canon), m):
        joint = row_masks[m_tuple[0]]
        for i in m_tuple[1:]:
            joint &= row_masks[i]
            if joint == 0:
                break
        num_kernels += 1
        if joint == 0:
            num_empty += 1
        else:
            constraints_set.add(joint)
    
    return constraints_set, num_kernels, num_empty


def remove_dominated(constraints_set):
    """Remove dominated constraints: c is dominated if some c' ⊂ c exists.
    
    Uses bitmask subset check. O(N²) in number of constraints.
    Skipped when N is too large.
    """
    sorted_by_popcount = sorted(constraints_set, key=lambda c: bin(c).count('1'))
    minimal = []
    for c in sorted_by_popcount:
        dominated = False
        for mc in minimal:
            if mc & c == mc:  # mc is subset of c → c is dominated
                dominated = True
                break
        if not dominated:
            minimal.append(c)
    return minimal


def remove_dominated_fast(constraints_set, n_bits, verbose=False):
    """Fast dominance removal using numpy batched by popcount level.
    
    Strategy:
      - Group constraints by popcount. Within a popcount level, no dominance is possible
        (subset requires strictly fewer bits set).
      - For each popcount level p (ascending), batch-check candidates against all
        previously-accepted minimals via numpy uint64 bitmask AND.
      - Chunk candidates to keep memory bounded.
    """
    n_words = (n_bits + 63) // 64
    UINT64_MAX = (1 << 64) - 1
    
    def masks_to_words_array(masks):
        n = len(masks)
        arr = np.zeros((n, n_words), dtype=np.uint64)
        for i, mask in enumerate(masks):
            for w in range(n_words):
                arr[i, w] = np.uint64((mask >> (w * 64)) & UINT64_MAX)
        return arr
    
    # Group by popcount
    by_pop = {}
    for c in constraints_set:
        p = bin(c).count('1')
        by_pop.setdefault(p, []).append(c)
    
    accepted_masks = []
    accepted_words = np.zeros((0, n_words), dtype=np.uint64)
    
    for p in sorted(by_pop):
        candidates = by_pop[p]
        cand_words = masks_to_words_array(candidates)
        n_cand = len(candidates)
        n_acc = len(accepted_masks)
        
        if n_acc == 0:
            dominated = np.zeros(n_cand, dtype=bool)
        else:
            # Chunk to bound memory of intermediate: chunk_n * n_acc * n_words * 8 bytes
            # Target ~200 MB per chunk
            max_bytes = 200 * 1024 * 1024
            per_cand_bytes = n_acc * n_words * 8
            chunk_size = max(1, max_bytes // max(per_cand_bytes, 1))
            
            dominated = np.zeros(n_cand, dtype=bool)
            for start in range(0, n_cand, chunk_size):
                end = min(start + chunk_size, n_cand)
                chunk = cand_words[start:end]  # (chunk_n, n_words)
                # Broadcast: (chunk_n, n_acc, n_words)
                anded = np.bitwise_and(chunk[:, None, :], accepted_words[None, :, :])
                eq = anded == accepted_words[None, :, :]
                is_subset = np.all(eq, axis=2)  # (chunk_n, n_acc)
                dominated[start:end] = is_subset.any(axis=1)
        
        # Accept non-dominated
        non_dom_idx = np.where(~dominated)[0]
        new_accepted = [candidates[i] for i in non_dom_idx]
        new_words = cand_words[non_dom_idx]
        accepted_masks.extend(new_accepted)
        accepted_words = np.vstack([accepted_words, new_words]) if len(new_accepted) > 0 else accepted_words
        
        if verbose:
            print(f"    popcount {p}: {n_cand} candidates → {len(new_accepted)} added, {len(accepted_masks)} total minimals")
    
    return accepted_masks


def mask_to_columns(mask, n_ys):
    """Convert bitmask to list of column indices."""
    return [j for j in range(n_ys) if mask & (1 << j)]


def solve_smus(m, n, verbose=True, use_dominance=True, dominance_threshold=15000):
    """Compute min SMUS via direct ILP.
    
    Suitable for small n. Time and memory grow with (2^n)^m matrix count.
    """
    t0 = time.time()
    Y = enumerate_canonical_balanced(n)
    n_ys = len(Y)
    
    if verbose:
        num_matrices_expected = comb(2**n, m)
        print(f"({m}, {n}): {n_ys} canonical balanced vectors, {num_matrices_expected} matrices to check")
    
    t1 = time.time()
    constraints_set, num_matrices, num_empty = build_constraints(Y, m, n)
    t2 = time.time()
    
    if verbose:
        print(f"  Constraints built in {t2-t1:.2f}s: {len(constraints_set)} unique of {num_matrices} matrices")
    
    if num_empty > 0:
        print(f"  ERROR: {num_empty} matrices have empty kernel — (m={m}, n={n}) is NOT UNSAT")
        return None, None
    
    # Optional: remove dominated constraints
    if use_dominance:
        t3 = time.time()
        if len(constraints_set) <= dominance_threshold:
            minimal = remove_dominated(constraints_set)
        else:
            if verbose:
                print(f"  Using fast numpy dominance removal ({len(constraints_set)} constraints)...")
            minimal = remove_dominated_fast(constraints_set, n_ys, verbose=verbose)
        t4 = time.time()
        if verbose:
            print(f"  Dominance removal in {t4-t3:.2f}s: {len(minimal)} minimal constraints")
    else:
        minimal = list(constraints_set)
    
    # Build ILP constraint matrix
    num_constraints = len(minimal)
    A = lil_matrix((num_constraints, n_ys))
    for i, mask in enumerate(minimal):
        for j in mask_to_columns(mask, n_ys):
            A[i, j] = 1
    A = A.tocsr()
    
    lb = np.ones(num_constraints)
    ub = np.full(num_constraints, np.inf)
    c_obj = np.ones(n_ys)
    integrality = np.ones(n_ys)
    bounds = Bounds(0, 1)
    
    if verbose:
        print(f"  Solving ILP with {n_ys} vars, {num_constraints} constraints...")
    t5 = time.time()
    result = milp(c_obj, constraints=LinearConstraint(A, lb=lb, ub=ub),
                  integrality=integrality, bounds=bounds)
    t6 = time.time()
    
    if not result.success:
        print(f"  ILP failed: {result.message}")
        return None, None
    
    min_smus = int(round(result.fun))
    S = [Y[i] for i in range(n_ys) if result.x[i] > 0.5]
    
    if verbose:
        print(f"  ILP solved in {t6-t5:.2f}s → min SMUS = {min_smus}")
        print(f"  Total time: {t6-t0:.2f}s")
    return min_smus, S


def format_y(y):
    """Format a ternary vector as a compact string."""
    return ''.join({1: '+', 0: '.', -1: '-'}[v] for v in y)


def weight_of(y):
    return sum(abs(v) for v in y)


def show_witness(S, max_show=30):
    """Display a witness cover, grouped by weight."""
    by_weight = {}
    for y in S:
        w = weight_of(y)
        by_weight.setdefault(w, []).append(y)
    for w in sorted(by_weight):
        print(f"    Weight {w}: {len(by_weight[w])} vectors")
        if sum(len(by_weight[k]) for k in by_weight) <= max_show:
            for y in by_weight[w]:
                print(f"      {format_y(y)}")


# =========================================================================
# Lazy constraint generation (CEGAR-style) for larger n
# =========================================================================

def compute_row_detect_masks(Y, n):
    """For each row r ∈ {0,1}^n, compute bitmask of y's it DETECTS (r·y ≠ 0)."""
    n_ys = len(Y)
    n_rows = 2 ** n
    Y_arr = np.array(Y, dtype=np.int8)
    detect_masks = [0] * n_rows
    kernel_masks = [0] * n_rows
    for r_int in range(n_rows):
        r = np.array([(r_int >> j) & 1 for j in range(n)], dtype=np.int8)
        dots = Y_arr @ r
        d_mask = 0
        k_mask = 0
        for j in range(n_ys):
            if dots[j] != 0:
                d_mask |= (1 << j)
            else:
                k_mask |= (1 << j)
        detect_masks[r_int] = d_mask
        kernel_masks[r_int] = k_mask
    return detect_masks, kernel_masks


def find_uncovered_matrix(S_indices, m, n, kernel_masks, detect_masks, n_ys):
    """Find a matrix A (m distinct rows) such that A y ≠ 0 for all y in S.
    
    A is not covered by S iff every y in S is DETECTED by some row of A.
    Equivalent to: exists m-subset of rows whose union of detect-masks ⊇ S.
    
    Solved via ILP: min sum x_r subject to Σ_{r detects y} x_r ≥ 1 for y ∈ S,
    Σ x_r ≤ m. Returns list of row-integers if feasible, else None.
    
    We use ILP to find MIN cover, then check if it fits in m rows.
    """
    n_rows = 2 ** n
    # Build set-cover ILP for the row-selection problem
    A_matrix = lil_matrix((len(S_indices), n_rows))
    for i, y_idx in enumerate(S_indices):
        y_bit = 1 << y_idx
        for r in range(n_rows):
            if detect_masks[r] & y_bit:
                A_matrix[i, r] = 1
    A_matrix = A_matrix.tocsr()
    
    lb = np.ones(len(S_indices))
    ub = np.full(len(S_indices), np.inf)
    c_obj = np.ones(n_rows)
    integrality = np.ones(n_rows)
    bounds = Bounds(0, 1)
    
    result = milp(c_obj, constraints=LinearConstraint(A_matrix, lb=lb, ub=ub),
                  integrality=integrality, bounds=bounds)
    if not result.success:
        return None  # S covers everything (no uncovered matrix exists)
    
    n_rows_needed = int(round(result.fun))
    if n_rows_needed > m:
        return None  # need more than m rows to detect S — so S covers all m-matrices
    
    # Extract selected rows
    selected = [r for r in range(n_rows) if result.x[r] > 0.5]
    return selected


def solve_smus_lazy(m, n, verbose=True, max_iter=500, seed_matrices=None):
    """Compute min SMUS via lazy constraint generation (CEGAR-style).
    
    Suitable for larger n where direct constraint enumeration is infeasible.
    """
    t0 = time.time()
    Y = enumerate_canonical_balanced(n)
    n_ys = len(Y)
    
    if verbose:
        num_matrices_expected = comb(2**n, m)
        print(f"({m}, {n}): {n_ys} canonical balanced vectors, "
              f"{num_matrices_expected} matrices in principle")
    
    if verbose:
        print("  Precomputing row masks...")
    t1 = time.time()
    detect_masks, kernel_masks = compute_row_detect_masks(Y, n)
    if verbose:
        print(f"  Row masks done in {time.time()-t1:.2f}s")
    
    # Bootstrap constraints
    constraints = set()  # frozensets of y-indices (or bitmasks)
    if seed_matrices is None:
        # Seed with all single-row "matrices": kernel of each row (only works for m=1)
        # For m > 1: seed with some random m-subsets
        n_rows = 2 ** n
        import random
        rng = random.Random(42)
        n_seeds = min(100, comb(n_rows, m))
        seed_matrices = set()
        while len(seed_matrices) < n_seeds:
            rows = tuple(sorted(rng.sample(range(n_rows), m)))
            seed_matrices.add(rows)
        
        for rows in seed_matrices:
            joint = kernel_masks[rows[0]]
            for r in rows[1:]:
                joint &= kernel_masks[r]
            if joint > 0:
                constraints.add(joint)
    
    if verbose:
        print(f"  Bootstrapped with {len(constraints)} initial constraints")
    
    prev_smus = 0
    for iteration in range(max_iter):
        # Solve current ILP
        constraint_list = list(constraints)
        A = lil_matrix((len(constraint_list), n_ys))
        for i, mask in enumerate(constraint_list):
            for j in mask_to_columns(mask, n_ys):
                A[i, j] = 1
        A = A.tocsr()
        
        lb = np.ones(len(constraint_list))
        ub = np.full(len(constraint_list), np.inf)
        c_obj = np.ones(n_ys)
        integrality = np.ones(n_ys)
        bounds = Bounds(0, 1)
        
        result = milp(c_obj, constraints=LinearConstraint(A, lb=lb, ub=ub),
                      integrality=integrality, bounds=bounds)
        if not result.success:
            print(f"  ILP failed at iter {iteration}: {result.message}")
            return None, None
        
        current_smus = int(round(result.fun))
        S_indices = [i for i in range(n_ys) if result.x[i] > 0.5]
        
        if verbose and (iteration % 10 == 0 or current_smus != prev_smus):
            print(f"  Iter {iteration}: |S|={current_smus}, constraints={len(constraint_list)}")
        prev_smus = current_smus
        
        # Check: does S cover everything? Find uncovered matrix if any
        uncovered = find_uncovered_matrix(S_indices, m, n, kernel_masks, detect_masks, n_ys)
        
        if uncovered is None:
            # S covers all matrices
            elapsed = time.time() - t0
            if verbose:
                print(f"  Converged at iter {iteration}: min SMUS = {current_smus} ({elapsed:.1f}s)")
            S = [Y[i] for i in S_indices]
            return current_smus, S
        
        # Add the constraint corresponding to the uncovered matrix
        joint = kernel_masks[uncovered[0]]
        for r in uncovered[1:]:
            joint &= kernel_masks[r]
        if joint == 0:
            print(f"  ERROR: found empty-kernel matrix — not UNSAT boundary")
            return None, None
        
        if joint in constraints:
            # Shouldn't happen, but bail if we keep adding same constraint
            print(f"  Constraint already present — bailing")
            return None, None
        
        constraints.add(joint)
    
    print(f"  Reached max_iter {max_iter} without converging")
    return None, None


# =========================================================================
# CLI: compute the sequence
# =========================================================================

if __name__ == "__main__":
    print("Minimum SMUS for balanced-detecting UNSAT at the boundary")
    print()
    
    values = []
    for n in [3, 4, 5, 6, 7, 8]:
        m = unsat_boundary(n)
        if m < 1:
            print(f"n={n}: m={m} skipped (trivial)")
            continue
        print(f"\n=== n = {n}, m = {m} (UNSAT boundary) ===")
        t_start = time.time()
        try:
            min_smus, S = solve_smus(m, n, verbose=True)
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            min_smus, S = None, None
        t_elapsed = time.time() - t_start
        
        if min_smus is not None:
            print(f"  Result: min SMUS({m}, {n}) = {min_smus}")
            values.append((n, m, min_smus))
            if len(S) <= 30:
                print(f"  Witness set S (size {len(S)}):")
                show_witness(S, max_show=30)
        else:
            print(f"  FAILED at n = {n}")
            break
    
    print("\n" + "=" * 50)
    print("SEQUENCE (n, m = α(n) - 1, min SMUS):")
    print(f"{'n':>4} {'m':>4} {'min SMUS':>10}")
    for n, m, v in values:
        print(f"{n:>4} {m:>4} {v:>10}")
