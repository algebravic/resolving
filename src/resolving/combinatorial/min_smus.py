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
# Quotient graph / symmetry-based analysis
# =========================================================================

def weight_class_indices(Y):
    """Partition Y indices by weight; returns dict weight -> list of indices."""
    classes = {}
    for i, y in enumerate(Y):
        w = sum(abs(v) for v in y)
        classes.setdefault(w, []).append(i)
    return classes


def constraint_profile(mask, weight_classes):
    """For a constraint bitmask, return the tuple (|K∩W_w| for w in sorted weights).
    
    Two constraints with the same profile are equivalent for INVARIANT covers.
    """
    profile = []
    for w in sorted(weight_classes):
        indices = weight_classes[w]
        count = sum(1 for j in indices if mask & (1 << j))
        profile.append(count)
    return tuple(profile)


def invariant_smus(m, n, verbose=True):
    """Compute the minimum G-invariant cover: S is a union of full weight classes.
    
    Returns (min_size, subset_of_weights, all_invariant_options).
    """
    Y = enumerate_canonical_balanced(n)
    classes = weight_class_indices(Y)
    weights = sorted(classes.keys())
    class_sizes = {w: len(classes[w]) for w in weights}
    
    if verbose:
        print(f"Weight class sizes at n={n}: {class_sizes}")
    
    # Build constraints (reuse existing pipeline)
    constraints_set, num_kernels, num_empty = build_constraints(Y, m, n)
    if num_empty > 0:
        print(f"  ERROR: {num_empty} empty-kernel matrices — not UNSAT boundary")
        return None, None, None
    
    # For each constraint, compute profile
    profiles = set()
    for mask in constraints_set:
        profiles.add(constraint_profile(mask, classes))
    if verbose:
        print(f"  {len(profiles)} unique constraint profiles across {len(constraints_set)} unique constraints")
    
    # Enumerate all 2^|weights| invariant subsets
    results = []
    for subset_bits in range(1, 1 << len(weights)):
        chosen_weights = [weights[i] for i in range(len(weights)) if subset_bits & (1 << i)]
        size = sum(class_sizes[w] for w in chosen_weights)
        # Check coverage: for each profile, at least one chosen weight has count ≥ 1
        chosen_indices = [i for i in range(len(weights)) if subset_bits & (1 << i)]
        covers = all(any(profile[i] >= 1 for i in chosen_indices) for profile in profiles)
        results.append((size, chosen_weights, covers))
    
    covering = [(size, weights_used) for size, weights_used, ok in results if ok]
    if not covering:
        return None, None, results
    covering.sort()
    return covering[0][0], covering[0][1], results


def lp_lower_bound(m, n, verbose=True):
    """LP relaxation of the invariant ILP: variables w_i ∈ [0,1] for each weight class.
    
    Since the full LP has an S_n-invariant optimum, this equals LP LB of full problem.
    Returns fractional lower bound on SMUS.
    """
    Y = enumerate_canonical_balanced(n)
    classes = weight_class_indices(Y)
    weights = sorted(classes.keys())
    class_sizes = [len(classes[w]) for w in weights]
    
    constraints_set, _, num_empty = build_constraints(Y, m, n)
    if num_empty > 0:
        return None
    
    profiles = set()
    for mask in constraints_set:
        profiles.add(constraint_profile(mask, classes))
    profiles = list(profiles)
    
    # LP: min sum(|W_i| w_i)
    # s.t. sum(profile[i] * w_i) ≥ 1 for each profile
    #      0 ≤ w_i ≤ 1
    from scipy.optimize import linprog
    c = np.array(class_sizes, dtype=float)
    A_ub = -np.array(profiles, dtype=float)  # -profile · w ≤ -1
    b_ub = -np.ones(len(profiles))
    bounds_lp = [(0, 1)] * len(weights)
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
    if not result.success:
        return None
    lb = result.fun
    if verbose:
        w_vals = result.x
        breakdown = ", ".join(f"w{weights[i]}={w_vals[i]:.4f}" for i in range(len(weights)))
        print(f"  LP lower bound: {lb:.4f}  ({breakdown})")
    return lb


def summarize_by_weight(S):
    """Group vectors in S by weight; return dict."""
    by_w = {}
    for y in S:
        w = sum(abs(v) for v in y)
        by_w.setdefault(w, []).append(y)
    return by_w


def adjacent_transposition_cuts(Y, n):
    """Generate symmetry-breaking constraints x_y ≥ x_{τy} for adjacent
    transpositions τ = (a, a+1), a = 0..n-2.
    
    Returns list of (i, j) pairs meaning x_i ≥ x_j (equivalently x_i - x_j ≥ 0).
    """
    Y_tuples = [tuple(y) for y in Y]
    y_index = {y: i for i, y in enumerate(Y_tuples)}
    
    def apply_swap(y, a):
        """Apply adjacent transposition (a, a+1) to y, then canonicalize under ±."""
        y_list = list(y)
        y_list[a], y_list[a + 1] = y_list[a + 1], y_list[a]
        y_swapped = tuple(y_list)
        return min(y_swapped, tuple(-x for x in y_swapped))
    
    cuts = set()
    for a in range(n - 1):
        for y in Y_tuples:
            y_swap = apply_swap(y, a)
            if y == y_swap:
                continue
            # Add x_y ≥ x_{y_swap} whenever y < y_swap lex
            if y < y_swap:
                i, j = y_index[y], y_index[y_swap]
                cuts.add((i, j))
    return sorted(cuts)


def solve_smus_with_symmetry(m, n, verbose=True, use_dominance=True):
    """Direct ILP with adjacent-transposition symmetry-breaking cuts.
    
    Adds x_y ≥ x_{τy} constraints, which don't change the optimum but tighten
    the LP relaxation and prune equivalent branches.
    """
    t0 = time.time()
    Y = enumerate_canonical_balanced(n)
    n_ys = len(Y)
    
    if verbose:
        print(f"({m}, {n}): {n_ys} canonical balanced vectors")
    
    t1 = time.time()
    constraints_set, num_kernels, num_empty = build_constraints(Y, m, n)
    t2 = time.time()
    if verbose:
        print(f"  Constraints built in {t2-t1:.2f}s: {len(constraints_set)} unique")
    if num_empty > 0:
        print(f"  ERROR: {num_empty} empty-kernel matrices")
        return None, None
    
    if use_dominance and len(constraints_set) <= 15000:
        minimal = remove_dominated(constraints_set)
        if verbose:
            print(f"  Dominance-minimal: {len(minimal)}")
    else:
        minimal = list(constraints_set)
    
    # Generate symmetry cuts
    sym_cuts = adjacent_transposition_cuts(Y, n)
    if verbose:
        print(f"  Symmetry cuts (adjacent-transposition): {len(sym_cuts)}")
    
    # Build combined constraint matrix
    num_cov = len(minimal)
    num_sym = len(sym_cuts)
    A = lil_matrix((num_cov + num_sym, n_ys))
    for i, mask in enumerate(minimal):
        for j in mask_to_columns(mask, n_ys):
            A[i, j] = 1
    for k, (i, j) in enumerate(sym_cuts):
        A[num_cov + k, i] = 1
        A[num_cov + k, j] = -1
    A = A.tocsr()
    
    lb = np.concatenate([np.ones(num_cov), np.zeros(num_sym)])
    ub = np.concatenate([np.full(num_cov, np.inf), np.full(num_sym, np.inf)])
    c_obj = np.ones(n_ys)
    integrality = np.ones(n_ys)
    bounds = Bounds(0, 1)
    
    if verbose:
        print(f"  Solving ILP: {n_ys} vars, {num_cov + num_sym} constraints "
              f"({num_cov} cover + {num_sym} sym)")
    t5 = time.time()
    result = milp(c_obj, constraints=LinearConstraint(A, lb=lb, ub=ub),
                  integrality=integrality, bounds=bounds)
    t6 = time.time()
    if verbose:
        print(f"  ILP solved in {t6-t5:.2f}s")
    
    if not result.success:
        print(f"  ILP failed: {result.message}")
        return None, None
    
    smus = int(round(result.fun))
    S = [Y[i] for i in range(n_ys) if result.x[i] > 0.5]
    if verbose:
        print(f"  min SMUS = {smus}, total time {t6-t0:.1f}s")
    return smus, S


def has_isolating_matrix(y_target, Y, m, n):
    """Does there exist an m×n 0/1 matrix A with ker(A) ∩ B(n)/± = {y_target}?
    
    Equivalent: some m-subset of canonical rows S with
      - every r ∈ S has r · y_target = 0  (y_target is in kernel)
      - for every other y' ∈ B(n)/±, some r ∈ S has r · y' ≠ 0  (y' is detected)
    
    Solved as set-cover feasibility ILP:
      variables x_r ∈ {0,1} for canonical rows r with r · y_target = 0
      constraints:  for each y' ≠ y_target,   sum_{r : r·y'≠0} x_r ≥ 1
                    sum x_r ≤ m
    Returns True if feasible.
    """
    n_ys = len(Y)
    Y_arr = np.array(Y, dtype=np.int8)
    y_target_idx = Y.index(tuple(y_target)) if isinstance(y_target, (list, np.ndarray)) else Y.index(y_target)
    y_t = np.array(Y[y_target_idx], dtype=np.int8)
    
    # Canonical rows: r ∈ {0,1}^n with r[0] = 0 (representatives under complement)
    n_canon = 2 ** (n - 1)
    admissible = []  # rows with r · y_target = 0
    dot_matrix = []  # for admissible rows, the dots r · y' for all y'
    for r_int in range(n_canon):
        r_bits = [(r_int >> j) & 1 for j in range(n - 1)]
        r = np.array([0] + r_bits, dtype=np.int8)
        if int(r @ y_t) != 0:
            continue
        admissible.append(r_int)
        dot_matrix.append(Y_arr @ r)  # length n_ys
    if len(admissible) < m:
        return False  # not enough admissible rows
    
    dot_matrix = np.array(dot_matrix)  # shape (n_admissible, n_ys)
    n_adm = len(admissible)
    
    # Constraints: for each y' ≠ y_target,  sum_{r : dot_matrix[r, y'] ≠ 0} x_r ≥ 1
    rows = []
    for y_idx in range(n_ys):
        if y_idx == y_target_idx:
            continue
        row_coeffs = (dot_matrix[:, y_idx] != 0).astype(float)
        if row_coeffs.sum() == 0:
            # No admissible row detects y'; can't isolate y_target
            return False
        rows.append(row_coeffs)
    A_ineq = np.array(rows)  # each row is a coverage constraint
    
    # Add cardinality constraint sum(x) ≤ m
    card_row = np.ones(n_adm)
    A_stack = np.vstack([A_ineq, card_row.reshape(1, -1)])
    lb = np.concatenate([np.ones(len(rows)), [-np.inf]])
    ub = np.concatenate([np.full(len(rows), np.inf), [float(m)]])
    
    c_obj = np.zeros(n_adm)  # feasibility only
    integrality = np.ones(n_adm)
    bounds = Bounds(0, 1)
    result = milp(c_obj, constraints=LinearConstraint(A_stack, lb=lb, ub=ub),
                  integrality=integrality, bounds=bounds)
    return result.success


def singleton_witness_analysis(m, n, verbose=True):
    """For each S_n orbit (= weight class), check if any representative has 
    an isolating matrix. Returns dict weight -> bool.
    
    SingY = union of weight classes for which this returns True.
    This gives a priori lower bound on min SMUS = sum of |W_w| for True orbits.
    """
    Y = enumerate_canonical_balanced(n)
    classes = weight_class_indices(Y)
    weights = sorted(classes.keys())
    
    isolating = {}
    for w in weights:
        y_rep = Y[classes[w][0]]  # first representative of weight class
        can_isolate = has_isolating_matrix(y_rep, Y, m, n)
        isolating[w] = can_isolate
        if verbose:
            marker = "✓" if can_isolate else "✗"
            print(f"    weight {w} (|W_{w}|={len(classes[w])}): isolable = {marker}")
    
    lb = sum(len(classes[w]) for w in weights if isolating[w])
    if verbose:
        print(f"    a priori lower bound |SingY| = {lb}")
    return isolating, lb


# =========================================================================
# Matrix-orbit deduplication of constraints (partial, via profile)
# =========================================================================

def orbit_dedup_constraints(constraints_set, Y, verbose=True):
    """Group constraints by S_n × row/column complement orbit.
    
    Simpler proxy: canonical bitmask under S_n coordinate permutation.
    Two matrices give equivalent constraints iff their kernel bitmasks
    are S_n-related. To canonicalize a bitmask: compute the multiset of 
    (weight, positions_pattern) for each y in the kernel, take lex-min 
    over σ ∈ S_n.
    
    For n ≤ 8 (n! ≤ 40320), direct enumeration is feasible.
    Returns list of canonical constraints (one per orbit).
    """
    n = len(Y[0])
    Y_tuples = [tuple(y) for y in Y]
    y_index = {y: i for i, y in enumerate(Y_tuples)}
    
    def apply_perm_to_mask(mask, perm):
        """Apply σ ∈ S_n to a constraint bitmask via induced action on B(n)/±."""
        new_mask = 0
        for j in range(len(Y)):
            if mask & (1 << j):
                y = Y_tuples[j]
                y_perm = tuple(y[perm[k]] for k in range(n))
                # canonicalize under ±
                canonical = min(y_perm, tuple(-x for x in y_perm))
                new_mask |= (1 << y_index[canonical])
        return new_mask
    
    # For each constraint, find canonical form (min over S_n)
    perms = list(itertools.permutations(range(n)))
    canonical_to_orig = {}
    for mask in constraints_set:
        canonical = min(apply_perm_to_mask(mask, p) for p in perms)
        canonical_to_orig.setdefault(canonical, mask)
    
    if verbose:
        print(f"  Orbit dedup: {len(constraints_set)} constraints → {len(canonical_to_orig)} orbit reps")
    return list(canonical_to_orig.keys())


def full_diagnostic_row(n, compute_smus=True, verbose=False, use_sym_breaking=False, use_dominance=False):
    """Compute the full diagnostic row for a given n at UNSAT boundary.
    
    Returns dict with: n, m, |B/±|, SingLB, isolable_per_weight, LP_LB, 
                       invSMUS, invSMUS_weights, SMUS, witness_composition.
    Set compute_smus=False to skip the expensive direct SMUS computation.
    Set use_sym_breaking=True to use the symmetry-broken ILP (needed for n≥8).
    """
    m = unsat_boundary(n)
    Y = enumerate_canonical_balanced(n)
    classes = weight_class_indices(Y)
    weights = sorted(classes.keys())
    
    # Singleton-witness (fast: ⌊n/2⌋ ILP feasibility problems)
    iso, sing_lb = singleton_witness_analysis(m, n, verbose=verbose)
    
    # LP relaxation (fast: only ⌊n/2⌋ variables, uses profiles)
    lp_lb = lp_lower_bound(m, n, verbose=False)
    
    # Invariant SMUS
    inv_smus, inv_weights, _ = invariant_smus(m, n, verbose=verbose)
    
    result = {
        'n': n, 'm': m, 'total_ys': len(Y),
        'sing_lb': sing_lb, 'isolable': dict(iso),
        'lp_lb': lp_lb,
        'inv_smus': inv_smus, 'inv_weights': inv_weights,
        'smus': None, 'witness': None
    }
    
    if compute_smus:
        if use_sym_breaking:
            smus, S = solve_smus_with_symmetry(m, n, verbose=verbose, use_dominance=use_dominance)
        else:
            smus, S = solve_smus(m, n, verbose=verbose)
        result['smus'] = smus
        if S is not None:
            result['witness'] = summarize_by_weight(S)
    
    return result


def print_summary_table():
    """Print the summary table for small n."""
    print(f"{'n':>3} {'m':>3} {'|B/±|':>7} {'SingLB':>7} {'LP LB':>9} "
          f"{'invSMUS':>8} {'SMUS':>6}  {'isolable':<20}  {'witness'}")
    print("-" * 110)
    for n in [3, 4, 5, 6, 7, 8]:
        try:
            r = full_diagnostic_row(n, compute_smus=(n <= 7))
        except Exception as e:
            print(f"n={n}: ERROR {e}")
            continue
        iso_str = ",".join(f"W{w}={'Y' if v else 'N'}" for w, v in r['isolable'].items())
        if r['witness']:
            wit_str = " + ".join(f"{len(r['witness'][w])}·W_{w}" for w in sorted(r['witness']))
        else:
            wit_str = "(not computed)"
        smus_str = str(r['smus']) if r['smus'] is not None else "?"
        print(f"{r['n']:>3} {r['m']:>3} {r['total_ys']:>7} {r['sing_lb']:>7} "
              f"{r['lp_lb']:>9.2f} {r['inv_smus']:>8} {smus_str:>6}  "
              f"{iso_str:<20}  {wit_str}")


# =========================================================================
# CLI: full analysis
# =========================================================================

if __name__ == "__main__":
    print("Minimum SMUS for balanced-detecting UNSAT — full quotient analysis\n")
    print_summary_table()


def find_isolating_matrix(y_target, Y, m, n):
    """Return the rows of an isolating m×n matrix for y_target if one exists,
    else None. Padded with all-zero rows to reach exactly m rows.
    
    Useful for independent brute-force verification of has_isolating_matrix results.
    """
    n_ys = len(Y)
    y_t = np.array(y_target, dtype=np.int8)
    Y_arr = np.array(Y, dtype=np.int8)
    y_target_idx = Y.index(tuple(y_target)) if not isinstance(y_target, tuple) else Y.index(y_target)
    
    n_canon = 2 ** (n - 1)
    admissible = []
    dot_matrix = []
    for r_int in range(n_canon):
        r_bits = [(r_int >> j) & 1 for j in range(n - 1)]
        r = np.array([0] + r_bits, dtype=np.int8)
        if int(r @ y_t) != 0:
            continue
        admissible.append(tuple(int(x) for x in r))
        dot_matrix.append(Y_arr @ r)
    if len(admissible) < 1:
        return None
    
    dot_matrix = np.array(dot_matrix)
    n_adm = len(admissible)
    
    rows_c = []
    for y_idx in range(n_ys):
        if y_idx == y_target_idx:
            continue
        row = (dot_matrix[:, y_idx] != 0).astype(float)
        if row.sum() == 0:
            return None
        rows_c.append(row)
    A_ineq = np.array(rows_c)
    card_row = np.ones(n_adm)
    A_stack = np.vstack([A_ineq, card_row.reshape(1, -1)])
    lb = np.concatenate([np.ones(len(rows_c)), [-np.inf]])
    ub = np.concatenate([np.full(len(rows_c), np.inf), [float(m)]])
    
    result = milp(np.zeros(n_adm), constraints=LinearConstraint(A_stack, lb=lb, ub=ub),
                  integrality=np.ones(n_adm), bounds=Bounds(0, 1))
    if not result.success:
        return None
    chosen = [admissible[i] for i in range(n_adm) if result.x[i] > 0.5]
    # Pad with all-zero rows
    while len(chosen) < m:
        chosen.append(tuple(0 for _ in range(n)))
    return chosen


def brute_force_kernel(A_rows, Y):
    """Return the list of y ∈ Y with A y = 0."""
    A = np.array(A_rows, dtype=np.int8)
    return [y for y in Y if all(int(np.array(r) @ np.array(y)) == 0 for r in A)]


def verify_isolating(y_target, Y, m, n, verbose=True):
    """Round-trip check: find isolating matrix and brute-force verify kernel = {y_target}."""
    rows = find_isolating_matrix(y_target, Y, m, n)
    if rows is None:
        if verbose:
            print(f"  {format_y(y_target)}: NOT isolable (has_isolating_matrix says infeasible)")
        return False
    ker = brute_force_kernel(rows, Y)
    ker_set = set(tuple(y) for y in ker)
    y_target_t = tuple(y_target)
    ok = (ker_set == {y_target_t})
    if verbose:
        marker = "✓" if ok else "✗ FALSE POSITIVE"
        print(f"  {format_y(y_target)}: kernel = {[format_y(y) for y in ker]}  {marker}")
    return ok
