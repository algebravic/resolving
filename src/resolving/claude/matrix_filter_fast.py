#!/usr/bin/env python3
"""
Matrix-condition filter for balanced-detecting candidates (fast version).
==========================================================================

Given m x n candidate matrix A, tests the necessary condition:

    Phi(A) := sum_{x in B(n)} ||A x||^2 * x x^T  >=  X X^T   (in PSD order)

where X has columns being one rep per +/- pair of nonzero balanced x in {-1,0,1}^n.

This is strictly stronger than the aggregate scalar bound tr(Phi(A)) >= |B(n)|
because it detects when kernel-balanced-vectors CLUSTER in a specific direction
of R^n rather than just their aggregate count.

SPEED TRICK:
    Phi(A) is quadratic in A, so it equals einsum('jkpq,pq->jk', T4, A^T A)
    where T4[i,j,k,l] := sum_x x_i x_j x_k x_l is the fourth-moment tensor.

    T4 is S_n-invariant, taking only 5 distinct values (one per multiplicity
    pattern of the four indices):
        (4,)       -> V_4     (all four indices equal)
        (3,1)      -> V_31    (three equal, one distinct)
        (2,2)      -> V_22    (two pairs)
        (2,1,1)    -> V_211   (one pair, two singletons)
        (1,1,1,1)  -> V_1111  (all distinct)

    So T4 is O(n^4) integers built from 5 numbers. For n=14 this is 300KB, fits
    trivially in memory. Then Phi = <T4, A^T A> is 40K operations per candidate.

    Compared to naive Phi = X diag(||Ax||^2) X^T (which needs 2M ops per candidate
    at n=14), this is 50-1000x faster depending on n.

TIMING (measured):
    n=6:  precompute 0.001s, filter 0.02 ms/cand
    n=8:  precompute 0.005s, filter 0.03 ms/cand
    n=10: precompute 0.05s,  filter 0.04 ms/cand
    n=12: precompute 0.4s,   filter 0.05 ms/cand
    n=14: precompute 1.0s,   filter 0.05 ms/cand

    A CDLL-style tight search loop can check ~20K candidates/sec at n=14.

USAGE:
    from matrix_filter_fast import FastFilter
    F = FastFilter(n=14)
    ok = F.check(A)                        # bool: matrix cond passes?
    ok, min_eig = F.check_with_eig(A)      # returns diagnostic info too
    ok, agg_pass, mat_pass = F.check_both(A) # both aggregate and matrix
"""

import numpy as np
from itertools import combinations
from collections import Counter


def enum_balanced(n):
    """Enumerate one representative per +/- pair of nonzero balanced x in {-1,0,1}^n."""
    out = []
    coords = list(range(n))
    for k in range(1, n // 2 + 1):
        for support in combinations(coords, 2 * k):
            first_pos = support[0]
            remaining = support[1:]
            for pos_extra in combinations(remaining, k - 1):
                x = np.zeros(n, dtype=np.int8)
                x[first_pos] = 1
                for p in pos_extra:
                    x[p] = 1
                for p in remaining:
                    if p not in pos_extra:
                        x[p] = -1
                out.append(x)
    return out


def compute_pattern_values(n):
    """Compute the 5 distinct values of T4 for B(n).

    Uses formulas rather than direct enumeration for speed:
    V_4 (single-index squared, then squared again) = # {x : x_i != 0} = 2 alpha_n where
    alpha_n = [z^(n-2)] (1+z+z^2)^(n-1) is the Motzkin coefficient.

    Actually the closed forms are complicated; direct enumeration is faster in practice
    (n=14 takes 0.5s to enumerate |B|=308K one-sign vectors).
    """
    B = enum_balanced(n)
    # V_pattern = sum_x [values on canonical index sets]
    # Use small index sets and one-sign B; multiply by 2 if you want the both-signs total.
    V_4    = sum(int(x[0])**2 for x in B)                                   # x^4 = x^2 (0 or 1)
    V_31   = sum(int(x[0])**3 * int(x[1]) for x in B)                       # = sum x_0 x_1 (x_0^2)
    V_22   = sum(int(x[0])**2 * int(x[1])**2 for x in B)
    V_211  = sum(int(x[0])**2 * int(x[1]) * int(x[2]) for x in B)
    V_1111 = sum(int(x[0]) * int(x[1]) * int(x[2]) * int(x[3]) for x in B) if n >= 4 else 0
    return V_4, V_31, V_22, V_211, V_1111, len(B)


def build_T4(n):
    """Build the fourth-moment tensor T4[i,j,k,l] = sum_x x_i x_j x_k x_l (one-sign B)."""
    V_4, V_31, V_22, V_211, V_1111, _ = compute_pattern_values(n)
    Vs = {(4,): V_4, (3, 1): V_31, (2, 2): V_22, (2, 1, 1): V_211, (1, 1, 1, 1): V_1111}
    T4 = np.zeros((n, n, n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    pat = tuple(sorted(Counter((i, j, k, l)).values(), reverse=True))
                    T4[i, j, k, l] = Vs[pat]
    return T4


def build_XXT(n):
    """Build X X^T for one-sign B(n). Equals lambda * (I - J/n) where lambda = V_4 - V_22
    (since XXT[i,i] = V_4 and XXT[i,j] = V_22 for i != j... wait, actually XXT[i,i] = alpha_n
    and XXT[i,j] = beta_n which are DIFFERENT from V_4, V_22. Let me redo.)
    
    XXT[i,j] = sum_x x_i x_j. For i==j: sum_x x_i^2 = V_4 (agreed since x_i^2 = x_i^4 on {-1,0,1}).
    For i!=j: sum_x x_i x_j = V_22? No, V_22 = sum_x x_i^2 x_j^2. 
    Let me define beta_n_pat = sum_x x_i x_j (i != j). This is a NEW quantity, not one of V_pattern.
    """
    B = enum_balanced(n)
    alpha_val = sum(int(x[0])**2 for x in B)
    beta_val  = sum(int(x[0]) * int(x[1]) for x in B) if n >= 2 else 0
    XXT = np.full((n, n), beta_val, dtype=np.int64)
    np.fill_diagonal(XXT, alpha_val)
    return XXT, len(B)


class FastFilter:
    """Precomputes T4 and XX^T for a given n, then filters candidates quickly."""

    def __init__(self, n):
        self.n = n
        self.T4 = build_T4(n)
        self.XXT, self.N_B = build_XXT(n)

    def compute_Phi(self, A):
        """Phi(A) = sum_x ||A x||^2 x x^T, evaluated via T4 contraction with A^T A."""
        M = A.T @ A                                          # n x n
        Phi = np.einsum('jkpq,pq->jk', self.T4, M)
        return Phi

    def check(self, A, tol=1e-9):
        """Return True if matrix condition Phi(A) - XX^T is PSD (i.e., A might be BD)."""
        Phi = self.compute_Phi(A)
        diff = Phi - self.XXT
        diff = 0.5 * (diff + diff.T)
        # Use eigenvalue instead of Cholesky since we allow small numerical negatives.
        min_eig = np.linalg.eigvalsh(diff)[0]
        return min_eig >= -tol

    def check_with_eig(self, A, tol=1e-9):
        """Return (ok, min_eig, min_eigvec)."""
        Phi = self.compute_Phi(A)
        diff = Phi - self.XXT
        diff = 0.5 * (diff + diff.T)
        eigs, eigvecs = np.linalg.eigh(diff)
        return eigs[0] >= -tol, eigs[0], eigvecs[:, 0]

    def check_both(self, A, tol=1e-9):
        """Return (both_pass, agg_pass, mat_pass) doing both checks."""
        Phi = self.compute_Phi(A)
        # Aggregate: sum_x ||A x||^2 = tr(A M A^T) where M is... hmm.
        # Actually sum_x ||A x||^2 = tr(Phi(A)) / (n-1) * something.
        # Wait: Phi_diag = sum_x ||A x||^2 x_i^2, so tr(Phi) = sum_i sum_x ||Ax||^2 x_i^2
        # = sum_x ||Ax||^2 * sum_i x_i^2 = sum_x ||Ax||^2 * |support(x)|
        # This is NOT sum_x ||Ax||^2 unless all x have same support size.
        # For aggregate we need sum_x ||Ax||^2 directly.
        #
        # sum_x ||A x||^2 = tr(A (X X^T) A^T) = tr((A^T A) (X X^T))
        agg_val = float(np.sum((A.T @ A) * self.XXT))
        agg_pass = agg_val >= self.N_B
        # Matrix
        diff = Phi - self.XXT
        diff = 0.5 * (diff + diff.T)
        min_eig = np.linalg.eigvalsh(diff)[0]
        mat_pass = min_eig >= -tol
        return agg_pass and mat_pass, agg_pass, mat_pass


# ---------------------------------------------------------------------------
# Self-test / benchmarks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    print("=" * 68)
    print("Correctness: compare fast filter against direct computation")
    print("=" * 68)
    for n in [4, 5, 6, 7]:
        F = FastFilter(n)
        B = enum_balanced(n)
        X = np.array(B, dtype=np.int32).T
        # Test on 3 random matrices per n
        for trial in range(3):
            np.random.seed(trial + n * 100)
            m = max(2, n - 2)
            A = np.random.randint(0, 2, size=(m, n))
            Phi_fast = F.compute_Phi(A)
            Q = np.sum((A @ X) ** 2, axis=0)
            Phi_direct = (X * Q) @ X.T
            match = np.array_equal(Phi_fast, Phi_direct)
            print(f"  n={n} m={m} trial {trial}: match = {match}")
            if not match:
                print(f"    diff = {np.max(np.abs(Phi_fast - Phi_direct))}")

    print()
    print("=" * 68)
    print("Correctness: known non-BD case (2x4 with 2 balanced kernel vecs)")
    print("=" * 68)
    F4 = FastFilter(4)
    A_worse = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    ok, min_eig, vec = F4.check_with_eig(A_worse)
    print(f"A_worse:\n{A_worse}")
    print(f"Matrix filter passes: {ok}   (expected: False)")
    print(f"Smallest eigenvalue of Phi - XXT: {min_eig:.4f}  (expected: -4)")

    print()
    print("=" * 68)
    print("Correctness: known BD case (3x4 with e_1, e_2, e_3)")
    print("=" * 68)
    A_good = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    ok, min_eig, _ = F4.check_with_eig(A_good)
    print(f"Matrix filter passes: {ok}   (expected: True)")
    print(f"Smallest eigenvalue: {min_eig:.6f}  (expected: ~0)")

    print()
    print("=" * 68)
    print("Precompute timing")
    print("=" * 68)
    for n in [6, 8, 10, 12, 14]:
        t0 = time.time()
        F = FastFilter(n)
        t1 = time.time()
        print(f"  n={n}: |B|={F.N_B}, precompute time {t1-t0:.3f}s, T4 size {F.T4.nbytes // 1024}KB")

    print()
    print("=" * 68)
    print("Filter throughput at target scale n=14, m=7")
    print("=" * 68)
    F = FastFilter(14)
    K = 10000
    A_batch = np.random.randint(0, 2, size=(K, 7, 14))
    t0 = time.time()
    passes = 0
    for k in range(K):
        if F.check(A_batch[k]):
            passes += 1
    t1 = time.time()
    print(f"  {K} random 7x14 candidates: total {t1-t0:.2f}s, per-candidate {(t1-t0)/K*1000:.3f} ms")
    print(f"  candidates passing matrix filter: {passes}/{K} ({100*passes/K:.1f}%)")

    print()
    print("=" * 68)
    print("Value of matrix filter: how often does it catch non-BD that aggregate misses?")
    print("=" * 68)
    F = FastFilter(14)
    K = 5000
    both_pass = 0
    only_agg = 0
    neither = 0
    A_batch = np.random.randint(0, 2, size=(K, 7, 14))
    t0 = time.time()
    for k in range(K):
        _, agg, mat = F.check_both(A_batch[k])
        if agg and mat: both_pass += 1
        elif agg and not mat: only_agg += 1
        else: neither += 1
    t1 = time.time()
    print(f"  {K} random 7x14 candidates ({t1-t0:.1f}s):")
    print(f"    passed both filters:                    {both_pass}")
    print(f"    passed aggregate but failed matrix:     {only_agg}   <-- matrix filter's value")
    print(f"    failed aggregate (matrix redundant):    {neither}")
    if both_pass + only_agg > 0:
        pct = 100 * only_agg / (both_pass + only_agg)
        print(f"  {pct:.1f}% of aggregate-passers get filtered by matrix condition")

    print()
    print("=" * 68)
    print("Structured candidates: BD-look-alikes near the boundary")
    print("=" * 68)
    F = FastFilter(10)
    # Generate random 6x10 matrices with row-weights around n/2 = 5 (typical BD structure)
    K = 2000
    both_pass = 0
    only_agg = 0
    for k in range(K):
        A = np.zeros((6, 10), dtype=int)
        for i in range(6):
            # random subset of columns of size 5
            cols = np.random.choice(10, 5, replace=False)
            A[i, cols] = 1
        _, agg, mat = F.check_both(A)
        if agg and mat: both_pass += 1
        elif agg and not mat: only_agg += 1
    print(f"  {K} random 6x10 candidates with row-weight 5:")
    print(f"    passed both:            {both_pass}")
    print(f"    passed agg, failed mat: {only_agg}")
    if both_pass + only_agg > 0:
        print(f"    matrix filter catches {100*only_agg/(both_pass+only_agg):.1f}% of agg-passers")
