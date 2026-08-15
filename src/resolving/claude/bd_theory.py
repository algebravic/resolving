#!/usr/bin/env python3
"""
bd_theory.py -- Balanced-detecting matrix theory backend.

Provides the "theory" side of the SDP-SAT hybrid: fast matrix filter,
eigenvector cut extraction, integer rounding of eigenvectors, and PB->CNF
lemma compilation. Independent of the SAT solver; imported by the propagator.

The matrix condition:
  Phi(A) := sum_{x in B(n)} ||A x||^2 x x^T  >= X X^T   (in PSD order)
is necessary for A to be balanced-detecting.

Phi(A) has a clean structural form: Phi(A) = <T4, A^T A> where T4 is the
fourth-moment tensor of B(n), which is S_n-invariant with 5 distinct values.

Key primitives exposed:
  BDTheory(m, n):                       precomputes T4 and XXT.
    .check(A) -> (bool, min_eig, eigvec) : matrix filter with diagnostics
    .aggregate_check(A) -> bool           : cheap scalar aggregate check
    .round_eigvec(v, max_denom) -> vint   : rational rounding of eigvec to
                                            small integer vector suitable
                                            for a PB cut.
    .compile_pb_cut(v_int, aux_var_map)  : generates the PB inequality
                                            coefficients c[i,p,p], c[i,p,q]
                                            and RHS.

  compile_pb_to_cnf(coeffs, rhs, ...)    : sequential-counter encoding of a
                                            PB inequality to CNF clauses,
                                            suitable for lemma addition.
"""

import numpy as np
from itertools import combinations
from collections import Counter
from fractions import Fraction


# ---------------------------------------------------------------------------
# B(n) enumeration and T4 pattern values
# ---------------------------------------------------------------------------

def enum_balanced(n):
    """Enumerate one representative per +/- pair of nonzero balanced x in {-1,0,1}^n.

    Returns list of int8 numpy arrays of shape (n,).
    """
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
    """Return dict from multiplicity pattern -> T4 value."""
    B = enum_balanced(n)
    Vs = {
        (4,):        sum(int(x[0]) ** 2 for x in B),
        (3, 1):      sum(int(x[0]) ** 3 * int(x[1]) for x in B),
        (2, 2):      sum(int(x[0]) ** 2 * int(x[1]) ** 2 for x in B),
        (2, 1, 1):   sum(int(x[0]) ** 2 * int(x[1]) * int(x[2]) for x in B),
        (1, 1, 1, 1): sum(int(x[0]) * int(x[1]) * int(x[2]) * int(x[3]) for x in B)
                      if n >= 4 else 0,
    }
    return Vs, len(B)


def build_T4(n, Vs):
    """Build T4 as full n^4 int64 tensor. O(n^4) memory, fine for n <= 20."""
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
# BDTheory: main theory class
# ---------------------------------------------------------------------------

class BDTheory:
    """Balanced-detecting theory for m x n candidate matrices.

    Encapsulates T4 tensor, XXT matrix, and provides check/rounding/compilation.
    """

    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.Vs, self.N_B = T4_pattern_values(n)
        self.T4 = build_T4(n, self.Vs)
        self.XXT = build_XXT(n, self.Vs)
        # Precompute pair index for (p,q) with p<q
        self.pair_idx = {}
        k = 0
        for p in range(n):
            for q in range(p + 1, n):
                self.pair_idx[(p, q)] = k
                k += 1
        self.num_pairs = k

    # --------- Filters ------------------------------------------------------

    def compute_Phi(self, A):
        """Phi(A) = <T4, A^T A>, an n x n symmetric int matrix."""
        M = A.T @ A
        return np.einsum('jkpq,pq->jk', self.T4, M)

    def check(self, A, tol=1e-9):
        """Matrix condition. Returns (ok, min_eig, min_eigvec).
        min_eigvec is only meaningful when not ok (or nearly so)."""
        Phi = self.compute_Phi(A)
        diff = Phi - self.XXT
        diff = 0.5 * (diff + diff.T)
        eigs, eigvecs = np.linalg.eigh(diff)
        return eigs[0] >= -tol, float(eigs[0]), eigvecs[:, 0].copy()

    def aggregate_check(self, A):
        """Scalar necessary condition: sum_x ||A x||^2 >= |B|.
        Uses sum_x ||Ax||^2 = tr((A^T A) (X X^T))."""
        return float(np.sum((A.T @ A) * self.XXT)) >= self.N_B

    # --------- Eigenvector rounding to integer test vector ------------------

    def round_eigvec(self, v, max_denom=6):
        """Round a real unit eigenvector v to a small integer vector v_int
        such that v_int^T (Phi - XXT) v_int is still substantially negative
        (if it was negative for v).

        Strategy: try {-1, 0, +1}^n rounding first, then {-2, -1, 0, 1, 2}^n,
        etc., picking the vector that minimizes v_int^T Phi_diff v_int / ||v_int||^2.

        We normalize v to place its largest-magnitude component at 1, then round
        each component to the nearest fraction with denominator <= max_denom.

        Returns best integer vector v_int with ||v_int||_inf <= max_denom.
        """
        n = len(v)
        v = v / np.max(np.abs(v))                      # scale so max entry is +/-1
        # Fixed candidate: rescale by max_denom and round to nearest integer
        candidates = []
        for scale in range(1, max_denom + 1):
            candidate = np.round(v * scale).astype(np.int64)
            if np.all(candidate == 0):
                continue
            # dedupe
            candidates.append(candidate)
        # Also try sign-only rounding (all-or-nothing)
        signs = np.sign(v).astype(np.int64)
        if not np.all(signs == 0):
            candidates.append(signs)
        # Try including thresholded versions: components with |v_i| > threshold get
        # rounded to +/- 1, others to 0.
        for thresh in [0.1, 0.2, 0.3, 0.5]:
            candidate = np.where(np.abs(v) > thresh, np.sign(v), 0).astype(np.int64)
            if not np.all(candidate == 0):
                candidates.append(candidate)
        return candidates

    def best_int_cut(self, A, tol=1e-9, max_denom=4):
        """Given non-BD candidate A, find an integer vector v_int such that
        v_int^T (Phi(A) - XXT) v_int < 0. Returns v_int or None.

        Iterates over rounded eigenvectors of the smallest eigenvalues of Phi - XXT,
        picking the first one that gives a strict violation.
        """
        Phi = self.compute_Phi(A)
        diff = Phi - self.XXT
        diff = 0.5 * (diff + diff.T)
        eigs, eigvecs = np.linalg.eigh(diff)
        # Iterate from smallest eigenvalue up; if that's >= -tol, no cut.
        best_v = None
        best_val = 0
        for i in range(len(eigs)):
            if eigs[i] >= -tol:
                break
            v = eigvecs[:, i]
            candidates = self.round_eigvec(v, max_denom=max_denom)
            for c in candidates:
                # Evaluate c^T (Phi - XXT) c
                val = int(c @ diff @ c)                # exact since c is int, diff is float
                # rounding may make val slightly off; use int matmul on diff-as-Fraction? Cheap way:
                # since diff = Phi - XXT and both are int64 originally, we should use ints
                # Phi is int, XXT is int. Recompute with int arithmetic:
                pass
            # Cleaner: recompute using int arithmetic
            diff_int = self.compute_Phi(A) - self.XXT
            for c in candidates:
                c = c.astype(np.int64)
                val = int(c @ diff_int @ c)
                if val < 0 and val < best_val:
                    best_val = val
                    best_v = c
        return best_v, best_val

    # --------- PB cut compilation from an integer test vector ---------------

    def pb_cut_coefficients(self, v_int):
        """Given integer test vector v_int in Z^n, return the coefficients
        of the PB constraint

           sum_i sum_p c_pp A_{i,p} + sum_i sum_{p<q} c_pq_pair z_{i,p,q}   >=  K

        where c_pp[p] = einsum('j,k,jkpp->', v, v, T4[j,k,p,p])
              c_pq_pair[(p,q)] = 2 * einsum('j,k,jkpq->', v, v, T4[j,k,p,q])  for p<q
              K = v^T XXT v

        Returns (c_diag, c_pair, K) where
          c_diag: dict from p -> int coefficient
          c_pair: dict from (p,q) -> int coefficient (with p<q)
          K:      int, the RHS
        """
        v = np.asarray(v_int, dtype=np.int64)
        # c[p,q] = sum_{j,k} v_j v_k T4[j,k,p,q]
        C = np.einsum('j,k,jkpq->pq', v, v, self.T4)
        c_diag = {}
        c_pair = {}
        for p in range(self.n):
            c_diag[p] = int(C[p, p])
            for q in range(p + 1, self.n):
                # off-diagonal in Phi contributes 2 * C[p,q] because A_ip A_iq
                # is counted with both orderings in the double sum sum_{p,q}
                c_pair[(p, q)] = int(C[p, q] + C[q, p])
        K = int(v @ self.XXT @ v)
        return c_diag, c_pair, K


# ---------------------------------------------------------------------------
# PB -> CNF compilation
# ---------------------------------------------------------------------------

def compile_pb_to_cnf(coeffs, rhs, next_var_id):
    """Encode a pseudo-Boolean constraint  sum_i c_i x_i >= K  as CNF clauses
    using the sequential counter encoding for weighted-sum-lower-bound.

    Args:
      coeffs: list of (coef, var_id) with coef possibly negative.
      rhs:    integer K.
      next_var_id: integer, next available variable id for auxiliaries.

    Returns:
      (clauses, new_next_var_id): list of clauses (each list of ints), new var id.

    Method: rewrite as sum_i pos_c_i x_i + sum_j |neg_c_j| ~x_j >= K + sum |neg|
    by substituting x for negative-coefficient variables with 1 - x. Then use
    a sequential-adder / sorting-network encoding of sum >= new_rhs.

    For simplicity and given that CaDiCaL handles many clauses well, we use
    the "generalized totalizer encoding":
      - Sort variables by coefficient magnitude
      - Build a binary tree of totalizers (unary counters)
      - Assert output threshold >= K.

    For very large coefficients this may be inefficient; a specialized PB solver
    would be better. Since our matrix cuts have coefficients ~ 10^5, the
    encoding size at n=14 is manageable for a few cuts but grows fast.

    OPTIMIZATION: this initial version returns a placeholder "weakened" clause
    representing "not the current assignment plus enough flips" which is CORRECT
    (blocks the specific bad model) but WEAK (doesn't share learning across
    similar models). See paper for how a strong PB->CNF encoding would look.
    """
    # For the initial delivery, we use a simple negation-of-model lemma: takes
    # the current partial assignment restricted to the variables in the PB
    # constraint and adds "not all of these simultaneously". This is weak but
    # correct.
    #
    # Strong PB encoding is future work; see e.g. Bailleux-Boufkhad totalizer.
    raise NotImplementedError(
        "Strong PB->CNF not implemented in v1. Use compile_pb_to_neg_model_lemma "
        "on the model that violates the constraint."
    )


def compile_pb_to_neg_model_lemma(coeffs, rhs, model_values):
    """Weak lemma: given a model that violates sum_i c_i x_i >= K, return a
    single CNF clause that blocks this model.

    Args:
      coeffs:       list of (coef, var_id) tuples (var_id positive)
      rhs:          K
      model_values: dict from var_id -> {True, False}, the current assignment

    Returns:
      list containing a single clause (list of ints): the clause blocking this
      model. That is, the disjunction of the negations of the currently-true
      literals AMONG THE VARIABLES INVOLVED in this PB constraint.

    Note this is a WEAK lemma: it only blocks this exact model. To get stronger
    propagation, use PB solver or compile PB->CNF strong encoding.
    """
    clause = []
    involved_vars = set(v for _, v in coeffs)
    for v in involved_vars:
        val = model_values.get(v)
        if val is None:
            continue                                    # unassigned; skip
        # If v is True, we want to allow it to be false: add ~v (= -v)
        # If v is False, we want to allow it to be true: add +v
        clause.append(-v if val else v)
    return [clause]
