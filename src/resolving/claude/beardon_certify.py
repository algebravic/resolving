#!/usr/bin/env python3
"""
Beardon-style resolving-set certifier.
=======================================

Given a candidate set V = {v_1, ..., v_m} of vertices of {0,1}^n (equivalently a
+/-1 matrix Vtilde of shape m x n), decide whether V resolves V^n and produce
one of two exact certificates:

  * RESOLVES: a rational proof.  The certificate is the statement
      "the null space W_perp of Vtilde over Q intersected with {-1,0,1}^n is {0}",
    together with a witnessing chain of implications that reduces this to
    trivial linear-algebraic and integrality facts.  This mirrors what Beardon
    does by hand (e.g. "adding equations gives 2z_2 + z_3 = 0, and z_3 in
    {-1,0,1} forces z_3 = 0, hence z_2 = 0, ...").

  * DOES NOT RESOLVE: an explicit nonzero z in {-1,0,1}^n with Vtilde @ z = 0.
    Equivalently: two distinct vertices x, y with x - y = z that give the same
    distance-tuple.

The decision is exact (no floats).  The workhorse is:
  1. Compute a rational RREF basis of W_perp = null(Vtilde), where each basis
     vector has a 1 at one "free coordinate" and 0s at other free coordinates.
     In this form, if z = sum_i t_i * b_i then z[free_col_i] = t_i.
  2. Since z in {-1,0,1}^n forces t_i in {-1,0,1} for every i, enumerate
     3^s candidate parameter tuples (s = dim null space) and check each.

CORRECTNESS NOTE: an earlier version used an integer-scaled basis and
enumerated t in {-1,0,1}^s over that basis.  That is *unsound*: after scaling
the basis to clear denominators, the true parameters may need to be
half-integers or worse, so nonzero ternary vectors get missed.  The RREF
rational basis version above is correct because the free-coordinate entries
of the basis ARE the parameters.

For infeasibility of ALL m-subsets (i.e. beta_n > m proofs), see
'certify_all_subsets' at the bottom.
"""

from fractions import Fraction
from itertools import product, combinations
import sys


# ============================================================================
# Rational linear algebra: null space of an integer matrix
# ============================================================================

def rational_rref(M):
    """Reduce a matrix of Fractions to reduced row echelon form.
    Returns (rref, pivot_cols)."""
    M = [row[:] for row in M]
    r = len(M); c = len(M[0]) if M else 0
    pivots = []
    row = 0
    for col in range(c):
        if row >= r: break
        # find pivot
        piv = None
        for k in range(row, r):
            if M[k][col] != 0:
                piv = k; break
        if piv is None: continue
        M[row], M[piv] = M[piv], M[row]
        # normalize
        lead = M[row][col]
        M[row] = [x / lead for x in M[row]]
        # eliminate
        for k in range(r):
            if k != row and M[k][col] != 0:
                factor = M[k][col]
                M[k] = [M[k][j] - factor * M[row][j] for j in range(c)]
        pivots.append(col); row += 1
    return M, pivots


def rational_null_basis(V):
    """Given integer matrix V (m x n) as list of lists, return list of basis
    vectors (each length n, Fraction entries) for the null space using RREF
    form: each basis vector has a 1 in one free-coordinate position and 0s in
    the other free-coordinate positions, and rational entries elsewhere.

    In this form, if z = sum_i t_i * b_i, then z[free_col_i] = t_i.
    So z in {-1,0,1}^n forces t_i in {-1,0,1} directly.
    """
    m = len(V); n = len(V[0]) if m else 0
    Q = [[Fraction(x) for x in row] for row in V]
    rref, pivots = rational_rref(Q)
    free_cols = [j for j in range(n) if j not in pivots]
    basis = []
    for fc in free_cols:
        v = [Fraction(0)] * n
        v[fc] = Fraction(1)
        for pi, pc in enumerate(pivots):
            v[pc] = -rref[pi][fc]
        basis.append(v)
    return basis, free_cols


def integer_null_basis(V):
    """Wrapper kept for backward compatibility.  Returns the RREF basis scaled
    to integer entries -- USE WITH CARE: the natural parameter of each basis
    vector is the scaling factor, not 1, which breaks naive ternary search.
    Prefer rational_null_basis for correctness."""
    rat, free_cols = rational_null_basis(V)
    out = []
    for v in rat:
        denom_lcm = 1
        for x in v:
            d = x.denominator
            denom_lcm = denom_lcm * d // gcd(denom_lcm, d)
        v_int = [int(x * denom_lcm) for x in v]
        for x in v_int:
            if x != 0:
                if x < 0: v_int = [-y for y in v_int]
                break
        out.append(v_int)
    return out


def gcd(a, b):
    while b: a, b = b, a % b
    return abs(a)


# ============================================================================
# Ternary search in the null space
# ============================================================================

def ternary_in_span(V, verbose=False):
    """Given the ORIGINAL integer matrix V (m x n) (NOT the null basis!), find
    a nonzero z in {-1,0,1}^n with V @ z = 0, or prove none exists.

    Uses the rational RREF null basis where the free coordinates ARE the
    parameters.  So z in {-1,0,1}^n forces each parameter t_i (=z[free_col_i])
    into {-1,0,1}, and we enumerate 3^(dim null) candidates.

    Returns (found_z_or_None, num_lattice_points_checked)."""
    basis, free_cols = rational_null_basis(V)
    s = len(basis)
    if s == 0:
        return None, 0
    n = len(basis[0])

    checked = 0
    for t in product((-1, 0, 1), repeat=s):
        checked += 1
        if all(x == 0 for x in t): continue
        z = [sum(t[i] * basis[i][j] for i in range(s)) for j in range(n)]
        if all(x in (Fraction(-1), Fraction(0), Fraction(1)) for x in z):
            return [int(x) for x in z], checked
    return None, checked


# ============================================================================
# The certifier
# ============================================================================

def vtilde_from_v(v, n):
    """Given v in {0,1}^n, return vtilde = 2v - 1 in {-1,+1}^n."""
    return [2 * x - 1 for x in v]


def certify(V, verbose=False):
    """V is a list of m vertices v_i in {0,1}^n.  Return a dict:
        {'resolves': True/False, 'reason': ..., 'witness': z (if False)}."""
    m = len(V); n = len(V[0]) if m else 0
    Vt = [vtilde_from_v(v, n) for v in V]

    basis, free_cols = rational_null_basis(Vt)
    if verbose:
        print(f"  m={m}, n={n}, null-space dim = {len(basis)}")
        for i, b in enumerate(basis):
            print(f"    basis[{i}] (rational) = {b}")

    if not basis:
        return {'resolves': True, 'reason': 'Vtilde has full column rank (null space = {0})',
                'null_dim': 0, 'witness': None}

    z, checked = ternary_in_span(Vt, verbose=verbose)
    if z is None:
        return {'resolves': True,
                'reason': f'Null space has dim {len(basis)}, but no nonzero {{-1,0,1}}^{n} '
                          f'vector lies in it (checked all {checked} candidates)',
                'null_dim': len(basis), 'witness': None}
    else:
        # Build the collision witness: any x, y in {0,1}^n with x - y = z.
        x = [max(zi, 0) for zi in z]
        y = [-min(zi, 0) for zi in z]
        assert [x[i] - y[i] for i in range(n)] == z
        return {'resolves': False, 'reason': f'Found nonzero z in {{-1,0,1}}^{n} with Vtilde @ z = 0',
                'null_dim': len(basis), 'witness': z, 'collision': (x, y)}


# ============================================================================
# Human-readable proof rendering (Beardon-style narrative)
# ============================================================================

def format_narrative(V, cert):
    """Produce a Beardon-style human-readable proof / disproof."""
    m = len(V); n = len(V[0]) if m else 0
    Vt = [vtilde_from_v(v, n) for v in V]
    out = []
    out.append(f"Candidate resolving set: {m} vertices in {{0,1}}^{n}")
    for i, v in enumerate(V):
        out.append(f"  v_{i+1} = {tuple(v)}     ~>     vtilde_{i+1} = {tuple(Vt[i])}")
    out.append("")
    out.append("By Beardon's Theorem 2.2, this set resolves V^n iff the linear system")
    out.append(f"    vtilde_j . z = 0   (j = 1..{m})")
    out.append(f"has only z = 0 as solution in {{-1,0,1}}^{n}.")
    out.append("")
    out.append("The linear system:")
    for j in range(m):
        terms = [f"{'+' if Vt[j][k] > 0 else '-'} z_{k+1}" for k in range(n)]
        eqn = " ".join(terms).replace("+ ", "+").replace("- ", "-").replace("+", "+ ").replace("-", "- ")
        if eqn.startswith("+ "): eqn = eqn[2:]
        out.append(f"    {eqn} = 0")
    out.append("")

    if cert['resolves']:
        out.append(f"CONCLUSION: this set RESOLVES V^{n}.")
        out.append(f"  Null-space dim over Q: {cert['null_dim']}.")
        if cert['null_dim'] == 0:
            out.append("  The vtilde vectors span R^n; the null space is trivial.")
        else:
            out.append(f"  Reason: {cert['reason']}")
        # Try to render a Beardon-style parametric solution
        basis = integer_null_basis(Vt)
        if basis and cert['null_dim'] <= 2:
            if cert['null_dim'] == 1:
                b = basis[0]
                # find a coordinate with large |b[k]| -- that's the integrality bottleneck
                maxk = max(range(n), key=lambda k: abs(b[k]))
                out.append(f"  The real solution space is {{ t * {tuple(b)} : t in R }}.")
                if abs(b[maxk]) >= 2:
                    out.append(f"  In {{-1,0,1}}^{n}, coord {maxk+1} = {b[maxk]}*t must lie in {{-1,0,1}};")
                    out.append(f"  since |{b[maxk]}*t| <= 1 forces |t| <= {Fraction(1, abs(b[maxk]))}, and if t is a rational")
                    out.append(f"  parameter with integer image, integrality of nearby coords forces t = 0.")
                else:
                    out.append(f"  Even |t*b_j| <= 1 alone doesn't force t=0, but the joint integrality on all")
                    out.append(f"  coords does; every ternary intersection with the line was checked.")
    else:
        z = cert['witness']
        x, y = cert['collision']
        out.append(f"CONCLUSION: this set DOES NOT resolve V^{n}.")
        out.append(f"  Witness z = {tuple(z)}  in {{-1,0,1}}^{n}, nonzero, with vtilde_j . z = 0 for all j.")
        out.append(f"  Corresponding collision:  x = {tuple(x)},  y = {tuple(y)},  x - y = z,")
        out.append(f"  and d(x, v_j) = d(y, v_j) for each j.")

    return "\n".join(out)


# ============================================================================
# Universal infeasibility: certify that NO m-subset of {0,1}^n resolves V^n
# ============================================================================

def certify_all_subsets(n, m, verbose=False, canonical_only=True):
    """Attempt to certify beta_n > m by exhaustively checking every candidate
    resolving set of size m from {0,1}^n, modulo symmetries.

    Uses the canonicalization from Beardon Lemma 3.1: complementing any v_i
    preserves the resolving property.  So we may WLOG assume each v_i has
    LAST coordinate = 0 (i.e., v_i in {0,1}^{n-1} x {0}).
    Additionally, we can fix v_1 = 0 (translation invariance is NOT present,
    but we CAN permute the set, so WLOG v_1 has the smallest lex order).

    This is exponential in m; for small (n, m) it's tractable.  For larger
    cases it's a stress-test that would find witnesses if beta_n <= m,
    otherwise it's a certificate that beta_n > m (subject to symmetry
    canonicalization being sound, which for the Lemma 3.1 part it is)."""
    from itertools import combinations
    # candidate vertices under Lemma 3.1 symmetry-break: last coord = 0
    if canonical_only:
        candidates = [tuple(list(bits) + [0]) for bits in product((0,1), repeat=n-1)]
    else:
        candidates = list(product((0,1), repeat=n))
    total = 0; found = 0
    for subset in combinations(candidates, m):
        total += 1
        V = list(subset)
        cert = certify(V)
        if not cert['resolves']:
            continue     # this subset doesn't resolve; try the next
        found += 1
        if verbose:
            print(f"    FOUND resolving set: {V}")
        return {'beta_upper': m, 'witness_set': V, 'total_checked': total}
    return {'beta_upper': None, 'total_checked': total,
            'conclusion': f"No {m}-subset (mod Lemma 3.1 symmetry) resolves V^{n}; hence beta_{n} > {m}."}


# ============================================================================
# Demo / smoke tests
# ============================================================================

def demo():
    print("="*70)
    print("Demo: reproduce Beardon's Erdos-Renyi example (Section 4)")
    print("="*70)
    # Four vectors in {0,1}^5 that resolve V^5
    V = [
        [1,1,1,1,1],
        [1,1,1,0,0],
        [0,1,0,1,0],
        [0,1,1,0,1],
    ]
    cert = certify(V, verbose=True)
    print()
    print(format_narrative(V, cert))
    print()

    print("="*70)
    print("Demo: Beardon's counter-example (Section 2) -- does NOT resolve V^4")
    print("="*70)
    V = [
        [1,0,0,0],       # e_1
        [1,0,1,0],       # e_1 + e_3
        [0,1,1,0],       # e_2 + e_3
    ]
    cert = certify(V, verbose=True)
    print()
    print(format_narrative(V, cert))
    print()

    print("="*70)
    print("Demo: {e_1, e_2, e_3, e_4} resolves V^5?")
    print("="*70)
    V = [
        [1,0,0,0,0],
        [0,1,0,0,0],
        [0,0,1,0,0],
        [0,0,0,1,0],
    ]
    cert = certify(V, verbose=True)
    print()
    print(format_narrative(V, cert))
    print()

    print("="*70)
    print("Universal infeasibility: certify beta_5 > 3 by exhausting 3-subsets")
    print("="*70)
    result = certify_all_subsets(n=5, m=3, verbose=False)
    print(result)
    print()

    print("="*70)
    print("Universal infeasibility: certify beta_6 > 4 by exhausting 4-subsets")
    print("(This one takes a few seconds; further cases scale worse.)")
    print("="*70)
    import time
    t0 = time.time()
    result = certify_all_subsets(n=6, m=4, verbose=False)
    print(result)
    print(f"({time.time()-t0:.1f}s)")
    print()

    print("Scaling: certify_all_subsets(n=6,m=4) enumerates ~36000 4-subsets in a few")
    print("seconds; (n,m)=(7,4) has ~635000 subsets and takes ~1 minute; (7,5) is far")
    print("worse (~7M subsets).  Beyond that the exhaustive route is impractical --")
    print("but per-instance certify() remains fast, and the per-instance search is the")
    print("real primitive.  A SAT-loop that pipes candidate resolving sets from a SAT")
    print("solver into certify() would let you use standard symmetry breaking on the")
    print("enumeration side while getting exact per-instance verdicts.")


if __name__ == "__main__":
    demo()
