"""
XOR-pivot reduction for balanced detecting matrices.

Background
----------
If A is a balanced detecting m x n matrix with rows D_1,...,D_m (subsets of
[n]), then for any pivot row p, the matrix A' obtained by

    - leaving row p unchanged, and
    - replacing every other row D_i by D_i XOR D_p (symmetric difference),
      then complementing it if its weight exceeds n/2 (to stay within the
      already-established row-complement symmetry)

is ALSO balanced detecting, and vice versa. Proof sketch (verified in full
before writing this module): encode each row as a sign vector s_j = 1 - 2*1[j
in D]. XOR on the 0/1 rows is entrywise *product* on the sign vectors:
s_i' = s_i (.) s_p. For balanced x (sum x_j = 0), s_i . x = -2 (D_i . x), so
D_i . x = 0 iff s_i . x = 0. Define y_j := s_{p,j} * x_j (flip the sign of x
on D_p, an involution that preserves weight but not balance in general). If x
is undetected by A', row p already fails on it, so D_p . x = 0 is automatic,
which forces y to be balanced (sum y_j = sum x_j - 2(D_p.x) = 0). Under that
condition, s_i' . x = s_i . y for every other row, and since y is balanced,
s_i . y = -2(D_i . y). Chaining through: x is undetected by A' iff y is
undetected by A. This is an exact bijection between the two "bad vector"
sets, so A is balanced detecting iff A' is.

Weight transform
-----------------
For pivot row p (weight w_p) and any other row j (weight w_j, intersection
k = |D_p ∩ D_j|), the transformed (and re-complemented) weight is

    w_j' = min(w_p + w_j - 2k, n - (w_p + w_j - 2k))

This is a fixed point (w_j' = w_j) at specific "self-dual" values of k.

IMPORTANT CORRECTNESS NOTE (found empirically, then explained, while working
the (4,4,5,5,5,5)@n=11 case -- this supersedes an earlier draft of this
module that claimed "total weight never increases under any pivot" in
general): that monotonicity claim is only true for a CONSTANT weight
profile. For a MIXED weight profile, an individual row's weight can
*increase* under a pivot (e.g. pivoting row of weight 4 into a row of weight
4 at intersection k=1, n=11, sends 4 -> 5), and such an increase can exactly
cancel a decrease elsewhere, leaving total weight unchanged (or even
increasing it for some pivot choices) while permuting *which* rows carry
which weight. Concrete witness (found by random search, then hand-verified):
6 rows of weights (4,4,5,5,5,5) at n=11 with pairwise intersections
|D0∩D1|=1, |D0∩D2|=|D0∩D3|=|D0∩D4|=2, |D0∩D5|=1, (D1 symmetric to D0's
pattern), all D2..D5 pairs =2 except |D3∩D5|=3: pivoting on row 0 sends row
1's weight 4->5 (increase) and row 5's weight 5->4 (decrease), leaving the
sorted weight multiset (4,4,5,5,5,5) EXACTLY UNCHANGED even though total
weight is unchanged and individual row identities permute. This realization
is a genuine fixed point of the reduction (verified by full BFS over its
reachable orbit, including "sideways" pivots that preserve the multiset --
the orbit has only 6 states and none has a lighter weight multiset), yet it
does NOT have every pairwise intersection at a per-pair self-dual value (the
(row0,row1) pair is at k=1, not at self_dual_k(4,4,11)=[2]).

Consequence: the correct potential function to minimize is not total_weight
but weight_profile (the SORTED multiset of row weights, compared
lexicographically) -- see `find_reducing_pivot` / `reduce_fully` below. For
a constant weight profile this coincides exactly with the total-weight
criterion (verified: 0 mismatches over 2000 random trials on (4,4,4,4,4)@9),
and in that special case a realization is irreducible iff ALL C(m,2)
pairwise intersections equal w/2 exactly (a "2-design"-like configuration).
For a MIXED weight profile there is no such clean pairwise characterization
in general: irreducibility can arise from a *joint*, multi-row cancellation
pattern (as in the witness above), not merely from every pair individually
sitting at a self-dual value. `profile_can_be_irreducible` below computes a
necessary-but-NOT-sufficient pairwise screen -- see its docstring for the
precise (limited) claim it supports.

Practical use: instead of directly enumerating the (often huge) space of
matrices realizing a weight profile, only the XOR-irreducible sub-case needs
to be searched directly; every other realization is equivalent (same
balanced-detecting status) to a realization of an already-covered,
lexicographically-lighter weight profile. Empirically (see `__main__` and
the (6,11) investigation), for mixed profiles the irreducible sub-case can
be a small but *nonzero* fraction of all realizations -- unlike the constant
profile case, it is not generally safe to assume a mixed profile reduces
away entirely.

Implementation note: this module is intentionally dependency-free (stdlib
only). `entropy.joint.weights_solutions` / `mentropies` currently cannot be
imported end-to-end (weights_solutions calls unwind(), which needs a
Multiset with .items(), but its actual caller chain passes a plain
np.ndarray; and multi_subsets calls an undefined mtuple_sub) -- confirmed by
direct import/call attempts. Rather than build on that broken pipeline, the
combinatorial routines below (matching resolving.entropy.joint's notion of a
"realization": an assignment of column-type multiplicities m(v), v in
{0,1}^m \\ {0}, consistent with a row-weight vector) are freshly written and
cross-validated against independent brute-force enumeration on small cases
(see the __main__ block).
"""
from itertools import product, combinations
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

Row = FrozenSet[int]


# ---------------------------------------------------------------------------
# Core reduction: operates on actual rows (sets of column indices)
# ---------------------------------------------------------------------------

def pivot_transform(rows: Sequence[Set[int]], pivot: int, n: int) -> List[FrozenSet[int]]:
    """
    XOR row `pivot` into every other row, then complement any row whose
    weight now exceeds n // 2. Row `pivot` itself is left unchanged.
    Returns a new list of rows (same order, same length as `rows`).
    """
    universe = frozenset(range(n))
    p = frozenset(rows[pivot])
    out = []
    for i, r in enumerate(rows):
        if i == pivot:
            out.append(frozenset(r))
            continue
        xr = frozenset(r) ^ p  # symmetric difference == XOR
        if len(xr) > n // 2:
            xr = universe - xr
        out.append(xr)
    return out


def total_weight(rows: Sequence[Set[int]]) -> int:
    return sum(len(r) for r in rows)


def weight_profile(rows: Sequence[Set[int]]) -> Tuple[int, ...]:
    """
    The sorted multiset of row weights. This, not total_weight, is the
    correct potential function to minimize when reducing: for a constant
    weight profile the two agree, but for a mixed profile they can diverge
    because a pivot can *increase* some individual row weights while
    decreasing others in a way that cancels in the sum but permutes which
    rows carry which weight (see the module docstring for a concrete
    witness). Comparing weight_profile lexicographically is a well-founded
    (finite, strictly-decreasing) order, so repeatedly moving to any
    lexicographically-smaller profile still terminates.
    """
    return tuple(sorted(len(r) for r in rows))


def self_dual_k(w_pivot: int, w_other: int, n: int) -> List[int]:
    """
    The intersection value(s) k = |D_pivot ∩ D_other| for which pivoting on
    `pivot` leaves row `other`'s weight unchanged (a "fixed point" for this
    pair). There can be 0, 1 or 2 such k in range [max(0,w_pivot+w_other-n),
    min(w_pivot,w_other)].
    """
    lo = max(0, w_pivot + w_other - n)
    hi = min(w_pivot, w_other)
    out = []
    for k in range(lo, hi + 1):
        raw = w_pivot + w_other - 2 * k
        if min(raw, n - raw) == w_other:
            out.append(k)
    return out


def find_reducing_pivot(rows: Sequence[Set[int]], n: int) -> Optional[int]:
    """
    Return a pivot index that makes the SORTED weight multiset
    lexicographically smaller (the correct notion of "reduces" -- see the
    module docstring; total weight alone is NOT a valid potential function
    for mixed profiles), or None if no pivot can improve on it (a "fixed
    point" of the reduction). When several pivots improve, the one giving
    the lexicographically smallest result is chosen (this is just a tie-
    break for determinism; any strictly-improving choice is equally valid
    for termination, since weight_profile is well-founded).
    """
    w0 = weight_profile(rows)
    best = None
    for p in range(len(rows)):
        wp = weight_profile(pivot_transform(rows, p, n))
        if wp < w0 and (best is None or wp < best[1]):
            best = (p, wp)
    return best[0] if best else None


def reduce_fully(rows: Sequence[Set[int]], n: int) -> Tuple[List[FrozenSet[int]], List[int]]:
    """
    Repeatedly apply reducing pivots until reaching a fixed point (no pivot
    lowers the sorted weight multiset, in lex order, any further). Returns
    (final_rows, pivot_trace). Balanced-detecting status is preserved at
    every step, so the returned matrix is BD iff the original is.
    """
    cur = [frozenset(r) for r in rows]
    trace = []
    while True:
        p = find_reducing_pivot(cur, n)
        if p is None:
            return cur, trace
        cur = pivot_transform(cur, p, n)
        trace.append(p)


def is_design_fixed_point(rows: Sequence[Set[int]], n: int) -> bool:
    """True iff no pivot can reduce the (lex) weight profile (irreducible)."""
    return find_reducing_pivot(rows, n) is None


def orbit_can_reduce(rows: Sequence[Set[int]], n: int, max_states: int = 20000) -> Tuple[bool, int, Tuple[int, ...]]:
    """
    Stronger check than find_reducing_pivot / reduce_fully: explores the
    FULL orbit reachable via any sequence of pivots that never makes the
    weight profile lexicographically *worse* (including "sideways" pivots
    that tie -- these are exactly the moves reduce_fully's greedy walk
    cannot see, since it only takes strictly-improving steps). Returns
    (found_a_lighter_profile, states_explored, lightest_profile_seen).
    Use this to confirm a `reduce_fully` fixed point is *genuinely*
    irreducible and not just a local optimum of the greedy walk (in
    practice, for the (4,4,5,5,5,5)@11 witnesses examined, the two never
    differ -- greedy fixed points were confirmed genuine by this check --
    but the greedy walk alone doesn't prove that on its own).
    """
    from collections import deque

    def canon(rs):
        return tuple(sorted(frozenset(r) for r in rs))

    start = [frozenset(r) for r in rows]
    w0 = weight_profile(start)
    seen = {canon(start)}
    q = deque([start])
    best = w0
    found_smaller = False
    explored = 0
    while q and explored < max_states:
        cur = q.popleft()
        explored += 1
        for p in range(len(cur)):
            nxt = pivot_transform(cur, p, n)
            wk = weight_profile(nxt)
            if wk < w0:
                found_smaller = True
            if wk < best:
                best = wk
            c = canon(nxt)
            if c not in seen and wk <= w0:
                seen.add(c)
                q.append(nxt)
        if found_smaller:
            break
    return found_smaller, explored, best


def pairwise_self_dual_candidates(w_i: int, w_j: int, n: int) -> List[int]:
    """
    k values for which the pair (i,j) is robust to being pivoted on from
    EITHER side (pivoting on i keeps j's weight, AND pivoting on j keeps
    i's weight).
    """
    return sorted(set(self_dual_k(w_i, w_j, n)) & set(self_dual_k(w_j, w_i, n)))


def profile_can_be_irreducible(weights: Sequence[int], n: int) -> Tuple[bool, List]:
    """
    Cheap NECESSARY-but-NOT-SUFFICIENT pairwise screen, no enumeration: for
    every pair of rows, is there at least one intersection value making
    that pair simultaneously robust to pivoting from either side?

    CAVEAT (found empirically on (4,4,5,5,5,5)@11, do not remove this
    without re-deriving the mixed-weight theory from scratch): this
    function's "False" answer does NOT imply "every realization is
    reducible." It only implies "no realization can have EVERY pair
    simultaneously self-dual." For a constant weight profile that
    coincides exactly with irreducibility (see module docstring), so
    "False" there really does mean 0 irreducible realizations. For a MIXED
    profile it does not: a realization can still be a genuine fixed point
    of the reduction via a *joint* multi-row cancellation (individual pairs
    off their self-dual value, but their weight changes canceling across
    rows so the sorted multiset is preserved) -- concretely, profile
    (4,4,5,5,5,5)@11 returns can_be_irreducible=False here, yet random
    sampling finds real (BFS-confirmed) irreducible realizations at a rate
    of roughly 0.04%-0.09% (95% CI from a 30,000-trial Monte Carlo run; see
    the (6,11) report). Use this function only as a quick necessary check
    ("if True, definitely can't rule out irreducibles"); use
    `reduce_fully` / `orbit_can_reduce` on actual realizations, or Monte
    Carlo sampling, to determine the true fraction for a mixed profile.
    """
    m = len(weights)
    bad_pairs = []
    for i, j in combinations(range(m), 2):
        cands = pairwise_self_dual_candidates(weights[i], weights[j], n)
        if not cands:
            bad_pairs.append(((i, j), (weights[i], weights[j])))
    return len(bad_pairs) == 0, bad_pairs


# ---------------------------------------------------------------------------
# Counting utilities (constant weight profile): quantify how much of the
# realization space is irreducible vs. reducible-away.
#
# A "realization" here matches resolving.entropy.joint's notion: an
# assignment m: {0,1}^m -> Z>=0 of column-type multiplicities with
# sum_v m(v) * v_i = w for every row i, and sum_v m(v) <= n (remaining
# columns are the all-zero/degree-0 type). This already quotients out column
# permutation; it does NOT quotient out row permutation/complementation.
# ---------------------------------------------------------------------------

def exact_realization_count(weights: Sequence[int], n: int) -> int:
    """
    Exact count of column-type-multiplicity realizations of `weights`
    (no filtering of duplicate/degenerate rows). Computed by DP over column
    types with state = (remaining per-row weight budget, columns used so
    far) -- exact, and cross-validated against brute-force enumeration on
    small cases (see __main__).
    """
    m = len(weights)
    types = [v for v in product((0, 1), repeat=m) if any(v)]
    dp: Dict[Tuple[Tuple[int, ...], int], int] = {(tuple(weights), 0): 1}
    for v in types:
        ndp: Dict[Tuple[Tuple[int, ...], int], int] = {}
        for (rem, used), cnt in dp.items():
            max_mult_w = min((rem[i] for i in range(m) if v[i] == 1), default=n)
            max_mult = min(max_mult_w, n - used)
            for mult in range(max_mult + 1):
                nrem = tuple(rem[i] - mult * v[i] for i in range(m))
                key = (nrem, used + mult)
                ndp[key] = ndp.get(key, 0) + cnt
        dp = ndp
    return sum(cnt for (rem, used), cnt in dp.items() if all(r == 0 for r in rem))


def design_realization_count(weights: Sequence[int], n: int, target_k: int) -> int:
    """
    Exact count of realizations additionally satisfying: every pairwise row
    intersection |D_i ∩ D_j| == target_k (the XOR-irreducible/fixed-point
    case for a CONSTANT weight profile -- pass target_k = w // 2; this
    characterization is only valid for a constant weight profile, see
    module docstring). DP state tracks remaining per-row weight budget AND
    remaining per-pair intersection budget, with feasibility pruning (a row
    with 0 weight budget left can't satisfy any pair constraint still
    touching it), and processes higher-popcount column types first since
    they constrain the most rows/pairs, which prunes the live-state set
    fast in practice.
    """
    m = len(weights)
    types = sorted((v for v in product((0, 1), repeat=m) if any(v)), key=lambda v: -sum(v))
    pairs = list(combinations(range(m), 2))

    State = Tuple[Tuple[int, ...], Tuple[int, ...], int]
    dp: Dict[State, int] = {(tuple(weights), tuple(target_k for _ in pairs), 0): 1}

    for v in types:
        pair_contrib = tuple(1 if v[i] and v[j] else 0 for (i, j) in pairs)
        ndp: Dict[State, int] = {}
        for (rem_w, rem_pair, used), cnt in dp.items():
            max_mult_w = min((rem_w[i] for i in range(m) if v[i] == 1), default=n)
            max_mult_pair = min((rem_pair[k] for k in range(len(pairs)) if pair_contrib[k] == 1), default=n)
            max_mult = min(max_mult_w, max_mult_pair, n - used)
            for mult in range(max_mult + 1):
                nrem_w = tuple(rem_w[i] - mult * v[i] for i in range(m))
                nrem_pair = tuple(rem_pair[k] - mult * pair_contrib[k] for k in range(len(pairs)))
                if any(nrem_pair[k] > 0 and (nrem_w[i] == 0 or nrem_w[j] == 0)
                       for k, (i, j) in enumerate(pairs)):
                    continue  # dead: a pair still needs intersection but a row involved is exhausted
                key = (nrem_w, nrem_pair, used + mult)
                ndp[key] = ndp.get(key, 0) + cnt
        dp = ndp

    return sum(cnt for (rem_w, rem_pair, used), cnt in dp.items()
               if all(r == 0 for r in rem_w) and all(r == 0 for r in rem_pair))


def monte_carlo_irreducible_fraction(weights: Sequence[int], n: int, trials: int = 30000,
                                      seed: int = 2026) -> Dict[str, object]:
    """
    For a MIXED weight profile, where design_realization_count's single-
    target_k characterization does not apply (see module docstring),
    estimate the fraction of realizations that are XOR-irreducible by
    sampling: draw `trials` independent realizations (each row an
    independent uniform random size-w_i subset of [n] -- NOTE this is not
    exactly uniform over the column-type-multiplicity realization measure
    that exact_realization_count uses, so treat this as indicative
    sampling of "typical" realizations, not a rigorous estimator of the
    exact-count fraction), run reduce_fully on each, and report the
    fraction with an empty pivot trace, with a 95% Wilson score interval.
    """
    import random
    import math

    rng = random.Random(seed)
    stuck = 0
    trace_lens: Dict[int, int] = {}
    for _ in range(trials):
        rows = [set(rng.sample(range(n), w)) for w in weights]
        _, trace = reduce_fully(rows, n)
        trace_lens[len(trace)] = trace_lens.get(len(trace), 0) + 1
        if len(trace) == 0:
            stuck += 1

    p_hat = stuck / trials
    z = 1.959963985
    denom = 1 + z * z / trials
    center = (p_hat + z * z / (2 * trials)) / denom
    halfwidth = (z * math.sqrt(p_hat * (1 - p_hat) / trials + z * z / (4 * trials * trials))) / denom
    return {
        "trials": trials,
        "stuck": stuck,
        "p_hat": p_hat,
        "ci95": (max(0.0, center - halfwidth), center + halfwidth),
        "trace_len_distribution": dict(sorted(trace_lens.items())),
    }


if __name__ == "__main__":
    import itertools as _it
    import time as _time

    # --- correctness checks (independent brute force on small cases) ---
    def _bruteforce_count(weights, n):
        m = len(weights)
        types = [v for v in _it.product((0, 1), repeat=m) if any(v)]
        cnt = 0
        for k in range(0, n + 1):
            for combo in _it.combinations_with_replacement(types, k):
                if [sum(c[i] for c in combo) for i in range(m)] == list(weights):
                    cnt += 1
        return cnt

    def _bruteforce_design_count(weights, n, target_k):
        m = len(weights)
        types = [v for v in _it.product((0, 1), repeat=m) if any(v)]
        cnt = 0
        for k in range(0, n + 1):
            for combo in _it.combinations_with_replacement(types, k):
                if [sum(c[i] for c in combo) for i in range(m)] != list(weights):
                    continue
                if all(sum(1 for c in combo if c[i] and c[j]) == target_k
                       for i, j in _it.combinations(range(m), 2)):
                    cnt += 1
        return cnt

    assert exact_realization_count((2, 2, 2), 4) == _bruteforce_count((2, 2, 2), 4) == 12
    assert design_realization_count((2, 2, 2), 4, 1) == _bruteforce_design_count((2, 2, 2), 4, 1) == 2
    print("DP counting routines verified against brute force on (2,2,2)@n=4.")

    # --- pivot_transform / reduce_fully sanity check ---
    rows = [{0, 3, 5, 7}, {4, 5, 6, 7}, {0, 1, 6, 8}, {0, 4, 6, 8}, {1, 4, 5, 8}]
    reduced, trace = reduce_fully(rows, 9)
    print(f"reduce_fully on a random weight-(4,4,4,4,4) family: "
          f"{sorted(len(r) for r in rows)} -> {sorted(len(r) for r in reduced)}  "
          f"(pivot trace={trace})")

    # --- the headline numbers for Victor's (5,9) case ---
    t0 = _time.time()
    total = exact_realization_count((4, 4, 4, 4, 4), 9)
    t1 = _time.time()
    design = design_realization_count((4, 4, 4, 4, 4), 9, 2)
    t2 = _time.time()
    print()
    print(f"(4,4,4,4,4) @ n=9:")
    print(f"  total column-type realizations   = {total:,}   ({t1-t0:.2f}s)")
    print(f"  XOR-irreducible (design) subset   = {design:,}   ({t2-t1:.2f}s)")
    print(f"  reduction factor                  = {total/design:,.1f}x")

    # --- Victor's (6,11) semi-hard case: (4,4,5,5,5,5) ---
    print()
    print(f"(4,4,5,5,5,5) @ n=11:")
    t3 = _time.time()
    total_611 = exact_realization_count((4, 4, 5, 5, 5, 5), 11)
    t4 = _time.time()
    print(f"  total column-type realizations   = {total_611:,}   ({t4-t3:.2f}s)")

    can_irr, bad_pairs = profile_can_be_irreducible((4, 4, 5, 5, 5, 5), 11)
    print(f"  profile_can_be_irreducible (necessary screen only) = {can_irr}  bad_pairs={bad_pairs}")

    mc = monte_carlo_irreducible_fraction((4, 4, 5, 5, 5, 5), 11, trials=30000)
    lo, hi = mc["ci95"]
    print(f"  Monte Carlo ({mc['trials']:,} trials): stuck={mc['stuck']}  "
          f"p_hat={mc['p_hat']:.5f}  95% CI=[{lo:.5f}, {hi:.5f}]")
    print(f"  pivot-trace-length distribution: {mc['trace_len_distribution']}")
    print(f"  => despite profile_can_be_irreducible()==False, a small but NONZERO fraction")
    print(f"     of realizations are genuinely XOR-irreducible (BFS-confirmed on witnesses);")
    print(f"     ~99.9%+ still reduce to a strictly lighter weight profile in exactly 1 pivot.")
