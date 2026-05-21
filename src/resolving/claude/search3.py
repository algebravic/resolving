"""
Set cover formulation:
  Universe: balanced vectors B^(n)
  For each candidate row r, define C_r = {x in B^(n) : r.x != 0}
  Find minimum cover by C_r sets.

For "full" version, candidate rows = those of weight floor(n/2).
For "unrestricted", complement-normalized = weight <= n/2.
"""
import sys
from itertools import combinations, product
import time

def popcount(x):
    return bin(x).count('1')

def balanced_masks(n):
    out = []
    for v in product((-1, 0, 1), repeat=n):
        if sum(v) == 0 and any(v):
            pos = 0; neg = 0
            for i, vi in enumerate(v):
                if vi == 1: pos |= (1 << i)
                elif vi == -1: neg |= (1 << i)
            out.append((pos, neg))
    return out

def full_rows(n):
    w = n // 2
    return [sum(1<<i for i in s) for s in combinations(range(n), w)]

def normalized_rows(n):
    return [r for r in range(1, 1<<n) if popcount(r) <= n // 2]

def coverage(rows, masks):
    """For each row r, return integer bitmask of which masks indices it catches."""
    cov = []
    nmasks = len(masks)
    for r in rows:
        bm = 0
        for idx, (p, ng) in enumerate(masks):
            if popcount(r & p) != popcount(r & ng):
                bm |= (1 << idx)
        cov.append(bm)
    return cov

def min_cover(coverage_list, universe_mask, m_target):
    """Find a cover of universe_mask using m_target sets from coverage_list. Return cover (list of indices) or None."""
    # Greedy ordering: try sets that cover most uncovered elements first.
    n_sets = len(coverage_list)

    def backtrack(remaining, picks, max_picks, start_idx):
        if remaining == 0:
            return list(picks)
        if max_picks == 0:
            return None
        # Pruning: cardinality of remaining vs. max coverage of a single set
        # Pick element with fewest covering sets to bound branching
        # Find an arbitrary uncovered element e
        # Determine all sets that cover e — must use one of them
        # Pick e to be the one with fewest covers (forced choice principle)
        rem_bits = remaining
        e = (rem_bits & -rem_bits).bit_length() - 1
        # Sets covering e (with index >= start_idx for ordering)
        # Actually for set cover, we don't need ordering by index; but to break symmetry,
        # we can require picks to be increasing. But that might miss solutions if same set is used twice — let's allow non-strict.
        # For now: any set covering e
        candidates = [i for i in range(n_sets) if (coverage_list[i] >> e) & 1]
        # Sort by coverage of remaining (greedy)
        candidates.sort(key=lambda i: -popcount(coverage_list[i] & remaining))
        for ci in candidates:
            new_remaining = remaining & ~coverage_list[ci]
            picks.append(ci)
            res = backtrack(new_remaining, picks, max_picks - 1, 0)
            if res is not None:
                return res
            picks.pop()
        return None

    return backtrack(universe_mask, [], m_target, 0)

def test(n, m_target, use_full=True, time_limit=600):
    masks = balanced_masks(n)
    rows = full_rows(n) if use_full else normalized_rows(n)
    print(f"n={n}, m_target={m_target}, |rows|={len(rows)}, |B|={len(masks)}", flush=True)
    cov = coverage(rows, masks)
    universe = (1 << len(masks)) - 1
    t0 = time.time()
    result = min_cover(cov, universe, m_target)
    t1 = time.time()
    print(f"  result: {'FOUND' if result else 'NONE'} (t={t1-t0:.1f}s)", flush=True)
    if result:
        for i in result:
            r = rows[i]
            bits = bin(r)[2:].zfill(n)[::-1]
            print(f"    row: {bits}")
    return result is not None

# First reconfirm n=8 at m=5
# print("=== n=8, m=5, full ===")
# test(8, 5, use_full=True)
# print("=== n=9, m=6, full ===")
# test(9, 6, use_full=True, time_limit=600)
