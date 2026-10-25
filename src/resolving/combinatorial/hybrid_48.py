"""
Hybrid entropy + enumeration proof that α(8) ≥ 5 (no 4×8 balanced-detecting matrix).

Pipeline:
  1. Level-1 entropy filter: 35 profiles → 9 univariate-feasible
  2. Level-2 entropy filter: enumerate (profile + intersection matrix) configs;
     ~46K → 778 pass bivariate Shearer
  3. Per-signature enumeration: for each surviving signature, enumerate all matrices
     with that structure and check if any is balanced-detecting
  4. If NONE has a balanced-detecting matrix → α(8) ≥ 5 proved.
"""

from math import log2, comb
from itertools import combinations_with_replacement, product, combinations
import time

n = 8
m = 4

# ---------- Entropy machinery ----------

def H_hyper(d, s, n):
    if d == 0 or d == n:
        return 0.0
    lo, hi = max(0, s + d - n), min(d, s)
    total = comb(n, s)
    H = 0.0
    for j in range(lo, hi + 1):
        p = comb(d, j) * comb(n - d, s - j) / total
        if p > 0:
            H -= p * log2(p)
    return H

def H_pair(d1, d2, k, s, n):
    r_none = n - d1 - d2 + k
    if r_none < 0 or k < 0 or k > min(d1, d2):
        return None
    if d1 - k < 0 or d2 - k < 0:
        return None
    joint = {}
    total = comb(n, s)
    for a in range(min(k, s) + 1):
        for b in range(min(d1 - k, s - a) + 1):
            for c in range(min(d2 - k, s - a - b) + 1):
                d = s - a - b - c
                if d < 0 or d > r_none:
                    continue
                p = comb(k, a) * comb(d1 - k, b) * comb(d2 - k, c) * comb(r_none, d) / total
                if p > 0:
                    key = (a + b, a + c)
                    joint[key] = joint.get(key, 0) + p
    return -sum(p * log2(p) for p in joint.values() if p > 0)

# ---------- Level 1 + 2 entropy filtering ----------

def level1_feasible():
    """Enumerate profiles that pass univariate entropy."""
    profiles = list(combinations_with_replacement(range(1, n // 2 + 1), m))
    feasible = []
    for p in profiles:
        ok = True
        for s in range(1, n):
            need = log2(comb(n, s))
            got = sum(H_hyper(k, s, n) for k in p)
            if got < need - 1e-9:
                ok = False
                break
        if ok:
            feasible.append(p)
    return feasible

pairs_idx = [(i, j) for i in range(m) for j in range(i + 1, m)]

def realizable_triple(profile, k_pairs):
    """Check that pairwise intersections are consistent (triple-existence)."""
    for a, b, c in combinations(range(m), 3):
        k_ab = k_pairs[pairs_idx.index((a, b))]
        k_ac = k_pairs[pairs_idx.index((a, c))]
        k_bc = k_pairs[pairs_idx.index((b, c))]
        lb = max(0, k_ab + k_ac - profile[a],
                    k_ab + k_bc - profile[b],
                    k_ac + k_bc - profile[c])
        ub = min(k_ab, k_ac, k_bc)
        if lb > ub:
            return False
    return True

def bivariate_feasible(profile, k_pairs):
    """Check that bivariate Shearer bound is satisfied at all s."""
    for s in range(2, n - 1):
        need = (m - 1) * log2(comb(n, s))
        got = 0
        for t, (i, j) in enumerate(pairs_idx):
            Hp = H_pair(profile[i], profile[j], k_pairs[t], s, n)
            if Hp is None:
                return False
            got += Hp
        if got < need - 1e-9:
            return False
    return True

def level2_feasible():
    """Enumerate (profile, pairwise intersections) that pass bivariate Shearer."""
    L1 = level1_feasible()
    result = []
    for p in L1:
        ranges = []
        for i, j in pairs_idx:
            lo = max(0, p[i] + p[j] - n)
            hi = min(p[i], p[j])
            ranges.append(range(lo, hi + 1))
        for k_pairs in product(*ranges):
            if not realizable_triple(p, k_pairs):
                continue
            if bivariate_feasible(p, k_pairs):
                result.append((p, k_pairs))
    return result

# ---------- Balanced-detecting check ----------

def enumerate_balanced(n):
    """Enumerate B(n) as list of (plus_mask, minus_mask) pairs, canonical (first nonzero +1)."""
    result = []
    for x_plus in range(1 << n):
        pw = bin(x_plus).count('1')
        if pw == 0:
            continue
        remaining = ((1 << n) - 1) ^ x_plus
        # Iterate subsets of `remaining` of size pw
        # We want x_minus disjoint from x_plus, |x_minus| = |x_plus|.
        for x_minus in subsets_of_size(remaining, pw):
            # Canonicalize: first nonzero should be +1
            # first nonzero position: min bit of (x_plus | x_minus)
            comb_mask = x_plus | x_minus
            first_bit = (comb_mask & -comb_mask)
            if (x_plus & first_bit) == 0:
                continue  # first is -1, skip (canonical requires +1)
            result.append((x_plus, x_minus, pw))  # pw = weight/2 (half-weight)
    return result

def subsets_of_size(mask, k):
    """Yield all sub-bitmasks of `mask` with popcount exactly k."""
    # Extract bit positions of mask
    positions = []
    m = mask
    while m:
        positions.append((m & -m).bit_length() - 1)
        m &= m - 1
    if k > len(positions):
        return
    for combo in combinations(positions, k):
        s = 0
        for p in combo:
            s |= (1 << p)
        yield s

def popcount(x):
    return bin(x).count('1')

def is_balanced_detecting(rows, balanced_vecs):
    """rows = tuple of 4 bitmasks. balanced_vecs = list of (x_plus, x_minus, half_weight).
       Return True if the 4×n matrix (from these rows) detects every balanced vector."""
    for xp, xm, hw in balanced_vecs:
        detected = False
        for r in rows:
            if popcount(r & xp) != popcount(r & xm):
                detected = True
                break
        if not detected:
            return False
    return True

# ---------- Per-signature enumeration ----------

def bitmasks_of_size(n, k):
    """All n-bit bitmasks with popcount k."""
    result = []
    for mask in range(1 << n):
        if popcount(mask) == k:
            result.append(mask)
    return result

# Precompute bitmasks by size
by_size = {k: bitmasks_of_size(n, k) for k in range(1, n // 2 + 1)}
# Add sizes n-k (for possibly needing full range)
for k in range(n // 2 + 1, n + 1):
    by_size[k] = bitmasks_of_size(n, k)

# Precompute B(n)
print(f"Enumerating B({n})...")
B = enumerate_balanced(n)
print(f"|B({n})| = {len(B)}")

def enumerate_matrices_for_signature(profile, k_pairs):
    """Enumerate (up to S_n column permutation) all 4×n matrices with 
       given profile and pairwise intersections."""
    # Fix Row 0 to canonical form: first profile[0] bits set.
    # This exploits S_n column permutation, dividing by n!/(#column-stabilizers).
    r0 = (1 << profile[0]) - 1
    k01 = k_pairs[pairs_idx.index((0, 1))]
    k02 = k_pairs[pairs_idx.index((0, 2))]
    k03 = k_pairs[pairs_idx.index((0, 3))]
    k12 = k_pairs[pairs_idx.index((1, 2))]
    k13 = k_pairs[pairs_idx.index((1, 3))]
    k23 = k_pairs[pairs_idx.index((2, 3))]
    matrices = []
    # Enumerate r1 with |r1| = profile[1], |r0 ∩ r1| = k01
    r1_candidates = [r for r in by_size[profile[1]] if popcount(r & r0) == k01]
    for r1 in r1_candidates:
        # Enumerate r2 with size profile[2], |r0∩r2|=k02, |r1∩r2|=k12
        r2_candidates = [r for r in by_size[profile[2]] 
                         if popcount(r & r0) == k02 and popcount(r & r1) == k12]
        for r2 in r2_candidates:
            # Enumerate r3 with size profile[3], intersections k03, k13, k23
            r3_candidates = [r for r in by_size[profile[3]] 
                             if popcount(r & r0) == k03 
                             and popcount(r & r1) == k13 
                             and popcount(r & r2) == k23]
            for r3 in r3_candidates:
                matrices.append((r0, r1, r2, r3))
    return matrices

# ---------- Main pipeline ----------

def main():
    t0 = time.time()
    print(f"\nLevel-1 filter...")
    L1 = level1_feasible()
    print(f"  Univariate-feasible profiles: {len(L1)}")
    for p in L1:
        print(f"    {p}")

    print(f"\nLevel-2 filter...")
    L2 = level2_feasible()
    print(f"  Bivariate-feasible signatures: {len(L2)}")
    print(f"  Time: {time.time() - t0:.1f}s")

    print(f"\nPer-signature enumeration + balanced-detecting check...")
    print(f"  Checking {len(L2)} signatures. Expected: all UNSAT (α(8) = 5)")
    n_sat = 0
    n_unsat = 0
    n_matrices_total = 0
    solution_signatures = []
    t1 = time.time()
    for idx, (profile, k_pairs) in enumerate(L2):
        matrices = enumerate_matrices_for_signature(profile, k_pairs)
        n_matrices_total += len(matrices)
        found_bd = False
        for A in matrices:
            if is_balanced_detecting(A, B):
                found_bd = True
                solution_signatures.append((profile, k_pairs, A))
                break
        if found_bd:
            n_sat += 1
        else:
            n_unsat += 1
        if (idx + 1) % 100 == 0 or idx == len(L2) - 1:
            print(f"    Progress: {idx+1}/{len(L2)} | SAT: {n_sat}, UNSAT: {n_unsat}, "
                  f"total matrices checked: {n_matrices_total}, elapsed: {time.time() - t1:.1f}s")

    print(f"\n=== FINAL RESULT ===")
    print(f"Total signatures: {len(L2)}")
    print(f"UNSAT (no balanced-detecting matrix): {n_unsat}")
    print(f"SAT (balanced-detecting matrix exists): {n_sat}")
    print(f"Total matrices enumerated: {n_matrices_total}")
    print(f"Total time: {time.time() - t0:.1f}s")
    if n_sat == 0:
        print(f"\n✓ PROVED: α(8) ≥ 5 (no 4×8 balanced-detecting matrix exists)")
    else:
        print(f"\n✗ Found balanced-detecting matrix — check for bugs")
        for p, k, A in solution_signatures[:3]:
            print(f"  Signature: profile={p}, intersections={dict(zip(pairs_idx, k))}")
            print(f"  Matrix bitmasks: {A}")

if __name__ == "__main__":
    main()
