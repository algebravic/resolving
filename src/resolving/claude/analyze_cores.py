#!/usr/bin/env python3
"""
analyze_cores.py -- Structural analysis of balanced-ternary cores from CEGAR/OPTUX runs.

Analyzes one or more files, each containing a set of balanced ternary vectors (rows), and
reports structural statistics relevant to the matroid-hypothesis investigation of
detecting-matrix infeasibility proofs.

Usage:
    python analyze_cores.py FILE [FILE ...] [--options]

File format: each line is a balanced ternary vector like "(1, -1, 0, 0, -1, 1)" or "1 -1 0 0 -1 1".
Empty lines and lines starting with '#' are ignored.

Examples:
    # Analyze one file
    python analyze_cores.py min_10_6.txt

    # Compare multiple cores from the same instance
    python analyze_cores.py min_10_6.txt minx_10_6.txt miny_10_6.txt

    # Just show the intersection ("hard core") without individual stats
    python analyze_cores.py --intersection-only *.txt

    # Focus on weight-w structure
    python analyze_cores.py --focus-weight 4 min_10_6.txt

Requirements:
    numpy (required)
    networkx (optional, for spanning-tree checks; falls back to simple check if missing)

Author: Written to support Victor Miller's β(Q_n) research.
"""

from __future__ import annotations
import argparse
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from math import comb
from typing import Sequence

import numpy as np

# ----- Optional networkx -----
try:
    import networkx as nx
    HAVE_NX = True
except ImportError:
    HAVE_NX = False


# =============================================================================
# I/O and validation
# =============================================================================

def parse_file(path: str) -> list[tuple[int, ...]]:
    """Parse a file with one balanced ternary vector per line.

    Accepts formats like (1, -1, 0, ...) or 1 -1 0 ... — anything from which
    the regex -?\\d+ extracts the entries in order.
    """
    with open(path) as f:
        vectors = []
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            nums = [int(x) for x in re.findall(r'-?\d+', line)]
            if not nums:
                continue
            vectors.append(tuple(nums))
    return vectors


def validate(T: list[tuple[int, ...]], name: str = "T") -> tuple[bool, int]:
    """Validate that every vector is balanced ternary of the same length.

    Returns (is_valid, n). Prints any problems found.
    """
    if not T:
        print(f"WARNING: {name} is empty")
        return True, 0

    n = len(T[0])
    problems = []
    for i, t in enumerate(T):
        if len(t) != n:
            problems.append(f"  vec {i}: length {len(t)} != {n}")
            continue
        if any(v not in (-1, 0, 1) for v in t):
            problems.append(f"  vec {i}: non-ternary entry in {t}")
        if sum(t) != 0:
            problems.append(f"  vec {i}: sum = {sum(t)} != 0 in {t}")
        if all(v == 0 for v in t):
            problems.append(f"  vec {i}: zero vector")
    if problems:
        print(f"VALIDATION PROBLEMS in {name}:")
        for p in problems[:10]:
            print(p)
        if len(problems) > 10:
            print(f"  ... {len(problems)-10} more")
        return False, n
    return True, n


# =============================================================================
# Structural analyses
# =============================================================================

def weight(t: Sequence[int]) -> int:
    """Number of nonzero entries."""
    return sum(1 for v in t if v != 0)


def support(t: Sequence[int]) -> tuple[int, ...]:
    """Indices of nonzero entries."""
    return tuple(i for i, v in enumerate(t) if v != 0)


def signed_pairing_key(t: Sequence[int]) -> tuple:
    """Canonical form of a balanced vector's signed pairing.

    Two vectors are equivalent up to sign iff they have the same signed_pairing_key.
    We choose the sign in which the smallest nonzero index is positive.
    """
    first_nz = next((v for v in t if v != 0), None)
    if first_nz is None:
        return t
    if first_nz < 0:
        return tuple(-v for v in t)
    return tuple(t)


def q_rank(T: list[tuple[int, ...]]) -> int:
    if not T:
        return 0
    return int(np.linalg.matrix_rank(np.array(T, dtype=float)))


def f2_rank(rows: np.ndarray) -> int:
    """Row rank over F_2 via Gaussian elimination. Non-mutating."""
    M = rows.copy() % 2
    R, C = M.shape
    r = 0
    for c in range(C):
        piv = None
        for i in range(r, R):
            if M[i, c] == 1:
                piv = i
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(R):
            if i != r and M[i, c] == 1:
                M[i] = (M[i] + M[r]) % 2
        r += 1
        if r == R:
            break
    return r


def f2_support_rank(T: list[tuple[int, ...]]) -> int:
    """F_2 rank of the {0,1}-support characteristic vectors."""
    if not T:
        return 0
    S = np.array([[1 if v != 0 else 0 for v in t] for t in T], dtype=np.int8)
    return f2_rank(S)


def used_columns(T: list[tuple[int, ...]], n: int) -> set[int]:
    used = set()
    for t in T:
        for i, v in enumerate(t):
            if v != 0:
                used.add(i)
    return used


def weight_distribution(T: list[tuple[int, ...]]) -> Counter:
    return Counter(weight(t) for t in T)


def weight2_graph_summary(T: list[tuple[int, ...]], n: int) -> dict:
    """Analyze the weight-2 vectors as a graph on columns.

    Weight-2 says "columns i, j must differ." A spanning tree of K_n means the
    core forces pairwise distinguishability of all columns.
    """
    w2 = [t for t in T if weight(t) == 2]
    edges = set()
    for t in w2:
        supp = support(t)
        edges.add(frozenset(supp))

    used = set()
    for e in edges:
        used.update(e)

    result = {
        'n_vectors': len(w2),
        'n_distinct_edges': len(edges),
        'edges': sorted(sorted(e) for e in edges),
        'columns_covered': sorted(used),
        'n_columns_covered': len(used),
        'is_spanning_tree_of_all_n': False,
        'is_spanning_tree_of_covered': False,
        'connected_components': None,
        'degree_sequence': None,
    }

    if HAVE_NX:
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(sorted(sorted(e) for e in edges))
        # Restrict to covered nodes for the "spanning tree of covered" question
        Gc = G.subgraph(used).copy() if used else G
        result['connected_components_all'] = nx.number_connected_components(G)
        result['connected_components_covered'] = (
            nx.number_connected_components(Gc) if used else 0
        )
        result['is_spanning_tree_of_all_n'] = (
            nx.is_tree(G) and G.number_of_nodes() == n
        )
        result['is_spanning_tree_of_covered'] = (
            nx.is_tree(Gc) and Gc.number_of_nodes() == len(used) if used else False
        )
        result['degree_sequence_all'] = sorted(dict(G.degree()).values(), reverse=True)
    else:
        # Simple union-find fallback
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        n_edges_used = 0
        for e in edges:
            a, b = tuple(e)
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
                n_edges_used += 1
        components = len({find(i) for i in range(n)})
        result['connected_components_all'] = components
        # Spanning tree of all n iff #edges = n-1 and connected
        result['is_spanning_tree_of_all_n'] = (
            len(edges) == n - 1 and components == 1
        )
    return result


def weight4_support_summary(T: list[tuple[int, ...]], n: int) -> dict:
    """For each weight-4 support in T, how many balanced vectors on it are present?

    The "1 per support" pattern is a strong matroid-like signature.
    """
    w4 = [t for t in T if weight(t) == 4]
    by_supp: dict[tuple[int, ...], list[tuple[int, ...]]] = defaultdict(list)
    for t in w4:
        by_supp[support(t)].append(t)
    per_supp_dist = Counter(len(v) for v in by_supp.values())
    return {
        'n_vectors': len(w4),
        'n_supports': len(by_supp),
        'total_possible_supports': comb(n, 4) if n >= 4 else 0,
        'per_support_distribution': dict(sorted(per_supp_dist.items())),
        'is_one_per_support': (per_supp_dist == Counter({1: len(by_supp)})) if by_supp else True,
    }


def weight_layer_summary(T: list[tuple[int, ...]], w: int, n: int) -> dict:
    """Generic analysis for weight-w slice."""
    wv = [t for t in T if weight(t) == w]
    by_supp: dict[tuple[int, ...], list[tuple[int, ...]]] = defaultdict(list)
    for t in wv:
        by_supp[support(t)].append(t)
    per_supp_dist = Counter(len(v) for v in by_supp.values())
    return {
        'weight': w,
        'n_vectors': len(wv),
        'n_supports': len(by_supp),
        'total_possible_supports': comb(n, w) if w <= n else 0,
        'balanced_vecs_per_support_max': (comb(w, w // 2) // 2) if w >= 2 else 0,
        'per_support_distribution': dict(sorted(per_supp_dist.items())),
    }


def structural_summary(T: list[tuple[int, ...]], name: str = "T") -> dict:
    """Full structural summary of a single set of balanced ternary vectors."""
    ok, n = validate(T, name)
    result = {
        'name': name,
        'valid': ok,
        'n': n,
        'size': len(T),
        'weight_distribution': dict(sorted(weight_distribution(T).items())),
        'columns_used': sorted(used_columns(T, n)),
        'n_columns_used': len(used_columns(T, n)),
        'q_rank': q_rank(T),
        'f2_support_rank': f2_support_rank(T),
    }
    if n > 0:
        # Reference sizes
        b_size = sum(comb(n, 2*k) * comb(2*k, k) // 2
                     for k in range(1, n//2 + 1))
        result['B_size_reference'] = b_size
        result['fraction_of_B'] = len(T) / b_size if b_size > 0 else 0
        result['n_supports_possible'] = sum(
            comb(n, 2*k) for k in range(1, n//2 + 1)
        )

        # Layer-by-layer
        result['weight2'] = weight2_graph_summary(T, n)
        result['weight4'] = weight4_support_summary(T, n)
        # All present weights
        weights_present = sorted(weight_distribution(T).keys())
        result['layers'] = {
            w: weight_layer_summary(T, w, n) for w in weights_present
        }
    return result


# =============================================================================
# Multi-file comparison
# =============================================================================

def compare_pair(T1: list[tuple[int, ...]], T2: list[tuple[int, ...]],
                 name1: str, name2: str) -> dict:
    """Structural comparison of two cores."""
    S1, S2 = set(T1), set(T2)
    both = S1 & S2
    only1 = S1 - S2
    only2 = S2 - S1
    result = {
        'name1': name1, 'name2': name2,
        'size1': len(T1), 'size2': len(T2),
        'intersection': len(both),
        'unique_to_1': len(only1),
        'unique_to_2': len(only2),
        'symmetric_difference': len(only1) + len(only2),
        'jaccard': len(both) / len(S1 | S2) if (S1 or S2) else 1.0,
    }
    # By weight
    result['unique_to_1_by_weight'] = dict(sorted(Counter(weight(t) for t in only1).items()))
    result['unique_to_2_by_weight'] = dict(sorted(Counter(weight(t) for t in only2).items()))
    result['intersection_by_weight'] = dict(sorted(Counter(weight(t) for t in both).items()))
    return result


def multi_intersection(Ts: list[list[tuple[int, ...]]],
                       names: list[str]) -> dict:
    """Intersection across multiple cores — the persistent 'hard core'."""
    if not Ts:
        return {}
    sets = [set(T) for T in Ts]
    inter = sets[0].copy()
    for s in sets[1:]:
        inter &= s
    union = set()
    for s in sets:
        union |= s

    result = {
        'n_cores': len(Ts),
        'core_sizes': [len(T) for T in Ts],
        'names': names,
        'intersection_size': len(inter),
        'intersection_by_weight': dict(sorted(Counter(weight(t) for t in inter).items())),
        'union_size': len(union),
        'union_by_weight': dict(sorted(Counter(weight(t) for t in union).items())),
    }

    # Persistent weight-4 signed vectors: same signed pairing in ALL cores
    # For each weight-4 support in ALL cores, check whether the chosen pairing is unique
    w4_maps = []
    for T in Ts:
        m = {}
        for t in T:
            if weight(t) == 4:
                s = support(t)
                if s in m:
                    m[s].append(t)
                else:
                    m[s] = [t]
        w4_maps.append(m)
    # Supports in ALL cores at weight 4
    common_w4_supports = set(w4_maps[0].keys())
    for m in w4_maps[1:]:
        common_w4_supports &= set(m.keys())

    if common_w4_supports:
        consistent_signed = 0
        inconsistent = 0
        for s in common_w4_supports:
            vecs_across = set()
            for m in w4_maps:
                for v in m[s]:
                    vecs_across.add(v)
            if len(vecs_across) == 1:
                consistent_signed += 1
            else:
                inconsistent += 1
        result['common_weight4_supports'] = len(common_w4_supports)
        result['consistent_signed_pairing'] = consistent_signed
        result['inconsistent_signed_pairing'] = inconsistent

    return result


# =============================================================================
# Printing / reporting
# =============================================================================

def print_hr(char='=', width=70):
    print(char * width)


def print_summary(summary: dict, verbose: bool = False,
                  focus_weight: int | None = None):
    """Print a structural summary."""
    print_hr()
    print(f"SUMMARY: {summary['name']}")
    print_hr()
    print(f"  Size |T|:              {summary['size']}")
    print(f"  Dimension n:           {summary['n']}")
    if not summary['valid']:
        print("  ** VALIDATION FAILED — analysis may be incorrect **")

    n = summary['n']
    if n == 0:
        return

    print(f"  |B(n)| reference:      {summary.get('B_size_reference', '?')}"
          f"   fraction: {summary.get('fraction_of_B', 0):.3f}")
    print(f"  Columns used:          {summary['n_columns_used']}/{n} "
          f"{summary['columns_used']}")
    print(f"  Q-rank (max n-1 = {n-1}): {summary['q_rank']}"
          + ("  ← SATURATES" if summary['q_rank'] == n-1 else ""))
    print(f"  F_2 support rank:      {summary['f2_support_rank']}"
          + ("  ← SATURATES" if summary['f2_support_rank'] == n-1 else ""))
    print()
    print(f"  Weight distribution:   {summary['weight_distribution']}")

    # Weight-2 structure
    if 'weight2' in summary and summary['weight2']['n_vectors'] > 0:
        w2 = summary['weight2']
        print()
        print(f"  Weight-2 layer:")
        print(f"    # vectors: {w2['n_vectors']}, distinct edges: {w2['n_distinct_edges']}")
        print(f"    Columns covered: {w2['n_columns_covered']}/{n}")
        if HAVE_NX:
            print(f"    Spanning tree of all n columns? {w2['is_spanning_tree_of_all_n']}")
            print(f"    Spanning tree of covered? {w2.get('is_spanning_tree_of_covered', '?')}")
            print(f"    Connected components: {w2.get('connected_components_all', '?')}")
        else:
            print(f"    Spanning tree of all n? {w2['is_spanning_tree_of_all_n']}")
            print(f"    Connected components: {w2.get('connected_components_all', '?')}")
            print(f"    (Install networkx for more detailed graph analysis)")
        if verbose:
            print(f"    Edges: {w2['edges']}")

    # Weight-4 structure
    if 'weight4' in summary and summary['weight4']['n_vectors'] > 0:
        w4 = summary['weight4']
        print()
        print(f"  Weight-4 layer:")
        print(f"    # vectors: {w4['n_vectors']} on {w4['n_supports']} distinct supports "
              f"(of {w4['total_possible_supports']} possible)")
        print(f"    Vectors per support: {w4['per_support_distribution']}")
        print(f"    'One-per-support' pattern: {w4['is_one_per_support']}")

    # Other weight layers
    if 'layers' in summary:
        other_weights = [w for w in summary['layers'] if w not in (2, 4)]
        for w in other_weights:
            layer = summary['layers'][w]
            print()
            print(f"  Weight-{w} layer:")
            print(f"    # vectors: {layer['n_vectors']} on {layer['n_supports']} distinct supports "
                  f"(of {layer['total_possible_supports']} possible)")
            print(f"    Max balanced vecs per support: {layer['balanced_vecs_per_support_max']}")
            print(f"    Vectors per support: {layer['per_support_distribution']}")

    if focus_weight is not None:
        print()
        print(f"  --- Focus on weight {focus_weight} ---")
        if focus_weight in summary.get('layers', {}):
            layer = summary['layers'][focus_weight]
            print(f"    Details: {layer}")


def print_comparison(comp: dict):
    print_hr('-')
    print(f"PAIRWISE: {comp['name1']}  vs  {comp['name2']}")
    print_hr('-')
    print(f"  Sizes: {comp['size1']}  vs  {comp['size2']}")
    print(f"  Intersection: {comp['intersection']}")
    print(f"  Unique to {comp['name1']}: {comp['unique_to_1']}")
    print(f"  Unique to {comp['name2']}: {comp['unique_to_2']}")
    print(f"  Jaccard similarity: {comp['jaccard']:.3f}")
    print(f"  Intersection by weight:   {comp['intersection_by_weight']}")
    print(f"  Unique to {comp['name1']} by weight: {comp['unique_to_1_by_weight']}")
    print(f"  Unique to {comp['name2']} by weight: {comp['unique_to_2_by_weight']}")


def print_multi_intersection(mi: dict):
    print_hr()
    print(f"MULTI-CORE INTERSECTION ('hard core' across {mi['n_cores']} cores)")
    print_hr()
    print(f"  Cores: {mi['names']}")
    print(f"  Sizes: {mi['core_sizes']}")
    print()
    print(f"  Intersection (in ALL cores): {mi['intersection_size']} vectors")
    print(f"    By weight: {mi['intersection_by_weight']}")
    print()
    print(f"  Union (in AT LEAST ONE core): {mi['union_size']} vectors")
    print(f"    By weight: {mi['union_by_weight']}")
    if 'common_weight4_supports' in mi:
        print()
        print(f"  Weight-4 support intersection:")
        print(f"    Supports common to ALL cores: {mi['common_weight4_supports']}")
        print(f"    With CONSISTENT signed pairing (matroid coloop signature): "
              f"{mi['consistent_signed_pairing']}")
        print(f"    With inconsistent pairing across cores: "
              f"{mi['inconsistent_signed_pairing']}")


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('files', nargs='+', help='One or more core files to analyze')
    ap.add_argument('--verbose', '-v', action='store_true',
                    help='Print more detail (all weight-2 edges, etc.)')
    ap.add_argument('--focus-weight', type=int, default=None,
                    help='Extra detail for a specific weight class')
    ap.add_argument('--intersection-only', action='store_true',
                    help='Skip individual per-file summaries; show multi-intersection only')
    ap.add_argument('--pairwise', action='store_true',
                    help='Also print all pairwise comparisons (can be verbose)')
    args = ap.parse_args()

    cores = []
    names = []
    for path in args.files:
        try:
            T = parse_file(path)
        except FileNotFoundError:
            print(f"ERROR: {path} not found", file=sys.stderr)
            sys.exit(1)
        # Short display name
        import os
        name = os.path.basename(path)
        cores.append(T)
        names.append(name)

    # Individual summaries
    if not args.intersection_only:
        for T, name in zip(cores, names):
            summary = structural_summary(T, name)
            print_summary(summary, verbose=args.verbose,
                          focus_weight=args.focus_weight)
            print()

    # Pairwise comparisons (optional; only useful for ≤ 4-ish files)
    if args.pairwise or (len(cores) > 1 and args.intersection_only):
        if args.pairwise:
            for i in range(len(cores)):
                for j in range(i + 1, len(cores)):
                    print_comparison(compare_pair(cores[i], cores[j],
                                                  names[i], names[j]))
            print()

    # Multi-way intersection
    if len(cores) > 1:
        mi = multi_intersection(cores, names)
        print_multi_intersection(mi)


if __name__ == '__main__':
    main()
