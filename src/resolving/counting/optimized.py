from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import SymmetricGroup
from collections import Counter
import math

def get_sn_cycle_profile(n):
    """
    Algebraically generates the cycle profile for S_n without a tree search.
    Returns a Counter mapping cycle tuples to their exact element counts.
    """
    profile = Counter()
    
    # Recursive partition generator to find all cycle structures of S_n
    def get_partitions(n, max_val):
        if n == 0:
            yield []
            return
        for i in range(min(n, max_val), 0, -1):
            for p in get_partitions(n - i, i):
                yield [i] + p

    for part in get_partitions(n, n):
        counts = list(0 for _ in range(n + 1))
        for x in part:
            counts[x] += 1
            
        # Cauchy's formula for the size of the conjugacy class
        denominator = 1
        for k in range(1, n + 1):
            if counts[k] > 0:
                denominator *= (k ** counts[k]) * math.factorial(counts[k])
        class_size = math.factorial(n) // denominator
        
        cycle_tuple = tuple(counts.__getitem__(slice(1, None)))
        profile[cycle_tuple] = class_size
        
    return profile

from collections import Counter
from sympy.combinatorics import Permutation

def get_h_profile_with_pruning(group, max_len):
    """
    Traverses the Schreier-Sims coset tree of H with subtree memoization/pruning.
    Returns a Counter tracking cycle tuples mapped to total element counts.
    """
    transversals = group.basic_transversals
    memo = dict()

    def get_perm_cycle_tuple(perm):
        counts = list()
        for _ in range(max_len + 1):
            counts.append(0)
            
        for length, count in perm.cycle_structure.items():
            if length <= max_len:
                counts.pop(length)
                counts.insert(length, count)
                
        total_in_cycles = sum(l * c for l, c in enumerate(counts))
        fixed_points = perm.size - total_in_cycles
        
        counts.pop(1)
        counts.insert(1, fixed_points)
        return tuple(counts.__getitem__(slice(1, None)))

    def backtrack(level, current_perm):
        if level == len(transversals):
            res_counter = Counter()
            res_counter[get_perm_cycle_tuple(current_perm)] = 1
            return res_counter
        
        # Subtree Pruning Cache Key: level + hashable array form of the path permutation
        state_key = (level, tuple(current_perm.array_form))
        if state_key in memo:
            return memo[state_key]
            
        subtree_profile = Counter()
        for u in transversals[level].values():
            subtree_profile.update(backtrack(level + 1, current_perm * u))
            
        memo[state_key] = subtree_profile
        return subtree_profile

    identity = Permutation([], size=group.degree)
    return backtrack(0, identity)
# def get_h_profile_with_pruning(group, max_len):
#     """
#     Traverses the Schreier-Sims tree of H with subtree memoization/pruning.
#     """
#     transversals = group.basic_transversals
#     profile = Counter()
    
#     # Memoization table: maps (level, current_perm_tuple) to subtree results
#     memo = {}

#     def get_perm_cycle_tuple(perm):
#         # Create a list of zeros up to max_len safely
#         counts = list()
#         for _ in range(max_len + 1):
#             counts.append(0)
            
#         for length, count in perm.cycle_structure.items():
#             if length <= max_len:
#                 counts.pop(length)
#                 counts.insert(length, count)
        
#         total_in_cycles = sum(l * c for l, c in enumerate(counts))
#         fixed_points = perm.size - total_in_cycles
        
#         counts.pop(1)
#         counts.insert(1, fixed_points)
        
#         # Return a hashable, 1-indexed tuple of cycle counts
#         return tuple(counts.__getitem__(slice(1, None)))
#     # def get_perm_cycle_tuple(perm):
#     #     counts = list(0 for _ in range(max_len + 1))
#     #     for length, count in perm.cycle_structure.items():
#     #         if length  k) & 1)
#     #     bits.append(0)
#     #     return bits

#     def to_class_idx(vec):
#         if vec.__getitem__(-1) == 1:
#             vec = list(1 - b for b in vec)
#         idx = 0
#         for k in range(m - 1):
#             idx |= (vec.__getitem__(k) << k)
#         return idx

#     # generators_H = list()
#     # if m > 1:
#     #     perm_swap = list()
#     #     for i in range(num_classes):
#     #         v = to_vector(i)
#     #         val0 = v.pop(0)
#     #         val1 = v.pop(0)
#     #         v.insert(0, val0)
#     #         v.insert(0, val1)
#     #         perm_swap.append(to_class_idx(v))
#     #     generators_H.append(Permutation(perm_swap))
        
#     #     perm_cycle = list()
#     #     for i in range(num_classes):
#     #         v = to_vector(i)
#     #         last_val = v.pop(-1)
#     #         v.insert(0, last_val)
#     #         perm_cycle.append(to_class_idx(v))
#     #     generators_H.append(Permutation(perm_cycle))
        
#     # for r in range(m):
#     #     perm_flip = list()
#     #     for i in range(num_classes):
#     #         v = to_vector(i)
#     #         current_bit = v.pop(r)
#     #         v.insert(r, 1 - current_bit)
#     #         perm_flip.append(to_class_idx(v))
#     #     p = Permutation(perm_flip)
#     #     if not p.is_identity:
#     #         generators_H.append(p)
            
#     H = PermutationGroup(generators_H)
    
#     # 4. Optimized Profile Generation
#     profile_G = get_sn_cycle_profile(n)            # Instant algebraic evaluation
#     profile_H = get_h_profile_with_pruning(H, n)   # Memoized tree search
    
#     # 5. Evaluate de Bruijn Arithmetic
#     total_sum = 0
#     for g_cycles, g_count in profile_G.items():
#         for h_cycles, h_count in profile_H.items():
#             term_product = 1
#             for k_minus_1, c_k_g in enumerate(g_cycles):
#                 k = k_minus_1 + 1
#                 if c_k_g == 0:
#                     continue
                
#                 inner_sum = 0
#                 for d in range(1, k + 1):
#                     if k % d == 0 and d <= len(h_cycles):
#                         inner_sum += d * h_cycles.__getitem__(d - 1)
                
#                 term_product *= (inner_sum ** c_k_g)
                
#             total_sum += g_count * h_count * term_product

#     return total_sum // (math.factorial(n) * H.order())

def count_matrix_orbits_fast(m, n):
    """
    Calculates the matrix orbits instantly using algebraic S_n counts 
    and a memoized, pruned tree search for the codomain group.
    """
    if m == 0 or n == 0:
        return 1

    num_classes = 2**(m-1)
    
    def to_vector(idx):
        bits = list()
        for k in range(m - 1):
            bits.append((idx >> k) & 1)
        bits.append(0)
        return bits

    def to_class_idx(vec):
        if vec.__getitem__(-1) == 1:
            vec = list(1 - b for b in vec)
        idx = 0
        for k in range(m - 1):
            idx |= (vec.__getitem__(k) << k)
        return idx

    generators_H = list()
    if m > 1:
        # Row Swap (0 1) via safe list operations
        perm_swap = list()
        for i in range(num_classes):
            v = to_vector(i)
            val0 = v.pop(0)
            val1 = v.pop(0)
            v.insert(0, val0)
            v.insert(0, val1)
            perm_swap.append(to_class_idx(v))
        generators_H.append(Permutation(perm_swap))
        
        # Row Rotation Cycle (0 1 ... m-1)
        perm_cycle = list()
        for i in range(num_classes):
            v = to_vector(i)
            last_val = v.pop(-1)
            v.insert(0, last_val)
            perm_cycle.append(to_class_idx(v))
        generators_H.append(Permutation(perm_cycle))
        
    # Independent row complements (Flipping row r globally across all columns)
    for r in range(m):
        perm_flip = list()
        for i in range(num_classes):
            v = to_vector(i)
            current_bit = v.pop(r)
            v.insert(r, 1 - current_bit)
            perm_flip.append(to_class_idx(v))
        p = Permutation(perm_flip)
        if not p.is_identity:
            generators_H.append(p)
            
    H = PermutationGroup(generators_H)
    
    # 4. Fast Profile Generation
    profile_G = get_sn_cycle_profile(n)            # Instant algebraic evaluation
    profile_H = get_h_profile_with_pruning(H, n)   # Memoized tree search
    
    # 5. Evaluate de Bruijn Arithmetic
    total_sum = 0
    for g_cycles, g_count in profile_G.items():
        for h_cycles, h_count in profile_H.items():
            term_product = 1
            for k_minus_1, c_k_g in enumerate(g_cycles):
                k = k_minus_1 + 1
                if c_k_g == 0:
                    continue
                
                inner_sum = 0
                for d in range(1, k + 1):
                    if k % d == 0 and d <= len(h_cycles):
                        inner_sum += d * h_cycles.__getitem__(d - 1)
                
                term_product *= (inner_sum ** c_k_g)
                
            total_sum += g_count * h_count * term_product

    return total_sum // (math.factorial(n) * H.order())
# --- Execution ---
def xorbits(m, n):
    print(f"Orbits for {m}x{n} Matrix: {count_matrix_orbits_fast(m, n)}")

xorbits(3, 3)
xorbits(4, 4)
xorbits(6, 9)
xorbits(7, 11)

