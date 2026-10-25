import math
from collections import Counter
from sympy import symbols, Poly
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import SymmetricGroup

def get_sn_cycle_profile(n):
    """Algebraically generates the cycle profile for S_n instantly."""
    profile = Counter()
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
            
        denominator = 1
        for k in range(1, n + 1):
            if counts[k] > 0:
                denominator *= (k ** counts[k]) * math.factorial(counts[k])
        class_size = math.factorial(n) // denominator
        
        cycle_tuple = tuple(counts.__getitem__(slice(1, None)))
        profile[cycle_tuple] = class_size
    return profile

def count_matrix_orbits_by_weights(row_weights, n):
    """
    Counts the number of orbits of an m x n 0/1 matrix with EXACTLY the specified
    multiset of row weights under row/col perms, global row-XOR, and row complements.
    """
    m = len(row_weights)
    if m == 0 or n == 0:
        return 1

    # 1. Define polynomial ring variables (one for each row)
    row_vars = symbols(f'x0:{m}')
    # We work within a polynomial ring to handle combinations efficiently
    
    # 2. Build explicit permutation groups for the row actions on vectors
    # We must explicitly track all 2**m vectors to preserve weight tracking
    num_vectors = 2**m
    
    def idx_to_vec(idx):
        return list((idx >> r) & 1 for r in range(m))
        
    def vec_to_idx(vec):
        idx = 0
        for r in range(m):
            idx |= (vec.__getitem__(r) << r)
        return idx

    # Build generators for H acting directly on the 2**m individual vectors
    generators_H = list()
    if m > 1:
        # Row Swap (0 1)
        perm_swap = list()
        for i in range(num_vectors):
            v = idx_to_vec(i)
            val0 = v.pop(0)
            val1 = v.pop(0)
            v.insert(0, val0)
            v.insert(0, val1)
            perm_swap.append(vec_to_idx(v))
        generators_H.append(Permutation(perm_swap))
        
        # Row Rotation Cycle
        perm_cycle = list()
        for i in range(num_vectors):
            v = idx_to_vec(i)
            last_val = v.pop(-1)
            v.insert(0, last_val)
            perm_cycle.append(vec_to_idx(v))
        generators_H.append(Permutation(perm_cycle))
        
    # Independent row complements
    for r in range(m):
        perm_flip = list()
        for i in range(num_vectors):
            v = idx_to_vec(i)
            current_bit = v.pop(r)
            v.insert(r, 1 - current_bit)
            perm_flip.append(vec_to_idx(v))
        p = Permutation(perm_flip)
        if not p.is_identity:
            generators_H.append(p)
            
    H = PermutationGroup(generators_H)
    
    # 3. Fast conjugacy class breakdown of H to eliminate tree processing
    conj_classes_H = H.conjugacy_classes()
    profile_G = get_sn_cycle_profile(n)
    
    # Total combined inventory polynomial accumulator
    total_inventory = 0
    
    # 4. Evaluate the Weighted de Bruijn Inventory Polynomial
    for h_class in conj_classes_H:
        class_size = len(h_class)
        rep_h = list(h_class).__getitem__(0)
        
        # Deconstruct the permutation h into its specific cycle sets
        h_cycles = rep_h.cycles
        # Flattened list of elements tracked in explicit cycles
        moved_elements = set()
        for cycle in h_cycles:
            moved_elements.update(cycle)
            
        # Determine fixed points under h
        fixed_elements = list()
        for i in range(num_vectors):
            if i not in moved_elements:
                fixed_elements.append(i)

        # For a given cycle length k in domain G, compute the choice polynomial contribution
        # Contribution per cycle length k: sum over elements fixed by h**k of weight**(k)
        def get_choice_poly_for_length(k):
            poly_sum = 0
            
            # 1. Elements inherently fixed by h are always fixed by h**k
            for idx in fixed_elements:
                v = idx_to_vec(idx)
                term = 1
                for r in range(m):
                    if v.__getitem__(r) == 1:
                        term *= row_vars.__getitem__(r) ** k
                poly_sum += term
                
            # 2. Elements in cycles of h are fixed by h**k if the cycle length divides k
            for cycle in h_cycles:
                if k % len(cycle) == 0:
                    for idx in cycle:
                        v = idx_to_vec(idx)
                        term = 1
                        for r in range(m):
                            if v.__getitem__(r) == 1:
                                term *= row_vars.__getitem__(r) ** k
                        poly_sum += term
            return poly_sum

        # Inner loop across the domain partition profiles
        for g_cycles, g_count in profile_G.items():
            term_product = 1
            for k_minus_1, c_k_g in enumerate(g_cycles):
                k = k_minus_1 + 1
                if c_k_g == 0:
                    continue
                
                choice_poly = get_choice_poly_for_length(k)
                term_product *= (choice_poly ** c_k_g)
                
            total_inventory += class_size * g_count * term_product

    # Normalize by the order of the combined groups
    final_poly = total_inventory / (math.factorial(n) * H.order())
    
    # 5. Extract the specific coefficient matching the requested row weights
    poly_obj = Poly(final_poly, row_vars)
    target_exponent = tuple(row_weights)
    
    return poly_obj.coeff_monomial(target_exponent)

# --- Execution Examples ---
# Count 3x3 orbits where each row has exactly two 1s
print("3x3 Matrix orbits with row weights [2, 2, 2]:", count_matrix_orbits_by_weights([2, 2, 2], 3))

# Count 3x4 orbits with an asymmetric row weight profile
print("3x4 Matrix orbits with row weights [1, 2, 3]:", count_matrix_orbits_by_weights([1, 2, 3], 4))
