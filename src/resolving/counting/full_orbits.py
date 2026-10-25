import sympy
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import SymmetricGroup
from collections import Counter

def get_cycle_profile_via_tree(group, max_len):
    """Traverses the Schreier-Sims coset tree via backtracking to count cycle profiles."""
    transversals = group.basic_transversals
    profile = Counter()

    def get_perm_cycle_tuple(perm):
        # Create a list of zeros up to max_len safely
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
        
        # Return a hashable, 1-indexed tuple of cycle counts
        return tuple(counts.__getitem__(slice(1, None)))

    def to_class_idx(vec):
        if vec.__getitem__(-1) == 1:  # If last bit is 1, complement vector
            vec = list(1 - b for b in vec)
        idx = 0
        for k in range(m - 1):
            idx |= (vec.__getitem__(k) << k)
        return idx

    # 3. Build Generators for the Codomain Group H' acting on the 2**(m-1) classes
    generators_H = list()
    
    if m > 1:
        # Row Swap (0 1) via safe list extraction
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
            v.insert(0, last_val)  # Rotate elements right
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
    
    # 4. Independent Schreier-Sims Tree Traversals
    profile_G = get_cycle_profile_via_tree(G, n)
    profile_H = get_cycle_profile_via_tree(H, n)
    
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

    return total_sum // (G.order() * H.order())

def count_matrix_orbits(m, n):
    """
    Calculates the number of orbits of an m x n 0/1 matrix under:
    Row perms, Col perms, Global row-XOR, and Independent row complements.
    """
    if m == 0 or n == 0:
        return 1

    # 1. Domain Group G = S_n (Column Permutations)
    G = SymmetricGroup(n)
    
    # 2. Setup Codomain Y' (m-bit vectors modulo complementation)
    num_classes = 2**(m-1)
    
    def to_vector(idx):
        bits = list()
        for k in range(m - 1):
            bits.append((idx >> k) & 1)
        bits.append(0)  # Canonical representative ends in 0
        return bits

    def to_class_idx(vec):
        if vec.__getitem__(-1) == 1:  # If last bit is 1, complement vector
            vec = list(1 - b for b in vec)
        idx = 0
        for k in range(m - 1):
            idx |= (vec.__getitem__(k) << k)
        return idx

    # 3. Build Generators for the Codomain Group H' acting on the 2**(m-1) classes
    generators_H = list()
    
    if m > 1:
        # Row Swap (0 1) via safe list extraction
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
            v.insert(0, last_val)  # Rotate elements right
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
    
    # 4. Independent Schreier-Sims Tree Traversals
    profile_G = get_cycle_profile_via_tree(G, n)
    profile_H = get_cycle_profile_via_tree(H, n)
    
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

    return total_sum // (G.order() * H.order())

# --- Run calculations ---
#print("Orbits for 3x3 Matrix:", count_matrix_orbits(3, 3))
#print("Orbits for 4x4 Matrix:", count_matrix_orbits(4, 4))
