import sympy
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import SymmetricGroup
from collections import Counter

def get_cycle_profile_via_tree(group, max_len):
    """Traverses the Schreier-Sims coset tree via backtracking to count cycle profiles."""
    transversals = group.basic_transversals
    profile = Counter()

    def get_perm_cycle_tuple(perm):
        # Create a list of zeros to hold fixed lengths up to max_len
        counts = [0] * (max_len + 1)
        for length, count in perm.cycle_structure.items():
            if length  k) & 1)
        bits.append(0)  # Canonical representative always ends in 0
        return bits

    def to_class_idx(vec):
        if vec[-1] == 1:  # If the last bit is 1, flip the whole vector
            vec = [1 - b for b in vec]
        idx = 0
        for k in range(m-1):
            idx |= (vec[k] << k)
        return idx

    # 3. Build Generators for the Codomain Group H' acting on the 2**(m-1) classes
    generators_H = []
    
    if m > 1:
        # Row Swap (0 1)
        perm_swap = []
        for i in range(num_classes):
            v = to_vector(i)
            v[0], v[1] = v[1], v[0]
            perm_swap.append(to_class_idx(v))
        generators_H.append(Permutation(perm_swap))
        
        # Row Rotation Cycle (0 1 ... m-1)
        perm_cycle = []
        for i in range(num_classes):
            v = to_vector(i)
            v = [v[-1]] + v[:-1]  # Rotate elements right
            perm_cycle.append(to_class_idx(v))
        generators_H.append(Permutation(perm_cycle))
        
    # Independent row complements (Flipping row r globally across all columns)
    for r in range(m):
        perm_flip = []
        for i in range(num_classes):
            v = to_vector(i)
            v[r] = 1 - v[r]
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
                        inner_sum += d * h_cycles[d - 1]
                
                term_product *= (inner_sum ** c_k_g)
                
            total_sum += g_count * h_count * term_product

    return total_sum // (G.order() * H.order())

# --- Execution ---
#print(f"Orbits for 3x3 Matrix: {count_matrix_orbits(3, 3)}")
#print(f"Orbits for 4x4 Matrix: {count_matrix_orbits(4, 4)}")
