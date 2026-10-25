from collections import Counter
from sympy.combinatorics import PermutationGroup, Permutation
from sympy.combinatorics.named_groups import SymmetricGroup
from math import gcd

def debruijn_enumeration(group_G, group_H, domain_size):
    """
    Implements de Bruijn's generalization of Pólya's Theorem.
    Counts the number of orbits of functions X -> Y under the action of G on X and H on Y.
    """
    # Helper to extract cycle counts up to domain_size
    def get_cycle_counts(perm, max_len):
        counts = {i: 0 for i in range(1, max_len + 1)}
        # perm.cycle_structure returns {length: count}
        for length, count in perm.cycle_structure.items():
            if length <= max_len:
                counts[length] = count
        # Uncounted elements are fixed points (cycles of length 1)
        total_in_cycles = sum(l * c for l, c in counts.items())
        counts[1] += (perm.size - total_in_cycles)
        return counts

    total_sum = 0
    
    # Pre-calculate cycle structures for group H
    h_cycles = [get_cycle_counts(h, domain_size) for h in group_H.elements]
    
    for g in group_G.elements:
        g_counts = get_cycle_counts(g, domain_size)
        
        for h_counts in h_cycles:
            term_product = 1
            # Iterate through cycle lengths k in the domain
            for k, c_k_g in g_counts.items():
                if c_k_g == 0:
                    continue
                
                # Inner sum over divisors: sum_{d | k} d * c_d(h)
                inner_sum = 0
                for d in range(1, k + 1):
                    if k % d == 0 and d in h_counts:
                        inner_sum += d * h_counts[d]
                
                term_product *= (inner_sum ** c_k_g)
                
            total_sum += term_product

    # Divide by the product of the group orders
    return total_sum // (group_G.order() * group_H.order())

# Example usage:
# G = Cyclic group of size 3 acting on 3 items
# H = Symmetric group of size 2 (interchanging 2 colors)
def practical_debruijn_enumeration(group_G, group_H, domain_size):
    """
    An optimized implementation of de Bruijn's Theorem using Conjugacy Classes.
    The underlying classes are found efficiently via SymPy's Schreier-Sims implementation.
    """
    def get_cycle_counts(perm, max_len):
        counts = {i: 0 for i in range(1, max_len + 1)}
        for length, count in perm.cycle_structure.items():
            if length <= max_len:
                counts[length] = count
        total_in_cycles = sum(l * c for l, c in counts.items())
        counts[1] += (perm.size - total_in_cycles)
        return counts

    total_sum = 0
    
    # 1. Retrieve conjugacy classes (computed using Schreier-Sims structural data)
    classes_G = group_G.conjugacy_classes()
    classes_H = group_H.conjugacy_classes()
    
    # 2. Extract a single representative and class size for each class in H
    h_class_data = []
    for c_H in classes_H:
        rep_h = list(c_H)[0]  # Grab a representative element
        h_class_data.append({
            'size': len(c_H),
            'counts': get_cycle_counts(rep_h, domain_size)
        })
    
    # 3. Double-sum over the classes instead of individual elements
    for c_G in classes_G:
        rep_g = list(c_G)[0]
        g_counts = get_cycle_counts(rep_g, domain_size)
        size_g = len(c_G)
        
        for h_data in h_class_data:
            size_h = h_data['size']
            h_counts = h_data['counts']
            
            term_product = 1
            for k, c_k_g in g_counts.items():
                if c_k_g == 0:
                    continue
                
                # Inner sum over divisors: sum_{d | k} d * c_d(h)
                inner_sum = 0
                for d in range(1, k + 1):
                    if k % d == 0 and d in h_counts:
                        inner_sum += d * h_counts[d]
                
                term_product *= (inner_sum ** c_k_g)
            
            # Weight the result by the sizes of BOTH conjugacy classes
            total_sum += size_g * size_h * term_product

    return total_sum // (group_G.order() * group_H.order())

# This setup handles massively larger groups instantly
# Example: Rubik's cube subgroups or large symmetric extensions

def get_cycle_profile_via_tree(group, max_len):
    """
    Traverses the Schreier-Sims coset tree using backtracking
    to build a profile of cycle counts across the entire group.
    """
    # SymPy computes the Schreier-Sims structure under the hood.
    # basic_transversals is a list of lists containing the coset representatives.
    transversals = group.basic_transversals
    profile = Counter()

    def get_perm_cycle_tuple(perm):
        # Creates a hashable tuple representing the cycle structure
        counts = [0] * (max_len + 1)
        for length, count in perm.cycle_structure.items():
            if length <= max_len:
                counts[length] = count
        # Handle remaining fixed points (cycles of length 1)
        total_in_cycles = sum(l * c for l, c in enumerate(counts))
        counts[1] += (perm.size - total_in_cycles)
        return tuple(counts[1:])

    def backtrack(level, current_perm):
        # Base case: reached a leaf node (a unique group element)
        if level == len(transversals):
            cycle_tuple = get_perm_cycle_tuple(current_perm)
            profile[cycle_tuple] += 1
            return

        # Branching: multiply by each coset representative at this level
        for u in transversals[level].values():
            # Backtrack down the tree
            backtrack(level + 1, current_perm * u)

    # Start the tree traversal from the root with the identity permutation
    identity = Permutation([], size=group.degree)
    backtrack(0, identity)
    
    return profile

def tree_debruijn_enumeration(group_G, group_H, domain_size):
    """
    De Bruijn enumeration using independent Schreier-Sims backtracking trees.
    """
    # 1. Generate cycle profiles via independent tree searches
    profile_G = get_cycle_profile_via_tree(group_G, domain_size)
    profile_H = get_cycle_profile_via_tree(group_H, domain_size)
    
    total_sum = 0

    # 2. Combine the profiles via de Bruijn arithmetic
    for g_cycles, g_count in profile_G.items():
        for h_cycles, h_count in profile_H.items():
            term_product = 1
            
            # g_cycles is a 0-indexed tuple where index k-1 represents cycle length k
            for k_minus_1, c_k_g in enumerate(g_cycles):
                k = k_minus_1 + 1
                if c_k_g == 0:
                    continue
                
                # Inner sum over divisors: sum_{d | k} d * c_d(h)
                inner_sum = 0
                for d in range(1, k + 1):
                    if k % d == 0 and d <= len(h_cycles):
                        c_d_h = h_cycles[d - 1]
                        inner_sum += d * c_d_h
                
                term_product *= (inner_sum ** c_k_g)
            
            total_sum += g_count * h_count * term_product

    return total_sum // (group_G.order() * group_H.order())


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
