import math
from collections import Counter
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

def get_h_profile_via_conjugacy(group, max_len):
    """
    Bypasses the manual Python tree loop entirely. Uses SymPy's native,
    highly optimized Schreier-Sims backend to extract conjugacy classes directly.
    """
    profile = Counter()
    
    # SymPy computes these using specialized orbit-stabilizer stabilizer chains instantly
    conj_classes = group.conjugacy_classes()
    
    for c in conj_classes:
        class_size = len(c)
        # Grab any single representative permutation from this class
        rep_perm = list(c).__getitem__(0)
        
        # Calculate cycle lengths up to max_len
        counts = list(0 for _ in range(max_len + 1))
        for length, count in rep_perm.cycle_structure.items():
            if length <= max_len:
                counts.pop(length)
                counts.insert(length, count)
                
        total_in_cycles = sum(l * c for l, c in enumerate(counts))
        fixed_points = rep_perm.size - total_in_cycles
        
        counts.pop(1)
        counts.insert(1, fixed_points)
        cycle_tuple = tuple(counts.__getitem__(slice(1, None)))
        
        # Accumulate the weighted size of the entire structural family
        profile[cycle_tuple] += class_size
        
    return profile

def count_matrix_orbits_ultra(m, n):
    """Calculates orbits using algebraic G partitions and native H conjugacy classes."""
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
        perm_swap = list()
        for i in range(num_classes):
            v = to_vector(i)
            val0 = v.pop(0)
            val1 = v.pop(0)
            v.insert(0, val0)
            v.insert(0, val1)
            perm_swap.append(to_class_idx(v))
        generators_H.append(Permutation(perm_swap))
        
        perm_cycle = list()
        for i in range(num_classes):
            v = to_vector(i)
            last_val = v.pop(-1)
            v.insert(0, last_val)
            perm_cycle.append(to_class_idx(v))
        generators_H.append(Permutation(perm_cycle))
        
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
    
    profile_G = get_sn_cycle_profile(n)
    profile_H = get_h_profile_via_conjugacy(H, n)
    
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
