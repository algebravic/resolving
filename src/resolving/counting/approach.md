To solve this matrix problem efficiently using de Bruijn's Theorem, you can apply an elegant mathematical shortcut that drastically simplifies the group action and shrinks the state space.
Instead of viewing the $m \times n$ matrix as a flat field of entries, view the matrix column-by-column. The matrix becomes a function mapping each column index to an $m$-bit vector: $f: N \to \mathbb{F}_2^m$, where $N = \{1, \dots, n\}$.
## The Mathematical Reduction

   1. The Local Group Action ($L$): Operation (3) (XOR an arbitrary row into every row) means that for each column independently, you can choose to flip all of its bits simultaneously. This corresponds to adding the all-ones vector $\mathbf{1}_m$ to that column's vector.
   2. The Global Group Action ($H$): Operations (1) and (4) (permute rows and independent row complements) mean you can permute the rows of the vector space and flip individual coordinates globally across all columns. This group is $H = S_m \ltimes \mathbb{F}_2^m$.
   3. The Simplification: Because the all-ones vector $\mathbf{1}_m$ is perfectly invariant under row permutations and commutes with row-wise flipping, the local column-flipping group $L = \{0, \mathbf{1}_m\}$ commutes perfectly with the global group $H$.

This allows you to completely quotient out the local action. The problem reduces to a standard de Bruijn formulation:

* 
* Domain ($X$): The $n$ columns.
* Domain Group ($G$): $S_n$ (column permutations).
* Codomain ($Y'$): The space of $m$-bit column vectors modulo complementation ($\mathbb{F}_2^m / \{0, \mathbf{1}_m\}$), which has a size of exactly $2^{m-1}$.
* Codomain Group ($H'$): The action of $S_m \ltimes \mathbb{F}_2^m$ mapping onto these $2^{m-1}$ classes.
* 

------------------------------
## Backtracking Tree Python Implementation
The following complete program implements this reduction. It generates the permutation group $H'$ acting on the $2^{m-1}$ classes, extracts its Schreier-Sims basic transversals via SymPy, and runs the backtracking tree search to get the answer.

import sympyfrom sympy.combinatorics import Permutation, PermutationGroupfrom sympy.combinatorics.named_groups import SymmetricGroupfrom collections import Counter
def get_cycle_profile_via_tree(group, max_len):
    """Traverses the Schreier-Sims coset tree via backtracking to count cycle profiles."""
    transversals = group.basic_transversals
    profile = Counter()

    def get_perm_cycle_tuple(perm):
        counts = [0] * (max_len + 1)
        for length, count in perm.cycle_structure.items():
            if length  k) & 1 for k in range(m-1)]
        bits.append(0) # Canonical representative has 0 at the end
        return bits

    def to_class_idx(vec):
        if vec[-1] == 1: # If last bit is 1, complement the entire vector
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
            v = [v[-1]] + v[:-1] # Shift right/rotate
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
print(f"Orbits for 3x3 Matrix: {count_matrix_orbits(3, 3)}")
print(f"Orbits for 4x4 Matrix: {count_matrix_orbits(4, 4)}")

## Performance Traits

* 
* For a $4 \times 4$ matrix, the raw search space of combinations is $2^{16} = 65,536$. Your group order is $\vert{}G\vert{} \times \vert{}H\vert{} = 24 \times 384 = 9,216$. This program executes instantly because the codomain group $H'$ gets condensed into an action on only $2^{4-1} = 8$ elements.
* The backtracking tree processes the groups independently, completely skipping any nested execution combinatorial explosion.
* 

Would you like to examine the cycle index polynomial properties of these specific matrix spaces, or run this code on asymmetric matrices (e.g., a $3 \times 10$ grid)?

import sympyfrom sympy.combinatorics import Permutation, PermutationGroupfrom sympy.combinatorics.named_groups import SymmetricGroup
def solve_matrix_orbits(m, n):
    # 1. Define G = S_n acting on columns
    G = SymmetricGroup(n)
    
    # 2. Define Y' = space of column vectors of length m modulo complement
    # We represent each class by its vector where the last bit is 0 (or just int < 2**(m-1))
    # Let's map each integer 0 <= i < 2**(m-1) to its full m-bit representation (with 0 at end)
    num_classes = 2**(m-1) if m > 0 else 1
    
    def to_vector(idx):
        # returns list of m bits
        bits = [(idx >> k) & 1 for k in range(m-1)]
        bits.append(0)
        return bits

    def to_class_idx(vec):
        if vec[-1] == 1:
            vec = [1 - b for b in vec]
        idx = 0
        for k in range(m-1):
            idx |= (vec[k] << k)
        return idx

    # We need to find the generators of H = S_m \ltimes \mathbb{F}_2^m acting on Y'
    generators_H = []
    
    # Row permutations (S_m) acting on Y'
    # S_m can be generated by a adjacent transposition (0 1) and a cycle (0 1 ... m-1)
    if m > 1:
        # Transposition (0 1)
        perm_list_1 = []
        for i in range(num_classes):
            v = to_vector(i)
            v[0], v[1] = v[1], v[0]
            perm_list_1.append(to_class_idx(v))
        generators_H.append(Permutation(perm_list_1))
        
        # Cycle (0 1 ... m-1)
        perm_list_2 = []
        for i in range(num_classes):
            v = to_vector(i)
            # shift left
            v = [v[-1]] + v[:-1]
            perm_list_2.append(to_class_idx(v))
        generators_H.append(Permutation(perm_list_2))
        
    # Independent row complements (translations by e_i)
    # Note that adding e_i is a translation
    for r in range(m):
        perm_list = []
        for i in range(num_classes):
            v = to_vector(i)
            v[r] = 1 - v[r]
            perm_list.append(to_class_idx(v))
        p = Permutation(perm_list)
        if not p.is_identity:
            generators_H.append(p)
            
    H = PermutationGroup(generators_H)
    
    # Use de Bruijn's theorem via conjugacy classes or the tree method
    # Let's write the conjugacy class method here for testing
    def get_cycle_counts(perm, max_len):
        counts = {i: 0 for i in range(1, max_len + 1)}
        for length, count in perm.cycle_structure.items():
            if length <= max_len:
                counts[length] = count
        total_in_cycles = sum(l * c for l, c in counts.items())
        counts[1] += (perm.size - total_in_cycles)
        return counts

    total_sum = 0
    classes_G = G.conjugacy_classes()
    classes_H = H.conjugacy_classes()
    
    h_class_data = []
    for c_H in classes_H:
        rep_h = list(c_H)[0]
        h_class_data.append({
            'size': len(c_H),
            'counts': get_cycle_counts(rep_h, n)
        })
        
    for c_G in classes_G:
        rep_g = list(c_G)[0]
        g_counts = get_cycle_counts(rep_g, n)
        size_g = len(c_G)
        
        for h_data in h_class_data:
            size_h = h_data['size']
            h_counts = h_data['counts']
            
            term_product = 1
            for k, c_k_g in g_counts.items():
                if c_k_g == 0:
                    continue
                inner_sum = 0
                for d in range(1, k + 1):
                    if k % d == 0 and d in h_counts:
                        inner_sum += d * h_counts[d]
                term_product *= (inner_sum ** c_k_g)
                
            total_sum += size_g * size_h * term_product

    return total_sum // (G.order() * H.order())

print("3x3:", solve_matrix_orbits(3, 3))
print("4x4:", solve_matrix_orbits(4, 4))


