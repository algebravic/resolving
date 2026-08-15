import numpy as np
import scipy.optimize
import time
import random
import itertools
from fractions import Fraction

# ==========================================
# 1. Math & Formatting Utilities
# ==========================================

def to_fractions(arr):
    # Use a high limit to capture exact rationals from floats
    return np.array([Fraction(x).limit_denominator(10**12) for x in arr], dtype=object)

def format_vec(v):
    return "[" + ", ".join(str(x) for x in v) + "]"

def format_poly_vars(exponents, vars_list=['a','b','c','d','e']):
    parts = []
    for i, exp in enumerate(exponents):
        if exp == 0: continue
        v = vars_list[i] if i < len(vars_list) else f"x_{i+1}"
        if exp == 1: parts.append(f"{v}")
        else: parts.append(f"{v}^{{{exp}}}")
    return " ".join(parts) if parts else "1"

def format_sym_sum(exponents):
    return f"\\sum_{{sym}} {format_poly_vars(exponents)}"

# ==========================================
# 2. Unbiased Test Case Generator
# ==========================================

def generate_uniform_case(n: int, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate Alpha: Random integers
    alpha_ints = np.sort(np.random.randint(0, 10, n))[::-1]
    alpha = to_fractions(alpha_ints)
    
    # Generate Beta: Random convex combination of permutations
    perms = list(set(itertools.permutations(alpha)))
    # Use a common denominator to ensure Beta is exactly representable and readable
    denom = 24 
    cuts = sorted([random.randint(0, denom) for _ in range(len(perms)-1)])
    weights_int = []
    prev = 0
    for cut in cuts:
        weights_int.append(cut - prev)
        prev = cut
    weights_int.append(denom - prev)
    
    beta = np.zeros(n, dtype=object)
    for p, w_int in zip(perms, weights_int):
        if w_int > 0:
            beta += np.array(p, dtype=object) * Fraction(w_int, denom)
            
    beta = np.sort(beta)[::-1]
    return alpha, beta

# ==========================================
# 3. Algorithms (Standardized Output)
# ==========================================

# --- A. HLP / Greedy ---
def hlp_greedy(alpha_in, beta_in):
    alpha = np.array(alpha_in, dtype=object)
    beta  = np.array(beta_in, dtype=object)
    n = len(alpha)
    steps = []
    
    # Limit iterations just in case, though it converges finitely
    for _ in range(n*n*10):
        if np.all(alpha == beta): break
        diff = alpha - beta
        
        candidates_i = [k for k, x in enumerate(diff) if x > 0]
        if not candidates_i: break
        i = candidates_i[0]
        candidates_j = [k for k, x in enumerate(diff[i+1:]) if x < 0]
        if not candidates_j: break
        j = candidates_j[0] + (i + 1)
        
        limit_src = alpha[i] - beta[i]
        limit_dst = beta[j] - alpha[j]
        delta = min(limit_src, limit_dst)
        if j > i:
            cum_diff = np.cumsum(diff)
            min_slack = np.min(cum_diff[i:j])
            if min_slack < delta: delta = min_slack
        if delta == 0: break
        
        # t calculation: u_new = (1-t)u + t*v => t = delta/(u-v)
        u, v = alpha[i], alpha[j]
        t = delta / (u - v)
        
        prev = alpha.copy()
        alpha[i] -= delta
        alpha[j] += delta
        
        # Define Permutations: Identity + Swap(i,j)
        base_indices = list(range(n))
        swap_indices = base_indices.copy()
        swap_indices[i], swap_indices[j] = swap_indices[j], swap_indices[i]
        
        steps.append({
            "description": f"Pairwise Mix (indices {i},{j})",
            "input": prev,
            "output": alpha.copy(),
            "components": [
                {"weight": 1-t, "indices": base_indices, "desc": "Identity"},
                {"weight": t,   "indices": swap_indices, "desc": f"Swap({i},{j})"}
            ]
        })
        
    return steps

# --- B. Birkhoff Decomposition ---
def birkhoff_decomp(alpha_in, beta_in):
    alpha_f = np.array([float(x) for x in alpha_in])
    beta_f  = np.array([float(x) for x in beta_in])
    n = len(alpha_f)
    
    c = np.zeros(n*n)
    A_eq, b_eq = [], []
    # Constraints
    for r in range(n):
        row=np.zeros(n*n); row[r*n:(r+1)*n]=1; A_eq.append(row); b_eq.append(1)
    for c_idx in range(n):
        col=np.zeros(n*n); col[c_idx::n]=1; A_eq.append(col); b_eq.append(1)
    for r in range(n):
        row=np.zeros(n*n)
        for c_idx in range(n): row[r*n+c_idx] = alpha_f[c_idx]
        A_eq.append(row); b_eq.append(beta_f[r])
        
    res = scipy.optimize.linprog(c, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=(0,1), method='highs')
    if not res.success: return []
    
    D_float = res.x.reshape((n,n))
    # Recover exact fractions from floats (within high precision)
    D = np.vectorize(lambda x: Fraction(x).limit_denominator(10**12))(D_float)
    
    # Balance rows to ensure sums are exactly 1
    for i in range(n):
        diff = 1 - sum(D[i,:])
        if diff != 0: D[i, np.argmax(D[i,:])] += diff
            
    components = []
    rem_D = D.copy()
    for _ in range(n*n+50):
        if np.all(rem_D==0): break
        mask = np.array([[1 if x>0 else 0 for x in r] for r in rem_D])
        if np.sum(mask)==0: break
        
        r_ind, c_ind = scipy.optimize.linear_sum_assignment(-mask)
        vals = [rem_D[r,c] for r,c in zip(r_ind, c_ind)]
        if any(v==0 for v in vals): break
        w = min(vals)
        
        components.append({"weight": w, "indices": list(c_ind), "desc": "Permutation"})
        for r,c in zip(r_ind, c_ind): rem_D[r,c] -= w
    
    return [{
        "description": "Global Convex Combination",
        "input": alpha_in,
        "output": beta_in,
        "components": components
    }]

# --- C. Radial Scaling ---
def get_radial_steps(alpha_in, beta_in):
    steps = []
    
    def build_tree(curr_alpha, curr_beta, offset):
        n_block = len(curr_alpha)
        if n_block == 1 or np.all(curr_alpha == curr_beta): return
        
        b_mean = sum(curr_beta) / n_block
        a_c = curr_alpha - b_mean
        b_c = curr_beta - b_mean
        S = np.cumsum(a_c)
        T = np.cumsum(b_c)
        
        indices = [k for k in range(n_block-1) if S[k] > 0]
        if not indices: return
        ratios = [(T[k]/S[k], k) for k in indices]
        t, k_idx = max(ratios, key=lambda x: x[0])
        split_k = k_idx + 1
        
        # Intermediate: t*alpha + (1-t)*mean
        intermediate = t * curr_alpha + (1 - t) * b_mean
        
        # Components: Identity (weight t) + Cyclic Shifts (weight 1-t distributed)
        comps = []
        full_indices = list(range(len(alpha_in))) # Global indices
        
        # 1. Identity
        comps.append({"weight": t, "indices": full_indices.copy(), "desc": "Identity (Scale)"})
        
        # 2. Cyclic Shifts of the block
        shift_w = (1 - t) / n_block
        if shift_w > 0:
            for shift_amt in range(n_block):
                p_indices = full_indices.copy()
                block_inds = p_indices[offset : offset + n_block]
                shifted_block = block_inds[shift_amt:] + block_inds[:shift_amt]
                p_indices[offset : offset + n_block] = shifted_block
                
                comps.append({"weight": shift_w, "indices": p_indices, "desc": f"CyclicShift +{shift_amt}"})

        steps.append({
            "description": f"Block Scale (indices {offset}-{offset+n_block-1})",
            "input_block": curr_alpha,
            "output_block": intermediate,
            "offset": offset,
            "components": comps 
        })
        
        build_tree(intermediate[:split_k], curr_beta[:split_k], offset)
        build_tree(intermediate[split_k:], curr_beta[split_k:], offset + split_k)

    build_tree(np.array(alpha_in, dtype=object), np.array(beta_in, dtype=object), 0)
    
    # Reconstruct full vector chain for verification
    full_steps = []
    curr_full = np.array(alpha_in, dtype=object)
    
    for s in steps:
        next_full = curr_full.copy()
        off = s['offset']
        blk_len = len(s['input_block'])
        next_full[off : off+blk_len] = s['output_block']
        
        s['input'] = curr_full
        s['output'] = next_full
        full_steps.append(s)
        curr_full = next_full
        
    return full_steps

# ==========================================
# 4. Unified Verifier
# ==========================================

def verify_and_print_proof(start_vec, target_vec, steps, algo_name, runtime):
    print(f"\n{'='*20} {algo_name.upper()} {'='*20}")
    print(f"Time Taken: {runtime:.5f} seconds")
    print(f"Goal: {format_vec(start_vec)} -> {format_vec(target_vec)}")
    
    current_vec = np.array(start_vec, dtype=object)
    
    for idx, step in enumerate(steps):
        print(f"\nStep {idx + 1}: {step['description']}")
        
        # 1. Unify Components: Sum(w_k * P_k * v_in)
        calc_vec = np.zeros_like(current_vec, dtype=object)
        
        comps = step['components']
        print(f"   Operation: Weighted Average of {len(comps)} Permutations")
        
        # Print ALL components
        for c in comps:
            w = c['weight']
            p_idx = c['indices']
            permuted_input = current_vec[p_idx]
            calc_vec += w * permuted_input
            
            # Reconstruct pretty permutation string for display
            # If p_idx is [0,1,2], display "Identity"
            # If p_idx is [1,0,2], display "Swap(0,1)" or similar
            # For brevity, we just show the output vector of this component
            print(f"     - w={w}: {c['desc']} -> {format_vec(permuted_input)}")

        # 2. Compare Calculation vs Claimed Output
        exact_check = np.all(calc_vec == step['output'])
        status = "[ALGEBRA VERIFIED]" if exact_check else "[VERIFICATION FAILED]"
        
        print(f"   Result: {format_vec(step['input'])} -> {format_vec(step['output'])}")
        print(f"   Status: {status}")
        
        if not exact_check:
             print(f"     DEBUG Calc: {format_vec(calc_vec)}")
             print(f"     DEBUG Want: {format_vec(step['output'])}")
        
        current_vec = step['output']

    final_match = np.all(current_vec == target_vec)
    print(f"\nFinal Match: {'YES' if final_match else 'NO'}")

# ==========================================
# 5. Execution
# ==========================================

def run_suite():
    for n in [3, 4, 5]:
        alpha, beta = generate_uniform_case(n, seed=42+n*10)
        print("\n" + "#"*70)
        print(f"TEST INSTANCE N={n}")
        print("#"*70)
        
        # 1. HLP
        t0 = time.perf_counter()
        steps = hlp_greedy(alpha, beta)
        dt = time.perf_counter() - t0
        verify_and_print_proof(alpha, beta, steps, "HLP Greedy", dt)
        
        # 2. Birkhoff
        t0 = time.perf_counter()
        steps = birkhoff_decomp(alpha, beta)
        dt = time.perf_counter() - t0
        verify_and_print_proof(alpha, beta, steps, "Birkhoff Decomposition", dt)
        
        # 3. Radial
        t0 = time.perf_counter()
        steps = get_radial_steps(alpha, beta)
        dt = time.perf_counter() - t0
        verify_and_print_proof(alpha, beta, steps, "Radial Scaling", dt)

if __name__ == "__main__":
    run_suite()
