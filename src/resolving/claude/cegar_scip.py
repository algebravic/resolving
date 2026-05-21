"""
SCIP-based CEGAR for proving SAT/UNSAT of m x n FULL balanced detecting matrices.

Key design choices:
  - Variables y_r in {0,1} for each weight-floor(n/2) row r.   (binom(n, n/2) of them.)
  - Constraints: sum_{r : r.x != 0} y_r >= 1, for x in active set S.
  - Objective: minimize sum y_r  (or feasibility with sum y_r <= m).
  - SCIP symmetry detection ON (it should find the column-permutation S_n action automatically).
  - CEGAR loop:
       (a) Solve ILP for current S.
       (b) If opt > m, UNSAT.
       (c) Else extract A = {r : y_r = 1}. Enumerate B^(n) for violated x's.
       (d) If none, SAT — return A.
       (e) Else batch-add up to K violated x's (one per S_n orbit if possible), repeat.

Initial seed S: one representative per shape orbit (k = |x_+|, k = 1..n/2), both signs.
Conflictor: iterates over B^(n) in random order for diversity; returns up to K finds.

USAGE:  python3 cegar_scip.py <n> <m_target> [--feasibility]
"""
import sys
import time
import random
from itertools import combinations, product
from math import comb

# ---------- Combinatorics ----------

def popcount(x):
    return bin(x).count('1')

def balanced_masks_iter(n, randomize=False, seed=0):
    """Yield (pos, neg) for nonzero balanced ternary x in {-1,0,1}^n."""
    items = []
    for v in product((-1, 0, 1), repeat=n):
        if sum(v) == 0 and any(v):
            pos = 0; neg = 0
            for i, vi in enumerate(v):
                if vi == 1: pos |= (1 << i)
                elif vi == -1: neg |= (1 << i)
            items.append((pos, neg))
    if randomize:
        rng = random.Random(seed)
        rng.shuffle(items)
    return items

def full_rows(n):
    w = n // 2
    return [sum(1<<i for i in s) for s in combinations(range(n), w)]

def row_catches(r, p, ng):
    return popcount(r & p) != popcount(r & ng)

def shape_representative(n, k, with_zero_mode='compact'):
    """Build a canonical x of shape (1^k, (-1)^k, 0^{n-2k}): positions 0..k-1 are +1, k..2k-1 are -1."""
    pos = (1 << k) - 1
    neg = ((1 << (2*k)) - 1) ^ pos
    return (pos, neg)

# ---------- Conflictor: enumerate B^(n) for violated x ----------

def find_violated(chosen_rows, n, max_batch=10, seed=None):
    """Find up to max_batch balanced x's not detected by any row in chosen_rows.
    Returns list of (p, ng); empty if all detected."""
    found = []
    items = balanced_masks_iter(n, randomize=(seed is not None), seed=(seed or 0))
    for (p, ng) in items:
        if all(not row_catches(r, p, ng) for r in chosen_rows):
            found.append((p, ng))
            if len(found) >= max_batch:
                return found
    return found

# ---------- SCIP model builder (pyscipopt) ----------

def build_scip_model(n, m_target, S, feasibility=False, time_limit=None):
    """
    Build pyscipopt Model:
      y_r in {0,1} for each full row r
      sum_{r : r.x != 0} y_r >= 1, x in S
      Either minimize sum y_r (then read off opt), or fix sum y_r <= m_target as constraint
      and check feasibility.
    Returns (model, y_vars, rows_list).
    """
    from pyscipopt import Model, quicksum
    model = Model("balanced_detecting")
    if time_limit is not None:
        model.setParam('limits/time', time_limit)
    # Enable SCIP's symmetry detection — this is the important bit for our S_n action
    model.setParam('misc/usesymmetry', 7)        # bitmask: 1 + 2 + 4 = SST + orbital fixing + orbitope
    model.setParam('propagating/symmetry/sstleadervartype', 7)
    model.setParam('propagating/symmetry/usedynamicprop', True)
    model.setParam('display/verblevel', 0)
    rows = full_rows(n)
    # Decision variables
    y = [model.addVar(vtype="B", name=f"y_{j}") for j in range(len(rows))]
    # Cover constraints
    for (p, ng) in S:
        coverers = [y[j] for j, r in enumerate(rows) if row_catches(r, p, ng)]
        if not coverers:
            # x not coverable by any full row -> immediately UNSAT
            model.addCons(quicksum([]) >= 1, name=f"uncoverable")
        else:
            model.addCons(quicksum(coverers) >= 1, name=f"cov_{p}_{ng}")
    if feasibility:
        model.addCons(quicksum(y) <= m_target, name="budget")
        # Trivial objective
        model.setObjective(0)
    else:
        model.setObjective(quicksum(y), "minimize")
    return model, y, rows

def cegar_scip(n, m_target, feasibility=True, time_limit=600,
               seed_size=None, batch=5, verbose=True):
    """Returns dict with status: 'SAT', 'UNSAT', 'TIMEOUT', and witness."""
    try:
        from pyscipopt import Model
    except ImportError:
        return {'status': 'NO_SCIP'}

    rows = full_rows(n)
    nR = len(rows)
    # Initial seed S: one rep per orbit (k=1..n/2), positive sign.
    S = []
    for k in range(1, n//2 + 1):
        S.append(shape_representative(n, k))
    if seed_size is not None:
        # Pad with random extra to get more constraints initially.
        extra = balanced_masks_iter(n, randomize=True, seed=17)
        for x in extra:
            if x not in S:
                S.append(x)
            if len(S) >= seed_size:
                break

    iteration = 0
    t_start = time.time()
    while True:
        iteration += 1
        elapsed = time.time() - t_start
        rem = time_limit - elapsed
        if rem <= 1:
            return {'status': 'TIMEOUT', 'iters': iteration, 'S_size': len(S)}
        # Build & solve
        model, yvars, rows_list = build_scip_model(
            n, m_target, S, feasibility=feasibility, time_limit=rem
        )
        t0 = time.time()
        model.optimize()
        t1 = time.time()
        status = model.getStatus()
        if verbose:
            print(f"  iter {iteration}: |S|={len(S)}, SCIP status={status}, t={t1-t0:.2f}s", flush=True)
        if status == 'infeasible':
            return {'status': 'UNSAT', 'iters': iteration, 'S_size': len(S)}
        if status == 'timelimit':
            return {'status': 'TIMEOUT', 'iters': iteration, 'S_size': len(S)}
        if status not in ('optimal',):
            return {'status': 'OTHER', 'detail': status, 'iters': iteration, 'S_size': len(S)}
        if not feasibility:
            opt = round(model.getObjVal())
            if opt > m_target:
                return {'status': 'UNSAT', 'opt': opt, 'iters': iteration, 'S_size': len(S)}
        # Extract chosen rows
        chosen = []
        for j, yj in enumerate(yvars):
            if model.getVal(yj) > 0.5:
                chosen.append(rows_list[j])
        # Conflictor
        viol = find_violated(chosen, n, max_batch=batch, seed=iteration)
        if not viol:
            return {'status': 'SAT', 'iters': iteration, 'S_size': len(S), 'rows': chosen}
        for x in viol:
            if x not in S:
                S.append(x)

def cegar_scip_main():
    if len(sys.argv) < 3:
        print("usage: python3 cegar_scip.py <n> <m_target> [--feasibility|--opt]")
        sys.exit(1)
    n = int(sys.argv[1])
    m = int(sys.argv[2])
    feas = '--opt' not in sys.argv[3:]
    print(f"=== n={n}, m_target={m}, feasibility={feas} ===")
    result = cegar_scip(n, m, feasibility=feas, time_limit=600)
    print("RESULT:", {k: v for k, v in result.items() if k != 'rows'})
    if 'rows' in result:
        print("Rows:")
        for r in result['rows']:
            print(f"  {bin(r)[2:].zfill(n)[::-1]}  weight={popcount(r)}")
