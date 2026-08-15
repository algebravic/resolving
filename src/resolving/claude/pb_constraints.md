# Pseudo-Boolean constraints for balanced-detecting matrix search

## What these are

Sound constraints derived from the balanced-detecting condition itself, no enumeration
required. Each constraint holds for **every** balanced-detecting matrix (assuming
your existing row-weight cap symmetry break, which pins each row's weight to
`floor(n/2)`). Validated by checking against:
- The verified 7×13 balanced-detecting matrix from our DFS search
- Cantor–Mills' constructive E_3 (7×12)
Both have 0 violations.

## Variables assumed

- `x[i,j] ∈ {0,1}` for i ∈ [r], j ∈ [n] — the primary matrix entries
- `w_j = sum_i x[i,j]` — column j's weight (0 to r); typically encoded via
  sequential counter or totalizer auxiliary variables
- `I[j,j'] = sum_i (x[i,j] AND x[i,j'])` for j < j' — pairwise intersection;
  needs auxiliary variables for the AND products, or encode via
  `I[j,j'] = w_j + w_{j'} - w(c_j XOR c_{j'})` if you have XOR-weight auxiliaries

## Constraint families

### A1: Total weight

```
sum_{i,j} x[i,j] <= r * floor(n/2)
```

*Derivation:* Sum of row-weight cap `floor(n/2)` over `r` rows.

### A2: Column-weight histogram bounds

For each weight k ∈ {0, 1, ..., r} where `binom(r,k) < n`:

```
count_{j : w_j = k} <= binom(r,k)
```

*Derivation:* Columns must be distinct (else `c_j - c_{j'}` is a weight-2 kernel
vector). Columns of weight k are distinct vectors in `{0,1}^r` of that weight;
only `binom(r,k)` such vectors exist.

### B: Refined pair-(t,v) bounds

For each pair `(t, v)` with `0 ≤ v ≤ t/2 ≤ r` where the bound is
non-trivial (`< binom(n,2)`):

```
count_{(j,j') : j<j', w_j+w_{j'}=t, I[j,j']=v} <= binom(r,v) * binom(r-v, t-2v)
```

*Derivation:* The pair sum `c_j + c_{j'}` in `{0,1,2}^r` has coordinate-sum t
and exactly v entries equal to 2 (the intersection). There are exactly
`binom(r,v) * binom(r-v, t-2v)` such vectors, and balanced-detecting requires
all `binom(n,2)` pair sums to be distinct.

The **extreme bins** (v=0 or 2v=t) tend to be tightest. Notable ones:

- `t=0, v=0` (both columns are zero): ≤ 1 — this is the "at most one zero column"
- `t=1, v=0` (weight-1 with weight-0): ≤ r
- `t=2r, v=r` (both columns all-ones): ≤ 1
- `t=r, v=0` (disjoint supports summing to r): ≤ 1 (only the all-ones vector fits)

## Full constraint list for r=7, n=13

```
# A1
sum_{i,j} x[i,j] <= 42

# A2 (column-weight histogram)
count_{j : w_j = 0} <= 1
count_{j : w_j = 1} <= 7
count_{j : w_j = 6} <= 7
count_{j : w_j = 7} <= 1

# B (refined pair-(t,v) bounds) - 24 constraints
count_{(j,j') : w_j+w_j' = 0,  I[j,j'] = 0} <= 1   # extreme
count_{(j,j') : w_j+w_j' = 1,  I[j,j'] = 0} <= 7   # extreme
count_{(j,j') : w_j+w_j' = 2,  I[j,j'] = 0} <= 21  # extreme
count_{(j,j') : w_j+w_j' = 2,  I[j,j'] = 1} <= 7   # extreme
count_{(j,j') : w_j+w_j' = 3,  I[j,j'] = 0} <= 35  # extreme
count_{(j,j') : w_j+w_j' = 3,  I[j,j'] = 1} <= 42
count_{(j,j') : w_j+w_j' = 4,  I[j,j'] = 0} <= 35  # extreme
count_{(j,j') : w_j+w_j' = 4,  I[j,j'] = 2} <= 21  # extreme
count_{(j,j') : w_j+w_j' = 5,  I[j,j'] = 0} <= 21  # extreme
count_{(j,j') : w_j+w_j' = 6,  I[j,j'] = 0} <= 7   # extreme
count_{(j,j') : w_j+w_j' = 6,  I[j,j'] = 3} <= 35  # extreme
count_{(j,j') : w_j+w_j' = 7,  I[j,j'] = 0} <= 1   # extreme
count_{(j,j') : w_j+w_j' = 7,  I[j,j'] = 1} <= 42
count_{(j,j') : w_j+w_j' = 8,  I[j,j'] = 1} <= 7
count_{(j,j') : w_j+w_j' = 8,  I[j,j'] = 4} <= 35  # extreme
count_{(j,j') : w_j+w_j' = 9,  I[j,j'] = 2} <= 21
count_{(j,j') : w_j+w_j' = 10, I[j,j'] = 3} <= 35
count_{(j,j') : w_j+w_j' = 10, I[j,j'] = 5} <= 21  # extreme
count_{(j,j') : w_j+w_j' = 11, I[j,j'] = 4} <= 35
count_{(j,j') : w_j+w_j' = 11, I[j,j'] = 5} <= 42
count_{(j,j') : w_j+w_j' = 12, I[j,j'] = 5} <= 21
count_{(j,j') : w_j+w_j' = 12, I[j,j'] = 6} <= 7   # extreme
count_{(j,j') : w_j+w_j' = 13, I[j,j'] = 6} <= 7
count_{(j,j') : w_j+w_j' = 14, I[j,j'] = 7} <= 1   # extreme
```

## Full constraint list for r=7, n=14 (target: β_14)

Identical to n=13 above but with:

```
# A1
sum_{i,j} x[i,j] <= 49  # r * floor(n/2) = 7 * 7
```

All A2 and B constraints are the **same** (they don't depend on n, only on r).

## Which are likely to prune

I'd bet effort in this order:

1. **B extreme bins with bound = 1**: `t=0,v=0`; `t=7,v=0`; `t=14,v=7`. These are hard
   equality-ish constraints — at most one such pair. If your existing encoding
   already handles column-distinctness, some of these are redundant.

2. **A2 histogram bounds**: cheap to add, small clause count. The `N_0 <= 1` and
   `N_7 <= 1` are almost trivial; `N_1 <= 7` and `N_6 <= 7` cut real space at
   larger n.

3. **B mid-range bins with tight bounds** (e.g., `t=2, v=1: <= 7`): these are the
   ones our earlier data showed hitting 100% at r=5,6, so likely to bind in
   near-extremal solutions.

The remaining B constraints (bounds 21, 35, 42) are looser and probably slack
against your existing propagation. Adding them costs clause budget; monitor
whether they participate in learned clauses to see if they're doing work.

## Honest caveats

- These are **necessary** conditions from the BD condition alone. They will not
  prove UNSAT that isn't already implied by BD + distinctness. If your SAT solver
  is currently deriving UNSAT through many conflict clauses that essentially
  reconstruct these bounds, adding them explicitly won't help much (though it
  might speed up early conflicts).

- The auxiliary variables `I[j,j']` add a quadratic number of variables. For
  n=14 that's `binom(14,2) = 91` new variables plus their totalizer/counter
  clauses. Worth it only if the constraints prune.

- The B constraints are what could most plausibly improve on your current setup.
  A1 and A2 are so cheap you should probably add them regardless; B needs
  benchmarking.
