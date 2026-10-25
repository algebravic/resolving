# Orbits of m×n 0/1 matrices under row/column permutations and complementations

Files: `orbit_count.py` (algorithms + CLI), `verify.py` (brute force and Schreier–Sims cross-checks).

## 0. The group

Write g = (σ, τ, a, b) with σ∈S_m, τ∈S_n, a∈F_2^m, b∈F_2^n, acting by

    (g·M)_{σ(i),τ(j)} = M_{ij} + a_i + b_j   (mod 2).

(a,b) and (a+1,b+1) act identically, and this is the only redundancy (test on M=0), so
|G| = m! n! 2^{m+n-1}. Burnside over the extended group of order m! n! 2^{m+n} gives the same
orbit count, so I average over all (σ,τ,a,b).

Stabilizer-chain view: G = N ⋊ (S_m×S_n), N = F_2^{m+n}/<(1,1)> (order 2^{m+n-1}). A Schreier–Sims
chain has transversals = the coset representatives of N and of S_m×S_n's own chain, so summing over
"every element of G via the transversal product" is exactly summing over (a,b,σ,τ). That sum is
|G| ~ 10^{37} terms at (m,n)=(15,16), so it is used only as a cross-check (verify.py); the algorithm
itself collapses the sum to conjugacy classes (cycle types).

## 1. Part 1: fixed points of g (proved)

### 1.1 Row cycles and column cycles

Row indices are {0,…,m−1} and column indices are {0,…,n−1}. The permutation σ∈S_m acts on the row indices; the subgroup ⟨σ⟩ splits
them into orbits, and each orbit is a **row cycle**: C = {i, σ(i), σ²(i), …, σ^{p−1}(i)}, where p = |C| is the smallest p>0 with σ^p(i)=i.
A row fixed by σ is a row cycle of length 1. The row cycles partition the row set; their number is c(σ), and their lengths (as a multiset)
form the *cycle type* λ ⊢ m. A **column cycle** D (length q) is the same thing for τ∈S_n on the column indices; its type is μ ⊢ n.
Row cycles depend only on σ, column cycles only on τ (not on a or b).

*Example* (used throughout). m=5, σ = (0 1 2)(3 4): row cycles C₁={0,1,2} (p=3) and C₂={3,4} (p=2), λ=(3,2), c(σ)=2.
n=4, τ = (0 1)(2)(3): column cycles D₁={0,1} (q=2), D₂={2} (q=1), D₃={3} (q=1), μ=(2,1,1), c(τ)=3.

For each cycle we record two things: its length, and the parity of the complementation on it,

    A_C = Σ_{i∈C} a_i mod 2   (are an odd number of the rows of C complemented?),   B_D = Σ_{j∈D} b_j mod 2.

### 1.2 Cell orbits and the consistency condition

g sends the cell (i,j) to (σ(i),τ(j)) and adds a_i+b_j to the value: a matrix M is fixed iff M_{σ(i),τ(j)} = M_{ij} + a_i + b_j for all cells.
The cells of the block C×D (p·q cells) are permuted by (i,j)↦(σi,τj), which is a p-cycle times a q-cycle; this splits C×D into
g = gcd(p,q) orbits, each of length L = lcm(p,q) (since pq = gL). Summed over all blocks the cell orbits partition all mn cells.

Fix one cell orbit and walk once around it. Going around, the fixed-point equations force the value to change by
Σ_{k=0}^{L−1}(a_{σ^k i} + b_{τ^k j}), which must be 0 for M to exist. In L steps the row index runs L/p times through every element
of C, so the a-part is (L/p)·A_C = (q/g)·A_C (mod 2); likewise the b-part is (L/q)·B_D = (p/g)·B_D. This does not depend on which
cell of the block we started from, so all g orbits of the block C×D give the same condition

    (q/g)·A_C + (p/g)·B_D ≡ 0 (mod 2).                                                (★)

If (★) holds for every pair (C,D), each cell orbit carries exactly one free bit (the value at its start cell determines the rest), so

    Fix(g) = 2^{Σ_{C,D} gcd(p_C,q_D)}   if (★) holds for all (C,D),   and Fix(g) = 0 otherwise.

Note that Fix(g) depends on (a,b) only through the parity patterns A=(A_C)_C ∈ F_2^{c(σ)} and B=(B_D)_D ∈ F_2^{c(τ)}.

### 1.3 Reading (★): three cases

p/g and q/g are coprime, so they are not both even; v₂(g)=min(v₂p, v₂q), hence
* v₂(p)=v₂(q): p/g and q/g are both odd, (★) reads **A_C = B_D**;
* v₂(p)<v₂(q): p/g odd, q/g even, (★) reads **B_D = 0**;
* v₂(p)>v₂(q): p/g even, q/g odd, (★) reads **A_C = 0**.
(v₂ = exponent of 2 in the length.) So every pair (C,D) contributes exactly one equation, of the form x=y or x=0 over F_2.

### 1.4 The constraint graph

The system of all equations (★) is encoded by an undirected graph, the **constraint graph** Γ(λ,μ):

* **vertices:** one vertex for every row cycle C (standing for the unknown bit A_C), one for every column cycle D (the unknown B_D),
  and one extra vertex Z standing for the constant 0. Total c(σ)+c(τ)+1 vertices.
* **edges:** for every pair (C,D), exactly one edge, chosen by the case in 1.3:
  C—D if v₂(p)=v₂(q) (equation A_C=B_D);  D—Z if v₂(p)<v₂(q) (B_D=0);  C—Z if v₂(p)>v₂(q) (A_C=0).
  (Multiple edges between the same two vertices are irrelevant; there are no edges C—C′ or D—D′.)

Every equation says "the two endpoints have the same value", where Z has the value 0. Hence a parity pattern (A,B) satisfies (★) iff it
is constant on each connected component of Γ, and equal to 0 on the component of Z. A component not containing Z can carry either
value independently of the others. Therefore

    N(λ,μ) := #{(A,B) satisfying (★)} = 2^{(number of connected components of Γ) − 1}
           = 2^{#components not containing Z}.

*Structure of Γ (this gives the closed form used in the code, `n_parity_patterns`).* Group the cycles by level v=v₂(length); let R_v and
K_v be the numbers of row and column cycles at level v, and s_min, t_min the smallest levels occurring among row and column cycles.
A row cycle at level v is joined to every column cycle of level v, and to Z exactly when some column cycle has level < v, i.e. t_min < v;
symmetrically a column cycle at level v is joined to Z exactly when s_min < v. Hence the components are:
* if R_v>0 and K_v>0: all cycles of level v form one connected piece (complete bipartite); it is attached to Z unless s_min=t_min=v,
  contributing a factor 2 in that case and 1 otherwise;
* if only rows at level v (K_v=0): each of the R_v row cycles is a separate vertex, attached to Z iff t_min<v; factor 2^{R_v} if t_min>v, else 1;
* if only columns at level v (R_v=0): factor 2^{K_v} if s_min>v, else 1.
N is the product of these factors over all levels v.

*Worked example.* Levels: C₁ (p=3, v₂=0), C₂ (p=2, v₂=1), D₁ (q=2, v₂=1), D₂,D₃ (q=1, v₂=0). Equations by pair:
(C₁,D₁): 0<1 ⇒ B_{D₁}=0 (edge D₁—Z); (C₁,D₂),(C₁,D₃): 0=0 ⇒ A_{C₁}=B_{D₂}=B_{D₃} (edges C₁—D₂, C₁—D₃);
(C₂,D₁): 1=1 ⇒ A_{C₂}=B_{D₁} (edge C₂—D₁); (C₂,D₂),(C₂,D₃): 1>0 ⇒ A_{C₂}=0 (edges C₂—Z).
Components: {Z, D₁, C₂} and {C₁, D₂, D₃}, so N = 2^{2−1} = 2 (free bit: the common value of A_{C₁}=B_{D₂}=B_{D₃}). The level formula agrees:
level 0 has both kinds and s_min=t_min=0 ⇒ 2; level 1 has both kinds but s_min=0<1 ⇒ 1. Also Σgcd = 1+1+1+2+1+1 = 7.
Check by brute force (all 2⁹ pairs (a,b), counting fixed matrices by cell-orbit consistency): Σ_{a,b} Fix = 4096 = 2^{5−2+4−3}·N·2⁷ = 16·2·128. ✓

### 1.5 From fixed points to the orbit count

For a given σ and τ each parity pattern (A,B) is realized by exactly 2^{m−c(σ)}·2^{n−c(τ)} pairs (a,b): on a cycle of length p there are
2^{p−1} vectors in F_2^p of each parity. Hence Σ_{a,b} Fix(σ,τ,a,b) = 2^{m−c(σ)+n−c(τ)} · N(λ,μ) · 2^{Σ gcd}, which depends only on the
cycle types (λ,μ) of (σ,τ). There are m!/z_λ permutations of type λ (z_λ = ∏_k k^{m_k}m_k!, m_k = number of cycles of length k) and n!/z_μ
of type μ. Burnside over the (extended) group of order m!·n!·2^{m+n} gives

    #orbits = 1/(m! n! 2^{m+n}) Σ_{λ⊢m, μ⊢n} (m!/z_λ)(n!/z_μ) 2^{m−ℓ(λ)+n−ℓ(μ)} N(λ,μ) 2^{Σ gcd}
            = Σ_{λ⊢m, μ⊢n}  N(λ,μ) · 2^{Σ_{p∈λ,q∈μ} gcd(p,q) − ℓ(λ) − ℓ(μ)} / (z_λ z_μ),

with ℓ(λ)=c(σ) the number of cycles. Cost: p(m)·p(n)·O(ℓ(λ)ℓ(μ)) — for n=16 at most 231·176 pairs per (m,n), seconds in Python;
the code (`count_orbits`) uses exact integers and asserts divisibility. Test: m=n=1 gives 1; m=n=4 gives 12 (matches brute force).

## 2. Part 2: what "G acting on B" can mean — please read

B (prescribed row sums) is **not G-invariant**: a column complement changes row sums by
|J|−2|M_i∩J|, which varies over B unless J=∅ or J=all columns. So "G acting on B" is ambiguous:

* (b) orbits of the setwise stabilizer H = Stab_G(B) on B  — a genuine group action; Burnside applies.
* (a) number of G-orbits on S that meet B (= classes of B under the induced equivalence). This is a
  groupoid component count, Σ_{M∈B} 1/|O_M∩B|; I found no Burnside-type formula for it, and give
  only a brute-force routine for small sizes.

I implement (b) and use (a) only in small brute-force comparisons so the difference is visible.

### Structure of H (proved)
Let ρ_i = min(r_i, n−r_i). Complementing rows with r_i>n/2 is a bijection B→B_ρ conjugating H to
H' = S_n × Π_{w<n/2} S_{u_w} × (Z_2 ≀ S_{u_{n/2}}) (u_w = #rows with ρ_i=w; the last factor only when n even),
so #H-orbits on B = #H'-orbits on B_ρ. (Column complements other than ∅/all are not in H unless B is a single matrix, in which case the count is 1 either way.)

### Fixed points on B_ρ
For h=(σ,τ,a)∈H', the row cycles of σ decouple: a row cycle C (length p, base row i0, A=A_C) contributes
f = #{x∈F_2^n : x∘τ^p = x + A, wt(x)=ρ_{i0}}. τ^p splits a column q-cycle into g=gcd(p,q) cycles of length ℓ=q/g;
* A=0: x constant on each: factor (1+t^ℓ)^g;   * A=1: need ℓ even, factor (2t^{ℓ/2})^g, else 0.
f = [t^{ρ}] of the product over column cycles. Rows with ρ<n/2 have a=0 (A=0 only); rows with ρ=n/2 have
both parities, each realised by 2^{p-1} vectors a. Sum over σ∈S_u of Π_cycles φ(p) obeys
T_u = Σ_{p=1}^u (u−1)!/(u−p)! · φ(p) · T_{u−p}. Then

    #orbits = (1/|H'|) Σ_{μ⊢n} (n!/z_μ) Π_w T_{u_w}(μ),   |H'| = n! Π_{w<n/2} u_w! · 2^{u_{n/2}} u_{n/2}!.

Complexity: p(n) · (#distinct ρ classes) · O(m² n) — negligible for n≤16.

## 3. Verification and results
All run and passing:
* `verify.py`: closed-form N(λ,μ) = explicit F_2 count for all m,n ≤ 7; part 1 formula = union-find brute force
  = Burnside over every element of G enumerated by a Schreier–Sims chain (own implementation, |G| matches
  m!n!2^{m+n-1}) for (m,n) ∈ {(1,1),(1,2),(1,3),(2,2),(2,3),(3,3),(2,4),(3,4),(1,5),(2,5),(4,4)}.
* Part 2 (b): formula = brute-force Stab_G(B)-orbits on B for every sorted r with (m,n) ∈ {(1,4),(2,3),(2,4),(3,3),(3,4),(2,5)},
  plus `verify_n6.py` (n=6, m=3, several r including ρ=n/2 rows).
* Part 2 (a) differs from (b) already at m=3,n=4, r=(1,1,2): 3 vs 2 (see verify.py output). Proven vs empirical:
  the formulas are derived; the checks above are empirical confirmation only on small cases.
* Timing: full part-1 table for all 1≤m<n≤16 in ~2 s; part 2 at n=16, m=15: ~0.1 s per r-vector (pure Python).
* Full table: `orbit_table_n16.txt`.

## 4. T = G·B (second request): |T| and #orbits of G on T

T is G-invariant. With ρ_i = min(r_i, n−r_i), P = multiset {ρ_i}, and G_0 = <row perms, row complements, column perms> (index 2^{n−1} in G,
transversal = column complementations t_b):

    A_P := G_0·B = {M : multiset of folded row weights = P},   T = ⋃_{b ∈ F_2^n/<1>} t_b A_P,
    |A_P| = m!/∏u_w! · ∏_i cnt(ρ_i),  cnt(w) = 2C(n,w) (w<n/2), C(n,n/2) (w=n/2).

Equivalently, in ±1 entries: M ∈ T ⇔ ∃ ε∈{±1}^n with multiset |Mε| = {n−2ρ_i}.

Let μ(M) = #{good column-complement classes b : M+b ∈ A_P}. Then (proved)
* |T| = Σ_{M∈A_P} 2^{n−1}/μ(M), so |T| ≤ 2^{n−1}|A_P| with equality iff μ ≡ 1;
* #orbits(T) = β − Σ_O (k(O)−1) ≤ β, β = #G_0-orbits on A_P = the previously computed Stab_G(B)-orbit count, k(O) = number of
  G_M-orbits on the good-center set of an M in orbit O; equality if μ ≡ 1.
* There is no Burnside formula: [M∈T] is an existential over 2^{n−1} centres, and Σ 1/μ needs the whole distribution of μ
  (inclusion–exclusion Σ_k (−1)^k N_k/(k+1) over k-sets of simultaneously good centres). I do not have a polynomial algorithm.
  The bounds are strict already at m=3,n=4 (e.g. r=(1,1,2): |T|=3072 < 9216, orbits 2 < β=3).

Exact tools: `T_exact_smallm.c` (column-class enumeration; m ≤ 5 at n=16 in ≤ 30 s, m=6 for n ≲ 10), `T_bruteforce.py` (mn ≲ 28).
Validation: C tool = brute force on all 110 profiles of (m,n) ∈ {(2,6),(2,7),(3,6),(3,7),(4,5),(4,6)} plus (3,4),(3,5).
