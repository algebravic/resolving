"""
Exact joint entropy calculation for the following:

We are given m rows, R1,...,Rm, of length n of 0/1 vectors, each of which
has weight <= n/2, and a parameter 1 <= s <= n/2. We want to calculate
the probability distribution
Prob(#(R1 . X) = j) for j=0,...,min(s, floor(n/2)) where
X is the random variable that takes on each 0/1 vector of length n
and weight s with equal probability.

We wish to do this for all such combinations.

To this end we define certain quantites.

If we arrange the rows in an m x n matrix, we have
the columns C1,...,Cn of this matrix
Note that this probability distribution only depends on the multiset of columns that
appear in the matrix (by symmetry).

More specifically if the multiplicity of the column v in this multiset is m(v),
and for the r.v. X, there are k <= m(v) 1's in those columns this contributes
a factor of binom(m(v),k) to the probabilty.

If we have the weights of the rows, we can try to find all solutions to
w = sum_v m(v) v, where w is the vector of weights.

For each of those solutions, and every value of s, we can find all partitions
of s = sum_v s(v), so that s(v) <= m(v). For each such we can calculate
Prob((#(Ri . X)) = t), where t is a vector of counts.

We do this by adding up sum_{s(v) in partitions} ((Prod_v binom(m(v),s(v))))
to get the numerator. The denominator is always binom(n,s).

Namely, if a region has a non-zero vector, v, then the number of ways that
s(v) 1's can take up slots in binom(m(v), s(v)). The actual vector that
it contributes to is sum_v s(v) v.

Conditioining: Our ultimate goal is to find a tight upper bound
for H(X1,...,Xm), where Y^s is a random variable supported on all
subsets of [n] of cardinality s, and Xi = #(Di intersect Y^s)
where Di is the i-th row of the test matrix.

Since finding the exact value of the above entropy depends on the exact
intersection pattern (there are 2^m of them) of the Di, we restrict
ourselves to using entropy inequalities. The simplest is of the following sort:

H(X1,...,Xm) <= H(X1) + ... + H(Xm).

A better one would be of the form
H(X1,...,Xm) <= H(X1) + H(X2|X1) + ... + H(Xm|X1) (*)
where we heuristically choose X1 so that it minimizes
the left over sum.

We can elaborate cases in this situation:
each case is governed by #(D1 intersect Di).
Note that now we have the option of choosing another distribution
for Y^s more adapted to this. In particular we look at
distributions of the following form:
Let d = #D1. Then look at the family of distributions
which are mixture models: with probability pj
choose all subsets, S, of [n] such that #(S intersect D1) = j
uniformly, for all j=max(0,n-s),...,min(d,s).
Given that we then want to calculate the minimum
of the right hand side of (*).


In the original formulation, by Pippenger, and my elaboration
for balanced detecting matrices, the distribution for Y^s was uniform.
This allowed the calculation of H(Xi) to depend only on the cardinality
of Di. 
"""
from typing import List, Iterable, Dict, Tuple
from operator import __mul__
from itertools import product, combinations
from functools import cache, partial, reduce
from collections import defaultdict, Counter
from multiset import Multiset, FrozenMultiset
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from ..bounds import binary_entropy
from .bounds import optimal_weights

DISTRIB = Dict[Tuple[int, ...], int]

@cache
def binom(num: int, mnum: int) -> int:
    if mnum < 0 or num < 0:
        print(f"(m,n) = ({mnum},{num})")
        return 0
    elif mnum > num:
        return 0
    elif mnum == 0 or num == mnum:
        return 1
    else:
        return binom(num - 1, mnum) + binom(num - 1, mnum - 1)

def prod(arg: List[int]) -> int:
    return reduce(__mul__, arg, 1)

def unwind(arg: Multiset) -> Tuple:

    return tuple(sorted(chain(*(_[1] * [_[0]] for _ in arg.items))))

def limited_part(regions: List[int], tot: int) -> Iterable[Tuple[int, ...]]:

    if len(regions) == 0:
        if tot == 0:
            yield ()
    elif len(regions) == 1:
        if tot <= regions[0]:
            yield (tot,)
    else:
        reg = regions[0]
        rest = regions[1:]
        for ind in range(min(reg, tot) + 1):
            yield from (((ind,) + _ for _ in limited_part(rest, tot - ind)))

def weights_solutions_sub(wgts: np.ndarray, targets: List[np.ndarray], dim: int,
                          verbose: int = 0) -> Iterable[List[int]]:
    """
    targets is a list of nonzero 0/1 vectors.
    """
    if verbose > 0:
        print(f"sub({wgts}, {targets}")
    if dim < 0 or (dim == 0 & (wgts > 0).any()) or wgts.min() < 0:
        return
    elif len(targets) == 0:
        if (wgts == 0).all():
            yield []
    else:
        targ = targets[0]
        if verbose > 0:
            print(f"targ={targ}, type={targ.dtype}")
        rest = targets[1:]
        for ind in range(min(dim, wgts[targ == 1].min()) + 1):
            yield from ([ind] + _
                        for _ in weights_solutions_sub(wgts - ind * targ, rest, dim - ind))

def nzeros(arg: List[int]) -> int:
    return len([_ for _ in arg if _ != 0])

            
def weights_solutions(wgts: np.ndarray, dim: int,
                      verbose: int = 0) -> Iterable[List[int]]:

    # Omit the all 0's vector
    cols = list(map(lambda _: np.array(list(_), dtype=int),
                    product(range(2), repeat = len(wgts))))[1:]

    def nondup(arg) -> bool:
        "Test for duplicate rows"
        
        mat = set(map(tuple, np.array([_[1] for _  in zip(arg[1:], cols) if _[0] > 0]).T.tolist()))
        return len(mat) == len(wgts)
        
    yield from filter(nondup,([dim - sum(_)] + _
                              for _ in weights_solutions_sub(np.array(unwind(wgts)), cols, dim,
                                                             verbose = verbose)))
def joint_distrib(wgts: np.ndarray, dim: int) -> Iterable[DISTRIB]:
    """
    For each s value in [1, floor(n/2)] give a distribution for
    each multiplicity distribution
    """

    half = dim // 2
    nrows = len(wgts)
    vecs = np.array(list(product(range(2), repeat=wgts.shape[0])), dtype=int)
    for soln in weights_solutions(wgts, dim):
        # For each region multiplicity
        # We want distributions for each possible s value
        distribs = []
        for sval in range(1, half + 1):
            distrib = defaultdict(int)
            for mults in limited_part(soln, sval):
                indx = tuple((np.array(mults, dtype=int) @ vecs).tolist())
                distrib[indx] += prod((binom(*_) for _ in zip(soln, mults)))
            distribs.append(distrib)
        # distribs is a list of length floor(n/2) of distributions (unnormalized)
        yield soln, distribs

def multi_sub(mult: int, choices: Multiset) -> Iterable[Multiset]:

    if mult > len(choices):
        return
    elif mult == 0:
        yield Multiset()
    elif len(choices) > 0:
        elt, mlt = choices.popitem()
        for ind in range(mlt + 1):
            yield from (_.add(elt, ind) for _ in mtuple_sub(mult - ind, choices.copy()))
        
def multi_subsets(mult: int, choices: Multiset) -> Iterable[Multiset]:
    """
    Find all sub multisets of cardinality mult from the multiset choices
    """
    yield from mtuple_sub(mult, choices.copy())
    
def mentropies(mult: int, wgts: Multiset[int], dim: int):
    binoms = np.array([binom(dim, _) for _ in range(1, (dim // 2) + 1)])
    denom = np.log(binoms) / np.log(2)
    dct = dict()
    for wgt in multi_subsets(mult, wgts):
        ents = dict()
        for soln, distrib in joint_distrib(np.array(wgt, dtype=int), dim):
            # Find the entropies, one for each value of s
            sub_ent = []
            for ixx, svalues in enumerate(distrib):
                nvalue = binary_entropy(np.array(list(svalues.values()))) / denom[ixx]
                sub_ent.append(nvalue)
            ents[tuple(soln)] = np.array(sub_ent)
        dct[wgt] = ents

    return dct

def max_entropies(distribs: Dict[Tuple[int, ...], Dict[Tuple[int, ...], np.ndarray]]):
    max_vals = dict()
    for mset, entval in distribs.items():
        # find the max
        max_vals[mset] = np.concatenate([[_] for _ in entval.values()], axis=0).max(axis=0)
    return max_vals
    
def better_bound(mult: int, wgts: List[int], dim: int):
    """
    Use the linear programming bound use the max entropy
    for each value of s.
    """
    distribs = mentropies(mult, wgts, dim)
    # distribs is indexed by a multiset of cardinality m of the lists of row weights
    # Each value is a dictionary indexed by all possible region breakdowns
    # The value of each of them is a vector of normalized entropies.
    # So to get a rigorous bound we need the max of of all these vectors.
    # Set up the linear program
    max_vals = max_entropies(distribs)
    half = dim // 2
    for sval in range(half):
        # Build the directory
        # recall that the argument to optimal weights is a dictionary whose keys are frozensets
        # of length mult, corresponding to tuples of random variables.
        dct = dict()
        for idx in combinations(range(len(wgts)), mult):
            dct[frozenset(idx)] = max_vals[tuple(sorted([wgts[_] for _ in idx]))][sval]
        opt, coeffs = optimal_weights(dct)
        print(f"Bound for {sval + 1} = {opt}")
        print(f"coeffs = {coeffs}")

def conditional(mult: int, wgts: List[int], dim: int):
    """
    For each weight choice in wgts, calculate the
    entropy bound for conditioning all other rows on that chosen row.
    """
    distribs = mentropies(mult + 1, wgts, dim)
    uni = mentropies(mult, wgts, dim)

    dct = dict()
    for ind in range(len(wgts)):
        arg = wgts[ind]
        if arg not in dct:
            rest = wgts[: ind] + wgts[ind + 1:]
            xtra = uni[(arg,)]
            dct[arg] = arg + sum((distribs[tuple(sorted([arg, _]))] - xtra for _ in rest))
    return dct

def min_func(coeffs: np.ndarray, probs: np.ndarray):

    # Stay in range
    p_clipped = np.clip(probs, 1e-15,1.0)
    entropy = - np.sum(p_clipped * np.log(p_clipped) / np.log(2))
    linear = coeffs @ p_clipped
    # If E, L > 0. Minimize (E/(L+E))
    # is the same as maximize 1 - (E/(L+E)) = L/(L+E)
    # Which is the same as minimize (L+E)/L = 1 + E/L.
    return entropy / (linear + entropy)
    

def one_row_opt(dim: int, sval:int, weight:int):

    pval = cp.Variable(min(sval, weight) + 1, nonneg=True, name='p')
    prob = [cp.sum(pval) == 1]
    nvals = min(sval, weight) + 1
    cards = [binom(weight, _) * binom(dim - weight, sval - _)
             for _ in range(nvals)]
    constraints = {'type': 'eq', 'fun': lambda _: np.sum(_) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(nvals)]
    initial = np.ones(nvals) / nvals
    result = minimize(partial(min_func, np.log(cards) / np.log(2)), initial,
                              method = 'SLSQP',
                              bounds = bounds,
                              constraints = constraints)

    if result.success:
        print(f"p = {result.x}")
        print(f"Minimized ratio = {result.fun}")
    else:
        print(f"Message: {result.message}")

    # We want to maximize bot/top = 1 + linear/entropy
    # Or minimize entropy / linear
    # Set t = 1 / linear
    # Then we want to minimize
    # t H(y/t) = - sum_i y[i] log(y[i]/t)
    # Use Dinkelbach's algorithm
    # In the general problem we have min
    # E = main entropy, L=linear
    # Z = sum of other entropies
    # Minimize (E+Z)/(E+L) = 1 + (Z-L)/(E+L)
    # Maximize (E+L)/(Z-L) = E/(Z-L) + L/(Z-L)
    # Or, if Z=0, minimize E/L
    # Aux: Maximize (E+L) - lambda * (Z-L)
    # top = sum entropies (concave) bot = linear + entropy
    # bot / top - 1 = (sum entropies - linear) / (linear + entropy)
    # inverting (linear + entropy) / (sum entropies - linear)
    # linear = np.log(cards) @ pval
    # top = cp.sum(cp.entr(pval))
    # bot = top + linear
    # obj = top / bot
    # multiplier = cp.Parameter(nonneg=True, name='lambda')
    # problem = cp.Problem(cp.Minimize(linear - multiplier * top), prob)
    # multiplier.value = 0.0
    # tolerance = 1.0e-6
    # while True:
    #     opt = problem.solve()
    #     print(f"lambda = {multiplier.value}, opt={opt}, p={pval.value}")
    #     new_mult = linear.value / top.value
    #     if abs(multiplier.value - new_mult) < tolerance:
    #         return opt, pval.value
    #     multiplier.value = new_mult

def region_splits(wgt1: int, wgt2: int) -> Iterable[Tuple[int, int]]:
    """
    Give all region splits (u,v) so that u+v = w2, 0<= u <= w1
    """
    yield from ((_, wgt2 - _) for _ in range(min(wgt1, wgt2) + 1))

def ik_tuples(nval, wval, sval, jval, uval, vval):
    def ik_value(ival, kval):
        return (binom(wval-uval, ival-kval)*
                binom(uval, kval)*
                binom(vval,jval-kval)*
                binom(nval-wval-vval,sval+kval-ival-jval))

    # i - w + u <= k <= i
    # 0 <= k <= u
    # j-v <= k <= j
    # i + j - s <= k <= n - w - v - s + i + j
    for ind in range(wval + 1):
        yield from ((ind, ik_value(ind,_))
                    for _ in range(max(0,
                                       ind - wval + uval,
                                       jval - vval,
                                       ind + jval - sval),
                                   min(ind, uval, jval,
                                       nval - wval - vval - sval + ind + jval) + 1))

def big_coeffs(nval: int, sval: int, wgt_main: int, uval: int, vval: int):

    # Optimize all functions
    nvals = min(sval, wgt_main) + 1
    icoeffs = [binom(wgt_main, _) * binom(nval - wgt_main, sval - _)
               for _ in range(nvals)]
    coeffs = np.zeros(nval, dtype=np.float64)
    for jval in range(nvals):
        for ival, icoeff in ik_tuples(nval, wgt_main, sval, jval, uval, vval):
            coeffs[jval] += icoeff / icoeffs[ival]
    return coeffs
                
def big_func(nval: int, sval: int, wgt_main: int, other: List[int], pval: np.ndarray):
    bcoeffs = []
    for uval, val in zip(*product(*(region_splits(wgt_main,_) for _ in other))):
        bcoeffs.append(big_coeffs(nval, sval, wgt_main, uval, vval))

def distrib_opt(dim: int, sval: int, main_weight: int, weights: Tuple[Tuple[int, int], ...]):
    """
    Given the
    dimension: d
    sval: the weight of the tested rows.
    main weight: w (in range(1, floor(d/2) + 1))
    subsidiary weights: pairs (k,r), where k is the main weight, and r is the
    cardinality of the intersection of the main rows with this row
    so k in range(1, floor(d/2) + 1), r <= k.

    We define Y as a mixture model. With probability
    p[i] we choose a subset of [n] of cardinality s
    such that its cardinality of intersection with the
    first row is i, uniformly among all such subsets, S[i].
    The number of such subsets is binom(w,i)*binom(n-w,s-i).
    Call that quantity n[i]. Z is an r.v. with values in [0, w]

    Pr(Y=v,Z=i) = (1/n[i] if v in S[i] else 0)

    If V is a subset
    So Prob(Y = V, Z=i) = sum_i p[i] * 
    So the entropy of Y is
    sum_i - n[i] * p[i]/n[i] * log(p[i]/n[i])
    = - sum_i p[i] * log(p[i]) + sum_i p[i] * log(n[i])

    The entropy of X1 is

    - sum_i p[i] * log(p[i])
    
    Namely, once we've chosen i, the probability of each
    set is 1/n[i], and there are n[i] such choices, so
    each of them has probability p[i]/n[i].

    The probability of intersecting one of the other rows
    | 0 ... 0 | 1 ... 1 | 1...1 | 0...0|
    The first region is where the other rows does not intersect
    the first row. It's of size w-r. The second is the
    intersection, of size r. The third is where
    the other row intersects the complement
    of the first row. This is of size k-r.
    The last section is of size n-k+r-w.
    The only sections that matter are the second and third.

    Let T[i] = collection of subsets V in S[i] such that
    the intersection sizes with D' are (u,v),
    where u = #(D' intersect V intersect D)
    and v = #(D' intersect V intersect ~D)

    If Z=i, the number of V in S[i] so that #(V intersect D') = j
    and #(V intersect D intersect D') = k
    is binom(w-u,i-k)*binom(u,k)*binom(v,j-k)*binom(n-w-v,s+k-i-j)
    := z(w,u,v)[j,i,k] with probability 1/n[i]
    and 0 otherwise

    So
    Prob(X=j) = sum_{V,i} Prob(X=j | Y=V, Z=i) Prob(Y=V | Z=i) p[i]
    sum_{V in S[i]} Prob(X=j | Y=V) p[i]/n[i]

    So Pr(X=j) = sum_i sum_k z(w,u,v)[j,i,k] * p[i]/n[i]
    where the sum over k is
    i-w+u <= k, 0 <= k <= u, 0 <= k <= j, k >= j-v, k <= n-w-v-s+i+j
    Or max(0,i-w+u,j-v) <= k <= min(u,j,n-w-v-s+i+j)
    Call this set of k, K[i]
    """
    
    pass
