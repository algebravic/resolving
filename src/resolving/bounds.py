"""
Calculate some lower bounds for the metric dimension.

See Pippenger:

His argument: Let D_1, ..., D_r be a resolving set,
with d_i = #D_i.  Let M be a random variable taking on each binary
n-tuple equally likely, and let M_i be the random variable #(D_i intersect M).

Then M_i is binomially distributed: Pr(M_i = k) = binom(d_i,k) 2^(-d_i),
So the binary entropy of M_i is H(M_i) = - sum_{0 <= k <= d_i} P(M_i=k) log_2(P(M_i=k))
Let m = (M_1, ..., M_r)
We have I(M; m) >= H(M) = n. But I(M;m) <= H(m) <= sum_i H(M_i).

Note that without loss of generality, we may assume that d_i <= floor(n/2).
"""
from typing import List, Iterable, Tuple
from functools import cache
from itertools import chain
from math import ceil, log, floor
import numpy as np
from sympy import binomial

def binomial_distr(weight: int) -> np.ndarray:
    """
    Unnormalized binomial distribution.
    """
    return np.array([binomial(weight, _) for _ in range(weight + 1)], dtype=np.float64)

def entr(arg: float) -> float:

    return - arg * log(arg) if arg > 0 else 0.0

def binary_entropy(distr: np.ndarray) -> float:
    """
    Given a distribution (all positive), compute
    the binary entropy.
    Let d = sum_i a[i]

    - sum_i (a[i]/d) * log(a[i]/d) = - (1/d) sum_i a[i] log(a[i)) + log(d)
       = - (1/d) sum_i a[i] log(a[i)) + (n/d) log(d)
    """
    
    denom = distr.sum()
    num = distr.shape[0]
    return sum(map(entr, distr)) / (denom * log(2.0)) + log(denom) / log(2.0)

def binomial_entropy(weight) -> float:
    """
    Entropy of binomial distribution.
    """
    return binary_entropy(binomial_distr(weight))

def lower_bound(dim: int) -> float:
    """
    Lower bound, per Pippenger, and using the fact that each element of a resolving
    set may be taken to have weight <= n/2.
    """
    return int(ceil(dim / binomial_entropy( dim // 2)))

def pippenger_bound(dim: int) -> float:

    entropies = map(binomial_entropy, range(1, dim // 2 + 1))
    return ceil(dim / max(entropies))

def subset_distr(num: int, dval: int, sval: int) -> List[float]:

    return np.array([binomial(dval, _) * binomial(num - dval, sval - _)
            for _ in range(min(dval, sval) + 1)], dtype=np.float64)

def entropy_list(dim: int, halve: bool =  True) -> np.ndarray:

    dbound = dim // 2 if halve else dim - 1
    lhs = np.array([log(binomial(dim, sval)) / log(2) for sval in range(1, (dim // 2) + 1)])
    rhs = np.array([[binary_entropy(subset_distr(dim, dval, sval))
                     for sval in range(1, (dim // 2) + 1)]
                    for dval in range(1, dbound + 1)])
    return rhs / lhs.reshape((1, -1))

def improved_bound(dim: int) -> float:
    # The matrix is indexed by (s,d) we must minimize over d
    return 1 + ceil(1 / entropy_list(dim).max(axis=0).min())
    # metric dimension = 1 + # rows in minimal balanced detecting

def violated_sub(arr: np.ndarray, rhs: np.ndarray, mval: int):
    """
    Given an d x s array A, with all positive entries, and a target sum v,
    find all nonnegative integer vectors, u, so that
    u^T A >= rhs, and sum(u) = v.

    A is d x s, rhs is an s-vector

    If the last rows has multiplicity j, then remaining rows
    must have multiplicity (m-j). The best that they can do
    is to have (m-j) * the max in each column be the remaining
    contribution (it could be worse)
    """
    # Look for the contributions of the max
    # Assume the rows are monotone non-decreasing
    dval, sval = arr.shape
    if dval == 0 or mval < 0:
        return
    if dval == 1:
        # we must have j * row[i] >= rhs[i], or j >= rhs[i]/row[i]
        yield from ((_,) for _ in range(max(0, ceil((rhs / arr[0]).max())), mval + 1))
        return
    # There probably is a better way of pruning
    # For each row can find the lower bound to what's necessary. If the sum of the lower bounds is > m
    # Then prune

    # Find all possible multiplicites for d
    # if multiplicity if j then we must have (m-j) * A[:-1].max(axis=0) + j * A[-1] > rhs
    # Or rhs - m * A[:-1].max(axis=0) < j * (A[-1] - A[:-1].max(axis=0))
    # rest = arr[:-1].max(axis=0) if dval > 1 else np.zeros(sval)
    # rest is (d-1) x s
    # jlow = max(0, ceil((rhs  / arr[-1]).max()))
    # jlow = max(0, 1 + floor(((rhs - mval * rest) / (arr[-1] - rest)).max()))
    jlow = 0
    for ind in range(jlow, mval + 1):
        # Try the multiplicity
        newrhs = rhs - ind * arr[-1]
        # By induction violated_sub will return all contintuations with the current ind value
        yield from (_ + (ind,) for _ in violated_sub(arr[:-1], newrhs, mval - ind))

def violated(arr: np.ndarray, mval: int):

    yield from violated_sub(arr, np.ones(arr.shape[1]), mval)

def possible_row_sums(dim: int, mval: int, halve: bool = True):
    elist = entropy_list(dim, halve = halve)
    yield from (_ for _ in violated(elist, mval)
                if (np.array(_) @ elist >= 1).all())

def multiset_row_sums(dim: int, mval: int, halve: bool = True) -> Iterable[Tuple[int,...]]:

    def _multiset(arg: Tuple[int,...]) -> Tuple[int,...]:

        return tuple(chain(*(_[1] * (_[0],)
                             for _ in enumerate(arg, start=1))))
    return map(_multiset, possible_row_sums(dim, mval, halve = halve))

def tuple_count(dim: int, mval: int) -> int:

    """
        The number of nonegative d tuples that sum to m.
    """
    return binomial(dim + mval - 1, dim - 1)

def num_pairs(num: int, subw: int) -> int:

    return (0 if num < 2 * subw
            else binomial(num, subw) * binomial(num - subw, subw))

def unresolved(num: int, wnum: int) -> int:
    """
    Calculate the number of equal weight pairs of vectors
    that are unresolved by a vector of weight k.
    Input:
       num: the length of the bit vectors
       wnum: the weight that we're testing.
    Output:
       The number of pairs of nonzero 0/1 vectors of equal weight
       which have the same weight when intersected with
       a vector of weight k.

    Method:
       For each weight w, the weight of that in the first k
       coordinates is v <= w
       such that 2 * v <= k, we count it in binom(k,v)*binom(k-v,v)
       ways (we'll later divide by 2).  To that we must multiply
       by binom(n-k, w-v) * binom(n-k-w+v,w-v)
       and sum over all (v,w), with 0 <= v <= w, 1 <= w.
    """
    return (sum((num_pairs(wnum, jnum) * num_pairs(num - 2 * wnum, rnum)
               for jnum in range((wnum // 2) + 1)
               for rnum in range(num // 2 - jnum) +1)) - 1)

def unresolved_table(num: int) -> List[int]:
    """
    Table of number of unresolved elements for weights
    in [0, floor(n/2)].
    """
    return [unresolved(num, _) for _ in range((num // 2) + 1)]

def central_trinomial(num: int) -> int:
    """
    The coefficient of 1 in (1 + x + 1/x)^n.
    Recurrence: (n+1)T[n+1] = (2n+1) T[n] + 3n T[n-1]
    T[0] = T[1] = 1
    """

    return sum((binomial(num, 2 * knum) * binomial(2 * knum, knum)
                for knum in range(1, num // 2 + 1)))

def balanced(num: int) -> int:
    """
    The number of nonzero vectors with coordinates in 0/1/-1
    whose sum of coordinates is 0, where a vector and its negative
    are identified. If the n-th value if U[n], then
    U[n] = (T[n] - 1)/2, so
    (2n+2) U[n+1] + (n+1) = (2n+2) U[n] + (2n+1) + 6n U[n-1] + 3n
    or
    (n+1) U[n+1] = (n+1) U[n] + 3n U[n-1] + 2n
    And U[0] = U[1] = 0.
    """

    return sum(binomial(num, knum)
               * binomial(num - knum, knum)
               for knum in range(1, num // 2 + 1)) // 2

@cache
def binary_ones(num: int) -> int:
    """
    Number of 1's in all binary number <= n.
    """
    if num <= 1:
        return 0
    bnd = 1
    cnt = -1
    while bnd <= num:
        bnd *= 2
        cnt += 1
    bnd //= 2
    rem = num - bnd
    return 2 ** (cnt - 1) * cnt + rem + binary_ones(rem)

def naive_bounds(num: int) -> int:
    " For j=1,n/2, we have (j+1)^m >= binom(n,j) "

    lbounds = (ceil(log(binomial(num, jind)) / log(jind + 1))
        for jind in range(1, num // 2 + 1))
    return max(lbounds)

def bipartite_rhs(diam: int, dim: int) -> int:
    """
      upper = (D-3-k)/3, where k=0,1,-1 (3)
      = (D+1) mod 3 - 1
    """
    top = (diam + 1 - (diam + 1) % 3) // 3
    lead = dim * sum((_+1) ** (dim - 1) for _ in range(top))

    match diam % 3:
        case 0:
            rest = ((diam + 3) // 3) ** dim + (diam // 3) ** dim
        case 1:
            rest = 2 * ((diam + 2) // 3) ** dim
        case 2:
            rest = 2 * ((diam + 1) // 3) ** dim
    return lead + rest

def bipartite_bound(num: int, diam: int) -> int:
    """
      Use the method of Dankelmann et. al. to find
      a bound on the metric dimension of a bipartite graph.
    """
    dim = 1
    while True:
        if num <= bipartite_rhs(diam, dim):
            return dim
        else:
            dim += 1
