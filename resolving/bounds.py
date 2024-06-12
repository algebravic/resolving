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
from typing import List
from functools import cache
from math import ceil, log
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

    sum_i (a[i]/d) * log(a[i]/d) = (1/d) sum_i a[i] log(a[i)) - (1/d) sum_i log(d)
       = (1/d) sum_i a[i] log(a[i)) - log(d)
    """
    
    denom = distr.sum()
    return (sum(map(entr, distr)) / denom - log(denom)) / log(2.0)

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

def entropy_list(dim: int) -> List[float]:

    return [log(binomial(dim, sval)) / max([binary_entropy(subset_distr(dim, dval, sval))
            for dval in range(1, (dim // 2) + 1)]) for sval in range(1, (dim // 2) + 1)]

def improved_bound(dim: int) -> float:

    entropies = entropy_list(dim)
    return ceil(max(entropies) / log(2))

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

def balanced(num: int) -> int:
    """
    The number of nonzero vectors with coordinates in 0/1/-1
    whose sum of coordinates is 0.
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
