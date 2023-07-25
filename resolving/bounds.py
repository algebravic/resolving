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
import numpy as np
from sympy import binomial

def binomial_distr(weight: int) -> np.ndarray:
    """
    Unnormalized binomial distribution.
    """
    return np.array([binomial(weight, _) for _ in range(weight + 1)], dtype=np.float64)

def binary_entropy(distr: np.ndarray) -> float:
    """
    Given a distribution (all positive), compute
    the binary entropy.

    sum_i (a[i])/d * log(a[i]/d) = (1/d) sum_i a[i] log(a[i)) - (1/d) sum_i log(d)
    """
    
    normalize = distr / distr.sum()
    return - (1/(np.log(2))) * (np.log(normalize) * normalize).sum()

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
    return int(np.ceil(dim / binary_entropy( dim // 2)))

def unresolved(num: int, knum: int) -> int:
    """
    Calculate the number of equal weight pairs of vectors
    that are unresolved by a vector of weight k.
    Input:
       num: the length of the bit vectors
       knum: the weight that we're testing.
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
    return sum(binomial(knum, vnum)
               * binomial(knum - vnum, vnum)
               * binomial(num - knum, wnum - vnum)
               * binomial(num - knum - wnum + vnum, wnum - vnum)
               for wnum in range(1, (num // 2) + 1)
               for vnum in range(min(wnum, knum // 2) +1)) // 2

def unresolved_table(num: int) -> List[int]:
    """
    Table of number of unresolved elements for weights
    in [0, floor(n/2)].
    """
    return [unresolved(num, _) for _ in range((num // 2) + 1)]
