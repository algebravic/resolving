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
