"""
Routines for entropy bounds.

We should have classes for random variables.

If our random variables are going to be subsets of [n].

We need operations like:

1) Subsets of [n]: range is 2^[n]
2) Union, Intersecton and complement.
2) cardinality of a random variables: range is nonNegative integers

Given an r.v. and an element of its range return the probability.

We need a good way to describe partitions of [n]: maybe ZDD's? Yes.


"""
from typing import Set, List, Tuple, FrozenSet, Dict, Iterable
from itertools import chain, combinations
from collections import Counter
import cvxpy as cp
import numpy as np
from sympy import binomial
from ..bounds import binary_entropy, subset_distr, distr_subsets

INT = int | np.int8 | np.int16 | np.int32 | np.int64
FLOAT = float | np.float16 | np.float32 | np.float64

class JointEntropy:
    """
    This class will be for collections of random variables.

    
    """
    def __init__(self, nvar):

        self._nvar = nvar # The number of random variables

    def entropy(self, subset: Set[int]) -> FLOAT:
        raise NotImplemented

    def entropy_effort(self, subset: Set[INT]) -> FLOAT:
        raise NotImplemented

def optimal_weights(covering: Dict[FrozenSet[INT], FLOAT]) -> Dict[FrozenSet[INT], FLOAT]:
    """
    Given entropies of subsets of a collection of variables
    Find the optimal weights for Fractional Shearer inequality.
    """
    rinds = dict(enumerate(covering.keys()))
    inds = {_[1]: _[0] for _ in rinds.items()}

    wgts = cp.Variable(len(rinds), name='w', nonneg=True)
    univ = set(chain(*map(list,covering.keys())))
    # make sure they're 
    # Form the covering weights
    cov_constraints = [
        sum((wgts[_] for _ in rinds if ind in rinds[_]), start=0.0) >= 1
        for ind in univ
        ]
    obj = cp.Minimize(sum((wgts[inds[_[0]]] * _[1] for _ in covering.items())))

    problem = cp.Problem(obj, cov_constraints)
    opt = problem.solve()
    # get the values
    return opt, {_: wgts.value[inds[_]] for _ in covering}

def test_order0(dim: int, dvals: List[int], sval: int):

    entrs = {frozenset([_[0]]): binary_entropy(subset_distr(dim, _[1], sval))
             for _ in enumerate(dvals)}
    return optimal_weights(entrs)

def v_vectors(mat: np.ndarray, rows: Iterable[int]) -> List[int]:
    "Given rows of mat, find partition count widths"
    return list(Counter(map(tuple, (mat[list(rows)].T).tolist())).values())

def row_order(mat: np.ndarray, tord: int, sval: int):
    """
    Given a 0/1 matrix, look at all possible intersection patterns
    with a uniform s-set.

    First find the partition for each of the nord subsets of rows.

    If the partition is (v_1,...,v_r) where sum_i v_i n

    then the probability is the multinomial coefficient (n;v_1,...,v_r)
    
    """

    return {frozenset(_): list(distr_subsets(v_vectors(mat, _), sval))
            for _ in combinations(range(mat.shape[0]), tord)}

def row_entropy(mat: np.ndarray, tord: int, sval: int):

    opt = optimal_weights({_[0]: binary_entropy(np.array(_[1])) for _ in
                           row_order(mat, tord, sval).items()})
    return opt[0]/(np.log(int(binomial(mat.shape[1], sval)))/np.log(2)), opt[0], opt[1]
