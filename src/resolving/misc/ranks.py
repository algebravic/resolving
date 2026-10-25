r"""
calculate ranks of various subsets of balanced vectors.

Given m,n and d[1],...,d[m] for m <= n
Enumerate all cardinalities of intersection patterns
between subsets D[1],...,D[m] of {1,...,n}
with #D[i] = d[i].

For each S subset [m], let
a[S] = #intersection_{j=1}^m C^S[j](D[j])
= sum_{k=1}^n [k in intersecton_{j=1}^m C^S[j](D[j])]
= sum_{k=1}^n (1 - product_{j=1}^m (-1)^{D[j][k] + S[j]} )/2
= n - (1/2) sum_{k=1}^n product_{j=1}^m (-1)^{D[j][k] + S[j]}
where C is the complementation operator.

The fourier coefficient of a is

sum_{X subset [m]} a[S] (-1)^{#(X intersect S)}
= sum_{X subset [m]} 


Let T[j] = {S subset [m] : S[j] = 0}
U[i,j] = {S subset [m] : S[j] = S[i] = 0}
Then we have

d[i] = sum_{S in T[j]} a[S]

"""
from typing import Tuple, Iterable, List, Dict, Set
from itertools import combinations, accumulate
from math import log2, floor
from collections import defaultdict
import numpy as np
from sympy import binomial, prod
from ..bounds import binary_entropy, int_entropy

FLOAT = (float, np.float16, np.float32, np.float64)

VEC = Tuple[int, ...]

def delta_vectors(num: int, support: int, excess: int) -> Iterable[VEC]:
    """
    Given a dimension n,
    support: the number of non-zero coordinates (all +/- 1)
    excess: the sum of the coordinates.

    Generate all such vectors
    """
    if num == 0 and support == 0 and excess == 0:
        yield ()
    elif support <= num and abs(excess) <= num:
        for fst in [0,1,-1]:
            yield from ((fst,) + _ for _ in delta_vectors(num - 1, support - abs(fst), excess - fst))

def rank_check(num: int, weight: int) -> int:
    """
    Form the matrix of balanced vectors in dimension n
    of weight w (the weight is the number of +/- 1
    pairs).
    """
    mat = np.array(list(delta_vectors(num, 2 * weight, 0)))
    return np.linalg.matrix_rank(mat)

def order0(dim: int, sval: int, dval: int) -> List[int]:
    """
    Distribution of #(Y intersect D) where Y is uniform
    over all subsets of [n] of cardinality s
    and D is a subset of [n] of cardinality d
    Pr(#(D intersect Y) = j) unnormalized

    """
    min_size = min(dval, sval)
    return [binomial(dval, _) * binomial(dim - dval, sval - _) for _ in range(min_size + 1)]

def quadruples(sval: int, d1: int, d2: int):
    for d12 in range(min(d1, d2) + 1):
        min1 = min(d1 - d12, sval)
        min2 = min(d2 - d12, sval)
        min12 = min(d12, sval)
        yield from ((d12, j1, j12, j2)
                    for j1 in range(min1 + 1)
                    for j2 in range(min2 + 1)
                    for j12 in range(min12 + 1)
                    if j1 + j2 + j12 <= sval)

def multinomial(*arg: Tuple[int, ...]) -> int:
    tot = sum(arg)
    return prod((binomial(tot - _[1], _[0]) for _ in zip(arg, accumulate((0,) + arg))))

def marginalize(arg: Dict[Tuple[int, ...], int], coords: Set[int]) -> Dict[Tuple[int, ...], int]:

   res = defaultdict(int)
   for key, val in arg.items():
       remain = sorted(set(range(len(key))).difference(coords))
       res[tuple((key[_] for _ in remain))] += val
   return res

def pairwise_intersections(dim: int, sval: int, d1: int, d2: int) -> Dict[Tuple[int,...], int]:
    r"""
    Given a dimension, n, values of s <= n/2
    and sizes d1, d2 of subsets of [n],
    Let D1, D2 be random variables taking on subsets of sizes d1,d2 of n uniformly
    and Y a random variable taking on subsets of size s of n uniformly.

    We want to generate the unnormalized distribution of
    (X1, X2, X3, X4),
    where X1 = #((D1\D2) intersection Y)
    X2 = #((D2\D1) intersection Y)
    X3 = #(D1 intersection D2 intersection Y)
    X4 = #([n]\(D1 union D2) intersection Y)

    Let d1=#D1, d2=#D2, d12 = #(D1 intersection D2)

    The number of pairs (D1, D2) with a given d12
    is multinomial(d1-d12, d2-d12, d12, n-(d1+d2-d12))
    The number of quadruples for (X1=s1, X2=s2, X3=s3, X4=s4=(s-s1-s2-s3)) given d12
    is binomial(d1 - d12, s1)*binomial(d2-d12,s2)*binomial(d12, s3)*
    binomial(n-d1-d2+d12, s4)

    Note that the sets that the Xi are counting are disjoint.
    """
    pass

def quadruple_distr(dim: int, sval: int, d1: int, d2: int) -> List[int]:
    r"""
    Find the probability distribution (unnormalized)
    subsets of [n] of cardinalities d1 and d2 with intersection size d12
    intersected with a random subset of size s.
    """
    # Want (j1, j2) so that 0 <= j1 <= min1, 0 <= j2 <= min2, j1 + j2 <= s
    return {(d12, j1, j12, j2):
            multinomial(d1-d12,d2-d12,d12,dim-d1-d2+d12) *
            binomial(d1 - d12, j1) *
            binomial(d2 - d12, j2) *
            binomial(d12, j12) *
            binomial(dim - (d1 + d2 - d12), sval - (j1 + j2 + j12))
            for d12, j1, j12, j2 in quadruples(sval, d1, d2)}

def intersection_counts(dim: int, d1: int, d2: int) -> Dict[int, int]:
    r"""
    | D1 \ D2 | D2 \ D1 | [n]\(D1 cup D2)|
    """
    return {_: multinomial((d1 - _, d2 - _, _, dim - d1 - d2 + _))
            for _ in range(min(d1, d2) + 1)}

def all_entropies(dim: int, d1: int, d2: int):

                            
    for sval in range(1, (dim // 2 ) + 1):
        d1_ent = int_entropy(order0(dim, sval, d1))
        d2_ent = int_entropy(order0(dim, sval, d2))
        total_ent = d1_ent + d2_ent
        yield [total_ent - int_entropy(list(binary_information(dim, sval, d1, d2, _).values()))
                              for _ in range(min(d1, d2) + 1)]
        
def all_solutions(amat: np.ndarray, vec: np.ndarray,
                  verbose: int = 0) -> Iterable[Tuple[int,...]]:
    """
    Given an mxn 0/1 matrix: A, and an m vector of positive integers, v
    Given all solutions to Ax = v, where x>=0 is integral.
    No all 0 columns allowed in A
    """
    if verbose > 0:
        print(f"{amat}@x = {vec}")
    mval, nval = amat.shape
    fst = amat[:, 0]
    if (fst == 0).all():
        raise ValueError("Columns must be nonzero")
    if not ((fst == 0) | (fst == 1)).all():
        raise ValueError("Columns must be 0/1")
    if (vec < 0).any():
        raise ValueError(f"vec must be nonnegative {vec}")
    if nval == 1: # One column, vec must be constant
        if ((vec[fst == 0] == 0).all() and
              len(set(vec[fst == 1].tolist()))) == 1:
            yield (int(vec[fst == 1][0]),)
    else:
        # try all values in first column
        # if any coordinate of vec is 0
        # and a corresponding coord in first col is > 0
        # top = 0
        for ind in range(vec[fst == 1].min() + 1):
            yield from ((ind,) + _ for _ in all_solutions(amat[:,1:], vec - ind * fst,
                                                         verbose = verbose))
def intersection_matrix(nset: int) -> np.ndarray:

    # All intersection patterns
    mat = np.ones((nset + 1, 1 << nset), dtype = int)
    for ind in range(nset):
        insert = 1 << ind
        idx = [_ for _ in range(1 << nset)
               if (_ & insert) == 0]
        mat[ind, idx] = 0
    return mat
    
def isolutions(univ: int, subs: Tuple[int,...],
               verbose: int = 0) -> Iterable[Tuple[int, ...]]:
    mat = intersection_matrix(len(subs))
    yield from all_solutions(mat, np.array(subs + (univ,), dtype=int), verbose=verbose)

def entropy_solution(soln: List[Tuple[int,...]]) -> FLOAT:
    """
    Given a solution to the set intersection problem return the entropy
    """
    nset = floor(log2(len(soln[0])))
    mat = intersection_matrix(nset)
    soln_dict = {_[:-1]: multinomial(*(_[:-1])) for _ in soln}
    soln_mat = np.array([list(_) for _ in soln], dtype=object)
    # Accumulate all but the last set
    # For variable i, the indices, j, are the ones with (1<<i) & j != 0
    indices = [[_ for _ in range(1<<nset) if (1 << ind) & _ != 0
                and (1 << (nset - 1)) & _ != 0]
               for ind in range(nset - 1)]
    out = defaultdict(int)
    for sln in soln:
        contrib = multinomial(*(sln[:-1]))
        key = ()
        for vals in indices:
            key += (sum((sln[_] for _ in vals)),)
        out[key] += contrib
    return out
