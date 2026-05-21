"""
  Try to describe the facets of the following polytope:

  P[n] is the convex hull of the following set of 0/1 vectors:

  v in {0,1}^(2n). All vectors are 0-based. For all j < n
  we have v[2j] + v[2j+1] <= 1.
  sum_j v[2j] = sum_j v[2j+1] >= 1.
"""
from typing import Iterable, Tuple, List
from  cdd.gmp import RepType, polyhedron_from_matrix, Polyhedron, matrix_from_array
from cdd.gmp import Matrix, copy_inequalities
from itertools import combinations
import numpy as np

def balanced_vectors(num: int) -> Iterable[Tuple[int, ...]]:
    """
      Produce all non-zero balanced vectors of dimension n.
      For all weights 1 <= k <= n/2, find all disjoint
      subsets of cardinality k in [n]
    """
    for wgt in range(1, 1 + num):

        for first in combinations(range(num), wgt):
            for second in combinations(set(range(num)).difference(first), wgt):
                vec = np.zeros((num,2), dtype=np.int8)
                vec[first,0] = 1
                vec[list(second), 1] = 1
                yield tuple(vec.reshape((-1,)).astype(int).tolist())

def balanced_constraints(num: int) -> Matrix:
    """
      Constraints of the form a^T x +  b >= 0

      Note: These constraints are *not* all the facet constraints
      for the balanced 0/1 vectors. There are a number of fractional
      vertices with denominator 2.
    """
    # construc the array
    myarr = []
    tmp = np.zeros((num, 2), dtype=int)
    tmp[:, 0] = 1
    tmp[:, 1] = -1
    myarr.append([0] + tmp.reshape((-1,)).astype(int).tolist())
    tmp[:, 1] = 0
    myarr.append([-1] + tmp.reshape((-1,)).astype(int).tolist())
    for ind in range(num):
        # x[i] >= 0
        for col in range(2):
            tmp = np.zeros((num, 2), dtype=int)
            tmp[ind, col] = 1
            myarr.append([0] + tmp.reshape((-1,)).astype(int).tolist())
        for col in range(2):
            tmp = np.zeros((num, 2), dtype=int)
            tmp[ind, col] = -1
            myarr.append([1] + tmp.reshape((-1,)).astype(int).tolist())
        # exclusion
        tmp = np.zeros((num, 2), dtype=int)
        tmp[ind] = -1
        myarr.append([1] + tmp.reshape((-1,)).astype(int).tolist())
    return matrix_from_array(myarr, lin_set = {0}, rep_type = RepType.INEQUALITY)

def check_symmetry(vecs: List[Tuple[int, ...]]) -> bool:

    """
      All elements should have the same length, which is even
      They should all be 0/1 vectors
      When reshaped into shape (-1,2) the row sums should be <=1
      and the column sums =.

      For symmetries, column swap should leave the set invariant
      and row rotation, and swapping first two rows
      since they are the generators of the symmetric group.
    """
    lens = set(map(len, vecs))
    if len(lens) != 1:
        return -2
    xnum = list(lens)[0]
    if xnum % 2 == 1:
        return -1
    num = xnum // 2
    rot = [num - 1] + list(range(num - 1))
    swap = [1, 0] + list(range(2, num))
    univ = set(vecs)
    failure = 0
    def fail(wvec: np.ndarray) -> bool:
        return tuple(wvec.reshape((-1,)).tolist()) not in univ
    for vec in vecs:
        # convert to numpy
        mat = np.array(vec, dtype=int).reshape((-1,2))
        failure += int((mat.sum(axis=1) > 1).any())
        failure += int((mat[:,0] - mat[:, 1]).sum() != 0)

        # Now check symmetries
        failure += int(fail(mat[rot]))
        failure += int(fail(mat[swap]))
        failure += int(fail(mat[: [1, 0]]))
    return failure

def balanced_polytope(num: int) -> Polyhedron:

    mat = matrix_from_array([[1] + list(_) for _ in balanced_vectors(num)],
        rep_type = RepType.GENERATOR)
    return polyhedron_from_matrix(mat)

def balanced_inequalities(num: int):

    ineq = copy_inequalities(balanced_polytope(num))
    lin = ineq.lin_set
    mat = np.array(ineq.array, dtype=int)
    for ind in range(mat.shape[0]):
        if ind in lin:
            continue
        # All pseudo boolean constraints
        # If all coefficients are 1 then its an atleast
        myvars = [(abs(int(val)), (idx + 1) * (2 * int(val > 0) - 1))
            for idx, val in enumerate(mat[ind, 1:])
            if val != 0]
        if len(myvars) <= 1:
            continue
        bound = int(sum([-_ for _ in mat[ind, 1:] if _ < 0])
            - mat[ind, 0])
        # Check if it's PB or >=
        if all((_[0] == 1 for _ in myvars)):
            yield ('CARD',) + tuple((_[1] for _ in myvars)) + ('>=', bound,)
        else:
            yield ('PB',) + tuple(myvars) + ('>=',) + (bound,)
