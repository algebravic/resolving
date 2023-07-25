"""
Use cdd lib to find the facets of the following polyedra:

For each pair of positive integers n >= m > r,
Generate all 2n tuples, where the first n-tuple
has weight m, and the second has weight r and is disjoint
from the first.
is of the form m * (1,) + (n-m) * (0,)
"""
from typing import Iterable, Tuple, Callable, List
from itertools import product, combinations
import numpy as np
from cdd import Matrix, RepType, Polyhedron

VEC = Tuple[int, ...]
CLAUSE = List[int]

def constant_weight(num: int, mnum: int) -> Iterable[VEC]:
    if num >= 0 and mnum == 0:
        yield num * (0,)
    elif num > 0 and mnum == num:
        yield num * (1,)
    elif 0 < mnum < num:
        for elt in range(2):
            for stem in constant_weight(num - 1, mnum - elt):
                yield (elt,) + stem

def big_points(num: int) -> Iterable[VEC]:

    for mnum in range(1, num // 2 + 1):
        for first in constant_weight(num, mnum):
            afirst = np.array(first, dtype=np.int8)
            place = afirst == 0
            for rnum in range(1, mnum):
                for second in constant_weight(num - mnum, rnum):
                    # intercalate
                    vec = np.zeros(num, dtype=np.int8)
                    vec[place] = second
                    yield first + tuple(vec.tolist())

def balanced(num: int) -> Iterable[VEC]:

    for mnum in range(1, num // 2 + 1):
        for skel in constant_weight(num, 2 * mnum):
            place = np.array(skel, dtype=np.int8) == 1
            for signs in constant_weight(2 * mnum, mnum):
                vec = np.zeros(num, dtype=np.int8)
                vec[place] = 2 * np.array(signs, dtype=np.int8) - 1
                yield tuple(vec.tolist())

def unbalanced_local(mnum: int) -> Iterable[VEC]:

    num = 2 * mnum
    for first in range(mnum + 1):
        for second in range(mnum + 1):
            if first != second:
                yield (first * (1,) + (mnum - first) * (0,)
                       + second * (1,) + (mnum - second) * (0,))
                
def unbalanced(mnum: int) -> Iterable[VEC]:

    num = 2 * mnum
    for first in product(range(2), repeat=mnum):
        for second in product(range(2), repeat=mnum):
            if sum(first) != sum(second):
                yield first + second
                    
def equal_points(num: int) -> Iterable[VEC]:

    for mnum in range(1, num // 2 + 1):
        for first in constant_weight(num, mnum):
            afirst = np.array(first, dtype = np.int8)
            place = afirst == 0
            for second in constant_weight(num - mnum, mnum):
                vec = np.zeros(num, dtype=np.int8)
                vec[place] = second
                yield first + tuple(vec.tolist())
                    
def make_poly(num: int,
              fun: Callable[[int], Iterable[VEC]] = big_points) -> Matrix:
    mat = Matrix([[1] + list(_) for _ in fun(num)],
                 number_type='fraction')
    mat.rep_type = RepType.GENERATOR
    return Polyhedron(mat)

def get_cutoffs(poly: Polyhedron) -> Iterable[Tuple[int, List[int]]]:
    """
    Get the cardinality constraints of the form:
    card(lits) >= bound
    """
    mat = poly.get_inequalities()
    lin = mat.lin_set
    for ind in range(mat.row_size):
        if ind in lin:
            continue
        coeffs = set(mat[ind][1:])
        if not coeffs.issubset([0,1,-1]):
            continue
        bound = (len([_ for _ in mat[ind][1:] if _ == -1])
                 - mat[ind][0])
        lhs = [(2*int(elt > 0) - 1) * (indx + 1)
               for indx, elt in enumerate(mat[ind][1:])
               if elt != 0]
        if len(lhs) > 1:
            yield (bound, lhs)
        

                                              
