"""
Use cdd lib to find the facets of the following polyedra:

For each pair of positive integers n >= m > r,
Generate all 2n tuples, where the first n-tuple
has weight m, and the second has weight r and is disjoint
from the first.
is of the form m * (1,) + (n-m) * (0,)
"""
from typing import Iterable, Tuple, Callable
from itertools import product, combinations
import numpy as np
from cdd import Matrix, RepType, Polyhedron
VEC = Tuple[int, ...]

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
