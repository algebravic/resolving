"""
Implement snake lex
"""
from typing import Iterable, List
import numpy as np
from pysat.formula import IDPool
from .logic import CLAUSE
from .lex import lex_compare, Comparator

def zig_zag(mat: np.ndarray) -> Iterable[int]:
    """
    Zig zag from two rows
    """
    _, ndim = mat.shape
    for ind in range(ndim):
        yield int(mat[ind % 2, ind])
          
def snake_lex(pool: IDPool,
              mat: np.ndarray) -> Iterable[CLAUSE]:
    """
    Implement the Snake Lex constraint
    Input matrix contain the variable numbers
    corresponding to the unknowns.

    Warning: must convert np.int to int!
    """
    mdim, ndim = mat.shape
    # Column constraints
    # Two cases, mdim, even/odd
    for ind in range(0,mdim,2):
        if ind+1 < mdim:
            yield from lex_compare(pool,
                                   list(map(int,mat[:,ind])),
                                   list(map(int,mat[:,ind+1])),
                                   Comparator.LESSEQUAL)
        if ind+2 < mdim:
            yield from lex_compare(pool,
                                   list(map(int,mat[:,ind])),
                                   list(map(int,mat[:,ind+2])),
                                   Comparator.LESSEQUAL)
            yield from lex_compare(pool,
                                   reversed(list(map(int,mat[:,ind+1]))),
                                   reversed(list(map(int,mat[:,ind+2]))),
                                   Comparator.LESSEQUAL)
        if ind+3 < mdim:
            yield from lex_compare(pool,
                                   reversed(list(map(int,mat[:,ind+1]))),
                                   reversed(list(map(int,mat[:,ind+3]))),
                                   Comparator.LESSEQUAL)
        
    # Now the rows
    for ind in range(0,ndim,2):
        if ind+2 < ndim:
            yield from lex_compare(pool,
                                   list(zig_zag(mat[ind: ind+2])),
                                   list(zig_zag(mat[ind+1: ind+3])),
                                   Comparator.LESSEQUAL)
