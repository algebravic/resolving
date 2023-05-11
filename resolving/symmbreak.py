r"""
Implement snake lex
"""
from typing import Iterable, List
import numpy as np
from pysat.formula import IDPool
from .logic import CLAUSE
from .lex import lex_compare, Comparator

def double_lex(pool: IDPool,
               mat: np.ndarray) -> Iterable[CLAUSE]:
    """
    The Double Lex symmetry breaking constraint.
    """

    mdim, ndim = mat.shape
    for ind in range(mdim - 1):
        yield from lex_compare(pool,
                               list(map(int, mat[ind])),
                               list(map(int, mat[ind+1])),
                               Comparator.LESS)
    for jind in range(ndim - 1):
        yield from lex_compare(pool,
                               list(map(int, mat[:, jind])),
                               list(map(int, mat[:, jind+1])),
                               Comparator.LESS)

def zig_zag(mat: np.ndarray, par: int) -> Iterable[int]:
    """
    Zig zag from two rows
    """
    _, ndim = mat.shape
    for ind in range(ndim):
        yield int(mat[(ind + par) % 2, ind])
          
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
    # Count = m//2 + 2 * ((m-1) // 2) + (m-2) // 2
    # = 2 * (m // 2) + 2 * ((m-1) // 2) - 1
    # If m odd, this is 4*((m-1)/2) - 1 = 2 * m - 3
    # If m even, this is m + 2*((m-1)/2) - 1 = 2 * m - 2

    for ind in range(2 * ndim - 3):
        arow = ind // 2
        if ind % 4 < 2:
            yield from lex_compare(pool,
                                   list(map(int,mat[:,arow])),
                                   list(map(int,mat[:,arow+1+ind%2])),
                                   Comparator.LESS)
        else:
            yield from lex_compare(pool,
                                   list(reversed(list(map(int,mat[:,arow])))),
                                   list(reversed(list(map(int,mat[:,arow+1+ind % 2])))),
                                   Comparator.LESS)
            
    # Now the rows
    for ind in range(0,mdim - 1):
        yield from lex_compare(pool,
                               list(zig_zag(mat[ind: ind+2], 0)),
                               list(zig_zag(mat[ind: ind+2], 1)),
                               Comparator.LESS)
