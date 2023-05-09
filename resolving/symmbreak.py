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
    for ind in range(0, mdim - 1, 2):
        yield from lex_compare(pool,
                               list(map(int,mat[ind])),
                               list(map(int,mat[ind+1])),
                               Comparator.LESS)
    for ind in range(0, mdim - 2, 2):
        yield from lex_compare(pool,
                               list(map(int,mat[ind])),
                               list(map(int,mat[ind+2])),
                               Comparator.LESS)
        yield from lex_compare(pool,
                               list(reversed(list(map(int,mat[ind+1])))),
                               list(reversed(list(map(int,mat[ind+2])))),
                               Comparator.LESS)
    for ind in range(0, mdim - 3, 2):
        yield from lex_compare(pool,
                               list(reversed(list(map(int,mat[ind+1])))),
                               list(reversed(list(map(int,mat[ind+3])))),
                               Comparator.LESS)

    # Now the rows
    for ind in range(0,ndim - 1):
        yield from lex_compare(pool,
                               list(zig_zag(mat[:, ind: ind+2], ind % 2)),
                               list(zig_zag(mat[:, ind: ind+2], (ind +1) % 2)),
                               Comparator.LESS)
