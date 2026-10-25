"""
Use Ganak to count the number of m x n 0/1 matrices so that
1) Rows are sorted lexicographically
2) Columns are sorted lexicographically
3) Optionally rows weights are <= n/2
4) Optionally the first column = 0
"""
from typing import Dict, Any, Tuple, Iterable
from enum import Enum
from itertools import product
import numpy as np
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType
from .lex import lex_less
from .ganak import ganak_count_models

class Weights(Enum):

    HALF = 0
    ZEROED = 1
    NEVER = 2

def count_matrix_classes(mval: int, nval:int,
                         bound: Weights = Weights.NEVER,
                         encoding: str = 'totalizer',
                         verbose: int = 0,
                         debug: bool = False,
                         approx: bool = False,
                         options: Dict[str, Any] = {}):
    """
    First 
    """
    pool = IDPool()
    cnf = CNF()
    amat = np.array([[pool.id(('A', ind, jind)) for jind in range(nval)]
                     for ind in range(mval)], dtype=np.int64)
    # First row is non-zero
    cnf.append(list(map(int, amat[0])))
    for ind in range(mval - 1):
        cnf.extend(lex_less(pool, list(map(int, amat[ind])),
                            list(map(int, amat[ind + 1]))))
    for jind in range(nval - 1):
        cnf.extend(lex_less(pool, list(map(int, amat[:, jind])),
                            list(map(int, amat[:, jind + 1]))))
    
    match bound:
        case Weights.ZEROED:
            cnf.extend([[- int(_)] for _ in amat[:, 0]])
        case Weights.HALF:
            enctype = getattr(EncType, encoding, EncType.totalizer)
            for ind in range(mval):
                cnf.extend(CardEnc.atmost(lits = map(int, amat[ind]),
                                          bound = nval // 2,
                                          encoding = enctype,
                                          vpool = pool))
        case Weights.NEVER:
            pass
    # Now call Ganak
    return ganak_count_models(cnf, pool, projected_prefixes = ['A'],
                              verbose = verbose,
                              options = options,
                              debug = debug,
                              approx = approx)
            
def lex_bigger(val: Tuple[int, ...]) -> Iterable[Tuple[int, ...]]:
    """
    Generate all tuples of the same length which are > lex
    Assume that the generated string has equality for exactly
    k elements. Then the next element will need to be a 1 if in
    the original string it's a 0.
    """
    num = len(val)
    for ind in range(num - 1):
        if val[ind] == 0:
            fst = val[: ind] + (1,)
            yield from (fst + _ for _ in product(range(2), repeat = num - ind - 1))

def exhaust_sub(mval: int, nval: int):
    """
    All m x n 0/1 matrices with lex increasing rows
    and non-decreasing columns.

    Note the first row must then have an initial segment of 0's
    followed by all 1's.
    """
    if mval == 1:
        yield from (np.array([(_ * (0,) + (nval - _) * (1,))], dtype=int) for _ in range(nval))
    else:
        # recursively get a matrix with one less row, and add columns so that it's lex non-decreasing
        for lmat in exhaust_sub(mval -  1, nval):
            for nrow in lex_bigger(tuple(lmat[-1])):
                nmat = np.concatenate([lmat,[list(nrow)]])
                if all((tuple(nmat[:, _]) <= tuple(nmat[:, _ + 1])
                        for _ in range(nval - 1))):
                    yield nmat

def increasing(mat: np.ndarray):
    return all((tuple(mat[_]) < tuple(mat[_ + 1])
                for _ in range(mat.shape[0] - 1)))
    
def exhaust_survivors(mval: int, nval: int, bound: Weights = Weights.NEVER):
    """
    Exhaust over all matrices that are row and column lex sorted.
    Row by row.
    """
    yield from (_ for _ in exhaust_sub(mval, nval)
                if increasing(_) and increasing(_.T)
                and ((bound == Weights.NEVER)
                    or (bound == Weights.HALF
                    and (_.sum(axis=1) <= nval // 2).all())))
