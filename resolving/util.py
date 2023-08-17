"""
SAT utilities for dealing with vectors and matrices.
"""
from typing import Iterable, List, Tuple, Dict
from itertools import product, combinations
import numpy as np
from pysat.formula import CNF, IDPool, WCNF
from .logic import MODEL, CLAUSE, FORMULA

def get_prefix(pool: IDPool, prefix: str, model: MODEL) -> List[Tuple[str, int,...]]:
    """
    Get all prefixes which are defined.
    """
    name_val = [(pool.obj(abs(_)), int(_ > 0)) for _ in model]
    return [(_[0][1:], _[1])
            for _ in name_val if isinstance(_[0], tuple) and _[0][0] == prefix]

def extract_mat(pool: IDPool, prefix: str, model: MODEL) -> np.ndarray:
    """
    Get the A matrix
    """
    values = sorted(get_prefix(pool, prefix, model))
    mdim = max((_[0][0] for _ in values)) + 1
    ndim = max((_[0][1] for _ in values)) + 1
    return  np.array([_[1] for _ in values], dtype=np.int8).reshape((mdim, ndim))

def getvec(pool: IDPool, prefix: str, model: MODEL) -> np.ndarray:
    """
    Get a vector
    """
    values = sorted(get_prefix(pool, prefix, model))
    # Check for consistency
    if {_[0][0] for _ in values} != set(range(len(values))):
        raise ValueError(f"{prefix}: {values}")
                                          
    return np.array([_[1] for _ in values], dtype=np.int8)

def makevec(pool: IDPool, prefix: str, size: int) -> Dict[int, int]:
    """
    Make a vector of values.
    """
    return {_ : pool.id((prefix, _)) for _ in range(size)}

def makemat(pool: IDPool, prefix: str, dim1: int, dim2: int) -> Dict[Tuple[int, int], int]:
    """
    Make a matrix of values.
    """
    return {_ : pool.id((prefix, _)) for _ in product(range(dim1),
                                                      range(dim2))}

def makecomp(pool: IDPool, prefix: str, dim1: int, dim2: int) -> Dict[Tuple[int, Tuple[int, int]], int]:

    return {_ : pool.id((prefix, _)) for _ in product(range(dim1),
                                                      combinations(range(dim2), 2))}



    
def _check_diff_cols(mat: np.ndarray) -> List[Tuple[int, int]]:
    """
    Check if a matrix has distinct columns.
    """
    _, ndim = mat.shape
    return [(ind, jind) for jind in range(1, ndim) for ind in range(jind)
            if (mat[:, ind] == mat[:, jind]).all()]
