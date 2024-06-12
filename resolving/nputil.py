"""
Numpy utilities.
"""
from typing import Tuple, Type
import numpy as np

def _check_mat(amat: np.ndarray, name: str, dtypes: Tuple[Type, ...] = tuple()) -> Tuple[int, int]:
    """
    Check if the argument is a 2d numpy array.
    If not, raise an exception.
    If it is, return the dimensions.
    """
    if not (isinstance(amat, np.ndarray)
            and len(amat.shape) == 2
            and (len(dtypes) == 0 or amat.dtype in dtypes)):
        raise ValueError(f"The argument {name} is not a 2D matrix of required type")
    return amat.shape

def _check_vec(avec: np.ndarray, name: str, dim: int,
               dtypes: Tuple[Type, ...] = tuple()):
    if not (isinstance(avec, np.ndarray)
            and len(avec.shape) == 1
            and (len(dtypes) == 0 or amat.dtype in dtypes)
            and avec.shape[0] == dim):
        raise ValueError(f"The argument {name} is not a vector of required type and dimension")
