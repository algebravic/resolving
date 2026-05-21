"""
Use Linear Programming to check for resolvability
"""
from typing import Tuple
import numpy as np
import cvxpy as cp

INTS = (np.int8, np.int16, np.int32, np.int64)

def _check_mat(mat: np.ndarray) -> Tuple[int, int]:
    """
    Check if the matrix is 2d 0/1 integers.
    If so, return shape.
    If not, raise an error
    """
    if not (isinstance(mat, np.ndarray)
            and len(mat.shape) == 2
            and mat.dtype in INTS
            and ((mat == 0) | (mat == 1)).all()):
        raise ValueError("Matrix must be 0/1 integers")
    return mat.shape

def check_resolve(mat: np.ndarray,
                  integral: bool = False,
                  solver: str | None = None,
                  **kwds) -> Tuple[np.ndarray, np.ndarray]:
    """
    Check resolvability.
    """
    mdim, ndim = _check_mat(mat)
    xvar = cp.Variable(ndim, name='x', boolean = integral)
    yvar = cp.Variable(ndim, name='y', boolean = integral)
    objective = cp.Minimize(sum(xvar))
    constraints = [mat @ (xvar - yvar) == 0,
        xvar >= 0, yvar >= 0,
        xvar <= 1, yvar <= 1,
        xvar + yvar <= 1,
        sum(xvar) == sum(yvar),
        sum(xvar) >= 1]
    prob = cp.Problem(objective, constraints)
    if solver is not None:
        kwds['solver'] = solver
    result = prob.solve(**kwds)
    return result, (xvar.value, yvar.value)
    
