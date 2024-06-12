"""
Calculate the maximum entropy distribution of a polytope contained
in [0,1]^n.
"""
from typing import Tuple, List
import numpy as np
import cvxpy as cp
from .nputil import _check_mat, _check_vec

def maxent(amat: np.ndarray, bvec: np.ndarray,
           cmat: np.ndarray, dvec: np.ndarray) -> np.ndarray:
    """
    Ax = b, C x <= d
    """
    mdim, ndim = _check_mat(amat, 'A')
    _check_vec(bvec, 'b', mdim)
    mdimp, ndimp = _check_mat(cmat, 'C')
    _check_vec(dvec, 'd', mdimp)
    if ndim != ndimp:
        raise ValueError(
            f"A and C are not compatible, {ndim} != {ndimp}")
    xvar = cp.Variable(ndim, name='x')
    objective = cp.Maximize(cp.sum(cp.entr(xvar)))
    constraints = [amat @ xvar == bvec,
                   cmat @ xvar <= dvec]
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True)
    return xvar.value
