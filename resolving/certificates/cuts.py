"""
Generate 0, 1/2 cuts to try to get a short proof that a polytope
contains no integral points, using the Farkas Lemma.
"""
from typing import Iterable, Tuple, List, Type
import numpy as np
import cvxpy as cp
from .nputil import _check_mat, _check_vec

INTS = (np.int8, np.int16, np.int32, np.int64)
CUT = Tuple[np.ndarray, int]
POLY = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def preprocess(amat: np.ndarray, bvec: np.ndarray,
               xstar: np.ndarray,
               slack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    Preprocess the reduced array/vector
    """
    amatc = amat.copy()
    bvecc = bvec.copy()
    good = (xstar != 0) & (amatc != 0).any(axis=0)
    amatc = amatc[:, good] # remove column with xtar = 0
    good_rows = ((amat != 0).any(axis=1) | (bvec != 0)) & (slack < 1)
    amatc = amatc[good_rows]
    bvecc = bvecc[good_rows]

    return amatc, bvecc

def zero_half(amat: np.ndarray, bvec: np.ndarray,
              cmat: np.ndarray, cvec: np.ndarray) -> Iterable[CUT]:
    """
    We have a system A x <= b, Cx = d, x >= 0,
    where A,C, b, d are all integral.

    We solve the LP: minimize sum_i x_i
    subject to x being in the above polyhedron
    (actually always a polytope).

    We use the algorithms of Koster et. al.
    to generated 0,1/2 cuts.

    We then add the cuts, and see if the polyhedron
    is empty.  If it is, we solve the dual
    to generate a Farkas vector.  If not,
      we then try to generate more cuts, etc.
    """
    mdim, ndim = _check_mat(amat, 'A', dtypes = INTS)
    _check_vec(bvec, 'b', mdim, dtypes=INTS)
    mdimp, ndimp = _check_mat(cmat, 'C', dtypes = INTS)
    _check_vec(dvec, 'd', mdimp, dtypes=INTS)
    if ndim != ndimp:
        raise ValueError(
            f"A and C are not compatible, {ndim} != {ndimp}")
    # Now form the LP
    xvar = cp.Variable(ndim, name='x')
    objective = cp.Minimize(sum(xvar))
    constraints = [amat @ xvar <= bvec,
                   cmat @ xvar == dvec,
                   xvar >= 0, yvar >= 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

def resolving_polyhedron(amat: np.ndarray) -> POLY:
    """
    A is a 0/1 matrix.  We have the polytope
    A (x-y) = 0, 0 <= x, y <= 1,
    sum(x) == sum(y), sum(x) + sum(y) >= 2
    for all j < n, we have
    AND (x[i] == 0 AND y[i] == 0, i < j) ==> (y[j] == 0)
    (x[j] == 0 OR y[j] == 0)
    """
    mdim, ndim = _check_mat(amat, 'A', dtypes = INTS)
    dtype = amat.dtype
    zblock = np.zeros((mdim, ndim), dtype=dtype)
    oblock = np.ones((1, ndim), dtype=dtype)
    fmat = np.concatenate([np.concatenate([amat, -amat], axis=1),
        np.concatenate([oblock, -oblock], axis=1)],
        axis = 0)
    fvec = np.zeros(mdim + 1, dtype=dtype)
    # Inequalities
    ineq = [
        (ndim * [-1] + ndim * [0], -1),
        (ndim * [0] + ndim * [-1], -1)]
    zeros = ndim * [0]
    for ind in range(ndim):
        chunk = ind * [0] + [1] + (ndim - ind - 1) * [0]
        # nchunk = ind * [0] + [-1] + (ndim - ind - 1) * [0]
        nshft = ind * [-1]
        zshft = (ndim - ind) * [0]
        # ineq.append((chunk + zeros, 1))
        # ineq.append((zeros + chunk, 1))
        ineq.append((chunk + chunk, 1))
        # Symmetry breaking
        ineq.append((nshft + zshft + nshft + [1] + zshft[1:], 0))
    rows, vec = zip(*ineq)
    cmat = np.array(rows, dtype=dtype)
    dvec = np.array(vec, dtype=dtype)
    return fmat, fvec, cmat, dvec

def normal_form(amat: np.ndarray, bvec: np.ndarray,
                cmat: np.ndarray, dvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    Input:
      A
      b
      C
      d
      defining the polyhedron Ax = b, Cx <= d
      return inequality form, by turing Ax=b into
      Ax <= b and -A x <= -b.
    """
    bigmat = np.concatenate([amat, -amat, cmat], axis=0)
    bigvec = np.concatenate([bvec, -bvec, dvec])
    return bigmat, bigvec
