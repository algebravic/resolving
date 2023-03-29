"""
Check if a collection of vectors is resolving.

According to Beardon: 'Resolving the Hypercube'
let Z be a set of +/-1 n-vectors.  Then Z resolves the n hypercube
if and only if the only solution to z[i] . w = 0, where all coordinates
of w are in {0,1,-1} is the 0 vector.  This is actually pretty straightforward.


We can do this by SAT solving with cardinality constraints.

Represent each coordinate of w by two boolean variables u,v
w = u - v, and u,v can't both be 1 simultaneously.

Then we have equations z[i] . u = z[i] . v, for all i.
Recall, that we know the values of z[i].  So we then add the
condition that u,v can't be all 0's.

We can check the lower bound if we use a QBF (quantified boolean formula)
solver

We want forall x exists y,z f(x,y,z)

where f(x,y) encodes the following:
the x variables will encode the putative resolving set,
possibly with symmetry breaking constraints.
the y,z variables will be distinct pairs that are not resolved by x.
"""

from typing import List, Tuple, Optional
from itertools import chain
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver
import numpy as np
from .generate import VECTOR
IDENT = None | Tuple[str, int]

def resolvable_model(potential: np.ndarray,
                     encode: str = 'pairwise') -> Tuple[CNF, IDPool]:
    """
    Check if a subset is resolvable
    """

    mval, nval = potential.shape
    pmat = 1 - 2 * potential
    cnf = CNF()
    pool = IDPool()

    encoding = getattr(EncType, encode, EncType.pairwise)
    # Exclusion
    for ind in range(nval):
        cnf.append([- pool.id(('x', ind)), - pool.id(('y', ind))])
    # Forbid 0
    cnf.append(chain(*((pool.id(('x', _)), pool.id(('y', _)))
                       for _ in range(nval))))
    for ind in range(mval):
        lits = list(chain(*((pmat[ind, _] * pool.id(('x', _)),
                             - pmat[ind, _] * pool.id(('y', _)))
                             for _ in range(nval))))
        cnf.extend(CardEnc.equals(lits = lits,
                                  bound = potential[ind].sum(),
                                  encoding = encoding,
                                  vpool = pool))
    return cnf, pool

def _getvec(val: List[IDENT], dim: int, stem: str) -> VECTOR:
    """
    Get the vector from the list of positive variables
    """
    value = [_[1] for _ in val if isinstance(_, tuple) and _[0] == stem]
    vec = np.zeros(dim, dtype=np.int8)
    vec[value] = 1
    return tuple(value)

def check_resolvable(points: List[VECTOR],
                     solver = 'cd15',
                     encode = 'pairwise') -> Tuple[VECTOR, VECTOR] | None:
    """
    Check if a set of vectors is a resolving set.
    If it is, return None, otherwise a countexample.
    """
    potential = np.array(points)
    _, nval = potential.shape
    cnf, pool = resolvable_model(potential, encode = encode)

    with Solver(name = solver, bootstrap_with = cnf, use_timer = True) as solve:
        status = solve.solve()
        if status:
            positive = [pool.obj(_) for _ in solve.get_model() if _ > 0]
            return _getvec(positive, nval, 'x'), _getvec(positive, nval, 'y')
        else:
            return None
