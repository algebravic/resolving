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

from typing import List, Tuple, Optional, Iterable
from itertools import chain
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver
import numpy as np
from .generate import VECTOR
IDENT = None | Tuple[str, int]


def _validate_lits(lits: List[int]) -> bool:
    """
    Validate that lits has the right form.
    """

    return (isinstance(lits, list)
            and all((isinstance(_, int) for _ in lits))
            and all((_ != 0 for _ in lits)))

def encode_equation(univ: Set[int], pos: Set[int],
                    variables: List[int],
                    idx: List[int],
                    encoding: int,
                    pool: IDPool) -> None:
    """
    Generate cardinality encodings.
    """

    lits = ([variables[_] for _ in pos]
            + [-variables[_] for _ in univ.difference(pos)])
    if len(univ) > 0:
        yield from CardEnc.equals(lits = lits,
                                  bound = len(univ),
                                  encoding = encoding,
                                  vpool = pool)

def resolvable_model(potential: np.ndarray,
                     pool: IDPool,
                     use_subset: bool = False,
                     encode: str = 'totalizer') -> Iterable[List[int]]:
    """
    Check if a subset is resolvable

    Sanity check:

    We want sum_i epsilon[i] x[i] = 0
    We want sum_i x[i]^epsilon[i] = #{i, epsilon[i] < 0}
    Let s[i] = +/- 1
    sum_i s[i] * (1 - 2 * x[i]) = sum_i s[i] - 2 * sum_i s[i] * x[i]
    sum_i s[i] * (1 - 2 * y[i]) = sum_i s[i] - 2 * sum_i s[i] * y[i]
    Difference is 2 * sum_i s[i] * (y[i] - x[i])
    """

    mval, nval = potential.shape
    pmat = (1 - 2 * potential).tolist()
    encoding = getattr(EncType, encode, EncType.totalizer)
    xvars = [pool.id(('x', _)) for _ in range(nval)]
    yvars = [pool.id(('y', _)) for _ in range(nval)]
    # Exclusion: both can't be 1 simultaneously
    yield from ([[- _[0], - _[1]] for _ in zip(xvars, yvars)])

    # Forbid 0
    yield from xvars + yvars

    for ind in range(mval):
        if use_subset:
            ones = {_ for _, val in enumerate(potential[ind]) if val == 1}
            zeroes = set(range(nval)).difference(ones)
            lits = ([xvars[_] for _ in ones]
                    + [-yvars[_] for _ in ones])
            if sum(ones):
                yield from CardEnc.equals(lits = lits,
                                          bound = sum(ones),
                                          encoding = encoding,
                                          vpool = pool)
            lits = ([xvars[_] for _ in zeroes]
                    + [-yvars[_] for _ in zeroes])
            if sum(zeroes):
                yield from CardEnc.equals(lits = lits,
                                          bound = sum(zeroes),
                                          encoding = encoding,
                                          vpool = pool)

        else:
            signs = pmat[ind]
            # Exactly half of these lits must be true
            lits = ([_[0] * _[1] for _ in zip(signs, xvars)]
                    + [ - _[0] * _[1] for _ in zip(signs, yvars)])

            if not _validate_lits(lits):
                print(f"lits = {lits}")
            yield from CardEnc.equals(lits = lits,
                                      bound = nval,
                                      encoding = encoding,
                                      vpool = pool)

def _getvec(val: List[IDENT], dim: int, stem: str) -> VECTOR:
    """
    Get the vector from the list of positive variables
    """
    value = [_[1] for _ in val if isinstance(_, tuple) and _[0] == stem]
    vec = np.zeros(dim, dtype=np.int8)
    vec[value] = 1
    return tuple(vec)


def check_resolvable(points: List[VECTOR],
                     solver = 'cd15',
                     encode = 'totalizer') -> Tuple[VECTOR, VECTOR] | None:
    """
    Check if a set of vectors is a resolving set.
    If it is, return None, otherwise a countexample.
    """
    potential = np.array(points)
    _, nval = potential.shape
    cnf = CNF()
    pool = IDPool()
    cnf.extend(resolvable_model(potential, pool, encode = encode))

    with Solver(name = solver, bootstrap_with = cnf, use_timer = True) as solve:
        status = solve.solve()
        if status:
            positive = [pool.obj(_) for _ in solve.get_model() if _ > 0]
            xvec = _getvec(positive, nval, 'x')
            yvec = _getvec(positive, nval, 'y')
            # xdist = (np.array(xvec).reshape((1, -1)) ^ points).sum(axis=1)
            # ydist = (np.array(yvec).reshape((1, -1)) ^ points).sum(axis=1)
            # print((xdist == ydist).all())
            return xvec, yvec
        else:
            return None

def random_check(num: int, target: int, times = 100) -> bool:
    """
    Generate a bunch of random sets of a given size.
    If any succeeds return True
    """

    for ind in range(times):
        tst = np.random.randint(0, 2, size = (target, num), dtype=np.int8)
        res = check_resolvable(tst)
        if res is None:
            print(f"There were {ind + 1} trials")
            return True
    return False
