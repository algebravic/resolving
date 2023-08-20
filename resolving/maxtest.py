"""
Test using Max Sat to come up with good conflict clauses
"""
from typing import List, Dict, Tuple
import numpy as np
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from pysat.card import EncType, CardEnc
from .symmbreak import snake_lex, double_lex
from .util import makemat

def _pool_clone(pool:IDPool) -> IDPool:
    """
    Clone a pool
    """
    npool = IDPool(occupied=pool._occupied)
    for ind in range(1, pool.top + 1):
        if ind not in occupied:
            npool.id(pool.obj(ind))
    return npool

def unequal(pool: IDPool,
            cnf: CNF,
            lits: List[int],
            bound: int,
            encoding: int = EncType.seqcounter) -> List[int]:
    """
    Add an unequal constraint to the CNF
    and return a controlling literals.
    """
    indic1 = pool.id()
    indic2 = pool.id()
    cnf.append([-indic1, -indic2])
    cnf.extend([-indic1] + _
               for _ in CardEnc.atmost(
                   lits = lits,
                   bound = bound - 1,
                   encoding = encoding,
                   vpool = pool).clauses)
    cnf.extend([-indic2] + _
               for _ in CardEnc.atleast(
                   lits = lits,
                   bound = bound + 1,
                   encoding = encoding,
                   vpool = pool).clauses)
    return [indic1, indic2]

def amat_constraints(mdim: int, dim: int,
                     pool: IDPool,
                     cnf: CNF,
                     snake: int = 0,
                     encoding: int = EncType.seqcounter):
    """
    Add weight and symmetry constraints.
    """
    avar = makemat(pool, 'A', mdim, dim)
    # No row is all 0
    cnf.extend([[avar[_, ind] for ind in range(dim)]
                for _ in range(mdim)])
    for kind in range(mdim):
        cnf.extend(CardEnc.atmost(
            lits = [avar[kind, _] for _ in range(dim)],
            bound = dim // 2,
            encoding = encoding,
            vpool = pool))
    vmat = np.array([
        [avar[ind, jind] for jind in range(dim)]
        for ind in range(mdim)],
        dtype = int)
    breaker = snake_lex if snake > 0 else double_lex
    cnf.extend(list(breaker(pool,
                            vmat.T if snake > 1 else vmat)))
    cnf.extend(list(snake_lex(pool, vmat.T)))

def get_min_core(cnf: CNF,
                 assump: List[int],
                 solver: str,
                 verbose: int = 1) -> List[int] | None:
    """
    Get a minimal core if UNSAT else None
    """
    solveit = Solver(name = solver, bootstrap_with=cnf)
    status = solveit.solve(assumptions = assump)
    if verbose > 1:
        print(f"len(assump) = {len(assump)}")
        print(f"status = {status}")
    # Should be UNSAT
    if status:
        return None
    # Now try to minimize the core
    core = solveit.get_core()
    if verbose > 1:
        print(f"initial core size = {len(core)}")
    good = [] # These are necessary lits
    action = {True: 'keeping', False: 'discarding'}
    for ind in range(len(core)):
        status = solveit.solve(assumptions = good + core[ind+1:])
        if verbose > 1:
            print(f"{action[status]} {core[ind]}")
        if status:
            # core[ind] is needed
            good.append(core[ind])
    return good

def min_conflict(conflicts: List[np.ndarray],
                 amat: np.ndarray,
                 solver: str = 'cd153',
                 encode: str = 'seqcounter',
                 verbose: int = 0,
                 **kwds) -> List[int] | None:
    """
    Find a minimum cardinality set of values for the A matrix
    which are still 
    """
    mdim, dim = amat.shape
    if verbose > 1:
        print(f"considering {amat}")
    pool = IDPool()
    cnf = CNF()
    encoding = getattr(EncType, encode, EncType.seqcounter)
    avar = makemat(pool, 'A', mdim, dim)
    # amat_constraints(mdim, dim, pool, cnf,
    #                  snake = snake,
    #                  encoding = encoding)
    for conf in conflicts:
        if not ((amat @ conf) == 0).all():
            print(f"{conf} isn't valid.")
        indics = []
        pos = np.arange(dim)[conf == 1].tolist()
        neg = np.arange(dim)[conf == -1].tolist()
        for kind in range(mdim):
            lits = ([avar[kind, _] for _ in pos]
                + [-avar[kind, _] for _ in neg])
            indics += unequal(pool, cnf, lits, len(neg),
                encoding = encoding)
        cnf.append(indics)
    assump = [int(2 * amat[ind, kind] - 1) * avar[ind, kind]
        for ind in range(mdim) for kind in range(dim)]
    core = get_min_core(cnf, assump, solver, verbose=verbose)
    return ([pool.obj(abs(_))[1] for _ in core]
            if core else None)
