"""
Use a ping-pong solver for the metric dimension problem.

We will have two instances of SAT:

1) This instance will search for a sensing matrix of a given dimension.
An m by n sensing matrix will be a 0/1 matrix whose rows are nonzero
and whose columns are distinct and are lexicographically strictly increasing.
We also need to break symmetries:
If A is the matrix then if A[:k, i] == A[:k, j] for i < j and A[k,i] == 1
then A[k,j] == 1.
We will dynamically also add constraints coming from the second instance.

2) The second instance will take as assumptions a value for the A matrix
from instance (1).  Its variables will be x[i], y[i], i=1,...,n
with sum(x) + sum(y) >= 4, OR(x \/ y), ~x[i] \/ ~y[i] for i=1,...,n
and sum(A[k,i] * x[i], i=1,..,n) + sum(A[k,i] * ~y[i], i=1,..,n)
     >= 1 + sum(A[k,i], i=1,..n) for all k

We will also, optionally as solver (2) to provide at most r solutions.

"""
from typing import Iterable, List, Tuple
from itertools import product
import numpy as np
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from .lex import lex_compare, Comparator

def set_equal(lit: int, lit1: int, lit2:int) -> Iterable[List[int]]:
    """
    CNF for lit := (lit1 == lit2)
    """
    yield from ([-lit, lit1, lit2],
                [-lit, -lit1, -lit2],
                [lit, lit1, -lit2],
                [lit, -lit1, lit2])

def set_xor(lit: int, lit1: int, lit2:int) -> Iterable[List[int]]:
    """
    CNF for lit := (lit1 XOR lit2)
    """
    yield from ([lit, lit1, lit2],
                [lit, -lit1, -lit2],
                [-lit, lit1, -lit2],
                [-lit, -lit1, lit2])

def set_and(lit: int, lit1: int, lit2: int) -> Iterable[List[int]]:
    """
    lit <-> lit1 and lit2
    """
    yield from ([-lit, lit1],
                [-lit, lit2],
                [lit, -lit1, -lit2])

def negate(formula: List[List[int]], pool: IDPool) -> Iterable[List[int]]:
    """
    Negate a formula
    """
    negation = []
    for clause in formula:
        lit = pool._next()
        negation.append(- lit)
        yield [-lit] + clause
        yield from ([lit, -_] for _ in clause)
    yield negation # Can't be all true

def _check_diff_cols(mat: np.ndarray) -> List[Tuple[int, int]]:
    """
    Check if a matrix has distinct columns.
    """
    _, ndim = mat.shape
    return [(ind, jind) for jind in range(1, ndim) for ind in range(jind)
            if (mat[:, ind] == mat[:, jind]).all()]
    
def model1(dim: int, mdim: int) -> Tuple[CNF, IDPool]:
    """
    Use the Ping pong method.
    """
    cnf1 = CNF()
    pool1 = IDPool()


    # Non zero first row
    # Everything is increasing so others are nonzero
    cnf1.append([pool1.id(('A', 0, _)) for _ in range(dim)])
    for kind in range(mdim - 1):
        cnf1.extend(list(lex_compare([pool1.id(('A', kind, _)) for _ in range(dim)],
                                     [pool1.id(('A', kind + 1, _)) for _ in range(dim)],
                                     Comparator.LESS,
                                     # Comparator.GREATER,
                                     pool = pool1)))
    # Create the column constraints
    # Row -1 everything is equal
    cnf1.extend([[pool1.id(('E', -1, ind, jind))]
                 for jind in range(1, dim)
                 for ind in range(jind)])
    # The remaining rows
    for kind in range(mdim):
        for jind in range(1, dim):
            for ind in range(jind):
                # Z[k,i,j] <-> A[k,i] == A[k,j]
                cnf1.extend(set_equal(pool1.id(('Z', kind, ind, jind)),
                                      pool1.id(('A', kind, ind)),
                                      pool1.id(('A', kind, jind))))
                # E[k,i,j] <-> Z[k,i,j] /\ E[k-1,i,j]
                cnf1.extend(set_and(pool1.id(('E', kind, ind, jind)),
                                    pool1.id(('E', kind-1, ind, jind)),
                                    pool1.id(('Z', kind, ind, jind))))
                # E[k-1,i,j] /\ A[k,i] -> A[k,j]
                cnf1.append([-pool1.id(('E',kind-1, ind, jind)),
                             -pool1.id(('A', kind, ind)),
                             pool1.id(('A', kind, jind))])
    # Distinct columns
    cnf1.extend([[-pool1.id(('E', mdim-1, ind, jind))]
                 for jind in range(1, dim)
                 for ind in range(jind)])

    return cnf1, pool1

def model2(dim: int, mdim: int, encode='totalizer') -> Tuple[CNF, IDPool]:
    """
    The second model.
    """

    cnf2 = CNF()
    pool2 = IDPool()
    xlits = [pool2.id(('X', _)) for _ in range(dim)]
    ylits = [pool2.id(('Y', _)) for _ in range(dim)]

    # x and y can't both be 1
    cnf2.extend([[-_[0], - _[1]] for _ in zip(xlits, ylits)])

    cnf2.extend(CardEnc.atleast(lits = xlits + ylits,
                               bound = 4,
                               encoding = getattr(EncType, encode,
                                                  EncType.totalizer),
                               vpool = pool2))
    cnf2.extend(CardEnc.equals(lits = xlits + [-_ for _ in ylits],
                               bound = dim,
                               encoding = getattr(EncType, encode,
                                                  EncType.totalizer),
                               vpool = pool2))
    for kind in range(mdim):
        for ind in range(dim):
            cnf2.extend(set_xor(pool2.id(('B', kind ,ind)),
                                pool2.id(('A', kind, ind)),
                                pool2.id(('X', ind))))
            cnf2.extend(set_xor(pool2.id(('C', kind ,ind)),
                                pool2.id(('A', kind, ind)),
                                pool2.id(('Y', ind))))
        cnf2.extend(CardEnc.equals(
            lits=([pool2.id(('B', kind, _)) for _ in range(dim)]
                  + [- pool2.id(('C', kind, _)) for _ in range(dim)]),
            bound = dim,
            encoding = getattr(EncType, encode,
                               EncType.totalizer),
            vpool = pool2))
    return cnf2, pool2

def extract_mat(prefix: str, mdim: int, ndim: int,
                model: List[int], pool: IDPool) -> np.ndarray:
    """
    Get the A matrix
    """
    mat = np.zeros((mdim, ndim), dtype=np.int8)
    values = [pool.obj(_) for _ in model if _ > 0]
    for elt in values:
        if isinstance(elt, tuple) and elt[0] == prefix:
            mat[elt[1:]] = 1
    return mat

def getvec(prefix: str, dim: int, model: List[int], pool: IDPool) -> np.ndarray:
    """
    Get a vector
    """
    vec = np.zeros(dim, dtype=np.int8)
    values = [pool.obj(_) for _ in model if _ > 0]
    for elt in values:
        if isinstance(elt, tuple) and elt[0] == prefix:
            vec[elt[1]] = 1
    return vec
    
def ping_pong(dim: int, mdim: int,
              times: int = 1,
              verbose: int = 0,
              encode = 'totalizer',
              solver = 'cd15',
              **kwds) -> np.ndarray | None:
    """
    Ping Pong method.
    """
    cnf1, pool1 = model1(dim, mdim)
    if verbose > 0:
        print(f"model1 # clauses = {len(cnf1.clauses)}")
    cnf2, pool2 = model2(dim, mdim, encode=encode)
    if verbose > 0:
        print(f"model2 # clauses = {len(cnf2.clauses)}")
    solve1 = Solver(name = solver,
                    bootstrap_with = cnf1,
                    use_timer = True, **kwds)
    solve2 = Solver(name = solver,
                    bootstrap_with = cnf2,
                    use_timer = True, **kwds)
    while True:

        status1 = solve1.solve()
        if verbose > 0:
            print(f"status model1 = {status1}")
        if not status1:
            return None
        # Extract the A matrix
        amat = extract_mat('A', mdim, dim, solve1.get_model(), pool1)
        if verbose > 0:
            print(f"amat = {amat}")
        col_diffs = _check_diff_cols(amat)
        if col_diffs:
            raise ValueError(f"Columns not distinct: {col_diffs}!")
        bmat = 2 * amat - 1

        assumptions = [int(bmat[_]) * pool2.id(('A',) + _)
                       for _ in product(range(mdim), range(dim))]
        if verbose > 1:
            print(f"assumptions = {assumptions}")
        counter = 0
        for _ in range(times):
            status2 = solve2.solve(assumptions = assumptions)
            if verbose > 0:
                print(f"status model2 = {status2}")
            if not status2:
                break
            # get counterexample
            # forbid it in solve2
            # add a conflict in solve1
            counter += 1
            mod2 = solve2.get_model()
            xval = getvec('X', dim, mod2, pool2)
            yval = getvec('Y', dim, mod2, pool2)
            if verbose > 0:
                print(f"counterexample = {xval - yval}")
            # Forbid this value
            forbid = ([int(1-2*xval[_]) * pool2.id(('X', _)) for _ in range(dim)]
                      + [int(1-2*yval[_]) * pool2.id(('Y', _)) for _ in range(dim)])
            solve2.add_clause(forbid)
            # Now add the clauses to solve1
            xind = np.arange(dim)[xval == 1]
            yind = np.arange(dim)[yval == 1]
            for kind in range(mdim):
                lhs = ([pool1.id(('A', kind, _)) for _ in xind]
                       + [- pool1.id(('A', kind, _)) for _ in yind])
                equality = list(CardEnc.equals(lits = lhs,
                                               bound = len(yind),
                                               encoding = getattr(EncType, encode,
                                                                  EncType.totalizer),
                                               vpool = pool1))
                solve1.append_formula(negate(equality, pool1))
        if counter == 0: # success!
            return amat
