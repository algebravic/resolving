"""
Write a QBF model for the hypercube.
"""
from typing import Tuple, Iterable, List, Dict
from collections import Counter
from itertools import product, chain
import numpy as np
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType
from .logic import set_and
from .qbf import QBF
from .symmbreak import double_lex, snake_lex

CLAUSE = List[int]
FORMULA = Iterable[CLAUSE]
VARDEF = Dict[int, int]
MATDEF = Dict[Tuple[int, int], int]

def large_or(pool: IDPool, formulas: Iterable[FORMULA]) -> FORMULA:
    """
    Using generated variables, return a formula
    so that it is satsifiable if at least one of the component
    formulas is satisfiable.
    """
    indicators = []
    for form in formulas:
        indic = pool._next()
        indicators.append(indic)
        yield from ([-indic] + _ for _ in form)
    yield indicators

def vec_def(pool: IDPool, num: int, stem: str) -> VARDEF:
    """
    Definition of X/Y vars.
    """
    return {ind: pool.id((stem, ind)) for ind in range(num)}

def mat_def(pool: IDPool, mdim: int, ndim: int, stem: str) -> MATDEF:
    """
    Define a matrix
    """
    return  {(ind, jind) : pool.id((stem, ind , jind)) for ind in range(mdim)
             for jind in range(ndim)}












def xy_pos(pool: IDPool, num: int,
           encoding: int = EncType.totalizer) -> FORMULA:
    """
    Positive restrictions for xy.
    """
    xvars = vec_def(pool, num, 'x')
    yvars = vec_def(pool, num, 'y')

    xyvars = list(xvars.values()) + list(yvars.values())
    yield from CardEnc.atleast(lits = xyvars,
                               bound = 4,
                               encoding = encoding,
                               vpool = pool).clauses
    xnyvars = list(xvars.values()) + [-_ for _ in yvars.values()]
    yield from CardEnc.equals(lits = xnyvars,
                              bound = num,
                              encoding = encoding,
                              vpool = pool).clauses
    yield from ([-xvars[ind], -yvars[ind]] for ind in range(num))

def a_pos(pool: IDPool, num: int, bound: int,
          snake: bool = False,
          encoding: int = EncType.totalizer) -> FORMULA:
    """
    The Positive restrictions on the A matrix.
    """
    avars = {(ind, jind) : pool.id(('a', ind , jind)) for ind in range(bound-1)
             for jind in range(num)}
    # Column/row 0 is not 0
    yield [avars[0, _] for _ in range(num)]
    yield [avars[_, 0] for _ in range(bound - 1)]
    # Double lex, or snake lex for the a variables
    amat = np.array([[avars[ind, jind] for jind in range(num)]
                     for ind in range(bound - 1)], dtype=int)
    breaker = snake_lex if snake else double_lex
    yield from breaker(pool, amat.T)
    for ind in range(bound - 1):
        yield from CardEnc.atmost(lits = [avars[ind, _] for _ in range(num)],
                                  bound = num // 2,
                                  encoding = encoding,
                                  vpool = pool).clauses

def bc_formula(pool: IDPool, num: int, bound: int) -> FORMULA:
    """
    The constraints on B and C.
    """
    avars = mat_def(pool, bound-1, num, 'a')
    xvars = vec_def(pool, num, 'x')
    yvars = vec_def(pool, num, 'y')
    bvars = mat_def(pool, bound-1, num, 'b')
    cvars = mat_def(pool, bound-1, num, 'c')
    # b = a /\ x, c = a /\ y
    for ind, jind in product(range(bound - 1), range(num)):
        yield from set_and(bvars[ind, jind], avars[ind, jind], xvars[jind])
        yield from set_and(cvars[ind, jind], avars[ind, jind], yvars[jind])

def bc_neg(pool: IDPool, num: int, bound: int,
           encoding: int = EncType.totalizer) -> Iterable[FORMULA]:
    """
    At least one rows has a nonzero dot product.
    """
    bvars = mat_def(pool, bound-1, num, 'b')
    cvars = mat_def(pool, bound-1, num, 'c')
    # There is a row which resolves (X,Y)
    for ind in range(bound - 1):
        lits = ([bvars[ind, _]
                 for _ in range(num) ]
                + [- cvars[ind, _]
                   for _ in range(num)])
        yield CardEnc.atmost(
            lits = lits, bound = num - 1,
            encoding = encoding,
            vpool = pool).clauses
        yield CardEnc.atleast(
            lits = lits, bound = num + 1,
            encoding = encoding,
            vpool = pool).clauses

def bc_pos(pool: IDPool, num: int, bound: int,
           encoding: int = EncType.totalizer) -> FORMULA:
    """
    At least one rows has a nonzero dot product.
    """
    bvars = mat_def(pool, bound-1, num, 'b')
    cvars = mat_def(pool, bound-1, num, 'c')
    # All rows fail to resolve (X,Y)

    for ind in range(bound - 1):
        lits = ([bvars[ind, _]
                 for _ in range(num) ]
                + [- cvars[ind, _]
                   for _ in range(num)])
        yield from CardEnc.equals(
            lits = lits, bound = num,
            encoding = encoding,
            vpool = pool).clauses

def xy_neg(pool: IDPool, num: int,
           encoding: int = EncType.totalizer) -> Iterable[FORMULA]:
    """
    The conflict formulas.
    """
    xvars = vec_def(pool, num, 'x')
    yvars = vec_def(pool, num, 'y')
    # Column/row 0 is not 0
    # Constraints on the X/Y variables (universally quantified)
    # Our problem is of the form exists A forall X,Y F(X,Y,A)
    # But we only want to quantify over a restricted set of X,Y
    # If G(X,Y) is the restriction for X,Y we should write
    # exists A forall X,Y ~G(X,Y) \/ F(X,Y,A)
    # complement is needed
    # totcnf will be a list of formulas
    # We want at leasat one of the formulas to be satisfied.
    # indic -> ~(\/ xy[i]) == indic -> /\ ~xy[i]
    # This formula is all of the xy variables are 0
    xyvars = list(xvars.values()) + list(yvars.values())
    yield CardEnc.atmost(lits = xyvars,
                         bound = 3,
                         encoding = encoding,
                         vpool = pool).clauses
    # (X,Y) sums to 0
    xnyvars = list(xvars.values()) + [-_ for _ in yvars.values()]
    yield CardEnc.atmost(lits = xnyvars,
                         bound = num - 1,
                         encoding = encoding,
                         vpool = pool).clauses

    yield CardEnc.atleast(lits = xnyvars,
                          bound = num + 1,
                          encoding = encoding,
                          vpool = pool).clauses
    # complement of x and y disjoint
    # indic -> ~(~x \/ ~y) <-> indic -> (x /\ y)
    # x and y can't be 1 simultaneously
    yield from ([[xvars[ind]], [yvars[ind]]] for ind in range(num))

def hypercube_model(num: int, bound: int,
                    dependencies: bool = False,
                    snake: bool = False,
                    verbose: int = 0,
                    encode: str = 'totalizer') -> Tuple[QBF, IDPool]:
    """
    Create a quantified boolean formula to test if there is a resolving
    set of size bound.
    """
    pool = IDPool()
    # acnf will be the formula of restrictions on the A variables
    # The B and C variables are auxilliary.
    encoding = getattr(EncType, encode, EncType.totalizer)
    qbf = QBF()
    xvars = vec_def(pool, num, 'x')
    yvars = vec_def(pool, num, 'y')
    avars = mat_def(pool, bound-1, num, 'a')
    bvars = mat_def(pool, bound-1, num, 'b')
    cvars = mat_def(pool, bound-1, num, 'c')

    qbf.exists(list(avars.values()))
    a_restriction = list(a_pos(pool, num, bound, encoding = encoding, snake = snake))
    qbf.exists(qbf.unquantified(a_restriction))
    qbf.forall(list(xvars.values()))
    qbf.forall(list(yvars.values()))
    # B and C depend on X and Y, so they must be quantified after.
    if dependencies:
        for ind, jind in product(range(bound - 1), range(num)):
            qbf.dependency(bvars[ind, jind], [xvars[jind]])
            qbf.dependency(cvars[ind, jind], [yvars[jind]])
    else:
        qbf.exists(list(bvars.values()))
        qbf.exists(list(cvars.values()))

    nxy_restriction = list(large_or(pool, xy_neg(pool, num, encoding = encoding)))
    bc_def = list(bc_formula(pool, num, bound))
    bc_restriction = list(large_or(
        pool, bc_neg(pool, num, bound, encoding = encoding)))
    qbf.exists(qbf.unquantified(nxy_restriction))
    qbf.exists(qbf.unquantified(bc_def))
    qbf.exists(qbf.unquantified(bc_restriction))

    cnf = CNF()
    # Exists A such that Forall X F(X,A)
    # F(X,A) = ((a restriction) /\ (bc_def) /\ bc_restriction) \/ (nxy_restriction))
    cnf.extend(list(
        large_or(pool, [a_restriction + bc_def + bc_restriction,
                  nxy_restriction])))
    if verbose > 0:
        print(f"Census = {Counter(map(len,cnf.clauses))}")
    qbf.add_model(cnf)
    return qbf, pool

def inverse_hypercube_model(num: int, bound: int,
                            dependencies: bool = False,
                            snake: bool = False,
                            verbose: int = 0,
                            encode: str = 'totalizer') -> Tuple[QBF, IDPool]:
    """
    For the UNSAT version:
    Forall A Exists X F(A,X)
    where A is an m by n matrix (with symmetry breaking constraints)
    X is a test vector, and F(A,X) is true if A@X = 0
    """
    pool = IDPool()
    # The B and C variables are auxilliary.
    encoding = getattr(EncType, encode, EncType.totalizer)
    qbf = QBF()

    xvars = vec_def(pool, num, 'x')
    yvars = vec_def(pool, num, 'y')
    avars = mat_def(pool, bound-1, num, 'a')
    bvars = mat_def(pool, bound-1, num, 'b')
    cvars = mat_def(pool, bound-1, num, 'c')

    qbf.forall(list(avars.values()))
    a_restriction = list(a_pos(pool, num, bound, encoding = encoding, snake = snake))
    qbf.exists(qbf.unquantified(a_restriction))
    qbf.exists(list(xvars.values()))
    qbf.exists(list(yvars.values()))
    xy_restriction = list(xy_pos(pool, num, encoding = encoding))
    qbf.exists(qbf.unquantified(xy_restriction))
    bc_def = list(bc_formula(pool, num, bound))
    qbf.exists(qbf.unquantified(bc_def))
    if dependencies:
        for ind, jind in product(range(bound - 1), range(num)):
            qbf.dependency(bvars[ind, jind], [xvars[jind]])
            qbf.dependency(cvars[ind, jind], [yvars[jind]])
    bc_restriction = list(bc_pos(pool, num, bound, encoding = encoding))
    qbf.exists(qbf.unquantified(bc_restriction))

    cnf = CNF()
    cnf.extend(a_restriction + bc_def + bc_restriction + xy_restriction)
    if verbose > 0:
        print(f"Census = {Counter(map(len,cnf.clauses))}")
    qbf.add_model(cnf)
    return qbf, pool
