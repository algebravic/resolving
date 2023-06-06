"""
Write a QBF model for the hypercube.
"""
from typing import Tuple, Iterable, List
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

def xy_pos(pool: IDPool, num: int,
           encoding: int = EncType.totalizer) -> FORMULA:
    """
    Positive restrictions for xy.
    """
    xvars = {ind : pool.id(('x', ind)) for ind in range(num)}
    yvars = {ind : pool.id(('y', ind)) for ind in range(num)}
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
    avars = {(ind, jind) : pool.id(('a', ind , jind)) for ind in range(bound-1)
             for jind in range(num)}
    xvars = {ind : pool.id(('x', ind)) for ind in range(num)}
    yvars = {ind : pool.id(('y', ind)) for ind in range(num)}
    bvars = {(ind, jind) : pool.id(('b', ind , jind)) for ind in range(bound-1)
             for jind in range(num)}
    cvars = {(ind, jind) : pool.id(('c', ind , jind)) for ind in range(bound-1)
             for jind in range(num)}
    # b = a /\ x, c = a /\ y
    for ind, jind in product(range(bound - 1), range(num)):
        yield from set_and(bvars[ind, jind], avars[ind, jind], xvars[jind])
        yield from set_and(cvars[ind, jind], avars[ind, jind], yvars[jind])

def bc_neg(pool: IDPool, num: int, bound: int,
           encoding: int = EncType.totalizer) -> Iterable[FORMULA]:
    """
    At least one rows has a nonzero dot product.
    """
    bvars = {(ind, jind) : pool.id(('b', ind , jind)) for ind in range(bound-1)
             for jind in range(num)}
    cvars = {(ind, jind) : pool.id(('c', ind , jind)) for ind in range(bound-1)
             for jind in range(num)}
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
    bvars = {(ind, jind) : pool.id(('b', ind , jind)) for ind in range(bound-1)
             for jind in range(num)}
    cvars = {(ind, jind) : pool.id(('c', ind , jind)) for ind in range(bound-1)
             for jind in range(num)}
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
    xvars = {ind : pool.id(('x', ind)) for ind in range(num)}
    yvars = {ind : pool.id(('y', ind)) for ind in range(num)}
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
                         bound = 2,
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
                    invert: bool = False,
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

    qbf.exists([pool.id(('a', ind, jind))
                for ind, jind in product(range(bound - 1), range(num))])
    a_restriction = list(a_pos(pool, num, bound, encoding = encoding, snake = snake))
    qbf.exists(qbf.unquantified(a_restriction))
    qbf.forall([pool.id(('x', jind)) for jind in range(num)])
    qbf.forall([pool.id(('y', jind)) for jind in range(num)])
    # B and C depend on X and Y, so they must be quantified after.
    if dependencies:
        for ind, jind in product(range(bound - 1), range(num)):
            qbf.dependency(pool.id(('b', ind, jind)),
                           [pool.id(('x', jind)),pool.id(('y', jind))])
            qbf.dependency(pool.id(('c', ind, jind)),
                           [pool.id(('x', jind)),pool.id(('y', jind))])
    else:
        qbf.exists([pool.id(('b', ind, jind))
                    for ind, jind in product(range(bound - 1), range(num))])
        qbf.exists([pool.id(('c', ind, jind))
                    for ind, jind in product(range(bound - 1), range(num))])

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
    X is a test vetctor, and F(A,X) is true if A@X = 0
    """
    pool = IDPool()
    # The B and C variables are auxilliary.
    encoding = getattr(EncType, encode, EncType.totalizer)
    qbf = QBF()

    qbf.forall([pool.id(('a', ind, jind))
                for ind, jind in product(range(bound - 1), range(num))])
    a_restriction = list(a_pos(pool, num, bound, encoding = encoding, snake = snake))
    qbf.exists(qbf.unquantified(a_restriction))
    qbf.exists([pool.id(('x', jind)) for jind in range(num)])
    qbf.exists([pool.id(('y', jind)) for jind in range(num)])
    xy_restriction = list(xy_pos(pool, num, encoding = encoding))
    qbf.exists(qbf.unquantified(xy_restriction))
    bc_def = list(bc_formula(pool, num, bound))
    qbf.exists(qbf.unquantified(bc_def))
    if dependencies:
        for ind, jind in product(range(bound - 1), range(num)):
            qbf.dependency(pool.id(('b', ind, jind)),
                           [pool.id(('x', jind)),pool.id(('y', jind))])
            qbf.dependency(pool.id(('c', ind, jind)),
                           [pool.id(('x', jind)),pool.id(('y', jind))])
    bc_restriction = list(bc_pos(pool, num, bound, encoding = encoding))
    qbf.exists(qbf.unquantified(bc_restriction))

    cnf = CNF()
    cnf.extend(a_restriction + bc_def + bc_restriction + xy_restriction)
    if verbose > 0:
        print(f"Census = {Counter(map(len,cnf.clauses))}")
    qbf.add_model(cnf)
    return qbf, pool
