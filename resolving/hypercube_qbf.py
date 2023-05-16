"""
Write a QBF model for the hypercube.
"""
from typing import Tuple
from itertools import product
import numpy as np
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType
from .logic import set_and
from .qbf import QBF
from .symmbreak import double_lex, snake_lex

def hypercube_model(num: int, bound: int,
                    snake: bool = False,
                    encode: str = 'totalizer') -> Tuple[QBF, IDPool]:
    """
    Create a quantified boolean formula to test if there is a resolving
    set of size bound.
    """

    qbf = QBF()
    cnf = CNF()
    pool = IDPool()

    # Set up variables
    avars = {(ind, jind) : pool.id(('a', ind , jind)) for ind in range(bound-1)
             for jind in range(num)}
    xvars = {ind : pool.id(('x', ind)) for ind in range(num)}
    yvars = {ind : pool.id(('y', ind)) for ind in range(num)}
    bvars = {(ind, jind) : pool.id(('b', ind , jind)) for ind in range(bound-1)
             for jind in range(num)}
    cvars = {(ind, jind) : pool.id(('c', ind , jind)) for ind in range(bound-1)
             for jind in range(num)}

    encoding = getattr(EncType, encode, EncType.totalizer)
    # x and y can't be 1 simultaneously
    # b = a /\ x, c = a /\ y
    # Double lex, or snake lex for the a variables
    qbf.exists(list(avars.values()))
    qbf.forall(list(xvars.values()) + list(yvars.values()))

    for ind, jind in product(range(bound - 1), range(num)):
        cnf.extend(set_and(bvars[ind, jind], avars[ind, jind], xvars[jind]))
        cnf.extend(set_and(cvars[ind, jind], avars[ind, jind], yvars[jind]))
    cnf.append(list(xvars.values()) + list(yvars.values())) # prohibit 0
    cnf.append([avars[0, _] for _ in range(num)])
    cnf.append([avars[_, 0] for _ in range(bound - 1)])
    # (X,Y) sums to 0
    cnf.extend(CardEnc.equals(lits = list(xvars.values()) + [-_ for _ in yvars.values()],
                              bound = num,
                              encoding = encoding,
                              vpool = pool))
    # Lex constraints.
    amat = np.array([[avars[ind, jind] for jind in range(num)]
                     for ind in range(bound - 1)], dtype=int)
    breaker = snake_lex if snake else double_lex
    cnf.extend(list(breaker(pool, amat.T)))
    for ind in range(bound - 1):
        cnf.extend(CardEnc.atmost(lits =
                                  [avars[ind, _]
                                   for _ in range(num) ],
                                  bound = num // 2,
                                  encoding = encoding,
                                  vpool = pool))
    indicators = []
    # There is a row which resolves (X,Y)
    for kind in range(bound - 1):
        lits = ([bvars[kind, _]
                 for _ in range(num) ]
                + [- cvars[kind, _]
                   for _ in range(num)])
        indic1 = pool._next()
        indicators.append(indic1)
        for clause in CardEnc.atmost(
                lits = lits, bound = num - 1,
                encoding = encoding,
                vpool = pool).clauses:
            cnf.append([-indic1] + clause)
        indic2 = pool._next()
        indicators.append(indic2)
        for clause in CardEnc.atleast(
                lits = lits, bound = num + 1,
                encoding = encoding,
                vpool = pool).clauses:
            cnf.append([-indic2] + clause)
    cnf.append(indicators)

    qbf.add_model(cnf)
    return qbf, pool
