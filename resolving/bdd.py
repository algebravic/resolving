"""
Create a BDD for the not equals cardinality constraint

The states correspond to the possibile cardinalities after
reading the first k literals.  In general, they will be
[0,k], but some may be trimmed, since the final outcome
will be known, since we know the cardinality.
"""
from typing import List, Iterable
from pysat.formula import IDPool
from .logic import CLAUSE

def not_equal_bdd(num: int, cardinality: int) -> List[List[int]]:
    """
    Generate not equal CNF.

    Represent the BDD by straight line progam.
    values are (level, card, zero, one) where card is the cardinality
    that we've seen, zero/one are the next state numbers transitioning
    on a zero or one, resp. and level is the number of literals read.

    State -1 is True, and -2 is False.
    """

    mybdd = [[0,0,None,None]]
    stack = [0]

    states = dict()
    states[0,0] = 0

    while stack:

        nxt = stack[-1]
        stack = stack[: -1]
        level, card, zero, one = mybdd[nxt]
        if card + num - level < cardinality or card > cardinality:
            zero = -1
            one = -1
        else:
            if level == num - 1:
                zero = -1 if (card != cardinality) else -2
                one = -1 if (card != cardinality - 1) else - 2
            else:
                # Create two new states
                if (level + 1, card) in states:
                    zero = states[level + 1, card]
                else:
                    zero = len(mybdd)
                    states[level + 1, card] = zero
                    mybdd.append([level + 1, card, None, None])
                    stack.append(zero)
                if (level + 1, card + 1) in states:
                    one = states[level + 1, card + 1]
                else:
                    one = len(mybdd)
                    states[level + 1, card + 1] = one
                    mybdd.append([level + 1, card + 1, None, None])
                    stack.append(one)

        mybdd[nxt][2] = zero
        mybdd[nxt][3] = one

    return mybdd

def not_equal(pool: IDPool, lits: List[int], cardinality: int,
              res: int) -> Iterable[CLAUSE]:
    """
    BDD for not equal. Result is the variable res.
    """

    # Now produce the Tseitin encoding
    # Each state has a boolean variable saying whether we reach it
    mybdd = not_equal_bdd(len(lits), cardinality)
    variables = {_:pool._next() for _ in range(len(mybdd))}
    variables[-1] = res
    variables[-2] = -res
    for ind, (level, _, zero, one) in enumerate(mybdd):
        var = variables[ind]
        if level == 0:
            yield [var] # start at the root

        yield [-var, lits[level], variables[zero]]
        yield [-var, -lits[level], variables[one]]
