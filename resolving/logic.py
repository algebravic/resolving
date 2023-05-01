"""
Various utilities for generating CNF
"""
from typing import Iterable, List, Tuple
from itertools import product, chain
from collections import Counter
from pysat.formula import IDPool
from pysat.card import CardEnc, EncType
from .lex import lex_compare, Comparator

CLAUSE = List[int]
FORMULA = List[CLAUSE]
MODEL = List[int]

def implies(pool: IDPool,
            form1: List[CLAUSE],
            form2: List[CLAUSE]) -> Iterable[CLAUSE]:
    """
    Clauses instantiating cl1 -> cl2.
    """
    avatars = []
    for clause in form1:
        # Make lit equisatisfiable with clause
        if len(clause) > 1:
            lit = pool._next()
            yield [-lit] + clause
            yield from ([lit, -elt] for elt in clause)
        else:
            lit = clause[0]
        avatars.append(- lit)
    yield from (avatars + elt for elt in form2)

def set_xor(lit: int, lit1: int, lit2:int) -> Iterable[CLAUSE]:
    """
    CNF for lit := (lit1 XOR lit2)
    """
    yield from ([-lit, lit1, lit2],
                [-lit, -lit1, -lit2],
                [lit, lit1, -lit2],
                [lit, -lit1, lit2])

def set_equal(lit: int, lit1: int, lit2:int) -> Iterable[CLAUSE]:
    """
    CNF for lit := (lit1 == lit2)
    """
    yield from set_xor(-lit, lit1, lit2)

def set_and(lit: int, lit1: int, lit2: int) -> Iterable[CLAUSE]:
    """
    lit <-> lit1 and lit2
    """
    yield from ([-lit, lit1],
                [-lit, lit2],
                [lit, -lit1, -lit2])

def negate(pool: IDPool, formula: FORMULA) -> Iterable[CLAUSE]:
    """
    Negate a formula.
    """
    yield from implies(pool, formula, [[]])

def special_less(pool: IDPool,
                 lit1: CLAUSE,
                 lit2: CLAUSE,
                 encode: str = 'totalizer') -> Iterable[CLAUSE]:
    """
    (wgt(lit1) >= wgt(lit2)) -> ((wgt(lit1) == wgt(lit2) and lit1 < lex lit2)
    """
    eqc = CardEnc.equals(lits = lit1 + [- _ for _ in lit2],
                         bound = len(lit2),
                         encoding = getattr(EncType, encode,
                                            EncType.totalizer),
                         vpool = pool).clauses
    eql = CardEnc.atmost(lits = [- _ for _ in lit1] + lit2,
                         bound = len(lit1),
                         encoding = getattr(EncType, encode,
                                            EncType.totalizer),
                         vpool = pool).clauses
    lexlt = list(lex_compare(lit1, lit2, Comparator.LESS, pool = pool))
    yield from implies(pool, eql, eqc + lexlt)
