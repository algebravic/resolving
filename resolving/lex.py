"""
Lexicographic CNF
"""
from typing import List, Iterable
from enum import Enum
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType
from .logic import set_xor, implies, CLAUSE

class Comparator(Enum):
    """
    Comparison operators.
    """
    EQUAL = 1
    LESS = 2
    GREATER = 3
    LESSEQUAL = 4
    GREATEREQUAL = 5

def lex_compare(pool: IDPool,
                op1: List[int],
                op2: List[int],
                operator: Comparator,
                reverse: bool = False) -> Iterable[List[int]]:
    """
    Generate CNF for lexicographic comparison
    between two lists of literals.

    We use a finite state automaton with 3 states:
    G, L, E.  G and L are absorbing.
    The alphabet is pairs of bits.
    (G, L) can be represented by one variable
    whose value = 1 => G, 0 => L
    There will be a different replica of E for every
    time step (an initial value at time 0 is always true).
    The three possible values are E for equal, (-E /\ G) for greater
    And (-E /\ -G) for less than.
    """

    ineq = pool._next() # 1 = greater, 0 = less
    equal = pool._next()
    yield [equal] # Things start out equal

    for elt1, elt2 in zip(op1, op2):

        nxt_equal = pool._next()
        # E' <==> (E /\ x1 /\ x2) \/ (E /\ -x1 /\ -x2)
         #     =   E /\ (x1 \/ -x2) /\ (-x1 \/ x2)
        # -E' <== (E /\ x1 /\ - x2)
        # G  <== (E /\ x1 /\ - x2)
        # -G  <== (E /\ - x1 /\ x2)
        # -E'  <== (E /\ - x1 /\ x2)
        yield from ([-equal, elt1, elt2, nxt_equal],
                    [-equal, -elt1, -elt2, nxt_equal],
                    [-equal, -elt1, elt2, ineq],
                    [-equal, -elt1, elt2, -nxt_equal],
                    [-equal, elt1, -elt2, - ineq],
                    [-equal, elt1, -elt2, -nxt_equal]
                    )
        equal = nxt_equal
    # check for unequal lengths
    if len(op1) != len(op2):
        nxt_equal = pool._next()
        yield [-equal, -nxt_equal]
        if len(op1) < len(op2):
            yield [-equal, - ineq]
        else:
            yield [-equal, ineq]
        equal = nxt_equal
    match operator:
       case Comparator.EQUAL:
          yield [equal]
       case Comparator.LESS:
          yield from ([-ineq], [-equal])
       case Comparator.GREATER:
          yield from ([ineq], [-equal])
       case Comparator.LESSEQUAL:
          yield [-ineq, equal]
       case Comparator.GREATEREQUAL:
          yield [ineq, equal]

def lex_less(pool: IDPool,
             op1: List[int],
             op2: List[int]) -> Iterable[List[int]]:
    yield from lex_compare(pool, op1, op2, Comparator.LESS)
    
def standard_lex(pool: IDPool,
                 op1: List[int],
                 op2: List[int]) -> Iterable[List[int]]:
    """
    Lexicographic <
    """
    equals = []
    for lit1, lit2 in zip(op1, op2):
        lit = pool._next()
        yield from set_xor(lit, lit1, lit2)
        equals.append(lit)
    # lita <= litb: ~lita /\ litb
    for ind, (lit1, lit2) in enumerate(zip(op1, op2)):
        yield equals[: ind] + [-lit1, lit2]
    yield equals # make it strict

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
    lexlt = list(lex_less(pool, lit1, lit2))
    yield from implies(pool, eql, eqc + lexlt)
