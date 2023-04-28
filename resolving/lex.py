"""
Lexicographic CNF
"""
from typing import List, Iterable
from enum import Enum
from pysat.formula import CNF, IDPool

class Comparator(Enum):
    """
    Comparison operators.
    """
    EQUAL = 1
    LESS = 2
    GREATER = 3
    LESSEQUAL = 4
    GREATEREQUAL = 5

def lex_compare(op1: List[int],
                op2: List[int],
                operator: Comparator,
                pool: IDPool,
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
