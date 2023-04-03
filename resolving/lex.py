"""
Lexicographic CNF
"""
from typing import List
from pysat.formula import CNF, IDPool

def lex_compare(op1: List[int],
                op2: List[int],
                cnf: CNF,
                pool: IDPool,
                reverse: bool = False) -> Tuple[int, int]:
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
    cnf.append([equal]) # Things start out equal

    for elt1, elt2 in zip(op1, op2):

        nxt_equal = pool._next()
        # E' <==> (E /\ x1 /\ x2) \/ (E /\ -x1 /\ -x2)
        #     =   E /\ (x1 \/ -x2) /\ (-x1 \/ x2)
        # -E' <== (E /\ x1 /\ - x2)
        # G  <== (E /\ x1 /\ - x2)
        # -G  <== (E /\ - x1 /\ x2)
        # -E'  <== (E /\ - x1 /\ x2)
        cnf.extend([[-equal, elt1, elt2, nxt_equal],
                    [-equal, -elt1, -elt2, nxt_equal],
                    [-equal, -elt1, elt2, - ineq],
                    [-equal, -elt1, elt2, -nxt_equal],
                    [-equal, elt1, -elt2, ineq],
                    [-equal, elt1, -elt2, -nxt_equal]
                    ])
        equal = nxt_equal
    # check for unequal lengths
    if len(op1) > len(op2):
        cnf.append([-equal, ineq])
    elif len(op1) < len(op2):
        cnf.append([-equal, -ineq])
    return equal, ineq
