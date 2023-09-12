"""
Lexicographic CNF
"""
from typing import List, Iterable
from enum import Enum
from pysat.formula import IDPool
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
    The three possible values are E for equal, (-E & G) for greater
    And (-E & -G) for less than.
    """

    ineq = pool._next() # 1 = greater, 0 = less
    equal = pool._next()
    yield [equal] # Things start out equal

    for elt1, elt2 in zip(op1, op2):

        nxt_equal = pool._next()
        # E' <==> (E & x1 & x2) | (E & -x1 & -x2)
        #     =   E & (x1 | -x2) & (-x1 | x2)
        # -E' <== (E & x1 & - x2)
        # G  <== (E & x1 & - x2)
        # -G  <== (E & - x1 & x2)
        # -E'  <== (E & - x1 & x2)
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
    """ Lexicographic <lex"""
    yield from lex_compare(pool, op1, op2, Comparator.LESS)

def standard_lex(pool: IDPool,
                 op1: List[int],
                 op2: List[int],
                 strict: bool = True) -> Iterable[List[int]]:
    """
    See Knuth TAOCP 7.2.2.2 page 101
    x <=_lex y, each of length n:
    AND_{k=1}^{n-1} C[k] /\ F, where
      F =  (~x[n] \/ y[n] \/ ~a[n-1]), and
    C[k] = (~x[k] \/ y[k] \/ ~a[k-1]) /\ (~x[k] \/ a[k] \/ ~a[k-1]) /\ (y[k] \/ a[k] \/ ~a[k-1])
    where ~a[0] is omitted, i.e. a[0] is True

    If we want strict inequality we replade the term, F, by the term
    F' = (~x[n] \/ ~a[n-1]) /\ (y[n] \/ ~a[n-1]).

    Knuth says 'These formulas arise by considering the carries that can occur when
    (~x) + (1 or 0) is added to y.'  This brings up the possibility of using the fast
    lookahead carry for better propagation.

    Note that we can just treat zip(op1, op2).  If the lengths are different,
    then strict is always true, without adding the clause F'
    """
    nop1 = len(op1)
    nop2 = len(op2)
    num = min(nop1, nop2)
    if num == 1:
        if strict:
            if nop1 == nop2:
                # ~ [(1,0), (0,0), (1,1)]
                # ~([(1,0), (1,1)] = ~[(1,)]: 
                # ~x \/ (x /\ y) = (~x \/ x) /\ (~x \/ y) = (~x \/ y)
                # forbid (1,0)
                yield from [[-op1[0]], [op2[0]]]
        else:
            yield [-op1[0], op2[0]]
        return
    aux = [pool.id() for _ in range(num)]
    yield from [[-op1[0], op2[0]], [-op1[0], aux[0]], [op2[0], aux[0]]]
    for ind in range(1, num - 1):
        yield from [
            [-op1[ind], op2[ind], - aux[ind - 1]],
            [-op1[ind], aux[ind], - aux[ind - 1]],
            [op2[ind], aux[ind], -aux[ind - 1]]]
    if strict:
        if nop1 == nop2:
            yield from [[-op1[num - 1], -aux[num - 2]],
                        [op2[num - 1], -aux[num - 2]]]
    else:
        yield [-op1[num - 1], op2[num - 1], - aux[num - 2]]

def simple_lex(pool: IDPool,
               op1: List[int],
               op2: List[int],
               strict: bool = True) -> Iterable[List[int]]:
    """
    Lexicographic <
    """
    equals = []
    for lit1, lit2 in zip(op1, op2):
        lit = pool._next()
        yield from set_xor(lit, lit1, lit2)
        equals.append(lit)
    # lita <= litb: ~lita & litb
    for ind, (lit1, lit2) in enumerate(zip(op1, op2)):
        yield equals[: ind] + [-lit1, lit2]
    if strict:
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
