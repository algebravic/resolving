"""
Various utilities for generating CNF
"""
from typing import Iterable, List, Tuple
from pysat.formula import IDPool
from pysat.card import CardEnc, EncType

CLAUSE = List[int]
FORMULA = List[CLAUSE]
MODEL = List[int]

def big_or(pool: IDPool,
           formulas: List[FORMULA]) -> Iterable[CLAUSE]:
    """
    yield a CNF which is the OR of a list of CNF.
    """
    result = []
    for form in formulas:
        avatars = []
        olit = pool._next()
        result.append(olit)
        for clause in form:
            if len(clause) > 1:
                lit = pool._next()
                yield [-lit] + clause
                yield from ([-elt, lit] for elt in clause)
            else:
                lit = clause[0]
        avatars.append(lit)
        # make olit <-> AND(avatars)
        yield from ([-olit, _] for _ in avatars)
        yield [olit] + [- _ for _ in avatars]
    yield result
    
def implies(pool: IDPool,
            form1: Iterable[CLAUSE],
            form2: Iterable[CLAUSE]) -> Iterable[CLAUSE]:
    """
    Clauses instantiating cl1 -> cl2.
    """
    avatars = []
    for clause in form1:
        # Make lit equisatisfiable with clause
        if len(clause) > 1:
            lit = pool._next()
            yield [-lit] + clause
            yield from ([-elt, lit] for elt in clause)
        else:
            lit = clause[0]
        avatars.append(- lit)
    yield from (avatars + clause for clause in form2)

def sum_not_zero(pool: IDPool, pos: List[int], neg: List[int],
                 encode: str = 'totalizer') -> Iterable[CLAUSE]:
    """
    Clauses for a sum not = 0.

    (S >= 0) ==> (S >= 1)
    """
    encoding = getattr(EncType, encode,
                       EncType.totalizer)
    gt0 = CardEnc.atleast(lits = pos + [- _ for _ in neg],
                          bound = len(neg),
                          encoding = encoding,
                          vpool = pool).clauses
    gt1 = CardEnc.atleast(lits = pos + [- _ for _ in neg],
                          bound = len(neg) + 1,
                          encoding = encoding,
                          vpool = pool).clauses
    yield from implies(pool, gt0, gt1)
    
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

def negate(pool: IDPool, formula: Iterable[CLAUSE]) -> Iterable[CLAUSE]:
    """
    Negate a formula.
    """
    yield from implies(pool, formula, [[]])

def parity(num: int, par: int) -> Iterable[Tuple[int, ...]]:
    """
    Generate all odd/even parity of length n.
    """
    if num == 0:
        if par == 0:
            yield tuple()
        return
    yield from ((0,) + _ for _ in parity(num - 1, par))
    yield from ((1,) + _ for _ in parity(num - 1, 1 - par))

def parity_constraint(lits: List[int], par: int) -> Iterable[List[int]]:
    """
    Parity constrained clauses.
    """
    num = len(lits)
    # Forbid opposite parity elements
    yield from ( ( (1-2 * _[0]) * _[1] for _ in zip(cons, lits))
                 for cons in parity(num, 1 - par))

def pre_post(pre: List[CLAUSE],
             post: List[CLAUSE],
             pool: IDPool) -> Iterable[CLAUSE]:
    """
    Generate clauses for pre ==> post
    """
    cvars = []
    for clause in pre:
        if len(clause) == 1:
            cvars.append(- clause[0])
        else:
            new_var = pool._next()
            cvars.append(- new_var)
            yield [-new_var] + clause
            yield from ([new_var, -_] for _ in clause)
    yield from (cvars + _ for _ in post)

def equiv(lft: List[CLAUSE],
          rgt: List[CLAUSE],
          pool: IDPool) -> Iterable[CLAUSE]:
    """
    Generate clauses for pre <==> post
    """
    yield from pre_post(lft, rgt, pool)
    yield from pre_post(rgt, lft, pool)
