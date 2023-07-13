"""
Solve the metric dimension of the hypercube using SMT.
"""
from itertools import chain
from pysmt.shortcuts import Symbol, ForAll, Exists, And, Or, Not, Int
from pysmt.shortcuts import Equals, NotEquals, LE, GE, Plus
from pysmt.typing import INT, BVType

def popcount(xexpr: BVType) -> INT:
    """
    This only works if width <= 32

    From Hacker's Delight:

    int pop1(unsigned x) {
        x -= ((x >> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        x = (x + (x >> 4)) & 0x0F0F0F0F;
        x += (x >> 8);
        x += (x >> 16);    
        return x & 0x0000003F;
    }
    """
    width = xexpr.bv_width()
    if width > 32:
        raise ValueError("Only allow bitvecs of length <= 32")
    mask = 2 ** width - 1

    x1 = xexpr - ((xexpr >> 1) & (0x55555555 & mask))
    x2 = ((x1 & (0x33333333 & mask))
          + ((x1 >> 2) & (0x33333333 & mask)))
    x3 = (x2 + (x2 >> 4)) & (0x0F0F0F0F & mask)

    if width > 8:
        x4 = x3 + (x3 >> 8)
    else:
        x4 = x3
    if width > 16:
        x5 = x4 + (x4 >> 16)
    else:
        x5 = x4
    return x5 & (0x0000003F & mask)

def smt_bv_model(num: int, mnum: int, reverse: bool = True):
    """
    Use bitvector for the model.
    """
    # The a variables
    avars = [Symbol(f'a_{ind}', BVType(num))
             for ind in range(mnum - 1)]
    # Since they are ordered, we only need say that the first is nonzero
    # However, putting in the explicit condition helps
    a_not_empty = And(*(_.NotEquals(0) for _ in avars))
    a_bound = And(*(popcount(_) <= num // 2 for _ in avars))
    a_ord = And(*(_[0] < _[1] for _ in zip(avars[:-1], avars[1:])))
    a_cond = a_not_empty & a_bound & a_ord

    # The two x variables, and their conditions
    xvars = [Symbol(f'x_{_}', BVType(num)) for _ in range(2)]

    disjoint_x = (xvars[0] & xvars[1]).Equals(0)
    # This is implied by disjx but might be hard to deduce
    bound_x = (popcount(xvars[0]) <= num // 2
              & popcount(xvars[1]) <= num // 2)
    x_not_empty = xvars[0].NotEquals(0) & xvars[1].NotEquals(0)
    equal_card = popcount(xvars[0]).Equals(popcount(xvars[1]))
    x_ord = xvars[0] > xvars[1] # A simple symmetry breaker
    x_cond = disjoint_x & bound_x & x_not_empty & equal_card & x_ord

    # Linking of the x variables and a variables
    resolved = Or(*(
        popcount(_ & xvars[0]).NotEquals(
            popcount(_ & xvars[1]))
        for _ in avars))

    outer, inner = (Exists, ForAll) if reverse else (ForAll, Exists)
    phi = resolved if reverse else Not(resolved)
    return outer(avars, a_cond & inner(xvars, x_cond & phi))
    
def smt_model(num: int, mnum: int):
    """
    SMT model for Forall A, Exists X such phi(A,X)
    This is the UNSAT version.
    A is an m by n array of 0/1
    All of the rows are nonzero

    X is an n-vector of 0/-1/1 whose sum is 0,
    and is not identically 0.

    phi(A,X) = A@X = 0, plus the domain constraints.
    
    """

    xvars = [Symbol(f'x_{ind}', INT) for ind in range(num)]
    avars = [[Symbol(f'a_{ind}_{jind}', INT) for jind in range(num)]
             for ind in range(mnum - 1)]
    xnonzero = Not(And(*[Equals(xvars[_],Int(0)) for _ in range(num)]))
    xdomain = And(*[GE(xvars[_],Int(-1)) & LE(xvars[_],Int(1))
                    for _ in range(num)])
    xcond = xnonzero & xdomain & Equals(Plus(xvars),Int(0))

    adomain = And(*[And(*[GE(avars[ind][_], Int(0)) & LE(avars[ind][_], Int(1))
                          for _ in range(num)])
                    for ind in range(mnum - 1)])
    
    anonzero = And(*[Not(And(*[Equals(avars[ind][_],Int(0))
                               for _ in range(num)]))
                     for ind in range(mnum - 1)])
                  
    acond = And(*[
        Equals(Plus([avars[ind][_] * xvars[_] for _ in range(num)]),Int(0))
        for ind in range(mnum - 1)])

    phi = acond & adomain & anonzero & xcond

    return list(chain(*avars)),  xvars, phi
