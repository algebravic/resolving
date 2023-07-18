"""
Solve the metric dimension of the hypercube using SMT.
"""
from typing import Iterable, Callable, List, Tuple
from itertools import chain, product
from random import randint
from pysmt.shortcuts import Symbol, ForAll, Exists, And, Or, Not, Int
from pysmt.shortcuts import Equals, NotEquals, LE, GE, GT, LT, Plus, BVExtract 
from pysmt.shortcuts import is_sat, get_model, get_unsat_core
from pysmt.typing import INT, BVType
from pysmt.fnode import FNode
from pysmt.typing import _BVType

FUN = Callable[[FNode], FNode]

def _check_width(xexpr: FNode) -> int:
    """
    Check to make sure that the input is a bitvector expression
    and that the bitvector width is <= 32.
    """
    try:
        width = xexpr.bv_width()
    except AssertionError:
        width = None
    if width is None or width > 32:
        raise ValueError("Only allow bitvecs of length <= 32")
    return width

def naive_popcount(xexpr: FNode) -> FNode:
    """
    Naive Computation one bit at a time.
    """
    width = _check_width(xexpr)
    return (xexpr & 1) + sum((xexpr >> _) & 1 for _ in range(1,width))

def popcount(xexpr: FNode) -> FNode:
    """
    This only works if width <= 32

    From Hacker's Delight (RIP Hank W.)

    int pop1(unsigned x) {
        x -= ((x >> 1) & 0x55555555);
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
        x = (x + (x >> 4)) & 0x0F0F0F0F;
        x += (x >> 8);
        x += (x >> 16);    
        return x & 0x0000003F;
    }
    """
    width = _check_width(xexpr)
    mask = (1 << width) - 1
    masks = [_ & mask for _ in [0x55555555, 0x33333333, 0x0F0F0F0F]]
             
    # Pocketed calculation
    # First add, in parallel, every other bit
    x1 = (xexpr & masks[0]) + ((xexpr >> 1) & masks[0])
    # x1 = xexpr - ((xexpr >> 1) & (0x55555555 & mask))
    # Now 2 bit fields added in parallel
    if width <= 2:
        return x1 & masks[1]
        # x2 = (x1 & (0x33333333 & mask))
    else:
        x2 = (x1 & masks[1]) + ((x1 >> 2) & masks[1])
    # Now 4 bit fields added in parallel
    if width <= 4:
        return x2 & masks[2]
        # x3 = x2 & (0x0F0F0F0F & mask)
    else:
        x3 = (x2 & masks[2]) + ((x2 >> 4) & masks[2])
    # Now 8 bit fields added in parallel: max value fits in 5 bits
    if width <= 8:
        return x3 & 0x0F
        # x4 = x3
    else:
        x4 = x3 + (x3 >> 8)

    if width <= 16:
        return x4 & 0x1F
        # x5 = x4
    else:
        x5 = x4 + (x4 >> 16)

    return x5 & (0x0000003F & mask)

def smt_bv_setup(num: int, mnum: int,
                 pop_fun: FUN = popcount):
    """
    Use bitvector for the model.

    Inputs:
    num: the size of the bitvectors
    mnum: 1 + the number of bitvectors (1 is implicitly 0)
    pop_fun: the function which calculates popcount of a bitvector
    reverse: Whether to model Exists/Forall or Forall/Exists (latter for the UNSAT case)

    Output:
    the SMT formula.
    """
    # The a variables
    avars = [Symbol(f'a_{num}_{ind}', BVType(num))
             for ind in range(mnum - 1)]
    # Since they are ordered, we only need say that the first is nonzero
    # However, putting in the explicit condition helps
    # Every element of A must be nonzero and have weight <= n/2
    a_restrict = And(*(And(_.NotEquals(0), (pop_fun(_) <= num // 2))
                       for _ in avars))
    # Order doesn't matter, so break symmetry by making them increasing
    a_ord = And(*(_[0] < _[1] for _ in zip(avars[:-1], avars[1:])))
    a_cond = And(a_restrict, a_ord)

    # The two x variables, and their conditions
    xvars = [Symbol(f'x_{num}_{_}', BVType(num)) for _ in range(2)]
    
    disjoint_x = (xvars[0] & xvars[1]).Equals(0)
    equal_card = pop_fun(xvars[0]).Equals(pop_fun(xvars[1]))
    # This is implied by disjx but might be hard to deduce
    x_restrict = And(*(And(pop_fun(_) <= num // 2,
                           _.NotEquals(0))
                       for _ in xvars))
    x_ord = xvars[0] < xvars[1] # A simple symmetry breaker

    x_cond = And([disjoint_x , x_restrict , equal_card , x_ord])

    # Linking of the x variables and a variables
    resolved = Or(*(pop_fun(_ &  xvars[0]).
                    NotEquals(pop_fun(_ & xvars[1]))
                  for _ in avars))

    return avars, xvars, a_cond, x_cond, resolved

def transpose_matrix(avars) -> FNode:
    """
    Given a list of bitvectors, create new variables
    representing the transpose.

    This is the 'naive' method.
    """
    widths = set(map(_check_width, avars))
    if len(widths) != 1:
        raise ValueError("All variables must be the same width")
    num = widths.pop()
    mnum = len(avars)
    bvars = [Symbol(f'b_{num}_{ind}', BVType(mnum))
             for ind in range(num)]
    formula = []
    for ind, jind in product(range(mnum), range(num)):
        formula.append(BVExtract(avars[ind], jind, jind).Equals(
                          BVExtract(bvars[jind], ind, ind)))
    return bvars, And(formula)

def smt_bv_model(num: int, mnum: int,
                 pop_fun: FUN = popcount,
                 transpose: bool = True,
                 reverse: bool = True):
    avars, xvars, a_cond, x_cond, resolved = smt_bv_setup(
        num, mnum, pop_fun = pop_fun)
    if transpose:
        bvars, b_cond = transpose_matrix(avars)
        b_ord = And(*(_[0] < _[1] for _ in zip(bvars[:-1], bvars[1:])))
        avars += bvars
        a_cond = And(a_cond, b_cond, b_ord)
    if reverse:
        return Exists(avars, And(
            a_cond, ForAll(
                xvars, Or(Not(x_cond), resolved))))
    else:
        return ForAll(avars, Or(
            Not(a_cond), Exists(
                xvars, And(x_cond, Not(resolved)))))

def check_bv_model(num: int, mnum: int,
                   avalues: List[int],
                   pop_fun: FUN = popcount,
                   quantified: bool = False):

    avars, xvars, a_cond, x_cond, resolved = smt_bv_setup(
        num, mnum, pop_fun = pop_fun)

    a_check = And(a_cond, *[_[0].Equals(_[1])
                            for _ in zip(avars, avalues)])
    if not is_sat(a_check):
        return ('a_check', None)
    if quantified:
        formula = ForAll(xvars, And(a_check, x_cond, resolved))
    else:
        formula = And(a_check, x_cond, Not(resolved))
    model = get_model(formula)
    if quantified:
        return 'resolved' if model else 'unresolved'
    else:
        return (
            ('unresolved', list(map(model.get_py_value, xvars)))
            if model else ('resolved', None))
    # return (
    #     ('unresolved', list(map(model.get_py_value, xvars)))
    #     if model else ('resolved', get_unsat_core(formula)))

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
    xdomain = And(*[(xvars[_] >= Int(-1)) & (xvars[_] <= Int(1))
                    for _ in range(num)])
    xcond = xnonzero & xdomain & Equals(Plus(xvars),Int(0))

    adomain = And(*[And(*[(avars[ind][_] >= Int(0))
                          & (avars[ind][_] <= Int(1))
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


def check_popcount(width: int, fun: FUN = popcount, tries: int = 1):
    """
    Generate random integers, and check that the popcount
    formula really yields popcount.
    """
    top = (1 << width) - 1
    xvar = Symbol(f'x_{width}', BVType(width))
    for iteration in range(tries):

        value = randint(0, top)
        formula = (xvar.Equals(value)
                   & (fun(xvar).Equals(value.bit_count())))
        status = is_sat(formula)
        result = "Success" if status else "Failure"
        print(f"{result}: popcount({hex(value)}) ?= {value.bit_count()}")

def check_equivalent(fun1: FUN, fun2: FUN, width: int) -> None | int:
    """
    """
    xvar = Symbol(f'check_{width}', BVType(width))
    formula = fun1(xvar).NotEquals(fun2(xvar))
    model = get_model(formula)
    return model.get_py_value(xvar) if model else None
