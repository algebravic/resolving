"""
Solve the metric dimension of the hypercube using SMT.
"""
from typing import Iterable, Callable, List, Tuple
from itertools import chain, product, combinations
from random import randint
from math import log, ceil
from pysmt.shortcuts import Symbol, ForAll, Exists, And, Or, Not, Int
from pysmt.shortcuts import Equals, NotEquals, LE, GE, GT, LT, Plus, BVExtract, AllDifferent
from pysmt.shortcuts import BVNot, BVConcat, BVToNatural, BVZero, BVZExt, BVXor
from pysmt.shortcuts import is_sat, get_model, get_unsat_core
from pysmt.typing import INT, BVType, FunctionType
from pysmt.fnode import FNode
from pysmt.typing import _BVType

FUN = Callable[[FNode], FNode]

class TempName:

    ordinal = 0

    @staticmethod
    def name() -> str:

        tname = f'tmp_{TempName.ordinal}'
        TempName.ordinal += 1
        return tname

def _check_width(xexpr: FNode, maxwidth: int = 32) -> int:
    """
    Check to make sure that the input is a bitvector expression
    and that the bitvector width is <= 32.
    """
    try:
        width = xexpr.bv_width()
    except AssertionError:
        width = None
    if width is None or (maxwidth > 0 and width > maxwidth):
        raise ValueError(f"Only allow bitvecs of length <= {maxwidth}")
    return width

def min_width(width: int) -> int:

    return ceil(log(width + 1)/log(2))

def extend(xexpr: FNode, width: int) -> FNode:

    xwidth = _check_width(xexpr, maxwidth = 0)
    if xwidth > width:
        raise ValueError("x is longer than requested length")
    elif xwidth == width:
        return xexpr
    else:
        return BVZExt(xexpr, width - xwidth)

def recursive_popcount(xexpr: FNode) -> FNode:
    """
    Recursive splitting
    """
    width = _check_width(xexpr, maxwidth = 0)
    if width == 1:
        return xexpr
    nwidth = min_width(width)
    half = width // 2
    left = recursive_popcount(BVExtract(xexpr, 0, half - 1))
    right = recursive_popcount(BVExtract(xexpr, half, width - 1))
    return extend(left, nwidth) + extend(right, nwidth)
    
def naive_popcount(xexpr: FNode) -> FNode:
    """
    Naive Computation one bit at a time.
    """
    width = _check_width(xexpr, maxwidth = 0)
    nwidth = min_width(width)
    return sum((BVZExt(BVExtract(xexpr,_,_), nwidth-1) for _ in range(width)))

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
                 pop_fun: FUN = popcount,
                 xor_break: bool = False,
                 concat: bool = False):
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
    a_nonzero = And(*map(lambda _: _.NotEquals(BVZero(num)), avars))
    a_half_weight = And(*map(lambda _: pop_fun(_) <= num // 2,
                             avars))
    a_restrict = And(a_nonzero, a_half_weight)
    # Order doesn't matter, so break symmetry by making them increasing
    a_ord = And(*(_[0] < _[1] for _ in zip(avars[:-1], avars[1:])))
    a_cond = And(a_restrict, a_ord)

    # The two x variables, and their conditions
    xvars = [Symbol(f'x_{num}_{_}', BVType(num)) for _ in range(2)]
    
    disjoint_x = (xvars[0] & xvars[1]).Equals(BVZero(num))
    if concat:
        equal_card = pop_fun(BVConcat(xvars[0], BVNot(xvars[1]))).Equals(num)
    else:
        equal_card = pop_fun(xvars[0]).Equals(pop_fun(xvars[1]))
    # This is implied by disjx but might be hard to deduce
    x_nonzero = And(*map(lambda _: _.NotEquals(BVZero(num)), xvars))
    x_half_weight = And(*map(lambda _: pop_fun(_) <= num // 2,
                             xvars))

    x_restrict = And(x_nonzero, x_half_weight)

    x_ord = xvars[0] < xvars[1] # A simple symmetry breaker

    x_cond = And([disjoint_x , x_restrict , equal_card , x_ord])
    if xor_break:
        # add A[0] to others: (A[0], A[1] + A[0], ...)
        # add A[i] to others: (A[i], ..., A[j] + A[i], A[i] + A[0])
        # add A[0] values to others: (A[i], A[j], A[0])
        # yields a swap.
        # Recall that all A[i] are nonzero
        # So that A[i] + A[j] != A[i]
        # Add A[j] for j > 0 to all others
        extra = [avars[0] < BVXor(avars[0], avars[_])
            for _ in range(1, mnum - 1)]
        # Add A[0] to all others
        extra1 = [avars[1] < BVXor(avars[0], avars[1])]
        a_cond = And([a_cond] + extra + extra1)

    # Linking of the x variables and a variables
    if concat:
        unresolved = And(*(
            pop_fun(BVConcat(
                _ & xvars[0],
                BVNot(_ & xvars[1]))).
            Equals(num) for _ in avars))
    else:
        unresolved = And(*(
            pop_fun(
                _ &  xvars[0]).Equals(pop_fun(_ & xvars[1]))
            for _ in avars))

    return avars, xvars, a_cond, x_cond, unresolved

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
    bvars = [Symbol(f'b_{mnum}_{num}_{ind}', BVType(mnum))
             for ind in range(num)]
    formula = []
    for ind, jind in product(range(mnum), range(num)):
        formula.append(BVExtract(avars[ind], jind, jind).Equals(
                          BVExtract(bvars[jind], ind, ind)))
    return bvars, And(formula)

def smt_bv_model(num: int, mnum: int,
                 pop_fun: FUN = popcount,
                 transpose: bool = True,
                 concat: bool = False,
                 xor_break: bool = False,
                 reverse: bool = True):
    avars, xvars, a_cond, x_cond, unresolved = smt_bv_setup(
        num, mnum,
        pop_fun = pop_fun,
        concat = concat,
        xor_break = xor_break)
    if transpose:
        bvars, b_cond = transpose_matrix(avars)
        b_ord = And(*(_[0] < _[1]
                      for _ in zip(bvars[:-1], bvars[1:])))
        avars += bvars
        a_cond = And(a_cond, b_cond, b_ord)
        x_cond = And(x_cond,
                     pop_fun(xvars[0]) >= 2,
                     pop_fun(xvars[1]) >= 2)
    if reverse:
        return Exists(avars, And(
            a_cond, ForAll(
                xvars, Or(Not(x_cond), Not(unresolved)))))
    else:
        return ForAll(avars, Or(
            Not(a_cond), Exists(
                xvars, And(x_cond, unresolved))))

def check_bv_model(num: int, mnum: int,
                   avalues: List[int],
                   pop_fun: FUN = popcount,
                   quantified: bool = False):

    avars, xvars, a_cond, x_cond, unresolved = smt_bv_setup(
        num, mnum, pop_fun = pop_fun)

    a_check = And(a_cond, *[_[0].Equals(_[1])
                            for _ in zip(avars, avalues)])
    if not is_sat(a_check):
        return ('a_check', None)
    if quantified:
        formula = ForAll(xvars, And(a_check,
                                    Or(Not(x_cond), Not(unresolved))))
    else:
        formula = And(a_check, x_cond, unresolved)
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

# Axioms for Hamming Weight
def hamming_weight_and(fun, arg1, arg2):

    return And(fun(arg1) <= fun(arg1 & arg2),
               fun(arg2) <= fun(arg1 & arg2))
