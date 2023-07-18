"""
Use Z3 to try to solve things.

Rant: Although Z3 has extensive documentation they
don't have a useful list of the operations available.

One just has to guess!

From
https://microsoft.github.io/z3guide/programming/Z3%20Python%20-%20Readonly/Introduction
    
Instead, Z3 provides special signed versions of arithmetical
operations where it makes a difference whether the bit-vector is
treated as signed or unsigned. In Z3Py, the operators <, <=, >,
>=, /, % and >>; correspond to the signed versions. The
corresponding unsigned operators are ULT, ULE, UGT, UGE, UDiv,
URem and LShR.
"""
from typing import Callable
from z3 import BitVec, BitVecSort, BitVecRef, LShR, ULT, ULE
from z3 import And, Or, Not, Lambda, Exists, ForAll, Lambda

FUN = Callable[[BitVecRef], BitVecRef]

def _check_width(xexpr: BitVecRef) -> int:
    """
    Check to make sure that the input is a bitvector expression
    and that the bitvector width is <= 32.
    """
    width = (xexpr.sort().size() if isinstance(xexpr, BitVecRef)
             else None)
    if width is None or width > 32:
        raise ValueError("Only allow bitvecs of length <= 32")
    return width

def lambda_popcount(width: int):
    pass

def naive_popcount(xexpr: BitVecRef) -> BitVecRef:
    """
    Naive Computation one bit at a time.
    """
    width = _check_width(xexpr)
    return (xexpr & 1) + sum(LShR(xexpr,_) & 1 for _ in range(1,width))

def popcount(xexpr: BitVecRef) -> BitVecRef:
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
    x1 = (xexpr & masks[0]) + (LShR(xexpr, 1) & masks[0])
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
        x3 = (x2 & masks[2]) + (LShR(x2, 4) & masks[2])
    # Now 8 bit fields added in parallel: max value fits in 5 bits
    if width <= 8:
        return x3 & 0x0F
        # x4 = x3
    else:
        x4 = x3 + LShR(x3, 8)

    if width <= 16:
        return x4 & 0x1F
        # x5 = x4
    else:
        x5 = x4 + LShR(x4, 16)

    return x5 & (0x0000003F & mask)


def z3_bv_setup(num: int, mnum: int,
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
    avars = [BitVec(f'a_{num}_{ind}', num)
             for ind in range(mnum - 1)]
    # Since they are ordered, we only need say that the first is nonzero
    # However, putting in the explicit condition helps
    # Every element of A must be nonzero and have weight <= n/2
    a_restrict = And(*(And(_ != 0, ULE(pop_fun(_), num // 2))
                       for _ in avars))
    # Order doesn't matter, so break symmetry by making them increasing
    a_ord = And(*(ULT(_[0], _[1]) for _ in zip(avars[:-1], avars[1:])))
    a_cond = And(a_restrict, a_ord)

    # The two x variables, and their conditions
    xvars = [BitVec(f'x_{num}_{_}', num) for _ in range(2)]
    
    disjoint_x = (xvars[0] & xvars[1]) == 0
    equal_card = pop_fun(xvars[0]) == pop_fun(xvars[1])
    # This is implied by disjx but might be hard to deduce
    x_restrict = And(*(And(ULE(pop_fun(_), num // 2), _ != 0)
                     for _ in xvars))
    x_ord = ULT(xvars[0], xvars[1]) # A simple symmetry breaker

    x_cond = And(disjoint_x , x_restrict , equal_card , x_ord)

    # Linking of the x variables and a variables
    resolved = Or(*(pop_fun(_ &  xvars[0]) != pop_fun(_ & xvars[1])
                    for _ in avars))

    return avars, xvars, a_cond, x_cond, resolved

def z3_bv_model(num: int, mnum: int,
                 pop_fun: FUN = popcount,
                 reverse: bool = True):
    avars, xvars, a_cond, x_cond, resolved = z3_bv_setup(
        num, mnum, pop_fun = pop_fun)

    if reverse:
        return Exists(avars, And(
            a_cond, ForAll(
                xvars, Or(Not(x_cond), resolved))))
    else:
        return ForAll(avars, Or(
            Not(a_cond), Exists(
                xvars, And(x_cond, Not(resolved)))))
    # Remember: to restrict a forall with condition c
    # Must do Or(Not(c), phi)
