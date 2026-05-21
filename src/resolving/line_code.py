"""
Create an SMTLIB encoding of the 'line code' problem:

We're given positive integers d, w, n with d < w/2.

We want to find a n by w matrix of 0/1 (or prove it doesn't exist).

A, such that w(A[i] ^ A[j]) >= d, and W(~A[i] ^ ~A[j]) >= d.

This is really an independent set problem on the graph
whose nodes are bit vectors of length w, where two nodes
are connected if and only if they're distance is < d
or > w - d.

1) if z is any bit vector of length w, then A is feasible
if and only if z ^ A is feasible, where this means
to XOR each row with z.  Thus, WOLOG we may assume that
A[1] = 0 (and we don't need to make it explicit).

2) Any permutation of the rows and/or columns of a feasible
matrix is feasible. Thus we may assume that the rows are
in strictly increasing lexicographic order, and the columns
are in lexicographic order (not necessarily strict).
"""
from typing import Iterable, Callable, List, Tuple
from itertools import chain, product, combinations
from random import randint
from math import log, ceil
from pysmt.shortcuts import Symbol, ForAll, Exists, And, Or, Not, Int
from pysmt.shortcuts import Equals, NotEquals, LE, GE, GT, LT, Plus, BVExtract, AllDifferent
from pysmt.shortcuts import BVNot, BVConcat, BVToNatural, BVZero, BVZExt, BVXor
from pysmt.shortcuts import is_sat, get_model, get_unsat_core
from pysmt.typing import INT, BVType
from pysmt.fnode import FNode
from pysmt.typing import _BVType
from .smt import FUN, recursive_popcount, naive_popcount, transpose_matrix
import cvxpy as cp

def line_code_sub(num: int, mnum: int, dval: int,
                  pop_fun: FUN = recursive_popcount,
                  row_symm: bool = True,
                  col_symm: bool = True) -> Iterable[FNode]:
    """
    Inputs:
      num: the size of the bitvectors
      mnum: 1 + the number of bitvectors (1 is implicitly 0)
      dval: The desired weight
      pop_fun: the functions calculating popcount
    """
    avars = [Symbol(f'a_{num}_{ind}', BVType(num))
             for ind in range(mnum - 1)]
    yield from map(lambda _: pop_fun(_) >= dval, avars)
    yield_from map(lambda _: pop_fun(_) <= num - dval, avars)
    for ind, jind in combinations(range(mnum - 1), 2):
        wgt = pop_fun(BVXor(avars[ind], avars[jind]))
        yield wgt >= dval
        yield wgt <= num - dval
    if row_symm:
        yield from (_[0] < _[1] for _ in zip(avars[:-1], avars[1:]))
    if col_symm:
        bvars, links = transpose_matrix(avars)
        yield links
        yield from (_[0] <= _[1] for _ in zip(bvars[:-1], bvars[1:]))

def line_code_formula(num: int, mnum: int, dval: int,
                      pop_fun: FUN = recursive_popcount,
                      row_symm: bool = True,
                      col_symm: bool = True) -> Tuple[FNode,
                          Dict[str, FNode]]
    """
    A formula for the line code problem
    """
    return And(*line_code_sub(num, mnum, dval,
                              pop_fun = pop_fun,
                              row_symm = row_symm,
                              col_symm = col_symm))

def line_code_lp(num: int, mdim: int, dval: int, **kwds):


    wvar = cp.Variable((mdim - 1, mdim - 1),
        name='d', integer = True)
    nonneg = [wvar[_] >= 0 for _ in product(range(mdim - 1), repeat =2)    ]
    upper = [wvar[_] <= num for _ in product(range(mdim - 1), repeat =2)]
    diag = [wvar[_, _] for _ in range(mdim - 1)]
    basic = ([_ >= dval for _ in diag]
        + [_  <= num - dval for _ in diag])
    bounda = [wvar[ind, jind] <= wvar[ind, ind]
              for ind, jind in combinations(range(mdim -1), 2)]
    boundb = [wvar[ind, jind] <= wvar[jind, jind]
              for ind, jind in combinations(range(mdim -1), 2)]
    bounds = bounda + boundb
    weights = {(ind, jind) : wvar[ind, ind] + wvar[jind, jind] -  2 * wvar[ind, jind] for ind, jind in combinations(range(mdim - 1), 2)}
    restrictl = [_ >= dval for _ in weights.values()]
    restrictu = [_ <= num - dval for _ in weights.values()]
    constraints = nonneg + upper + basic + bounds + restrictl + restrictu
    objective = cp.Minimize(sum(diag))
    prob = cp.Problem(objective, constraints)
    result = prob.solve(**kwds)
    return result, [_.value for _ in diag]
