"""
Given a 0/1 array, generate all possible additional rows using
symmetries.

The input input matrix will by m by n 0/1 matrix
The allowed permutations are those that respect column values.
That is, columns with the same value may be permuted.  We call each
such subset of columns a "sector".

In addition each row has weight <= n/2, and the weights are non-increasing.
The candidates for an additonal row will be all those with weight <=
that of the last row.  In addition, within each sector, all of the 1's come
first.  We also reject a row which is identical to any of the previous rows.
"""
from typing import List, Tuple, Iterable, Dict, Set, FrozenSet
from itertools import product, chain
from more_itertools import bucket
import numpy as np

VECTOR = Tuple[int, ...]

def _restricted(cap: int, parts: List[int]) -> Iterable[List[FrozenSet[int]]]:
    """
    Recursive sub.  Given a list of pairs (cap, set),
    where cap >= len(set), and union of all sets is is
    set(range(0, |union(set)|)).
    """
    if cap == 0:
        yield [frozenset() for _ in range(len(parts))]
        return
    for elt in _restricted(cap - 1, parts):
        # place cap in the appropriate set, if there's room
        for ind, val in enumerate(elt):
            if len(val) < parts[ind]:
                yield elt[: ind] + [val.union([cap - 1])] + elt[ind+1:]

def restricted_partitions(parts: List[int]) -> Iterable[List[FrozenSet[int]]]:
    """
    Given a list of positive integers, let n be their sum.
    This yields a sequence of all ordered set partitions whose
    cardinalities correspond to the parts in the input.
    """
    yield from _restricted(sum(parts), parts)

def set_partitions(num: int, nparts: int) -> Iterable[List[FrozenSet[int]]]:
    """
    Set partitions of range(n) into at most nparts.
    """
    if num == 0:
        yield [frozenset() for _ in range(nparts)]
        return
    for elt in set_partitions(num - 1, nparts):
        isempty = False
        for ind in range(nparts):
            if not isempty:
                yield elt[: ind] + [elt[ind].union([num])] + elt[ind+1: ]
            isempty = isempty or len(elt[ind]) == 0 # only add to the first empty

def partitions(num: int, parts: List[int]) -> Iterable[Tuple[int]]:
    """
    Generate partitions of num into parts.
    """
    if num == sum(parts):
        yield len(parts) * (1,)
    elif num <= sum(parts):
        for val in range(min(num, parts[0]) + 1):
            yield from ((val,) + _ for _ in partitions(num - val, parts[1:]))

def find_sectors(mat: np.ndarray) -> List[List[int]]:
    """
    Find the sectors of an array
    """
    _, nval = mat.shape
    dist = bucket(range(nval), key=lambda _: tuple(mat[:, _]))
    return [list(dist[_]) for _ in list(dist)]

def old_find_sectors(mat: np.ndarray) -> List[List[int]]:
    """
    Find the sectors in an array.
    """
    dct = dict()
    _, num = mat.shape
    for ind in range(num):
        key = tuple(mat[:, ind].tolist())
        if key not in dct:
            dct[key] = []
        dct[key].append(ind)
    return list(dct.values())

def generate_row(mat: np.ndarray) -> Iterable[VECTOR]:
    """
    Generate an additional row subject to the above restrictions.
    """
    mval, num = mat.shape
    nhalf = num // 2
    sectors = find_sectors(mat)
    cards = list(map(len, sectors))
    wgt = mat[-1].sum()
    # Now distribute things
    forbidden = {tuple(mat[_]) for _ in range(mval)}
    for wgt2 in range(wgt, nhalf + 1):
        for part in partitions(wgt2, cards):
            row = np.zeros(num, dtype=np.int8)
            row[list(chain(*(places[: contrib]
                             for contrib, places in zip(part, sectors))))] = 1
            # for contrib, places in zip(part, sectors):
            #     row[places[: contrib]] = 1
            # Get rid of a duplicate
            trow = tuple(row)
            if trow not in forbidden:
                yield trow

def weight_range(num: int, low: int, high: int) -> Iterable[VECTOR]:
    """
    A generator for all 0/1 vectors of length n, whose weight in in [l,h]
    """
    if low > high or num < low:
        return
    if num == low:
        yield num * (1,)
        return
    if num == 0:
        return
    for elt in weight_range(num - 1, low, high):
        yield (0,) + elt
    for elt in weight_range(num - 1, max(0, low - 1), high - 1):
        yield (1,) + elt

def hybrid_range(op1: VECTOR, op2: VECTOR) -> Iterable[VECTOR]:
    """
    Generate all element x, op1 < x < op2, where the total order
    is the hybrid order: x <= y if and only if either wt(x) < wt(y)
    or wt(x) = wt(y) and x <= y in lexicographic order.
    """
    bot = sum(op1)
    top = sum(op2)
    num = len(op1)

    yield from (elt for elt in weight_range(num, bot + 1, top - 1))
    if bot == top:
        yield from (elt for elt in weight_range(num, bot, bot)
                    if op1 < elt < op2)
    else:
        yield from (elt for elt in weight_range(num, bot, bot)
                    if op1 < elt)
        yield from (elt for elt in weight_range(num, top, top)
                    if elt < op2)
        
def symmetry_breakers_sub(mat: np.ndarray, depth: int,
                          forbid: bool = True,
                          trace: int = 0) -> Iterable[
                              Tuple[List[VECTOR], List[VECTOR]]]:

    """
    Generate all symmetry breakers of depth.

    Input:
    mat: an m by n 0/1 array.  Its rows are the included tuples in
    the resolving set.
    depth: a nonnegative int indicating how many more rows should be
    generated

    Output:
    An iterable of clauses.  These clauses are of two kinds
    1) Conditional on the inclusion of the elements of mat, the
    allowed next elements, one representative per symmetry class
    which preserves the original matrix under column permutations
    2) Conditional on the inclusion of the elements of mat,
    forbid those elements of smaller weight than the last.
    """
    mval , _ = mat.shape
    # The first row is all 0's.  It's required.
    ante = list(map(tuple, mat.tolist()))
    # The list of candidates
    cand = list(generate_row(mat))
    if trace > 0:
        print(f"There were {len(cand)} candidates at depth {mval}")
    yield ante, cand
    if depth > 0:
        yield from chain(*(symmetry_breakers_sub(
            np.concatenate([mat, np.array(elt).reshape((1, -1))], axis = 0),
            depth - 1, forbid=forbid, trace=trace)
                           for elt in cand))
    # forbid all elements of smaller weight from the last
    if forbid and mval > 1:
        yield from ((ante + [elt], []) for elt in hybrid_range(tuple(mat[-2]),
                                                               tuple(mat[-1])))

def symmetry_breakers(num, depth: int,
                      forbid: bool = True,
                      trace: int = 0) -> Iterable[
        Tuple[List[VECTOR], List[VECTOR]]]:

    """
    Yield symmetry breaking clauses for hypercube metric dimension
    Result will be an iterable of a pair of lists.  Each list's
    elements will be a 0/1 tuple of length n.  The first list will
    be negative literals, the second list the positive.
    """
    # all 0's always present
    if depth > 0:
        for elt in weight_range(num, num // 2 + 1, num):
            yield ([elt], [])
        if num % 2 == 0:
            for elt in weight_range(num - 1, num // 2 - 1, num // 2 - 1):
                yield([(1,) + elt], [])

        yield ([], [num * (0,)])
    # elements of greater weight than n/2 are forbidden
    # If even dimension, forbid the top half of the half weight elements
    yield from symmetry_breakers_sub(np.zeros((1, num), dtype=np.int8),
                                     depth, forbid = forbid, trace=trace)
