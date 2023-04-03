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
from typing import List, Tuple, Iterable, Dict
from itertools import product, chain
from more_itertools import bucket
import numpy as np

VECTOR = Tuple[int, ...]


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

    Ouput:
    An iterable of clauses.  These clauses are of two kinds
    1) Conditional on the inclusion of the elements of mat, the
    allowed next elements, one representative per symmetry class
    which preserves the original matrix under column permutations
    2) Conditional on the inclusion of the elements of mat,
    forbid those elements of smaller weight than the last.
    """
    mval , num = mat.shape
    # The first row is all 0's.  It's required.
    ante = list(map(tuple, mat.tolist()))
    # The list of candidates
    cand = list(generate_row(mat))
    yield (ante, cand)
    if depth > 0:
        yield from chain(*(symmetry_breakers_sub(
            np.concatenate([mat, np.array(elt).reshape((1, -1))], axis = 0),
            depth - 1, forbid=forbid, trace=trace)
                           for elt in cand))
    # forbid all elements of smaller weight from the last
    if forbid and mval > 1:
        bot = mat[-2].sum()
        botval = tuple(mat[-2])
        top = mat[-1].sum()
        topval = tuple(mat[-1])
        if trace > 0:
            print(f"weight_range({num}, {bot}, {top})")
        yield from ((ante + [elt], []) for elt in weight_range(num, bot + 1, top - 1))
        if bot < top:
            yield from ((ante + [elt], []) for elt in weight_range(num, bot, bot)
                        if elt > botval)

            yield from ((ante + [elt], []) for elt in weight_range(num, top, top)
                    if elt < topval)
        else:
            yield from ((ante + [elt], []) for elt in weight_range(num, bot, bot)
                        if botval < elt < topval)

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
    yield ([], [num * (0,)])
    # elements of greater weight than n/2 are forbidden
    for elt in weight_range(num, num // 2 + 1, num):
        yield ([elt], [])
    yield from symmetry_breakers_sub(np.zeros((1, num), dtype=np.int8),
                                     depth, forbid = forbid, trace=trace)
