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
from pysat.formula import IDPool

def partitions(num: int, parts: List[int]) -> Iterable[List[int]]:
    """
    Generate partitions of num into parts.
    """
    if len(parts) == 0:
        if num == 0:
            yield []
        return
    first = parts[0]
    for val in range(min(num, first) + 1):
        yield from ([val] + _ for _ in partitions(num - val, parts[1:]))

def find_sectors(mat: np.ndarray) -> List[List[int]]:
    """
    Find the sectors of an array
    """
    mval, nval = mat.shape
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

def generate_row(mat: np.ndarray) -> Iterable[Tuple[int, ...]]:
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

def weight_range(num: int, low: int, high: int) -> Iterable[Tuple[int, ...]]:
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
                          pool: IDPool) -> Iterable[List[int]]:
    """
    Generate all symmetry breakers of depth.
    """
    mval , num = mat.shape
    # The first row is all 0's.  It's required.
    ante = [- pool.id(('x', tuple(mat[_]))) for _ in range(1, mval)]
    # The list of candidates
    cand = list(generate_row(mat))
    yield ante + list(map(lambda _: pool.id(('x', _)), cand))
    if depth > 0:
        yield from chain(*(symmetry_breakers_sub(np.concatenate([mat, np.array(elt).reshape((1, -1))], axis = 0),
                                                 depth - 1,
                                                 pool)
                            for elt in cand))
    # forbid all elements or larger weight from the last
    if mval > 1:
        forbidden = {tuple(mat[_])
                     for _ in range(mval - 1)}
        bot = mat[-2].sum()
        top = mat[-1].sum() - 1

        for elt in weight_range(num, bot, top):
            if elt not in forbidden:
                yield ante + [-pool.id(('x', elt))]

def symmetry_breakers(num, depth: int, pool:IDPool) -> Iterable[List[int]]:
    yield [pool.id(('x', num * (0,)))]
    for elt in weight_range(num, num // 2 + 1, num):
        yield [-pool.id(('x', elt))]
    yield from symmetry_breakers_sub(np.zeros((1, num), dtype=np.int8), depth, pool)
    
