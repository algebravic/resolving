"""
Greedy construction for a resolving set.
"""
from typing import Tuple
from itertools import chain
import numpy as np
from .generate import find_sectors, VECTOR
from .check import check_resolvable

def half_one(num: int) -> Tuple[int, ...]:
    """
    Return tuple of size num, the first half are 1
    and the remaining are 0
    """

    return (num // 2) * (1,) + (num - (num // 2)) * (0,)

def greedy(num: int, mnum: int) -> np.ndarray:
    """
    Use the greedy algorithm.
    mnum is the putative size.
    """

    val = np.zeros((mnum, num), dtype=np.int8)
    for ind in range(1, mnum):
        sect = find_sectors(val[:ind, :])
        # Find the sectors
        sizes = map(lambda _: _ // 2, map(len, sect))
        # Try to split each in 2
        val[ind, list(chain(*(places[: contrib]
                            for contrib, places in zip(sizes, sect))))] = 1
    return val

def check_greedy(num: int, mnum: int) -> Tuple[VECTOR, VECTOR] | None:
    """
    Compute the greedy solution and check it.
    """
    val = greedy(num, mnum)
    return check_resolvable(list(map(tuple, val)))
