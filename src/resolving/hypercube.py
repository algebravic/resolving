"""
Resolving sets for the hypercube.
"""
from typing import Iterable, List, Tuple
from itertools import chain, product
import numpy as np
from .generate import weight_range

VECTOR = Tuple[int,...]

def parity_vectors(num: int, parity: int) -> Iterable[VECTOR]:
    """
    All even weight 0/1 vectors of length n.
    parity: 0/1
    """
    if num == 0:
        if parity == 0:
            yield tuple()
    else:
        for elt in range(2):
            for vec in parity_vectors(num - 1, elt ^ parity):
                yield (elt,) + vec

def subvectors(vec: np.ndarray) -> Iterable[VECTOR]:
    """
    The values of R[v,0].
    """
    num = vec.shape[0]
    tst = np.zeros(num, dtype=np.int8)
    where = vec == 1
    wgt = sum(where)
    swgt = wgt // 2
    for subone in chain(*(weight_range(wgt, 0, swgt - 1),
                          weight_range(wgt, swgt + 1, wgt))):
        tst[where] = np.array(subone, dtype=np.int8)
        for other in product(range(2), repeat=num - wgt):
            tst[~where] = np.array(other, dtype=np.int8)
            val = tuple(tst.tolist())
            yield val

def basic_resolving_sets(num: int) -> Tuple[VECTOR, Iterable[List[VECTOR]]]:
    """
    Generate all resolving sets of the hypercube.
    """
    for vec in parity_vectors(num, 0):
        if sum(vec) > 0:
            yield vec, subvectors(np.array(vec, dtype=np.int8))

def translated_resolving_set(tst: List[Tuple[int, ...]],
                             vec: np.ndarray) -> Iterable[Tuple[int, ...]]:
    """
    Yield v + R[0,u], only allowing vectors v < 1 ^ v
    """
    for elt in tst:
        gend = np.array(elt, dtype=np.int8) ^ vec
        if tuple(gend) < tuple(1 ^ gend):
            yield tuple(gend)
            
def hypercube_resolving_sets(num: int) -> Iterable[Iterable[VECTOR]]:
    """
    All resolving sets.
    """
    for vec, basic in basic_resolving_sets(num):
        where = np.array(vec, dtype=np.int8) == 1
        lbasic = list(basic)
        for trans in product(range(2), repeat=sum(vec)):
            tst = np.zeros(num, dtype=np.int8)
            tst[where] = list(trans)
            yield translated_resolving_set(lbasic, tst)
