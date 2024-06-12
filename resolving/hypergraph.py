"""
Generate the hypergraph for hypercube metric dimension problem.

Networkx apparently doesn't have a hypergraph, but we can realize
this using a Bipartite graph, where the left nodes are the
hyperedges, and the right nodes are the nodes in the hyperedge.

Each hyperedge corresponds to a balanced 0/1/-1 vector
where we break symmetries by having the first nonzero be 1.
"""
from typing import Iterable, List, Tuple
import numpy as np
from .graphs import constant_weight

VEC = Tuple[int, ...]

def sectors(weights: List[int]) -> Iterable[VEC]:
    """
    Generate weighted sectors 0/1 vectors.
    """
    if not (isinstance(weights, list)
            and all((isinstance(_, int) and _ >= 0
                     for _ in weights))):
        raise ValueError(
            "weights must be a list of nonnegative integers")

    num = sum(weights)
    for idxs in product(*(range(_ + 1) for _ in weights)):
        yield from product(map(lambda _: constant_weight(_[0], _[1]) for _ in zip(weights, idxs)))

def hypercube_hypergraph(num: int) -> Iterable[List[int]]:
    """
    Emit the hyperedges of the reduced metric dimension
    hypergraph
    """
    for wgt in range(num // 2):
        for first in constant_weight(num, wgt):
            vec = np.array(first, dtype=np.int8)
            place = vec == 0
            for second in constant_weight(num - wgt, wgt):
                svec = np.zeros(num, dtype=np.int8)
                svec[place] = 1
