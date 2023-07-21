"""
Some graph families (not in networkx).
"""
from typing import Tuple, Iterable
from itertools import combinations
import networkx as nx

VEC = Tuple[int, ...]

def _validate(num: int, knum: int):
    """
    Validate the arguments.
    """
    if not(isinstance(num, int)
           and isinstance(knum, int)
           and num >= knum >= 1):
        raise ValueError(f"Parameters illegal n={num}, k={knum}")

def constant_weight(num: int, knum: int) -> Iterable[VEC]:
    """
    All 0/1 vectors of length n of weight k.
    """
    if knum == 0 and num >= 0:
        yield num * (0,)
    elif knum == num and num >= 0:
        yield num * (1,)
    elif 1 <= knum < num:
        for elt in range(2):
            for rest in constant_weight(num - 1, knum - elt):
                yield (elt,) + rest
                
def johnson_graph(num: int, knum: int) -> nx.Graph:
    """
    Nodes are 0/1 n-tuples of weight k.
    Two nodes are joined by an edge if their
    intersection as weight k-1
    """
    _validate(num, knum)
    gph = nx.Graph()
    for node1, node2 in combinations(constant_weight(num, knum), 2):
        if sum((_[0] & _[1] for _ in zip(node1, node2))) == knum - 1:
            gph.add_edge(node1, node2)
    return gph

def kneser_graph(num: int, knum: int) -> nx.Graph:
    """
    Nodes are 0/1 n-tuples of weight k.
    Two nodes are joined by an edge if their
    intersection as weight k-1
    """
    _validate(num, knum)
    gph = nx.Graph()
    for node1, node2 in combinations(constant_weight(num, knum), 2):
        if sum((_[0] & _[1] for _ in zip(node1, node2))) == 0:
            gph.add_edge(node1, node2)
    return gph
