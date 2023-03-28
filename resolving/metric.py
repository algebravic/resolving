"""
Class for metric dimension of a graph
"""
from typing import Tuple, List, Hashable, Iterable, Set
from itertools import combinations
import networkx as nx
import numpy as np

def distance_matrix(gph: nx.Graph) -> Tuple[np.ndarray, List[Hashable]]:
    """
    Create the all pairs distance matrix for the graph.
    """
    dist = nx.all_pairs_shortest_path_length(gph)
    ids, dst = zip(*dist)
    trans = list(ids)
    return np.array([[elt[ind] for ind in trans] for elt in dst],dtype=int), trans

class MetricDimension:
    """
    Calculation of Metric Dimension of an undirected graph.
    """

    def __init__(self, gph: nx.Graph):

        self._gph = gph.copy()
        self._dst, self._trans = distance_matrix(gph)
        self._num = self._dst.shape[0]

    def resolving_set(self) -> Iterable[Set[Hashable]]:
        """
        Return the resolving sets for a graph.
        """
        skel = np.arange(self._num)
        for ind, jind in combinations(range(self._num), 2):
            yield {self._trans[_] for _ in skel[self._dst[ind] != self._dst[jind]].tolist()}

    def __len__(self) -> int:
        """
        The size of the graph
        """
        return self._num

    @property
    def translation(self) -> List[Hashable]:
        """
        Translate ordinals to node names
        """
        return self._trans
