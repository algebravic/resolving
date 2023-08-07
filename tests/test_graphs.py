"""Make some tests"""

import networkx as nx
from pysat.formula import WCNF, IDPool
from resolving.resolve import resolve_maxsat
from resolving.graphs import johnson_graph

def graph(gph: nx.Graph, dim: int):

    result = resolve_maxsat(gph)
    assert (result is not None) and len(result) == dim

def cycle(num: int):
    """Test the metric dimension of an n-cycle

    The metric dimension of C_n is 2 if n>=3.

    Args:
    num: The size of the cycle
    """
    gph = nx.Graph()

    for ind in range(num):
        gph.add_edge(ind, (ind + 1) % num)

    answer = min(2, num-1)

    graph(gph, answer)

def complete(num: int):
    """Test the metric dimension of a complete graph

    Args:
        num: The degree of the graph
    """
    graph(nx.complete_graph(num), num - 1)

def johnson(num: int, knum: int):

    assert knum == 2
    gph = johnson_graph(num, knum)
    # Cf. Bailey et. al. Resolving sets for Johnson and Kneser graphs
    small = {3: 2, 4: 3, 5: 3}
    answer = small.get(num, (2 * (num - (num % 3))) // 3 + (num % 3))
    graph(gph, answer)
    
class TestGraph:

    def test_5cycle(self):
        cycle(5)

    def test_7cycle(self):
        cycle(7)

    def test_k5(self):
        complete(5)
        
    def test_k7(self):
        complete(7)
 
    def test_k15(self):
        complete(15)

    def test_j5(self):
          johnson(5,2)

    def test_j6(self):
          johnson(6,2)

    def test_j10(self):
          johnson(10,2)
