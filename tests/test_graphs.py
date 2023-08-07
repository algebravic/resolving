"""Make some tests"""

import networkx as nx
from pysat.formula import WCNF, IDPool
from resolving.resolve import resolving_model, solve_maxsat, get_answer

def graph(gph: nx.Graph, dim: int):
    """Test if a graph has the indicated metric dimension

    Args:
    gph: A networkx Graph
    dim: The asserted metric dimension
    """
    cnf = WCNF()
    pool = resolving_model(gph, cnf)
    soln = solve_maxsat(cnf)
    result = get_answer(soln, pool) if soln is not None else None
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
