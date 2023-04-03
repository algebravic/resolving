"""
Use MAXSAST to find a minimal cardinality resolving set for a finite
unidirected connected graph G.
"""
from itertools import combinations, product, chain
from typing import Set, Tuple, List, Hashable, Iterable, Union
from functools import partial
import numpy as np
import networkx as nx
from pysat.formula import CNF, WCNF, IDPool
from pysat.examples.rc2 import RC2, RC2Stratified
from pysat.examples.hitman import Hitman
from .metric import MetricDimension
from .generate import symmetry_breakers

VECTOR = Tuple[int, ...]
CLAUSE = List[int]

def resolving_model(gph: nx.Graph) -> Tuple[WCNF, IDPool]:
    """
    Create a MAXSAT model for the minimal resolving set.
    """
    pool = IDPool()
    cnf = WCNF()

    met = MetricDimension(gph)

    for elt in met.resolving_set():
        cnf.append([pool.id(('x', _)) for _ in elt])

    for elt in gph.nodes:
        cnf.append([-pool.id(('x', elt))], weight = 1)

    return cnf, pool

def _process(pool: IDPool,
             negpos: Tuple[List[VECTOR], List[VECTOR]]) -> List[CLAUSE]:
    """
    Make clause out of negative, positive
    """
    neg, pos = negpos
    return ([-pool.id(('x', ind)) for ind in neg]
            + [pool.id(('x', ind)) for ind in pos])

def resolve_hypercube_maxsat(num: int,
                             symm: int = 1,
                             forbid: bool = True,
                             trace: int = 0,
                             stratified: bool = False,
                             **kwds) -> Set[VECTOR]:
    """
    Minimal resolving set for the hypercube.
    """

    gph = nx.hypercube_graph(num)
    cnf, pool = resolving_model(gph)
    # for clause in symmetry_breaking_clauses(num, symm, pool):

    cnf.extend(map(partial(_process, pool), symmetry_breakers(num,
                                                              symm,
                                                              forbid=forbid,
                                                              trace=trace)))

    maxsat_solver = RC2Stratified if stratified else RC2
    solver = maxsat_solver(cnf, **kwds)
    soln = solver.compute()
    print("Time = {}".format(solver.oracle_time()))
    pos = [pool.obj(_) for _ in soln if _ > 0]
    return {_[1] for _ in pos if _ is not None and _[0] == 'x'}

def resolve_hypercube_hitman(num: int, **kwds) -> Set[int]:
    """
    Minimal resolving set for the hypercube.

    As part of partial symmetry breaking for the hypercube, we may
    assume that the resolving set includes (0, ..., 0) (all 0's).
    Furthermore, by permuting the coordinates, a second member will
    be 1^k 0^{n-k} for some k <= n.
    """

    gph = nx.hypercube_graph(num)
    met = MetricDimension(gph)
    hit = Hitman(bootstrap_with=met.resolving_set(), htype='sorted', **kwds)
    soln = hit.get()
    print("Time = {}".format(hit.oracle_time()))
    return set(soln)
