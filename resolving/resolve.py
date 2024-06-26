"""
Use MAXSAT to find a minimal cardinality resolving set for a finite
unidirected connected graph G.
"""
from typing import Set, Tuple, List, Any
from functools import partial
import networkx as nx
from pysat.formula import CNF, WCNF, IDPool
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver
from pysat.examples.rc2 import RC2, RC2Stratified
from pysat.examples.hitman import Hitman
from .metric import MetricDimension
from .generate import symmetry_breakers
from .timeit import Timer

VECTOR = Tuple[int, ...]
CLAUSE = List[int]
CADICAL = 'Cadical195'

def resolving_model(gph: nx.Graph, cnf: CNF | WCNF) -> IDPool:
    """
    Create a MAXSAT model for the minimal resolving set.
    """
    pool = IDPool()

    met = MetricDimension(gph)
    # Get rid of duplicates
    for elt in set(met.resolving_set()):
        # Don't use the trivial graph
        if len(elt) < len(gph):
            cnf.append([pool.id(('x', _)) for _ in elt])

    return pool

def _process(pool: IDPool,
             negpos: Tuple[List[VECTOR], List[VECTOR]]) -> List[CLAUSE]:
    """
    Make clause out of negative, positive
    """
    neg, pos = negpos
    return ([-pool.id(('x', ind)) for ind in neg]
            + [pool.id(('x', ind)) for ind in pos])

def solve_maxsat(cnf: WCNF,
                 stratified: bool = False, **kwds) -> List[int] | None:
    """
    Solve with a maxsat solver
    """
    maxsat_solver = RC2Stratified if stratified else RC2
    max_solver = maxsat_solver(cnf, **kwds)
    soln = max_solver.compute()
    print("Time = {}".format(max_solver.oracle_time()))
    return soln

def solve_sat(cnf: CNF, solver_name: str) -> Tuple[Solver, List[int]] | None:
    """
    Solve with a sat solver
    """
    solve = Solver(name = solver_name,
                   bootstrap_with = cnf, use_timer = True)
    status = solve.solve()
    print("Time = {}".format(solve.time()))
    return solve, solve.get_model() if status else None

def setup_hypercube(cnf: CNF | WCNF,
                    num: int,
                    symm: int = 1,
                    forbid: bool = True,
                    trace: int = 0) -> Tuple[IDPool, nx.Graph]:
    """
    Setup the problem
    """
    gph = nx.hypercube_graph(num)
    with Timer('model'):
        pool = resolving_model(gph, cnf)
    with Timer('symmetry'):
        cnf.extend(map(partial(_process, pool), symmetry_breakers(num,
                                                                  symm,
                                                                  forbid=forbid,
                                                                  trace=trace)))
    return pool, gph

def get_answer(soln: List[int] | None,
               pool: IDPool) -> List[VECTOR] | None:
    """
    Get the answer.
    """
    if soln is None:
        return None

    pos = [pool.obj(_) for _ in soln if _ > 0]
    return {_[1] for _ in pos if _ is not None and _[0] == 'x'}

def resolve_maxsat(gph: nx.Graph, **kwds) -> List[Any] | None:
    """Test if a graph has the indicated metric dimension

    Args:
    gph: A networkx Graph
    dim: The asserted metric dimension
    """
    cnf = WCNF()
    pool = resolving_model(gph, cnf)
    for node in gph.nodes:
        cnf.append([-pool.id(('x', node))], weight=1)
    soln = solve_maxsat(cnf, **kwds)
    return get_answer(soln, pool) if soln is not None else None


def resolve_hypercube_maxsat(num: int,
                             symm: int = 1,
                             forbid: bool = True,
                             trace: int = 0,
                             stratified: bool = False,
                             **kwds) -> Set[VECTOR] | None:
    """
    Use maxsat to find the minimal resolving set for the hypecube.
    """
    cnf = WCNF()
    pool, gph = setup_hypercube(cnf, num, symm = symm,
                                    forbid = forbid, trace = trace)
    for elt in gph.nodes:
        cnf.append([-pool.id(('x', elt))], weight = 1)
    return get_answer(solve_maxsat(cnf, stratified = stratified, **kwds), pool)
    
def resolve_hypercube_sat(num: int,
                          bound: int,
                          symm: int,
                          forbid: bool = True,
                          encode: str = 'totalizer',
                          core: bool = False,
                          **kwds) -> Set[VECTOR] | None:
    """
    Minimal resolving set for the hypercube.
    """

    # for clause in symmetry_breaking_clauses(num, symm, pool):

    cnf = CNF()
    pool, gph = setup_hypercube(cnf, num, symm = symm, forbid = forbid)

    cnf.extend(CardEnc.atmost(lits = [pool.id(('x', _)) for _ in gph.nodes],
                              bound = bound,
                              encoding = getattr(EncType, encode,
                                                 EncType.totalizer),
                              vpool = pool))
    solve, model = solve_sat(cnf, solver_name = kwds.get('solver', CADICAL))
    soln = get_answer(model, pool)
    if soln is None:
        print(f"core = {core}")
        print(f"status = {solve.get_status()}")
        if core:
            mcore = solve.get_core()
            print(f"core = {type(mcore)}")
            return mcore
        return None

    return soln
    

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
