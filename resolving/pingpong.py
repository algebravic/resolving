"""
Use a ping-pong solver for the metric dimension problem.
We will have two instances of SAT:

1) This instance will search for a sensing matrix of a given dimension.
An m by n sensing matrix will be a 0/1 matrix whose rows are nonzero
and whose columns are distinct and are lexicographically strictly increasing.
We also need to break symmetries:
If A is the matrix then if A[:k, i] == A[:k, j] for i < j and A[k,i] == 1
then A[k,j] == 1.
We will dynamically also add constraints coming from the second instance.

2) The second instance will take as assumptions a value for the A matrix
from instance (1).  Its variables will be x[i], y[i], i=1,...,n
with sum(x) + sum(y) >= 4, OR(x \/ y), ~x[i] \/ ~y[i] for i=1,...,n
and sum(A[k,i] * x[i], i=1,..,n) + sum(A[k,i] * ~y[i], i=1,..,n)
     >= 1 + sum(A[k,i], i=1,..n) for all k

We will also, optionally as solver (2) to provide at most r solutions.

"""
from typing import Iterable, List, Tuple, Dict
from itertools import islice
import numpy as np
from .conflictor import Conflict
from .resolver import Resolve

CONFLICT = Tuple[int,...]

def main_loop(resolver: Resolve, conflict: Conflict,
              verbose = 0,
              getcore: int = 0,
              times = 1, rtimes = 1) -> Tuple[bool,np.ndarray | None]:
    """
    The main ping/pong loop.
    """
    conflicts = []
    soln = None
    found = False
    reasons = []
                
    for amat in islice(resolver.get(), rtimes):
        if verbose > 2:
            print(f"A:\n{amat}")
        # Check validity
        # Give A (as assumptions) to model2 to find conflicts.
        lconf = list(conflict.get_conflicts(amat, times))
        reasons.append((amat, lconf))
        if verbose > 2:
            print(f"conflicts={lconf}")
        soln = amat
        if len(lconf) == 0: # amat is a solution!
            found = True
            break
        if verbose > 1:
            print(f"conflicts: {lconf}")
        conflicts += lconf
    if verbose > 2:
        print(f"found = {found}, amat = {soln}")
    if not found:
        for amat, lconf in reasons:
            resolver.add_conflicts(amat, lconf)

    return found, soln

def ping_pong(dim: int, mdim: int,
              times: int = 1,
              rtimes: int = 1,
              verbose: int = 1,
              encode: str = 'totalizer',
              solver: str = 'cd15',
              resolver_opts: Dict | None = None,
              smallest: int = 0, # favor smaller weight conflicts
              minimal: int = 0,
              trace: int = 0,
              mverbose: int = 0,
              getcore: int = 0,
              solver_kwds: Dict | None = None) -> np.ndarray | None | List[CONFLICT]:
    """
    Test if the a set of size m is a resolving set for the
    n-dimensional hypercube.  If it is, return such a set.  If not,
    return none.

    There are two cooperating SAT solvers: the Resolver, and
    Conflictor.  The resolver produces a putative resolving set given
    the current set of constraints.  This is then handed to the
    Conflictor, which either finds one or more constraints that the
    current solution does not satisfy, or states that there are none,
    in which case, the putative solution is a real solution.  In the
    former case the new constraints are added to the Resolver.  This
    back and forth continues until either a solution is found, or the
    Resolver produces UNSAT, in which case there is no resolving set.

    See classes Resolve for the Resolver and Conflict for the Conflictor.

    Input:
       dim: the dimension of the hypercube.
       mdim: the size of the putative resolving set.
       times: How many counterexamples for the putative resolving set.
       rtimes: How many putative resolving sets should be generated.
       verbose: Level of verbosity.
       encode: The encoding method for cardinality constraints.
       solver: The SAT solver to use.
       resolver_opts: a dictionary of options for the Resolver (cf. class)
       minimal: For Unsat the method of finding a minimal set of constraints.
       trace: the number of iterations for tracing.
       mverbose: The verbosity level of the minimal set solver.
       solver_kwds: options for the SAT solver.

    """

    if resolver_opts is None:
        resolver_opts = {}
    if solver_kwds is None:
        solver_kwds = {}

    resolver = Resolve(dim, mdim - 1,
                       verbose = verbose,
                       solver=solver,
                       encode=encode,
                       getcore = getcore,
                       solver_kwds = solver_kwds,
                       **resolver_opts)


    if verbose > 1:
        print(f"Resolve census = {resolver.census}")

    bound = (1 if resolver_opts.get('ss_cuts', False) or
        (resolver_opts.get('snake', 0) != 0) else 2)
    print(f"lower bound = {2 * bound}")
    conflict = Conflict(dim, mdim - 1,
                        verbose = verbose,
                        solver=solver,
                        encode=encode,
                        smallest = smallest,
                        bound = bound,
                        solver_kwds = solver_kwds)

    if verbose > 1:
        print(f"Conflict census = {conflict.census}")

    pass_no = 0
    found = False
    amat = None
    while not found:
        pass_no += 1
        if trace > 0 and pass_no % trace == 0:
            print(f"At pass {pass_no}, resolve time = {resolver.cum_time}")
            print(f"conflict time = {conflict.cum_time}, "
                  + f"conflicts = {resolver.num_conflicts}")
            if verbose > 1:
                print(f"resolver census = {resolver.census}")
                print(f"conflictor census = {conflict.census}")
        # Add new conflicts.  At the beginning there are none
        # found = True, means that problem is SAT, then amat is the solution
        # amat is None means problem is UNSAT
        found, amat = main_loop(resolver, conflict,
                                verbose = verbose,
                                times = times,
                                rtimes = rtimes)
        if not found and amat is None: # It is UNSAT
            break
    if getcore:
        print(f"Cores statistics = {resolver.core_stats()}")
    if verbose > 0:
        print(f"Final conflict time = {conflict.cum_time}, "
              + f"conflicts={resolver.num_conflicts}")
        # print(f"duplicates = {resolver.duplicates}")
        print(f"Total passes: {pass_no}, resolve time = {resolver.cum_time}.")
    if minimal > 0 and amat is None:
        minimal_conflicts = resolver.minimal(
            mverbose=mverbose, use_ux = minimal != 1)
        if verbose > 0:
            print(f"Final resolve time = {resolver.cum_time}")
        return minimal_conflicts
    if amat is not None:
        fconflict = Conflict(dim, mdim - 1,
                             solver=solver,
                             encode=encode,
                             smallest = smallest,
                             bound = resolver_opts.get('snake', 0) == 0,
                             solver_kwds = solver_kwds)
        check = list(fconflict.get_conflicts(amat, 1))
        print(f"Final check: {len(check) == 0}")

    return amat
