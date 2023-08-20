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
from itertools import product, islice, combinations
from collections import Counter
import numpy as np
from pysat.formula import CNF, IDPool, WCNF
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from pysat.examples.optux import OptUx
from pysat.examples.musx import MUSX
from .lex import lex_compare, Comparator, standard_lex
from .logic import MODEL, CLAUSE, FORMULA
from .logic import set_equal, set_and
from .bdd import not_equal
from .symmbreak import double_lex, snake_lex
from .util import get_prefix, extract_mat, getvec, makevec, makemat, makecomp
from .maxtest import min_conflict

CONFLICT = Tuple[int,...]

class Conflict:
    """
    A class to find vectors not distinguished by a matrix.
    """

    def __init__(self, dim: int, mdim: int,
                 verbose: int = 0,
                 solver: str = 'cd15',
                 encode: str = 'totalizer',
                 smallest: int = 0,
                 bound: bool = True,
                 solver_kwds: Dict | None = None):

        self._dim = dim
        self._mdim = mdim
        self._verbose = verbose
        self._smallest = smallest
        self._encoding = getattr(EncType, encode, EncType.totalizer)
        self._getter = (self._get_all_weights if smallest
                        else self._get_conflicts)
        self._cnf = CNF()
        self._pool = IDPool()

        self._avar = makemat(self._pool, 'A', self._mdim, self._dim)
        self._evar = makecomp(self._pool, 'E', self._mdim, self._dim)
        # F[i,j] means col(i) = col(j) and no equality in between
        self._fvar = makecomp(self._pool, 'F', 1, self._dim)
        self._gvar = makecomp(self._pool, 'G', 1, self._dim)
        self._xvar = makevec(self._pool, 'X', self._dim)
        self._yvar = makevec(self._pool, 'Y', self._dim)
        # Assumption variables to find smallest weight conflict
        self._wvar = {_: self._pool.id(('W', _))
                      for _ in range(2, self._dim // 2 + 1)}
        self._bound = bound
        self._generate()
        if solver_kwds is None:
            solver_kwds = {}
        self._solve = Solver(name = solver,
                             bootstrap_with = self._cnf,
                             use_timer = True, **solver_kwds)
        self._cum_time = 0.0

    def append(self, clause: CLAUSE):
        """
        append a clause to both the solver and the cnf
        """
        self._cnf.append(clause)
        self._solve.add_clause(clause)

    def extend(self, form: FORMULA):
        """
        append a clause to both the solver and the cnf
        """
        self._cnf.extend(form)
        self._solve.append_formula(form)

    def _generate(self):
        """
        The Conflict formula
        """
        # x and y can't both be 1
        xlits = [self._xvar[_] for _ in range(self._dim)]
        ylits = [self._yvar[_] for _ in range(self._dim)]
        self._cnf.extend([[-_[0], - _[1]] for _ in zip(xlits, ylits)])
        # Conditional clause to only allow one weight
        self._cnf.extend(CardEnc.atmost(lits = self._wvar.values(),
                                        bound = 1,
                                        encoding = EncType.ladder,
                                        vpool = self._pool))
        for wgt in range(2, self._dim // 2 + 1):
            self._cnf.extend(list(([- self._wvar[wgt]] + clause
                             for clause in CardEnc.equals(
                                     lits = xlits,
                                     bound = wgt,
                                     encoding = self._encoding,
                                     vpool = self._pool).clauses)))
        # Support (even) is >= 4
        self._cnf.extend(CardEnc.atleast(
            lits = xlits + ylits,
            bound = 4 if self._bound else 2,
            encoding = self._encoding,
            vpool = self._pool))
        # sum_i (x[i] - y[i]) = 0
        self._cnf.extend(CardEnc.equals(
            lits = xlits + [-_ for _ in ylits],
            bound = self._dim,
            encoding = self._encoding,
            vpool = self._pool))
        self._cnf.extend(list(standard_lex(self._pool,
                                          ylits, xlits)))
        bvar = makemat(self._pool, 'B', self._mdim, self._dim)
        cvar = makemat(self._pool, 'C', self._mdim, self._dim)
        for kind in range(self._mdim):
            for ind in range(self._dim):
                self._cnf.extend(set_and(bvar[kind, ind],
                                         self._avar[kind, ind],
                                         self._xvar[ind]))
                self._cnf.extend(set_and(cvar[kind, ind],
                                         self._avar[kind, ind],
                                         self._yvar[ind]))
            self._cnf.extend(CardEnc.equals(
                lits=([bvar[kind, _] for _ in range(self._dim)]
                      + [- cvar[kind, _] for _ in range(self._dim)]),
                bound = self._dim,
                encoding = self._encoding,
                vpool = self._pool))
        # Symmetry breaking constraints
        # If two columns of A are equal then the (x,y) values
        # between the two must be non-decreasing
        # E[i,(j,k)] is true <==> A[i,j] == A[i,k]
        for lft, rgt in combinations(range(self._dim), 2):
            self._cnf.append([-self._evar[ind, (lft, rgt)]
                              for ind in range(self._mdim)]
                             + [self._gvar[0, (lft, rgt)]])
            
                                          
            for ind in range(self._mdim):
                self._cnf.append([self._evar[ind, (lft, rgt)],
                                  -self._gvar[0, (lft, rgt)]])
                self._cnf.extend(set_equal(self._evar[ind, (lft, rgt)],
                                           self._avar[ind, lft],
                                           self._avar[ind, rgt]))
            # if any col strictly between lft and rgt are equal
            # to lft then f is false
            self._cnf.extend([[-self._gvar[0, (lft, btw)],
                               -self._fvar[0, (lft, rgt)]]
                              for btw in range(lft+1, rgt)])
            # if no column strictly between lft and rgt
            # is equal of lft, then f[lft,rgt] is false
            self._cnf.extend([[self._gvar[0, (lft, btw)],
                               -self._gvar[0, (lft, rgt)],
                               self._fvar[0, (lft, rgt)]]
                              for btw in range(lft+1, rgt)])
        # Note that we only call solve with assumptions for
        # all of the A values.  Then unit propagation makes
        # all of the e,f,g values constant.  Thus the only
        # active constraints are those that are strictly necessary.
        # Now conditional lex constraints.
        # If A[:,i] == A[:,j] then (x,y)[i] <= (x,y)[j]
        # Note that since the A variables are assumptions
        # Unit propagation will delete all non-applicable clauses

            # prefix = [-self._evar[ind, (lft, rgt)]
            #     for ind in range(self._mdim)]
            prefix = [-self._fvar[0, (lft, rgt)]]
            self._cnf.extend([prefix + _
                               for _ in standard_lex(
                                   self._pool,
                                   [self._xvar[lft], self._yvar[lft]],
                                   [self._xvar[rgt], self._yvar[rgt]],
                               strict=False)])
            

    def _get_soln(self, assumptions: List[int]) -> np.ndarray | None:
        """
        Get a solution.
        """
        status = self._solve.solve(assumptions = assumptions)
        self._cum_time += self._solve.time()
        if not status:
            return None
        model = self._solve.get_model()
        xval = getvec(self._pool, 'X', model)
        yval = getvec(self._pool, 'Y', model)
        # Don't ever get this solution again
        forbid = ([int(1 - 2*xval[_]) * self._xvar[_] for _ in range(self._dim)]
                  + [int(1 - 2*yval[_]) * self._yvar[_] for _ in range(self._dim)])
        self.append([-_ for _ in assumptions] + forbid)
        return xval - yval

    def _get_all_weights(self, assumptions: List[int]) -> Iterable[np.ndarray]:
        """
        Get weights in increasing order.
        """
        bot = 2
        found = False
        while bot <= self._dim // 2:
            result = self._get_soln(assumptions + [self._wvar[bot]])
            if result is None:
                if found and self._smallest > 1:
                    return
                bot += 1
            else:
                found = True
                yield result

    def _get_conflicts(self, assumptions: List[int]) -> Iterable[np.ndarray]:
        """
        Get conflicts in arbitrary order.
        """
        # Turn off specific weights
        assump = assumptions + [-_ for _ in self._wvar.values()]
        while True:
            result = self._get_soln(assump)
            if result is None:
                return
            yield result

    def get_conflicts(self,
                      amat: np.ndarray,
                      times: int) -> Iterable[np.ndarray]:
        """
        Given an A matrix get up to times counterexamples to resolvability.

        self._wvar[wgt] will be a selector variable to select
        conflicts of weight wgt
        """
        assumptions = [(2 * int(amat[key]) - 1) * lit
                       for key, lit in self._avar.items()]
        if self._verbose > 2:
            print(f"assumptions = {assumptions}")

        yield from islice(self._getter(assumptions), times)

    @property
    def census(self):
        """
        Get the initial model clause census.
        """
        return Counter(map(len, self._cnf.clauses))

    @property
    def cum_time(self):
        """ Cumulative solving times """
        return self._cum_time

class Resolve:
    """
    A class to find a putative resolving matrix.

    It keeps a state consisting of the global constraints on the A
    matrix (along with symmetry breaking constraints) and all
    conflicts that have been added so far.  Each conflict constraint
    is made conditional on a separate variable so that a later
    optional constraint minimizer may be run.

    After constraints are added, one or more A matrices compatible
    with the constraints may be requested.  If there are none, then
    the problem is UNSAT.
    """

    def __init__(self, dim: int, mdim: int,
                 verbose: int = 0,
                 alt_model: bool = False, # Use the alternate model
                 alt: bool = False, # Whether to use BDD for not equal
                 nozero: bool = False, # disallow 0 column
                 solver = 'cd15',
                 encode = 'totalizer',
                 snake: int = 0, # Use snake lex if > 0, 2 if Transpose
                 maxweight: bool = False, # Only use maximum weights
                 firstweight: bool = False,
                 getcore: int = 0,
                 solver_kwds: dict | None = None):

        
        self._dim = dim
        self._mdim = mdim
        self._verbose = verbose
        self._encoding = getattr(EncType, encode, EncType.totalizer)
        self._snake = snake
        self._maxweight = maxweight
        self._getcore = getcore
        self._cnf = CNF()
        self._pool = IDPool()
        self._avar = {_ : self._pool.id(('A',) + _)
                       for _ in product(range(self._mdim),range(self._dim))}
        self._nozero = nozero
        self._duplicates = 0
        self._firstweight = firstweight
        if firstweight:
            self._setfirst()
        else:
            self._weight_restriction()
        if alt_model:
            self._model1()
        else:
            self._model2()
        self._solve_name = solver
        if solver_kwds is None:
            solver_kwds = {}
        self._solve = Solver(name = solver,
                             bootstrap_with = self._cnf,
                             use_timer = True, **solver_kwds)
        self._cum_time = 0.0
        self._alt = alt
        self._conflicts = {}
        self._controls = [] # controlling variable for last found matrices
        self._cores = {}
        self._num_conflicts = 0

    def append(self, clause: CLAUSE):
        """
        Append a clause to both the solver and to self._cnf
        """
        self._cnf.append(clause)
        self._solve.add_clause(clause)

    def extend(self, form: FORMULA):
        """
        Append a clause to both the solver and to self._cnf
        """
        self._cnf.extend(form)
        self._solve.append_formula(form)

    @property
    def census(self):
        """
        The clause census.
        """
        return Counter(map(len, self._cnf.clauses))

    @property
    def cum_time(self):
        """ Cumulative time """
        return self._cum_time

    @property
    def num_conflicts(self):
        """ Return the number of distinct conflicts """
        return self._num_conflicts

    @property
    def duplicates(self):
        """ Number of duplicate conflict clauses """
        return self._duplicates

    def get(self) -> Iterable[np.ndarray]:
        """
        Get a resolving matrix.  With each call to get we have an indicator
        variable which can activate the forbidden clause.  During the main
        loop, since we're only adding constraints, we can disallow all of the
        previously found matrices.  Only when we're looking for a minimal
        set do we have to allow all of them.
        """
        # control is used to turn on/off counterexamples
        control = self._pool.id()
        prefix = (list(self._conflicts.values())
                  + self._controls + [control])
        self._controls.append(-control)
        while True:
            status = self._solve.solve(assumptions = prefix)
            stime = self._solve.time()
            self._cum_time += stime
            if self._verbose > 2:
                print(f"Resolve status = {status} time = {stime}")
            if not status:
                return
            model = self._solve.get_model()

            amat = extract_mat(self._pool, 'A', model)
            self.append([-control] +
                        [- (2 * int(amat[key]) - 1) * val
                         for key, val in self._avar.items()])
            yield amat

    def minimal(self, mverbose: int = 0, use_ux: bool = False) -> List[CONFLICT]:
        """
        Use either MUSX or OptUx to find a minimal set.
        """

        consider = list(self._conflicts.values())

        wcnf = WCNF()
        for elt in consider:
            wcnf.append([elt], weight=1)
        wcnf.extend(self._cnf.clauses)
        minimizer = OptUx if use_ux else MUSX
        min_opts = {('verbose' if use_ux else 'verbosity'): mverbose}
        msolver = minimizer(wcnf,**min_opts)
        answer = msolver.compute()
        self._cum_time += msolver.oracle_time()
        backwards = {_[1]:_[0] for _ in self._conflicts.items()}
        # return answer
        # indices are 1-origin?
        # To be strict we need to look at fn.soft
        good = [wcnf.soft[_ - 1] for _ in answer]
        if not all((len(_) == 1 for _ in good)):
            raise ValueError("Minimal result clauses not all unit")
        return [backwards[_[0]] for _ in good]

    def add_conflicts(self, amat: np.ndarray, xvals: List[np.ndarray]):
        """
        Add conflict clauses.
        Each conflict has an associated assumption variable.
        We have a dict whose key is the tuple for the conflict,
        and the value is a new variable associated to the conflict.
        We will use those variables later to find a minimal set of conflicts
        for the UNSAT case.
        """
        conflictor = (self._bdd_add_conflict
            if self._alt else self._add_conflict)
        self._num_conflicts += len(xvals)
        if self._getcore < 2:
            for xval in xvals:
                txval = tuple(xval.tolist())
                if self._verbose > 2:
                    print(f"+conflict = {txval}")
                if txval in self._conflicts:
                    self._duplicates += 1
                    continue
                assump = self._pool.id()
                # self.append([-assump, - self._controls[-1]])
                self._conflicts[txval] = assump
                conflictor(xval)
        if self._getcore:
            core = min_conflict(
                xvals,
                amat,
                solver = self._solve_name,
                snake = self._snake,
                verbose = self._verbose)
            for xval in xvals:
                txval = tuple(xval.tolist())
                self._cores[txval] = core
                if self._verbose > 1:
                    print(f"xval = {xval}, core = {core}")
            if core:
                self.append([int(1 - 2 * amat[_]) * self._avar[_]
                            for _ in core])

    def core_stats(self):
        """Statistics about cores"""
        return Counter(map(len, self._cores.values()))
    
    def _bdd_add_conflict(self, xval: np.ndarray):
        """
        Add clauses that forbid A@x = 0
        """
        inequalities = []
        indicators = [self._pool.id() for _ in range(self._mdim)]
        bound = (xval == -1).sum()
        pos = np.arange(self._dim)[xval == 1].tolist()
        neg = np.arange(self._dim)[xval == -1].tolist()
        for kind in range(self._mdim):
            lits = ([self._avar[kind, _] for _ in pos]
                    + [- self._avar[kind, _] for _ in neg])
            inequalities += list(
                not_equal(self._pool, lits, bound, indicators[kind]))
        assump = self._conflicts[tuple(xval.tolist())]

        self.extend([[-assump] + _ for _ in
                     inequalities + [indicators]])

    def _add_conflict(self, xval: np.ndarray):
        """
        Add clauses that forbid A@x = 0
        """
        indicators = []
        bound = (xval == -1).sum()
        pos = np.arange(self._dim)[xval == 1].tolist()
        neg = np.arange(self._dim)[xval == -1].tolist()
        assump = self._conflicts[tuple(xval.tolist())]
        # This conflict forbids the last bunch of matrices found.
        for kind in range(self._mdim):
            lits = ([self._avar[kind, _] for _ in pos]
                    + [- self._avar[kind, _] for _ in neg])
            indic1 = self._pool.id()
            indicators.append(indic1)
            for clause in CardEnc.atmost(
                    lits = lits, bound = bound - 1,
                    encoding = self._encoding,
                    vpool = self._pool).clauses:
                self.append([-assump, -indic1] + clause)
            indic2 = self._pool.id()
            indicators.append(indic2)
            for clause in CardEnc.atleast(
                    lits = lits, bound = bound + 1,
                    encoding = self._encoding,
                    vpool = self._pool).clauses:
                self.append([-assump, -indic2] + clause)
            # Both cannot be true
            self.append([-indic1, -indic2])
        # At least one row must yield a nonzero value
        self.append([-assump] + indicators)

    def _setfirst(self):
        """
        Make the last column all 1's.  An alternative
        to weight_restriction
        Note: this must the last column because of lex
        ordering.
        """
        self._cnf.extend([[self._avar[_, 0]] for _ in range(self._mdim)])
        # all other columns cannot be all 1's
        for ind in range(1, self._dim):
            self._cnf.append([-self._avar[_, ind]
                              for _ in range(1, self._mdim)])
        
        for ind in range(self._mdim):
            self._cnf.append([self._avar[ind, _] for _ in range(self._dim)])
        
    def _weight_restriction(self):
        """
        Restrict the weights of the rows to be <= n/2.
        Optionally make first column nonzero
        """
        for ind in range(self._mdim):
            self._cnf.append([self._avar[ind, _] for _ in range(self._dim)])
        if self._nozero:
            self._cnf.append([self._avar[_,0] for _ in range(self._mdim)])
        # for kind in range(self._mdim - 1):
        #     self._cnf.extend(list(special_less(self._pool,
        #         [self._avar[kind, _] for _ in range(self._dim)],
        #         [self._avar[kind + 1, _] for _ in range(self._dim)])))
        card_constraint = CardEnc.equals if self._maxweight else CardEnc.atmost
        for kind in range(self._mdim):
            self._cnf.extend(card_constraint(lits =
                                             [self._avar[kind, _]
                                              for _ in range(self._dim)],
                                             bound = self._dim // 2,
                                             encoding = self._encoding,
                                             vpool = self._pool))

    def _model1(self):
        """
        First model with symmetry breaking
        """
        # Non zero first row
        # Everything is increasing so others are nonzero
        # Create the column constraints
        # Row -1 everything is equal
        # E[k,i,j] = (A[:k+1,i] == A[:k+1,j]).all()
        evar = {_: self._pool.id(('E',) + _)
                for _ in product(range(self._mdim),
                                 range(self._dim - 1))}
        for kind, ind in product(range(self._mdim),
                                 range(self._dim-1)):

            # E[k,i] <-> (A[k,i] == A[k,i+1])
            self._cnf.extend(set_equal(evar[kind, ind],
                                       self._avar[kind, ind],
                                       self._avar[kind, ind + 1]))
            # E[k-1,i,j] & A[k,i] -> A[k,i+1]
            # Shift all 1's in a sector to the right
            self._cnf.append([-evar[jind, ind] for jind in range(kind)]
                             + [-self._avar[kind, ind], self._avar[kind, ind+1]])
        # Distinct columns
        self._cnf.extend([[-evar[jind,ind] for jind in range(self._mdim)]
                           for ind in range(self._dim-1)])
        # Lex increasing rows
        for ind in range(self._mdim-1):
            self._cnf.extend(standard_lex(self._pool,
                                          [self._avar[ind, jind]
                                           for jind in range(self._dim)],
                                          [self._avar[ind+1, jind]
                                           for jind in range(self._dim)]))

    def _model2(self):
        """
        Simpler model with double lexical ordering.
        """

        # Non zero first row
        # Everything is increasing so others are nonzero
        # Double sorted increasing
        amat = np.array([[self._avar[ind, jind] for jind in range(self._dim)]
                         for ind in range(self._mdim)], dtype=int)
        if self._firstweight:
            amat = amat[:, 1:] # avoid first column
        breaker = snake_lex if self._snake > 0 else double_lex
        self._cnf.extend(list(breaker(self._pool,
                                      amat.T if self._snake > 1 else amat)))

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

    conflict = Conflict(dim, mdim - 1,
                        verbose = verbose,
                        solver=solver,
                        encode=encode,
                        smallest = smallest,
                        bound = resolver_opts.get('snake', 0) == 0,
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
