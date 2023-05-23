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
from typing import Iterable, List, Tuple
from itertools import product, chain, combinations, islice
from collections import Counter
import numpy as np
from pysat.formula import CNF, IDPool, WCNF
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from pysat.examples.optux import OptUx
from .lex import lex_compare, Comparator, standard_lex, special_less
from .logic import MODEL
from .logic import negate, set_equal, set_and, big_or
from .bdd import not_equal
from .symmbreak import double_lex, snake_lex

CLAUSE = List[int]
FORMULA = List[CLAUSE]

def get_prefix(pool: IDPool, prefix: str, model: MODEL) -> List[Tuple[str, int,...]]:
    """
    Get all prefixes which are defined.
    """
    
    name_val = [(pool.obj(abs(_)), int(_ > 0)) for _ in model]
    
    return [(_[0][1:], _[1])
            for _ in name_val if isinstance(_[0], tuple) and _[0][0] == prefix]

def _check_diff_cols(mat: np.ndarray) -> List[Tuple[int, int]]:
    """
    Check if a matrix has distinct columns.
    """
    _, ndim = mat.shape
    return [(ind, jind) for jind in range(1, ndim) for ind in range(jind)
            if (mat[:, ind] == mat[:, jind]).all()]

def extract_mat(pool: IDPool, prefix: str, model: MODEL) -> np.ndarray:
    """
    Get the A matrix
    """
    values = sorted(get_prefix(pool, prefix, model))
    mdim = max((_[0][0] for _ in values)) + 1
    ndim = max((_[0][1] for _ in values)) + 1
    return  np.array([_[1] for _ in values], dtype=np.int8).reshape((mdim, ndim))

def getvec(pool: IDPool, prefix: str, model: MODEL) -> np.ndarray:
    """
    Get a vector
    """
    values = sorted(get_prefix(pool, prefix, model))
    # Check for consistency
    if {_[0][0] for _ in values} != set(range(len(values))):
        raise ValueError(f"{prefix}: {values}")
                                          
    return np.array([_[1] for _ in values], dtype=np.int8)

class Conflict:
    """
    A class to find vectors not distinguished by a matrix.
    """

    def __init__(self, dim: int, mdim: int,
                 solver: str = 'cd15',
                 encode: str = 'totalizer',
                 solver_kwds: dict = dict()):

        self._dim = dim
        self._mdim = mdim
        self._encoding = getattr(EncType, encode, EncType.totalizer)
        self._cnf = CNF()
        self._pool = IDPool()
        self._avar = {_ : self._pool.id(('A',) + _)
                      for _ in product(range(mdim), range(dim))}
        self._xvar = {_: self._pool.id(('X', _)) for _ in range(self._dim)}
        self._yvar = {_: self._pool.id(('Y', _)) for _ in range(self._dim)}
        # Assumption variables to find largest weight conflict
        self._wvar = {_: self._pool.id(('W', _))
                      for _ in range(2, self._dim // 2 + 1)}
        self._generate()
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
        self._solve.add_formula(form)
        
    def _generate(self):
        """
        The Conflict formula
        """
        # x and y can't both be 1
        xlits = [self._xvar[_] for _ in range(self._dim)]
        ylits = [self._yvar[_] for _ in range(self._dim)]
        self._cnf.extend([[-_[0], - _[1]] for _ in zip(xlits, ylits)])
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
            bound = 4,
            encoding = self._encoding,
            vpool = self._pool))
        # sum_i (x[i] - y[i]) = 0
        self._cnf.extend(CardEnc.equals(
            lits = xlits + [-_ for _ in ylits],
            bound = self._dim,
            encoding = self._encoding,
            vpool = self._pool))
        self._cnf.extend(list(lex_compare(self._pool,
                                          ylits, xlits,
                                          Comparator.LESS)))

        bvar = {_: self._pool.id(('B',) + _) for _ in product(range(self._mdim),
                                                           range(self._dim))}
        cvar = {_: self._pool.id(('C',) + _) for _ in product(range(self._mdim),
                                                           range(self._dim))}
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
        self.append(forbid)
        return xval - yval
        
    def _get_weight(self, wgt: int, assumptions: List[int]) -> np.ndarray | None:
        """
        Get a a conflict, given assumptions of weight = wgt or None if there
        is no such.
        """
        # Only consider those of weight = wgt
        weights = ([self._wvar[wgt]]
                   + [-self._wvar[_] for _ in range(2, self._dim // 2 + 1)
                      if _ != wgt])

        return self._get_soln(assumptions + weights)

    def _get_all_weights(self, assumptions: List[int]) -> np.ndarray:
        """
        Get weights in decreasing order.
        """
        top = self._dim // 2
        while top >= 2:
            result = self._get_weight(top, assumptions)
            if result is None:
                top -= 1
            else:
                yield result

    def _get_conflicts(self, assumptions: List[int]) -> Iterable[Tuple[int,...]]:
        """
        Get conflicts in arbitrary order.
        """
        assump = assumptions + [-self._wvar[_] for _ in range(2, self._dim // 2 + 1)]
        while True:
            result = self._get_soln(assump)
            if result is None:
                return
            yield result

    def get_conflicts(self,
                      amat: np.ndarray,
                      times: int,
                      largest: bool = False, # Find largest weight
                      verbose: int = 0) -> Iterable[np.ndarray]:
        """
        Given an A matrix get up to times counterexamples to resolvability.

        self._wvar[wgt] will be a selector variable to select
        conflicts of weight wgt
        """
        assumptions = [(2 * int(amat[key]) - 1) * lit
                       for key, lit in self._avar.items()]
        if verbose > 2:
            print(f"assumptions = {assumptions}")
        getter = self._get_all_weights if largest else self._get_conflicts
        yield from islice(getter(assumptions), times)

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
    """

    def __init__(self, dim: int, mdim: int,
                 alt_model: bool = False,
                 alt: bool = False,
                 nozero: bool = True,
                 solver = 'cd15',
                 encode = 'totalizer',
                 snake: bool = False,
                 maxweight: bool = True,
                 solver_kwds: dict | None = None):

        
        self._dim = dim
        self._mdim = mdim
        self._encoding = getattr(EncType, encode, EncType.totalizer)
        self._snake = snake
        self._maxweight = maxweight
        self._cnf = CNF()
        self._pool = IDPool()
        self._avar = {_ : self._pool.id(('A',) + _)
                       for _ in product(range(self._mdim),range(self._dim))}
        self._nozero = nozero
        self._duplicates = 0
        if alt_model:
            self._model1()
        else:
            self._model2()
        self._solve_name = solver
        if solver_kwds is None:
            solve_kwds = dict()
        self._solve = Solver(name = solver,
                             bootstrap_with = self._cnf,
                             use_timer = True, **solver_kwds)
        self._cum_time = 0.0
        self._alt = alt
        self._conflicts = dict()

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
        self._solve.add_formula(form)

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
        return len(self._conflicts)

    @property
    def duplicates(self):
        """ Number of duplicate conflict clauses """
        return self._duplicates

    def get(self, verbose: int = 0, times: int = 1) -> Iterable[np.ndarray]:
        """
        Get a resolving matrix.  With each call to get we have an indicator
        variable which can activate the forbidden clause.  During the main
        loop, since we're only adding constraints, we can disallow all of the
        previously found matrices.  Only when we're looking for a minimal
        set do we have to allow all of them.
        """
        indic = self._pool._next()
        control = list(self._conflicts.values())
        ncontrol = [-_ for _ in self._conflicts.values()]
        for _ in range(times):
            status = self._solve.solve(assumptions = control)
            stime = self._solve.time()
            self._cum_time += stime
            if verbose > 1:
                print(f"Resolve status = {status} time = {stime}")
                if not status:
                    break
            if not status:
                break
            
            model = self._solve.get_model()
            
            amat = extract_mat(self._pool, 'A', model)
            yield amat
            # Now forbid this one
            self.append(ncontrol +
                [- (2 * int(amat[key]) - 1) * val
                 for key, val in self._avar.items()])

            if verbose > 1:
                print(f"amat = {amat}")
            # Check that A has all columns distinct
            col_diffs = _check_diff_cols(amat)
            if col_diffs:
                evalues = get_prefix(self._pool, 'E', model)
                print(f"E array = {dict(evalues)}")
                avalues = get_prefix(self._pool, 'A', model)
                print(f"A array = {dict(avalues)}")
                raise ValueError(f"Columns not distinct: {col_diffs}!")

    def minimal_ux(self, verbose: int = 0) -> List[Tuple[int, ...]]:
        """
        Use OptUx to find a minimal set.
        """

        consider = list(self._conflicts.values())

        wcnf = WCNF()
        for elt in consider:
            wcnf.append([elt], weight=1)
        wcnf.extend(self._cnf.clauses)
        optux = OptUx(wcnf,verbose=verbose)
        answer = optux.compute()
        backwards = {_[1]:_[0] for _ in self._conflicts.items()}
        self._cum_time += optux.oracle_time()
        # return answer
        # indices are 1-origin?
        # To be strict we need to look at fn.soft
        good = [wcnf.soft[_ - 1] for _ in answer]
        if not all((len(_) == 1 for _ in good)):
            raise ValueError("Minimal result clauses not all unit")
        return [backwards[_[0]] for _ in good]

    def minimal(self, verbose: int = 0) -> List[Tuple[int,...]]:
        """
        Return a minimal set of conflicts that still
        cause the model to be UNSAT.

        Use the naive method with randomization.

        self._conflicts is a dict whose key is the actual conflict
        and value is the CNF variable number.
        """

        rng = np.random.default_rng()

        consider = rng.permutation(list(self._conflicts.values())).tolist()

        good = set(consider)
        bad = set()

        for elt in consider:

            good.remove(elt)

            status = self._solve.solve(
                assumptions = list(good) + [-elt] + list(bad))
            self._cum_time += self._solve.time()
            if status: # elt must be there
                good.add(elt)
            else:
                bad.add(-elt)
        rdict = {_[1]: _[0] for _ in self._conflicts.items()}
        return [rdict[_] for _ in good]

    def add_conflict(self, xval: np.ndarray):
        """
        Add conflict clauses.
        Each conflict has an associated assumption variable.
        We have a dict whose key is the tuple for the conflict,
        and the value is a new variable associated to the conflict.
        We will use those variables later to find a minimal set of conflicts
        for the UNSAT case.
        """
        txval = tuple(xval.tolist())
        if txval in self._conflicts:
            self._duplicates += 1
            return
        self._conflicts[txval] = self._pool._next()
        (self.bdd_add_conflict if self._alt else self.alt_add_conflict)(xval)
            
    def bdd_add_conflict(self, xval: np.ndarray):
        """
        Add clauses that forbid A@x = 0
        """
        inequalities = []
        indicators = [self._pool._next() for _ in range(self._mdim)]
        bound = (xval == -1).sum()
        for kind in range(self._mdim):
            lits = ([self._avar[kind, _]
                     for _ in range(self._dim) if xval[_] == 1]
                    + [- self._avar[kind, _]
                       for _ in range(self._dim) if xval[_] == -1])
            inequalities += list(
                not_equal(self._pool, lits, bound, indicators[kind]))
        assump = self._conflicts[tuple(xval.tolist())]
        
        self.extend([[-assump] + _ for _ in
                     inequalities + [indicators]])

    def alt_add_conflict(self, xval: np.ndarray):
        """
        Add clauses that forbid A@x = 0
        """
        indicators = []
        bound = (xval == -1).sum()
        assump = self._conflicts[tuple(xval.tolist())]
        for kind in range(self._mdim):
            lits = ([self._avar[kind, _]
                     for _ in range(self._dim) if xval[_] == 1]
                    + [- self._avar[kind, _]
                       for _ in range(self._dim) if xval[_] == -1])
            indic1 = self._pool._next()
            indicators.append(indic1)
            for clause in CardEnc.atmost(
                    lits = lits, bound = bound - 1,
                    encoding = self._encoding,
                    vpool = self._pool).clauses:
                self.append([-assump, -indic1] + clause)
            indic2 = self._pool._next()
            indicators.append(indic2)
            for clause in CardEnc.atleast(
                    lits = lits, bound = bound + 1,
                    encoding = self._encoding,
                    vpool = self._pool).clauses:
                self.append([-assump, -indic2] + clause)
        self.append([-assump] + indicators)

    def _model1(self):
        """
        First model with symmetry breaking
        """
        # Non zero first row
        # Everything is increasing so others are nonzero
        if self._nozero:
            self._cnf.append([self._avar[0, _] for _ in range(self._dim)])
        for kind in range(self._mdim - 1):
            self._cnf.extend(list(special_less(self._pool,
                [self._avar[kind, _] for _ in range(self._dim)],
                [self._avar[kind + 1, _] for _ in range(self._dim)])))
        card_constraint = CardEnc.equals if self._maxweight else CardEnc.atmost
        for kind in range(self._mdim):
            self._cnf.extend(card_constraint(lits =
                                             [self._avar[kind, _]
                                              for _ in range(self._dim)],
                                             bound = self._dim // 2,
                                             encoding = self._encoding,
                                             vpool = self._pool))
        # Create the column constraints
        # Row -1 everything is equal
        # E[k,i,j] = (A[:k+1,i] == A[:k+1,j]).all()
        evar = {_: self._pool.id(('E',) + _)
                for _ in product(range(-1, self._mdim),
                                 combinations(range(self._dim), 2))}
        zvar = {_: self._pool.id(('Z',) + _)
                for _ in product(range(self._mdim),
                                 combinations(range(self._dim), 2))}
        self._cnf.extend([[evar[-1, _]]
                           for _ in combinations(range(self._dim), 2)])
        # The remaining rows
        # defined = set()
        for kind, (ind, jind) in product(range(self._mdim),
                                         combinations(range(self._dim), 2)):
            
            # Z[k,i,j] <-> A[k,i] == A[k,j]
            self._cnf.extend(set_equal(zvar[kind, (ind, jind)],
                                       self._avar[kind, ind],
                                       self._avar[kind, jind]))
            # E[k,i,j] <-> Z[k,i,j] /\ E[k-1,i,j]
            # defined.add((kind, ind, jind))
            self._cnf.extend(set_and(evar[kind, (ind, jind)],
                                     evar[kind - 1, (ind, jind)],
                                     zvar[kind, (ind, jind)]))
            # E[k-1,i,j] /\ A[k,i] -> A[k,j]
            # Shift all 1's in a sector to the left
            self._cnf.append([-evar[kind-1, (ind, jind)],
                              -self._avar[kind, ind],
                              self._avar[kind, jind]])
        # Distinct columns
        # req = {(mdim - 1, ind, jind) for jind in range(1, dim) for ind in range(jind)}
        # print(f"req = {sorted(req)}")
        # if not req.issubset(defined):
        #    raise ValueError(f"Not defined {req.difference(defined)}")
        self._cnf.extend([[-evar[self._mdim - 1, _]]
                           for _ in combinations(range(self._dim), 2)])

    def _model2(self):
        """
        Simpler model with double lexical ordering.
        """

        # Non zero first row
        # Everything is increasing so others are nonzero
        self._cnf.append([self._avar[0, _] for _ in range(self._dim)])
        if self._nozero:
            self._cnf.append([self._avar[_, 0] for _ in range(self._mdim)])
        # Double sorted increasing
        amat = np.array([[self._avar[ind, jind] for jind in range(self._dim)]
                         for ind in range(self._mdim)], dtype=int)
        breaker = snake_lex if self._snake else double_lex
        self._cnf.extend(list(breaker(self._pool, amat.T)))
        card_constraint = CardEnc.equals if self._maxweight else CardEnc.atmost
        for ind in range(self._mdim):
            self._cnf.extend(card_constraint(lits =
                                             [self._avar[ind, _]
                                              for _ in range(self._dim)],
                                             bound = self._dim // 2,
                                             encoding = self._encoding,
                                             vpool = self._pool))

def ping_pong(dim: int, mdim: int,
              times: int = 1,
              rtimes: int = 1,
              verbose: int = 1,
              encode: str = 'totalizer',
              solver: str = 'cd15',
              resolver_opts: dict = dict(),
              largest=False, # favor larger weight conflicts
              minimal: int = 0,
              trace: int = 0,
              mverbose: int = 0,
              **solver_kwds) -> np.ndarray | None | List[Tuple[int,...]]:
    """
    Ping Pong method.
    """

    resolver_opts_defaults = dict(
        snake=False, # Use snake lex
        alt=False, # Whether to use BDD for not equal
        alt_model=False, # Use the alternate model
        nozero=True, # disallow 0 column
        maxweight=True # Only use maximum weights
    )
    resolver = Resolve(dim, mdim - 1,
                       solver=solver,
                       encode=encode,
                       solver_kwds = solver_kwds,
                       **(resolver_opts_defaults | resolver_opts))


    if verbose > 1:
        print(f"Resolve census = {resolver.census}")

    conflict = Conflict(dim, mdim - 1,
                        solver=solver,
                        encode=encode,
                        **solver_kwds)

    if verbose > 1:
        print(f"Conflict census = {conflict.census}")

    old_amat = np.zeros((mdim - 1, dim), dtype = np.int8)
    pass_no = 0
    found_solution = False
    while True:
        pass_no += 1
        if trace > 0 and pass_no % trace == 0:
            print(f"At pass {pass_no}, resolve time = {resolver.cum_time}")
            print(f"conflict time = {conflict.cum_time}, "
                  + f"conflicts = {resolver.num_conflicts}")
        # Add new conflicts.  At the beginning there are none
        amat_count = 0
        conflicts = []
        amat = None
        for amat in resolver.get(times = rtimes, verbose = verbose):
            amat_count += 1
            if verbose > 1:
                print(f"A:\n{amat}")
            # Check validity
            if (old_amat == amat).all():
                raise ValueError("A matrix didn't change!")
            old_amat = amat.copy()
            # Give A (as assumptions) to model2 to find conflicts.
            lconf = list(conflict.get_conflicts(amat, times,
                                                largest = largest,
                                                verbose = verbose))
            if len(lconf) == 0:
                found_solution = True
                break
            conflicts += lconf
        if found_solution or amat_count == 0:
            break
        # Only do this for the side effects
        list(map(resolver.add_conflict, conflicts))

    if verbose > 0:
        print(f"Final conflict time = {conflict.cum_time}, "
              + f"conflicts={resolver.num_conflicts}")
        print(f"duplicates = {resolver.duplicates}")
        print(f"Total passes: {pass_no}, resolve time = {resolver.cum_time}.")
    if minimal > 0 and amat is None:
        minimizer = resolver.minimal if minimal == 1 else resolver.minimal_ux
        minimal_conflicts = minimizer(verbose=mverbose)
        if verbose > 0:
            print(f"Final resolve time = {resolver.cum_time}")
        return minimal_conflicts
    return amat
