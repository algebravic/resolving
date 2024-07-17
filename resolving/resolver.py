"""
The Resolver.
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
from .logic import set_equal, set_and, set_xor
from .bdd import not_equal
from .symmbreak import double_lex, snake_lex
from .util import get_prefix, extract_mat, getvec, makevec, makemat, makecomp
from .maxtest import min_conflict
from .schreier_sims import schreier_sims_cuts
from .gensymm import encode_implications
from symmetry import row_snake_order, column_snake_order, reduce_impl
from symmetry import lex_double
from symmetry import get_extended_symmetry
from symmetry import make_implications

CONFLICT = Tuple[int,...]
CADICAL = 'Cadical195'

def xor_comp(pool: IDPool, rowl: np.ndarray, rowr: np.ndarray) -> Iterable[CLAUSE]:
    """
    Generate clauses for A < A XOR B
    """
    dim = rowl.shape[0]
    values = [pool.id() for _ in range(dim)]
    for ind, val in enumerate(values):
            yield from set_xor(val, int(rowl[ind]), int(rowr[ind]))
    yield from standard_lex(pool, list(map(int, rowl)), values)
        
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
                 solver = CADICAL,
                 encode = 'totalizer',
                 breaker: str = 'double_lex',
                 ss_cuts: bool = False,
                 xor_break: bool = False, # use xor symm break
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
        self._ss_cuts = ss_cuts
        self._xor_break = xor_break
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
            self.append([-assump, -indic1, -indic2])
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

    def _symmetry_break(self, breaker: str = 'double_lex'):
        """
          Use the indicated symmetry break

          Possiblities:
          'double_lex' : standard double lex
          'snake_row' : snake lex with row snake
          'snake_col' : snake lex with col snake
          other: the name of a file which contains the json of lex implicants
        """
        impl = []
        match breaker:
            case 'double_lex':
                impl = make_implications((self._mdim, self._dim),
                    row_wise, lex_double)
            case 'snake_row':
                impl = reduce_impl(make_implications((self._mdim, self._dim),
                    row_snake_order, lex_double))
            case 'snake_col':
                impl = reduce_impl(make_implications((self._mdim, self._dim),
                    column_snake_order, lex_double))
            case 'schreier-sims':
                impl = schreier_sims_cuts(self._dim, self._mdim)
            case ('covering', mnum, num):
                impl = get_extended_symmetry(
                    self._mdim, self._dim, mnum, num)
            case _:
                print(f"Unrecognized {breaker}, using default")
                impl = make_implications((self._mdim, self._dim),
                    row_wise, lex_double)
        self._cnf.extend(encode_implications(self._pool,
                                             self._avar,
                                             impl))
    def _nonzero_rows(self):

        return [[self._avar[ind, jind] for jind in range(self._dim)]
                for ind in range(self._mdim)]

    def _break_xor(self):
        " Only valid for rowwise order "
        for ind in range(1, self._mdim):
            self._cnf.extend(list(xor_comp(self._pool,
                                           amat[0], amat[ind])))
            self._cnf.extend(list(xor_comp(self._pool,
                                           amat[1], amat[0])))
    
    def _model2(self):
        """
        Simpler model with double lexical ordering.
        """

        # Non zero first row
        # Everything is increasing so others are nonzero
        # Double sorted increasing
        amat = np.array([[self._avar[ind, jind] for jind in range(self._dim)]
                         for ind in range(self._mdim)], dtype=int)
        # All rows nonzero
        for ind in range(self._mdim):
            # i-th row nonzero
            self._cnf.append(list(map(int, amat[ind])))
            
        if self._ss_cuts:
            self._cnf.extend([[-int(amat[leader]), int(amat[follower])]
                              for leader, follower
                              in schreier_sims_cuts(self._dim,
                                                    self._mdim)])
        else:
            if self._firstweight:
                amat = amat[:, 1:] # avoid first column
            breaker = snake_lex if self._snake > 0 else double_lex
            self._cnf.extend(list(breaker(self._pool,
                                        amat.T if self._snake > 1
                                          else amat)))
        if self._xor_break:
            for ind in range(1, self._mdim):
                self._cnf.extend(list(xor_comp(self._pool,
                                               amat[0], amat[ind])))
            self._cnf.extend(list(xor_comp(self._pool,
                                           amat[1], amat[0])))
