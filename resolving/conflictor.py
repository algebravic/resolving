"""
The Conflictor.
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
