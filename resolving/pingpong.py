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
from itertools import product, chain, combinations
from collections import Counter
import numpy as np
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from .lex import lex_compare, Comparator
from .logic import MODEL
from .logic import negate, set_equal, set_and, special_less, big_or
from .bdd import not_equal

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
                 **kwds):

        self._dim = dim
        self._mdim = mdim
        self._encoding = getattr(EncType, encode, EncType.totalizer)
        self._cnf = CNF()
        self._pool = IDPool()
        self._avar = {_ : self._pool.id(('A',) + _)
                      for _ in product(range(mdim), range(dim))}
        self._xvar = {_: self._pool.id(('X', _)) for _ in range(self._dim)}
        self._yvar = {_: self._pool.id(('Y', _)) for _ in range(self._dim)}
        self._generate()
        self._solve = Solver(name = solver,
                             bootstrap_with = self._cnf,
                             use_timer = True, **kwds)
        self._cum_time = 0.0

    def _generate(self):
        """
        The Conflict formula
        """
        # x and y can't both be 1
        xlits = [self._xvar[_] for _ in range(self._dim)]
        ylits = [self._yvar[_] for _ in range(self._dim)]
        self._cnf.extend([[-_[0], - _[1]] for _ in zip(xlits, ylits)])

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

    def get_conflicts(self,
                      amat: np.ndarray,
                      times: int,
                      verbose: int = 0) -> Iterable[np.ndarray]:
        """
        Given an A matrix get up to times counterexamples to resolvability.
        """
        assumptions = [(2 * int(amat[key]) - 1) * lit
                       for key, lit in self._avar.items()]
        if verbose > 2:
            print(f"assumptions = {assumptions}")
        for _ in range(times):
            status = self._solve.solve(assumptions = assumptions)
            stime = self._solve.time()
            self._cum_time += stime
            if verbose > 1:
                print(f"status Conflicts = {status} time = {stime}")
            if not status:
                return
            # get counterexample
            # forbid it in solve2
            # add a conflict in solve1
            model = self._solve.get_model()
            xval = getvec(self._pool, 'X', model)
            yval = getvec(self._pool, 'Y', model)
            if verbose > 1:
                print(f"counterexample = {xval - yval}")
                # Check it
                chk = amat @ (xval - yval)
                if not (chk == 0).all():
                    raise ValueError(f"residual = {chk}")
            # Forbid this value
            # Note: must convert np.int8 to int
            forbid = ([int(1 - 2*xval[_]) * self._xvar[_] for _ in range(self._dim)]
                      + [int(1 - 2*yval[_]) * self._yvar[_] for _ in range(self._dim)])
            # Make this conditional on the assumptions
            self._solve.add_clause([- _ for _ in assumptions] + forbid)

            yield xval - yval

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
                 solver = 'cd15',
                 encode = 'totalizer',
                 **kwds):

        self._dim = dim
        self._mdim = mdim
        self._encoding = getattr(EncType, encode, EncType.totalizer)
        self._cnf = CNF()
        self._pool = IDPool()
        self._avar = {_ : self._pool.id(('A',) + _)
                       for _ in product(range(self._mdim),range(self._dim))}
        if alt_model:
            self._model1()
        else:
            self._model2()
        self._solve_name = solver
        self._solve = Solver(name = solver,
                             bootstrap_with = self._cnf,
                             use_timer = True, **kwds)
        self._cum_time = 0.0
        self._alt = alt

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

    def get(self, verbose: int = 0) -> np.ndarray | None:
        """
        Get a resolving matrix
        """
        status = self._solve.solve()
        stime = self._solve.time()
        self._cum_time += stime
        if verbose > 1:
            print(f"RESOLVE status = {status} time = {stime}")
        if not status:
            return None
        model = self._solve.get_model()
        amat = extract_mat(self._pool, 'A', model)

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
        return amat
    def add_conflict(self, xval: np.ndarray):
        (self.alt_add_conflict if self._alt else self.bdd_add_conflict)(xval)
            
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
        self._solve.append_formula(inequalities + [indicators])

    def alt_add_conflict(self, xval: np.ndarray):
        """
        Add clauses that forbid A@x = 0
        """
        indicators = []
        bound = (xval == -1).sum()
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
                self._solve.add_clause([-indic1] + clause)
            indic2 = self._pool._next()
            indicators.append(indic2)
            for clause in CardEnc.atleast(
                    lits = lits, bound = bound + 1,
                    encoding = self._encoding,
                    vpool = self._pool).clauses:
                self._solve.add_clause([-indic2] + clause)
        self._solve.add_clause(indicators)

    def _model1(self):
        """
        First model with symmetry breaking
        """
        # Non zero first row
        # Everything is increasing so others are nonzero
        self._cnf.append([self._avar[0, _] for _ in range(self._dim)])
        for kind in range(self._mdim - 1):
            self._cnf.extend(list(special_less(self._pool,
                [self._avar[kind, _] for _ in range(self._dim)],
                [self._avar[kind + 1, _] for _ in range(self._dim)])))
        for kind in range(self._mdim):
            self._cnf.extend(CardEnc.atmost(lits =
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
        self._cnf.append([self._avar[_, 0] for _ in range(self._mdim)])
        # Double sorted increasing
        for ind in range(self._mdim-1):
            self._cnf.extend(list(lex_compare(self._pool,
                                              [self._avar[ind, _]
                                               for _ in range(self._dim)],
                                              [self._avar[ind + 1, _]
                                               for _ in range(self._dim)],
                                              Comparator.LESS)))
        for ind in range(self._dim-1):
            self._cnf.extend(list(lex_compare(self._pool,
                                              [self._avar[_, ind]
                                               for _ in range(self._mdim)],
                                              [self._avar[_, ind+1]
                                               for _ in range(self._mdim)],
                                              Comparator.LESS)))
        for ind in range(self._mdim):
            self._cnf.extend(CardEnc.equals(lits =
                                            [self._avar[ind, _]
                                             for _ in range(self._dim)],
                                            bound = self._dim // 2,
                                            encoding = self._encoding,
                                            vpool = self._pool))

def ping_pong(dim: int, mdim: int,
              times: int = 1,
              verbose: int = 0,
              encode: str = 'totalizer',
              solver: str = 'cd15',
              alt: bool = False,
              alt_model: bool = False,
              check: bool = False,
              trace: int = 0,
              **kwds) -> np.ndarray | None:
    """
    Ping Pong method.
    """
    resolver = Resolve(dim, mdim - 1,
                       solver=solver,
                       encode=encode,
                       alt = alt,
                       alt_model = alt_model,
                       **kwds)

    if verbose > 1:
        print(f"Resolve census = {resolver.census}")

    conflict = Conflict(dim, mdim - 1, solver=solver, encode=encode, **kwds)

    if verbose > 1:
        print(f"Conflict census = {conflict.census}")

    old_amat = np.zeros((mdim - 1, dim), dtype = np.int8)
    total_conflicts = 0
    pass_no = 0
    while True:
        pass_no += 1
        if trace > 0 and pass_no % trace == 0:
            print(f"At pass {pass_no}, resolve time = {resolver.cum_time}")
            print(f"conflict time = {conflict.cum_time}, conflicts = {total_conflicts}")
        # Add new conflicts.  At the beginning there are none
        amat = resolver.get(verbose = verbose)
        if verbose > 1:
            print(f"A:\n{amat}")
        if amat is None:
            break
        # Check validity
        if (old_amat == amat).all():
            raise ValueError("A matrix didn't change!")
        old_amat = amat.copy()

        # Give A (as assumptions) to model2 to find conflicts.
        con_count = 0
        for xval in conflict.get_conflicts(amat, times, verbose = verbose):
            con_count += 1
            resolver.add_conflict(xval)
        total_conflicts += con_count
        if con_count == 0: # success!
            break

    if verbose > 0:
        print(f"conflict time = {conflict.cum_time}, conflicts={total_conflicts}")
        print(f"Total passes: {pass_no}, resolve time = {resolver.cum_time}.")
    return amat
