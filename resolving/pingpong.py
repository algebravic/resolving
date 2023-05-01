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
from itertools import product, chain
from collections import Counter
import numpy as np
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from .lex import lex_compare, Comparator
from .logic import CLAUSE, FORMULA, MODEL
from .logic import negate, set_equal, set_xor, set_and, special_less

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
        self._alit = {_ : self._pool.id(('A',) + _)
                      for _ in product(range(mdim), range(dim))}
        self._xlit = {_: self._pool.id(('X', _)) for _ in range(self._dim)}
        self._ylit = {_: self._pool.id(('Y', _)) for _ in range(self._dim)}
        self._generate()
        self._solve = Solver(name = solver,
                             bootstrap_with = self._cnf,
                             use_timer = True, **kwds)

    def _generate(self):
        """
        The second model.
        """
        # x and y can't both be 1
        xlits = [self._xlit[_] for _ in range(self._dim)]
        ylits = [self._ylit[_] for _ in range(self._dim)]
        self._cnf.extend([[-_[0], - _[1]] for _ in zip(xlits, ylits)])
        # cnf.append(xlits + ylits) # Not identically 0
        # Support (even) is > 2
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
        self._cnf.extend(list(lex_compare(ylits, xlits,
                                          Comparator.LESS,
                                          self._pool)))
        for kind in range(self._mdim):
            for ind in range(self._dim):
                self._cnf.extend(set_and(self._pool.id(('B', kind ,ind)),
                                         self._pool.id(('A', kind, ind)),
                                         self._pool.id(('X', ind))))
                self._cnf.extend(set_and(self._pool.id(('C', kind ,ind)),
                                         self._pool.id(('A', kind, ind)),
                                         self._pool.id(('Y', ind))))
                self._cnf.extend(CardEnc.equals(
                    lits=([self._pool.id(('B', kind, _))
                           for _ in range(self._dim)]
                          + [- self._pool.id(('C', kind, _))
                             for _ in range(self._dim)]),
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
                       for key, lit in self._alit.items()]
        if verbose > 1:
            print(f"assumptions = {assumptions}")
        for _ in range(times):
            status = self._solve.solve(assumptions = assumptions)
            if verbose > 0:
                print(f"status CONFLICTS = {status} time = {self._solve.time()}")
            if not status:
                return
            # get counterexample
            # forbid it in solve2
            # add a conflict in solve1
            model = self._solve.get_model()
            xval = getvec(self._pool, 'X', model)
            yval = getvec(self._pool, 'Y', model)
            if verbose > 0:
                print(f"counterexample = {xval - yval}")
                # Check it
                chk = amat @ (xval - yval)
                if not (chk == 0).all():
                    raise ValueError(f"residual = {chk}")
            # Forbid this value
            # Should we make it conditional on the assumptions?
            forbid = ([int(1 - 2*xval[_]) * self._pool.id(('X', _))
                       for _ in range(self._dim)]
                      + [int(1 - 2*yval[_]) * self._pool.id(('Y', _))
                         for _ in range(self._dim)])
            # Make this conditional on the assumptions
            self._solve.add_clause([- _ for _ in assumptions] + forbid)

            yield xval - yval

    @property
    def census(self):
        """
        Get the initial model clause census.
        """
        return Counter(map(len, self._cnf.clauses))

class Resolve:
    """
    A class to find a putative resolving matrix.
    """

    def __init__(self, dim: int, mdim: int,
                 alt: bool = False,
                 solver = 'cd15',
                 encode = 'totalizer',
                 **kwds):

        self._dim = dim
        self._mdim = mdim
        self._encoding = getattr(EncType, encode, EncType.totalizer)
        self._cnf = CNF()
        self._pool = IDPool()
        self._alits = {_ : self._pool.id(('A',) + _)
                       for _ in product(range(self._mdim),range(self._dim))}
        if alt:
            self._model1()
        else:
            self._model2()
        self._solve = Solver(name = solver,
                             bootstrap_with = self._cnf,
                             use_timer = True, **kwds)

    @property
    def census(self):
        """
        The clause census.
        """
        return Counter(map(len, self._cnf.clauses))

    def get(self, verbose: int = 0) -> np.ndarray | None:
        """
        Get a resolving matrix
        """
        status = self._solve.solve()
        if verbose > 0:
            print(f"RESOLVE status = {status} time = {self._solve.time()}")
        if not status:
            return None
        model = self._solve.get_model()
        amat = extract_mat(self._pool, 'A', model)
        # Check that A has changed.
        if verbose > 0:
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
        """
        Add conflict clauses.
        """
        for clause in negate(
                self._pool,
                chain(*(CardEnc.equals(
                    lits = ([self._alits[kind, _]
                             for _ in range(self._dim) if xval[_] == 1]
                            + [- self._alits[kind, _]
                               for _ in range(self._dim) if xval[_] == -1]),
                    bound = (xval == -1).sum(),
                    encoding = self._encoding,
                    vpool = self._pool).clauses
                        for kind in range(self._mdim)))):
            self._solve.add_clause(clause)

    def _model1(self):
        """
        First model with symmetry breaking
        """
        # Non zero first row
        # Everything is increasing so others are nonzero
        self._cnf.append([self._pool.id(('A', 0, _)) for _ in range(self._dim)])
        for kind in range(self._mdim - 1):
            self._cnf.extend(list(special_less(self._pool,
                [self._pool.id(('A', kind, _)) for _ in range(self._dim)],
                [self._pool.id(('A', kind + 1, _)) for _ in range(self._dim)])))
        for kind in range(self._mdim):
            self._cnf.extend(CardEnc.atmost(lits =
                                            [self._pool.id(('A', kind, _))
                                             for _ in range(self._dim)],
                                            bound = self._dim // 2,
                                            encoding = self._encoding,
                                            vpool = self._pool))
        # Create the column constraints
        # Row -1 everything is equal
        # E[k,i,j] = (A[:k+1,i] == A[:k+1,j]).all()
        self._cnf.extend([[self._pool.id(('E', -1, ind, jind))]
                          for jind in range(1, self._dim)
                          for ind in range(jind)])
        # The remaining rows
        # defined = set()
        for kind in range(self._mdim):
            for jind in range(1, self._dim):
                for ind in range(jind):
                    # Z[k,i,j] <-> A[k,i] == A[k,j]
                    self._cnf.extend(set_xor(-self._pool.id(('Z', kind, ind, jind)),
                                             self._pool.id(('A', kind, ind)),
                                             self._pool.id(('A', kind, jind))))
                    # E[k,i,j] <-> Z[k,i,j] /\ E[k-1,i,j]
                    # defined.add((kind, ind, jind))
                    self._cnf.extend(set_and(self._pool.id(('E', kind, ind, jind)),
                                             self._pool.id(('E', kind-1, ind, jind)),
                                             self._pool.id(('Z', kind, ind, jind))))
                    # E[k-1,i,j] /\ A[k,i] -> A[k,j]
                    # Shift all 1's in a sector to the left
                    self._cnf.append([-self._pool.id(('E',kind-1, ind, jind)),
                                 -self._pool.id(('A', kind, ind)),
                                 self._pool.id(('A', kind, jind))])
        # Distinct columns
        # req = {(mdim - 1, ind, jind) for jind in range(1, dim) for ind in range(jind)}
        # print(f"req = {sorted(req)}")
        # if not req.issubset(defined):
        #    raise ValueError(f"Not defined {req.difference(defined)}")
        self._cnf.extend([[-self._pool.id(('E', self._mdim-1, ind, jind))]
                          for jind in range(1, self._dim)
                          for ind in range(jind)])

    def _model2(self):
        """
        Simpler model with double lexical ordering.
        """

        # Non zero first row
        # Everything is increasing so others are nonzero
        self._cnf.append([self._alits[0, _] for _ in range(self._dim)])
        # Double sorted increasing
        for ind in range(self._mdim-1):
            self._cnf.extend(list(lex_compare([self._alits[ind, _]
                                               for _ in range(self._dim)],
                                              [self._alits[ind + 1, _]
                                               for _ in range(self._dim)],
                                              Comparator.LESS, self._pool)))
        for ind in range(self._dim-1):
            self._cnf.extend(list(lex_compare([self._alits[_, ind]
                                               for _ in range(self._mdim)],
                                              [self._alits[_, ind+1]
                                               for _ in range(self._mdim)],
                                              Comparator.LESS, self._pool)))
        for ind in range(self._mdim):
            self._cnf.extend(CardEnc.equals(lits =
                                            [self._alits[ind, _]
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
              **kwds) -> np.ndarray | None:
    """
    Ping Pong method.
    """
    resolver = Resolve(dim, mdim - 1, solver=solver, encode=encode, alt = alt, **kwds)

    if verbose > 0:
        print(f"Resolve census = {resolver.census}")

    conflict = Conflict(dim, mdim - 1, solver=solver, encode=encode, **kwds)

    if verbose > 0:
        print(f"Conflict census = {conflict.census}")

    old_amat = np.zeros((mdim - 1, dim), dtype = np.int8)

    while True:
        # Add new conflicts.  At the beginning there are none
        amat = resolver.get()
        if verbose > 0:
            print(f"A:\n{amat}")
        if amat is None:
            return None
        # Check validity
        if (old_amat == amat).all():
            raise ValueError("A matrix didn't change!")
        old_amat = amat.copy()

        # Give A (as assumptions) to model2 to find conflicts.
        con_count = 0
        for xval in conflict.get_conflicts(amat, times, verbose = verbose):
            con_count += 1
            resolver.add_conflict(xval)
        if con_count == 0: # success!
            return amat
