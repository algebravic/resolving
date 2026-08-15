"""
Dynamic row/column symmetry breaking for a Boolean matrix, via PySAT's
IPASIR-UP user-propagator interface (CaDiCaL 1.9.5 / 3.0.0 / MinisatEP).

Instead of posting static symmetry-breaking clauses up front, this attaches a
propagator that, whenever the solver reaches a complete model, checks whether
the model is the lexicographically smallest element of its orbit under a given
set of row/column permutations. If some permutation g maps the model to a
lex-smaller matrix, the model is rejected and the corresponding lex-leader
conflict clause is handed back to the solver, which backtracks and continues
(keeping its learned clauses and the trail -- no restart from the root).

Convention
----------
- Matrix X = (x[i][j]), 0 <= i < r, 0 <= j < c.
- Cells are ordered ROW-MAJOR: rank(i, j) = i * c + j.
- A symmetry g = (row_perm, col_perm) maps cell (i, j) to its SOURCE cell
  (row_perm[i], col_perm[j]); the permuted matrix Y has
  Y[i][j] = X[row_perm[i]][col_perm[j]].
- CANONICAL = lexicographically minimal in row-major order.  We keep X iff
  X <=_lex Y for every g, i.e. g witnesses a violation when Y <_lex X.

The actual clause math lives in `lex_leader_clause`, which has NO pysat
dependency so it can be unit-tested standalone.
"""

try:
    from pysat.engines import Propagator
    _HAVE_PYSAT = True
except Exception:                      # pysat not installed -> still importable
    Propagator = object
    _HAVE_PYSAT = False


class RowColSymmetry:
    """A row/column permutation, 0-indexed.

    row_perm[i] = pi(i), col_perm[j] = sigma(j).
    `source(i, j)` returns the cell that supplies Y[i][j].
    """
    __slots__ = ("row_perm", "col_perm")

    def __init__(self, row_perm, col_perm):
        self.row_perm = tuple(row_perm)
        self.col_perm = tuple(col_perm)

    def source(self, i, j):
        return self.row_perm[i], self.col_perm[j]


def _dedup(literals):
    """Remove duplicate literals, preserving order.  Signs are always
    consistent by construction, so this never yields a tautology; we assert
    that to fail loudly if an assumption is ever violated."""
    seen = set()
    out = []
    for lit in literals:
        if lit in seen:
            continue
        assert -lit not in seen, "tautological lemma -- construction invariant broken"
        seen.add(lit)
        out.append(lit)
    return out


def lex_leader_clause(r, c, var, g, val):
    """Return the lex-leader conflict clause if permutation `g` makes the
    (complete) assignment `val` lex-greater than its image, else None.

    Parameters
    ----------
    r, c : matrix dimensions
    var  : callable (i, j) -> positive int variable id
    g    : object with .source(i, j) -> (si, sj)
    val  : callable var_id -> 0/1  (value of every matrix variable)
    """
    guards = []
    for i in range(r):
        for j in range(c):
            u_var = var(i, j)
            si, sj = g.source(i, j)
            v_var = var(si, sj)
            u = val(u_var)
            v = val(v_var)
            if u == v:
                if u_var != v_var:                  # prefix stays equal
                    if u == 0:
                        guards.append(u_var)
                        guards.append(v_var)
                    else:
                        guards.append(-u_var)
                        guards.append(-v_var)
                continue
            # first strict difference at (i, j)
            if u == 1 and v == 0:                   # X > Y here -> violation
                return _dedup(guards + [-u_var, v_var])
            return None                             # u == 0, v == 1 -> X < Y, ok
    return None                                     # g fixes the assignment


class RowColLexLeader(Propagator):
    """IPASIR-UP propagator enforcing lex-min canonicity under row/col perms.

    The check runs at complete models (`check_model`).  This is the simplest
    correct strategy ("lazy" / leaf-level lex-leader).  Partial pruning during
    search can be layered on later in `propagate`.
    """

    def __init__(self, nrows, ncols, var, symmetries, collect_all=False):
        self.r = nrows
        self.c = ncols
        self.var = var                       # (i, j) -> variable id
        self.symmetries = list(symmetries)
        self.collect_all = collect_all       # queue every violator vs. first only
        self.value = {}                      # var id -> 0/1
        self.trail = []
        self.levels = [0]
        self._queue = []
        self.cell_vars = [var(i, j) for i in range(nrows) for j in range(ncols)]

    # -- the variables the engine needs the solver to observe --
    def observed(self):
        return list(self.cell_vars)

    # -- notifications: keep the partial assignment incremental --
    def on_assignment(self, lit, fixed=False):
        self.trail.append(lit)
        self.value[abs(lit)] = 1 if lit > 0 else 0

    def on_new_level(self):
        self.levels.append(len(self.trail))

    def on_backtrack(self, to):
        target = self.levels[to]
        while len(self.trail) > target:
            self.value.pop(abs(self.trail.pop()), None)
        del self.levels[to + 1:]

    # -- callbacks --
    def decide(self):
        return 0                              # defer branching to the solver

    def propagate(self):
        return []                             # no eager propagation (yet)

    def provide_reason(self, lit):
        return []                             # unused unless propagate() fires

    def add_clause(self):
        return self._queue.pop(0) if self._queue else []

    def check_model(self, model):
        val = self.value.__getitem__          # all cell vars are observed/assigned
        rejected = False
        for g in self.symmetries:
            clause = lex_leader_clause(self.r, self.c, self.var, g, val)
            if clause is not None:
                self._queue.append(clause)
                rejected = True
                if not self.collect_all:
                    break
        return not rejected


def solve_with_symmetry(cnf, nrows, ncols, var, symmetries, name="cadical195"):
    """Convenience wrapper.  `var(i, j)` must agree with the variable ids used
    when building `cnf`.  Returns (is_sat, model_or_None)."""
    if not _HAVE_PYSAT:
        raise RuntimeError("pysat is required to actually solve")
    from pysat.solvers import Cadical195
    prop = RowColLexLeader(nrows, ncols, var, symmetries)
    with Cadical195(bootstrap_with=cnf) as solver:
        solver.connect_propagator(prop)
        # NOTE: method name is `observe` in current PySAT; verify against your
        # installed version (rolling release).  Harmless if connect already
        # observed them.
        for v in prop.observed():
            solver.observe(v)
        sat = solver.solve()
        return sat, (solver.get_model() if sat else None)
