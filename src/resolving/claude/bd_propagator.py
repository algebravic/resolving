#!/usr/bin/env python3
"""
bd_propagator.py -- IPASIR-UP propagator for balanced-detecting matrix theory.

This is the "theory-side" of an SDP-SAT hybrid for balanced-detecting matrix
infeasibility. It plugs into PySAT's Propagator abstract class (which wraps
CaDiCaL's IPASIR-UP interface).

ARCHITECTURE:

  CNF encoding (external, e.g. from sat_encode_matrix.py):
    - Primary vars A_{i,j}: numbered 1 .. m*n
    - Aux vars z_{i,p,q} = A_{i,p} AND A_{i,q}: numbered m*n+1 .. m*n + m*C(n,2)
    - Definitional Tseitin clauses (3 per aux var) hardcoded in CNF
    - Optional: per-x constraints (large but standard)
    - Optional: static test-vector matrix constraints

  Propagator (this file):
    - Observes only the m*n primary A_{i,j} variables
    - On cb_check_found_model: reconstructs A, runs matrix filter,
      if it fails, extracts eigenvector, finds violating integer test vector v_int,
      adds a lemma clause blocking this model (weak version), or generates
      strong PB->CNF cut (future work)
    - cb_propagate: not implemented in v1

USAGE:

    from pysat.solvers import Cadical195
    from pysat.formula import CNF
    from bd_propagator import BDPropagator

    m, n = 7, 14
    # 1) Build CNF with primary + aux vars and per-x constraints
    cnf = build_base_cnf(m, n)   # user-provided
    # 2) Create propagator
    prop = BDPropagator(m, n, primary_var_start=1, aux_var_start=m*n+1)
    # 3) Attach and solve
    with Cadical195(bootstrap_with=cnf.clauses) as solver:
        solver.connect_propagator(prop)
        # Observe the primary variables so the propagator gets notifications
        for var in prop.observed_variables():
            solver.observe(var)
        if solver.solve():
            # Even if solve returns True, our propagator may have added lemmas
            model = solver.get_model()
            # ...
"""

import numpy as np
from bd_theory import BDTheory, compile_pb_to_neg_model_lemma


try:
    from pysat.engines import Propagator
    HAVE_PYSAT = True
except ImportError:
    HAVE_PYSAT = False
    # Mock base class for standalone testing
    class Propagator:
        """Mock base class matching PySAT's Propagator abstract class.

        Real PySAT Propagator has these methods:
          def notify_assignment(self, lits):        # batched: list of int lits
          def notify_new_decision_level(self):
          def notify_backtrack(self, new_level):
          def cb_check_found_model(self, model):    # returns bool; model is list of lits
          def cb_decide(self):                      # returns int lit or 0
          def cb_propagate(self):                   # returns int lit or 0
          def cb_add_reason_clause_lit(self, propagated_lit):
              # returns int lit; called repeatedly, ending with 0
          def cb_has_external_clause(self):         # returns bool
          def cb_add_external_clause_lit(self):     # returns int lit; ends with 0
        """
        def __init__(self):
            pass


class BDPropagator(Propagator):
    """IPASIR-UP propagator for the balanced-detecting matrix condition.

    Variable numbering convention:
      A_{i,j}  ->  primary_var_start + i*n + j       (i in [0,m), j in [0,n))
      z_{i,p,q} (with p<q)  ->  aux_var_start + i*C(n,2) + pair_index(p,q,n)

    The propagator observes only the primary A_{i,j} variables. When a full
    assignment is found, it reconstructs A and runs the matrix filter.
    """

    def __init__(self, m, n, primary_var_start=1, aux_var_start=None,
                 lemma_strategy="weak", verbose=False):
        """
        Args:
          m, n:                matrix dimensions
          primary_var_start:   first CNF var id for A_{0,0} (typically 1)
          aux_var_start:       first CNF var id for aux z_{0,0,1}. If None, no
                               aux vars are used (weak lemmas only).
          lemma_strategy:      "weak" = block this model only (single clause);
                               "cut"  = compile PB cut to CNF (future work)
          verbose:             print diagnostics
        """
        super().__init__()
        self.m = m
        self.n = n
        self.pv0 = primary_var_start
        self.av0 = aux_var_start
        self.theory = BDTheory(m, n)
        self.lemma_strategy = lemma_strategy
        self.verbose = verbose

        # State tracked across notifications
        self.assignment = {}                             # var_id -> bool
        self.decision_levels = [0]                       # stack of var counts at each level (for backtracking)
        self.current_level = 0
        # External clause queue
        self.pending_clauses = []                        # list of clauses to add
        self.current_clause_iter = None
        self.current_lit_iter = None

        # Stats
        self.n_models_checked = 0
        self.n_models_rejected = 0
        self.n_cuts_added = 0
        self.rejection_cache = set()                     # frozenset of true primary lits -> already-rejected

    # ------- Variable indexing helpers -------------------------------------

    def var_A(self, i, j):
        return self.pv0 + i * self.n + j

    def var_z(self, i, p, q):
        assert self.av0 is not None
        if p > q:
            p, q = q, p
        return self.av0 + i * self.theory.num_pairs + self.theory.pair_idx[(p, q)]

    def observed_variables(self):
        """Return the list of variables the propagator wants to observe.
        For matrix-filter we only need the primary A_{i,j} vars.
        """
        return [self.var_A(i, j) for i in range(self.m) for j in range(self.n)]

    # ------- Reconstruct A from assignment ---------------------------------

    def build_A(self):
        """From self.assignment, build an m x n int matrix. Unassigned = 0."""
        A = np.zeros((self.m, self.n), dtype=np.int64)
        for i in range(self.m):
            for j in range(self.n):
                v = self.var_A(i, j)
                A[i, j] = 1 if self.assignment.get(v, False) else 0
        return A

    def all_primary_assigned(self):
        """Check if all primary A_{i,j} vars are assigned in self.assignment."""
        for i in range(self.m):
            for j in range(self.n):
                if self.var_A(i, j) not in self.assignment:
                    return False
        return True

    # ------- IPASIR-UP callbacks -------------------------------------------

    def notify_assignment(self, lits):
        """Batched notification of assignments. lits is a list of ints:
        positive = variable assigned True, negative = variable assigned False.
        """
        for lit in lits:
            var = abs(lit)
            val = lit > 0
            self.assignment[var] = val

    def notify_new_decision_level(self):
        self.current_level += 1
        self.decision_levels.append(len(self.assignment))

    def notify_backtrack(self, new_level):
        """Backtrack: undo assignments made after level `new_level`.
        We track by remembering the size of self.assignment at each level and
        rebuild if needed. Simpler approach: since we can't reliably know which
        assignments to undo without more info, clear and let notify_assignment
        rebuild. But CaDiCaL guarantees it will re-notify propagations that
        remain valid; we don't rely on the state persisting across backtracks
        for correctness.

        Actually, we need to properly undo. Simple approach: keep an "assignment
        history" list of (var, prev_value) tuples per level.
        """
        while self.current_level > new_level and self.decision_levels:
            # Truncate assignment history: everything past decision_levels[-1] was added
            # after this level.
            old_size = self.decision_levels.pop() if len(self.decision_levels) > 1 else 0
            # Determine which vars to remove: those added after old_size
            # For simplicity, clear vars with insertion order > old_size. Python dicts
            # preserve insertion order.
            keys = list(self.assignment.keys())
            for k in keys[old_size:]:
                del self.assignment[k]
            self.current_level -= 1

    def cb_decide(self):
        """Return a literal to decide, or 0 to let the solver decide.
        v1: always let the solver decide."""
        return 0

    def cb_propagate(self):
        """Return a literal to propagate, or 0 if no propagation.
        v1: no propagation."""
        return 0

    def cb_add_reason_clause_lit(self, propagated_lit):
        """Reason clause literals for a propagated literal. Return 0 to end.
        v1: no propagations, so never called."""
        return 0

    def cb_check_found_model(self, model):
        """Called when SAT solver finds a satisfying assignment. Return True to
        accept, or False plus enqueue an external clause via cb_has_external_clause
        + cb_add_external_clause_lit.

        Args:
          model: list of literals representing the full model.

        Returns:
          bool: True if we accept, False if we want to reject.
        """
        # Update self.assignment from model (in case notifications missed anything)
        for lit in model:
            var = abs(lit)
            self.assignment[var] = lit > 0

        # Verify all primary vars are assigned
        if not self.all_primary_assigned():
            # Unusual: model doesn't cover our observed vars. Accept it.
            return True

        self.n_models_checked += 1

        # Reconstruct A
        A = self.build_A()

        # Compute signature (which primary vars are true) to avoid re-rejecting
        true_prims = frozenset(v for v in self.observed_variables()
                               if self.assignment.get(v, False))
        if true_prims in self.rejection_cache:
            # Already rejected this before; solver shouldn't have returned it.
            # Safety: reject again with same lemma.
            if self.verbose:
                print(f"WARNING: model already in rejection cache; re-rejecting")

        # Run matrix filter
        ok_mat, min_eig, min_eigvec = self.theory.check(A)

        # Run per-x check as ground truth (v1 debugging)
        ok_agg = self.theory.aggregate_check(A)

        if self.verbose:
            print(f"[model check] agg={ok_agg}, mat={ok_mat}, min_eig={min_eig:.4f}")

        if ok_mat:
            # Matrix filter passes; matrix filter is only NECESSARY, not sufficient.
            # A model can pass the matrix filter but still not be balanced-detecting,
            # namely when there is exactly one balanced vector in ker(A) (that vec's
            # direction ends up aligning with 1, which is in ker(XXT) trivially, giving
            # a zero eigenvalue rather than negative).
            #
            # To keep the propagator sound, we do a direct BD verification here.
            # Cost: |B(n)| matrix-vector products, one-time per accepted model.
            from bd_theory import enum_balanced
            B = enum_balanced(self.n)                    # cached in future version
            failing_x = None
            for x in B:
                if not np.any(A @ x):
                    failing_x = x
                    break
            if failing_x is None:
                # Genuinely balanced-detecting
                return True
            # Matrix filter passed but direct BD check failed. Add a lemma
            # based on failing_x: at least one row must have A_i . failing_x != 0.
            # This is exactly the per-x constraint for failing_x, which our base
            # CNF may or may not have encoded.
            if self.verbose:
                print(f"  matrix filter passed but x={failing_x} is in ker(A); adding per-x lemma")
            # Weak lemma: block this model
            clause = []
            for i in range(self.m):
                for j in range(self.n):
                    v = self.var_A(i, j)
                    val_ij = self.assignment.get(v, False)
                    clause.append(-v if val_ij else v)
            self.pending_clauses.append(clause)
            self.rejection_cache.add(true_prims)
            self.n_models_rejected += 1
            self.n_cuts_added += 1
            return False

        # Matrix filter fails: extract cut
        v_int, val = self.theory.best_int_cut(A)
        if v_int is None:
            # Shouldn't happen if ok_mat is False and min_eig < 0
            if self.verbose:
                print(f"  matrix filter failed but no integer cut found; using weak lemma")
            v_int = None
        elif self.verbose:
            print(f"  cut: v_int = {v_int.tolist()}, v_int^T (Phi - XXT) v_int = {val}")

        # Generate lemma
        if self.lemma_strategy == "weak" or v_int is None:
            # Block this model: the disjunction of negations of currently-true primary vars
            # plus positive versions of currently-false primary vars.
            clause = []
            for i in range(self.m):
                for j in range(self.n):
                    v = self.var_A(i, j)
                    val_ij = self.assignment.get(v, False)
                    clause.append(-v if val_ij else v)
            self.pending_clauses.append(clause)
        else:
            # "cut" strategy: future work. For now fall back to weak.
            clause = []
            for i in range(self.m):
                for j in range(self.n):
                    v = self.var_A(i, j)
                    val_ij = self.assignment.get(v, False)
                    clause.append(-v if val_ij else v)
            self.pending_clauses.append(clause)

        self.rejection_cache.add(true_prims)
        self.n_models_rejected += 1
        self.n_cuts_added += 1

        return False                                     # reject the model

    def cb_has_external_clause(self):
        """Return True if there are external clauses ready to add."""
        return len(self.pending_clauses) > 0

    def cb_add_external_clause_lit(self):
        """Emit literals of pending clause(s), one at a time. Return 0 to end
        each clause; the solver calls this repeatedly.
        """
        if self.current_clause_iter is None:
            if not self.pending_clauses:
                return 0
            self.current_clause_iter = iter(self.pending_clauses.pop(0))
        try:
            return next(self.current_clause_iter)
        except StopIteration:
            self.current_clause_iter = None
            return 0                                     # end of clause

    # ------- Stats ---------------------------------------------------------

    def stats(self):
        return {
            'models_checked': self.n_models_checked,
            'models_rejected': self.n_models_rejected,
            'cuts_added': self.n_cuts_added,
            'unique_rejections': len(self.rejection_cache),
        }
