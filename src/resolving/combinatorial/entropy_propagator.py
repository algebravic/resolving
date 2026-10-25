class EntropyPropagator:
    """IPASIR-UP theory propagator for entropy pruning during SAT search."""
    
    def __init__(self, m, n):
        self.m, self.n = m, n
        # Precomputed entropy tables
        self.H_hyper = precompute_hypergeometric_entropies(n)
        self.H_pair = precompute_pair_entropies(n)
        # Partial profile state
        self.row_supports = [set() for _ in range(m)]     # which cols are 1
        self.row_zero_cols = [set() for _ in range(m)]    # which cols are known 0
        self.pending_clauses = []
    
    def notify_assignment(self, var, value):
        """Called when SAT solver sets a variable a_{i,j} to value."""
        i, j = decode(var)
        if value:
            self.row_supports[i].add(j)
        else:
            self.row_zero_cols[i].add(j)
        # Check whether current partial profile has entropy-feasible completion
        if not self.check_entropy_feasibility():
            # Derive explanation clause
            reason = self.explain_infeasibility()
            self.pending_clauses.append(reason)
    
    def notify_backtrack(self, level):
        """Rollback profile state."""
        # ...
    
    def check_propagation(self):
        """Return list of forced literals with their explanations."""
        # For each unassigned var, check if setting it to a value would break entropy
        forced = []
        for var in unassigned_vars():
            i, j = decode(var)
            if extending_row(i, j, True) breaks entropy:
                forced.append((-var, explanation))
            elif extending_row(i, j, False) breaks entropy:
                forced.append((var, explanation))
        return forced
    
    def check_entropy_feasibility(self):
        """Given partial row supports, check if any completion satisfies:
           - Univariate: Σ H(X_j, s) ≥ log₂C(n, s) for each s
           - Bivariate: Σ H(X_i, X_j, s) ≥ (m-1) log₂C(n, s)
           Under the constraint that unknown cells could go either way."""
        # For each row: current support size ∈ [known_ones, n - known_zeros]
        # For each pair: current intersection ∈ [known_common_ones, upper_bound]
        # Check whether ANY combination in these ranges is entropy-feasible.
        # Uses interval propagation on the profile lattice.
        ...
    
    def explain_infeasibility(self):
        """Produce a minimal set of literals whose assignment is entropy-infeasible."""
        # Extract the smallest subset of the current partial assignment that
        # already forces the entropy violation. This becomes the conflict clause.
        ...
