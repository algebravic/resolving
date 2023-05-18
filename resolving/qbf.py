"""
Make a QBF model for the hypercube metric dimension.

Plan:

To certify a lower bound of B, construct the following model

forall X[i,j] i in range(B-1), j in range(N)
This corresponds to a putative resolving set.
There Exists Y[j], Z[j], j in range(N).
This corresponds to a pair (y,z) which are *not* resolved by the
putative resolving set.

Define U[i,j] = X[i,j] xor Y[j], for i,j
       V[i,j] = X[i,j] xor Z[j]

We have constraints \/_i Y[i] \/ \/_i Z[i] (rule out 0)
                    ~Y[i]\/~Z[i] for all i (not simulatenously 1)
                    sum_j U[i,j] + sum_j ~V[i,j] = N for all i

We also need symmetry breaking constaints for the X[i,j]:

sum_j X[i,j] <= sum_j X[i+1, j] for i in range(B-2)
X[0,j] = 0 (the all 0 is in the set).

if sum_j X[i,j] == sum_j X[i+1,j] then X[i] < (lex) X[i+1]

For 0 <= i < k < N, 0 <= r < B-1, E[i,k,r] is true
if and only if X[s,i] == X[s,k] for 0 <= s <= r.

Then we want E[i,k,r] ==> ~(X[r,i] == 0 /\ X[r,k] == 1)
(this is the prime symmetry breaking).

Inductively, we have E[i,k,0] = 1 for all i,k
E[i,k, r + 1] = E[i,k,r] /\ (X[r,i] == X[r,k])

"""

from typing import List, Tuple, Set, Iterable
from itertools import chain
from pysat.formula import CNF
from pysat.solvers import Solver

CLAUSE = List[int]

def _validate(variables: List[int] | Set[int] | int) -> Set[int] | None:
    """
    Must be positive ints non-repeating
    """
    if isinstance(variables, int):
        return set([variables]) if variables > 0 else None
    return (set(variables) if (isinstance(variables, (list, set))
                and all(isinstance(_, int) and _ > 0 for _ in variables))
            else None)
class QBF:
    """
    Manage Quantified Boolean formulas
    """

    def __init__(self):

        self._model = []
        self._quantifiers = [] # Quantifiers
        self._quantified = set()

    def _add(self, variables: List[int] | int, quantifier: str):
        """
        Add variables with a given quantifier.
        """
        varset = _validate(variables)
        if varset is None:
            raise ValueError(f"Variables must be a positive int {variables}")
        overlap = self._quantified.intersection(varset)
        if overlap:
            raise ValueError(f"Variables have already been quantified: {overlap}")
        self._quantified.update(varset)
        if len(self._quantifiers) == 0:
            self._quantifiers.append((quantifier, varset))
        elif self._quantifiers[-1][0] == quantifier:
            self._quantifiers[-1] = (quantifier, self._quantifiers[-1][1].union(varset))
        else:
            self._quantifiers.append((quantifier, varset))

    def exists(self, variables: List[int] | int):
        """
        Add exists variables
        """
        self._add(variables, 'e')

    def forall(self, variables: List[int] | int):
        """
        Add exists variables
        """
        self._add(variables, 'a')

    def add_model(self, cnf: CNF):
        """
        Add Clauses to the existing model.
        """
        self._model += cnf.clauses.copy()

    def check(self, solver = 'cd15') -> bool:
        """
        Check the satisfiability of the raw CNF.
        """
        return Solver(name = solver,
            bootstrap_with = self._model).solve()

    def full_quantify(self):
        """
        Determine quantification levels for unbound variables.

        For every bound variable, determine its level.
        For every bound variable determine its tentative level
        as being the maximum of the levels of the bound variables present
        in any clause containing it or its negation.

        Then, if its tentative level is at forall level, put it into
        and existential level at one level further down (perhaps merging).
        If its tentative level is at an existential level, then merge it into
        that level.  I think that that is correct.
        """

        levels = dict()
        unquantified = dict()
        for level, (_, variables) in enumerate(self._quantifiers):
            for variable in variables:
                levels[variable] = level
        # Now go through all the clauses
        for clause in self._model:
            quants = set(map(abs, clause)).intersection(self._quantified)
            unquants = set(map(abs, clause)).difference(quants)
            tlevel = max([levels[_] for _ in quants])
            unquantified.update(unquants)
            for elt in unquants:
                if elt in unquantified:
                    unquantified[elt] = max(unquantified[elt], tlevel)
                else:
                    unquantified[elt] = tlevel
        # Now go through the unquantified variables assigning it to a level
        updated = dict()
        for elt, level in unquantified.items():
            quantifier = self._quantifiers[level][0]
            if quantifier == 'a':
                # bump to the next level
                if len(self._quantifiers) <= level + 1:
                    self._quantifiers.append(('e', set()))
                self._quantifiers[level + 1][1].update([elt])
                if level + 1 not in updated:
                    updated[level + 1] = 0
                updated[level + 1] += 1
            else:
                self._quantifiers[level][1].update([elt])
                if level not in updated:
                    updated[level] = 0
                updated[level] += 1
        print(f"Updated {updated}")
        
    def write(self, filename: str):
        """
        Write out in QDimacs format.
        """

        # First validate
        support = set(chain(*(map(abs, _) for _ in self._model)))
        if not self._quantified.issubset(support):
            alone = self._quantified.difference(support)
            print(f"Warning: variables not in model {alone}")
        nvars = max(max(support), max(self._quantified))
        with open("{}.cnf".format(filename), 'w') as fil:
            fil.write(f"p cnf {nvars} {len(self._model)}\n")
            fil.write('\n'.join((f"{quant[0]} {' '.join(map(str, quant[1]))} 0")
                                  for quant in self._quantifiers))
            fil.write('\n')
            fil.write('\n'.join((' '.join(map(str, _)) + ' 0') for _ in self._model))
            fil.write('\n')
