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

We have constraints |_i Y[i] | |_i Z[i] (rule out 0)
                    ~Y[i]|~Z[i] for all i (not simulatenously 1)
                    sum_j U[i,j] + sum_j ~V[i,j] = N for all i

We also need symmetry breaking constaints for the X[i,j]:

sum_j X[i,j] <= sum_j X[i+1, j] for i in range(B-2)
X[0,j] = 0 (the all 0 is in the set).

if sum_j X[i,j] == sum_j X[i+1,j] then X[i] < (lex) X[i+1]

For 0 <= i < k < N, 0 <= r < B-1, E[i,k,r] is true
if and only if X[s,i] == X[s,k] for 0 <= s <= r.

Then we want E[i,k,r] ==> ~(X[r,i] == 0 & X[r,k] == 1)
(this is the prime symmetry breaking).

Inductively, we have E[i,k,0] = 1 for all i,k
E[i,k, r + 1] = E[i,k,r] & (X[r,i] == X[r,k])

"""

from typing import List, Set, Iterable
from itertools import chain
from pysat.formula import CNF
from pysat.solvers import Solver

CLAUSE = List[int]
FORMULA = List[CLAUSE]

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
        self._quantified = dict()
        self._dependencies = dict()

    def _add(self, variables: List[int] | int, quantifier: str):
        """
        Add variables with a given quantifier.
        """
        varset = _validate(variables)
        if varset is None:
            raise ValueError(f"Variables must be a positive int {variables}")
        overlap = set(self._quantified.keys()).intersection(varset)
        if overlap:
            raise ValueError(f"Variables have already been quantified: {overlap}")
        self._quantified.update({_: quantifier for _ in varset})
        if len(self._quantifiers) == 0 or self._quantifiers[-1][0] != quantifier:
            self._quantifiers.append((quantifier, varset))
        else:
            self._quantifiers[-1] = (quantifier,
                                     self._quantifiers[-1][1].union(varset))

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

    def unquantified(self, form: FORMULA) -> List[int]:
        """
        Return all variables in the formula that are not quantified.
        """
        support = set(chain(*(map(abs, _) for _ in form)))
        return list(support.difference(set(self._quantified.keys())))

    def dependency(self, variable: int, varlist: List[int] | int):
        """
        Add dependencies to variable.
        """
        if variable in self._quantified:
            raise ValueError(f"variable {variable} is already quantified")
        if variable not in self._dependencies:
            self._dependencies[variable] = set()
        realvars = [varlist] if isinstance(varlist, int) else varlist
        if not all(_ in self._quantified and self._quantified[_] == 'a'
                   for _ in realvars):
            raise ValueError("Not all variables universally quantified")
        self._dependencies[variable].update(realvars)
        
    def push_down(self):
        """
        Make all the explicitly unquantified variables existential
        at the lowest level.
        """

        support = set(chain(*(map(abs, _) for _ in self._model)))
        not_quant = support.difference(
            set(self._quantified.keys()).union(self._dependencies.keys()))
        self.exists(list(not_quant))
        print(f"# unquantified = {len(not_quant)}")

    def _render(self) -> Iterable[str]:
        """
        Output the lines for the model.
        """
        # Output the quantifier lines
        yield from ((f"{quant[0]} {' '.join(map(str, quant[1]))} 0")
                                  for quant in self._quantifiers)
        # Output the dependency lines, if any
        yield from ((f"d {key} {' '.join(map(str,val))} 0")
                                for key, val in self._dependencies.items())
        # Finally output the model
        yield from ((' '.join(map(str, _)) + ' 0') for _ in self._model)

    def write(self, filename: str):
        """
        Write out in QDimacs format.
        """

        # First validate
        # Make sure everything is quantified
        self.push_down()
        support = set(chain(*(map(abs, _) for _ in self._model)))
        if not set(self._quantified.keys()).issubset(support):
            alone = set(self._quantified.keys()).difference(support)
            print(f"Warning: variables not in model {alone}")
        nvars = max(chain(support, self._quantified.keys()))
        with open(f"{filename}.cnf", 'w', encoding='utf8') as fil:
            fil.write(f"p cnf {nvars} {len(self._model)}\n")
            fil.write('\n'.join(self._render()))
            fil.write('\n')
