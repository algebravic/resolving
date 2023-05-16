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

CLAUSE = List[int]

def _validate(variables: List[int] | Set[int]) -> Set[int] | None:
    """
    Must be positive ints non-repeating
    """
    if not (isinstance(variables, (list, set))
            and all(isinstance(_, int) and _ > 0 for _ in variables)):
        return None
    return set(variables)

class QBF:
    """
    Manage Quantified Boolean formulas
    """

    def __init__(self):

        self._model = []
        self._quantifiers = [] # Quantifiers
        self._quantified = set()

    def _add(self, variables: List[int], quantifier: str):
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
        elif self._quantifiers[0][0] == quantifier:
            self._quantifiers[0] = (quantifier, self._quantifiers[0][1].union(varset))
        else:
            self._quantifiers.append((quantifier, varset))

    def exists(self, variables: List[int]):
        """
        Add exists variables
        """
        self._add(variables, 'e')

    def forall(self, variables: List[int]):
        """
        Add exists variables
        """
        self._add(variables, 'a')

    def add_model(self, cnf: CNF):
        """
        Add Clauses to the existing model.
        """
        self._model += cnf.clauses.copy()

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
