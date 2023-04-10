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
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType
from itertools import chain
from .lex import lex_compare
from .check import resolvable_model

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
            raise ValueError("Variables must be a positive int")
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
            fil.write(f"p cnf {len(self._model)} {nvars}\n")
            fil.write('\n'.join((f"{quant[0]} {' '.join(map(str, quant[1]))} 0")
                                  for quant in self._quantifiers))
            fil.write('n')
            fil.write('\n'.join(' '.join(map(str, _)) + '0') for _ in self._model)
            fil.write('n')

def parity(num: int, par: int) -> Iterable[Tuple[int, ...]]:
    """
    Generate all odd/even parity of length n.
    """
    if num == 0:
        if par == 0:
            yield tuple()
        return
    yield from ((0,) + _ for _ in parity(num - 1, par))
    yield from ((1,) + _ for _ in parity(num - 1, 1 - par))

def parity_constraint(lits: List[int], par: int) -> Iterable[List[int]]:
    """
    Parity constrained clauses.
    """
    num = len(lits)
    # Forbid opposite parity elements
    yield from ( ( (1-2 * _[0]) * _[1] for _ in zip(cons, lits))
                 for cons in parity(num, 1 - par))

def pre_post(pre: List[CLAUSE],
             post: List[CLAUSE],
             pool: IDPool) -> Iterable[CLAUSE]:
    """
    Generate clauses for pre ==> post
    """
    cvars = []
    for clause in pre:
        if len(clause) == 1:
            cvars.append(- clause[0])
        else:
            new_var = pool._next()
            cvars.append(- new_var)
            yield [-new_var] + clause
            yield from ([new_var, -_] for _ in clause)
    yield from (cvars + _ for _ in post)

def equiv(lft: List[CLAUSE],
          rgt: List[CLAUSE],
          pool: IDPool) -> Iterable[CLAUSE]:
    """
    Generate clauses for pre <==> post
    """
    yield from pre_post(lft, rgt, pool)
    yield from pre_post(rgt, lft, pool)
    
def quantified_hypercube(num: int, bound: int) -> Tuple[QBF, IDPool]:
    """
    Create a QBF formula for checking if beta_n = bound.
    """
    qbf = QBF()
    cnf = CNF()
    pool = IDPool()

    for ind in range(bound - 1):
        for jind in range(num):
            cnf.extend(parity_constraint([pool.id(('u', ind, jind)),
                                          pool.id(('x', ind, jind)),
                                          pool.id(('y', jind))], 1))
            cnf.extend(parity_constraint([pool.id(('v', ind, jind)),
                                          pool.id(('x', ind, jind)),
                                          pool.id(('z', jind))], 1))
    cnf.append([pool.id(('x', _)) for _ in range(num)] +
               [pool.id(('y', _)) for _ in range(num)])

    cnf.extend([[-pool.id(('y', _)), - pool.id(('z', _))]
                for _ in range(num)])
    for jind in range(num):
        cnf.extend(CardEnc.equals(lits =
                                  [pool.id(('u', jind, _))
                                   for _ in range(num)] +
                                  [- pool.id(('v', jind, _))
                                   for _ in range(num)],
                                  encoding = encoding,
                                  bound = num,
                                  vpool = pool))
        
    for ind in range(bound - 2):
        cnf.extend(CardEnc.atleast(lits =
                                   [pool.id(('x', ind+1, _))
                                    for _ in range(num)] +
                                   [-pool.id(('x', ind, _))
                                    for _ in range(num)],
                                   encoding = encoding,
                                   bound = num,
                                   vpool = pool))
    for ind in range(bound - 2):
        equ = CardEnc.equals(lits =
                             [pool.id(('x', ind+1, _))
                              for _ in range(num)] +
                             [-pool.id(('x', ind, _))
                              for _ in range(num)],
                             encoding = encoding,
                             bound = num,
                             vpool = pool)

        lexical = lex_less([pool.id(('x', ind, _)) for _ in range(num)],
                           [pool.id(('x', ind + 1, _)) for _ in range(num)])

        cnf.extend(pre_post(equ, lexical, pool))
        
