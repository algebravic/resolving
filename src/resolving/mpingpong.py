"""
Use Ping Pong with Max Sat.

The original problem is

Exists a, g(a), forall x, g(x) ==> ~phi(x,a).

This will be solved by two interacting solvers.
Solver (A) will be Exists a, g(a), C_i
where C_i is a list of forbidden subconfigurations of the a variables.

Solver (A) will either be UNSAT (in which case the entire problem is UNSAT),
or will produce a putative solution A.

The Max Sat solver (B) will have hard clauses g(x) /\ phi(x,a), and
soft unit clauses, one for each element of A.  It will try to find
the minimum number of assignments of the a variables that will will lead
to a SAT instance.  It will then pass that subset of assignments back
to solver (A) which will add a clause to forbid those.

Note: the subset indicated by the b_{i,j} variables must be nonempty.
So we need a hard clause of OR_{i,j} b_{i,j}

We would like to know if some subset of assignments to the a variables
will have a resolving set (regardless of the assignment to the other variables).

Again, I'm having problems with quantifiers.
What we're after is to find a subset of the assignment of the a variables,
so that *for all* assignments to the remaining variables there is an x satsifying
g(x) /\ phi(a,x).
So that means that that we want that assignment to provoke SAT for g(x) /\ phi(a,x).

Clearing up the confusion: Suppose we have a strict subset of the a variables, S,
and an assignment to them, so that there is an assignment to the a variables in ~S,
and to the x variables so that g(x) /\ phi(a,x) is SAT.  This does not mean that
there is an assignment to the variables in ~S so that the above expression is not
UNSAT.  So we want all such assignments to be SAT, i.e. a tautology.  This means
that we want g(x) ==> ~phi(a,x) to be UNSAT.  We want the mininum sized set S
for partial assignment, which does this.  If we solve the MAXSAT problem, maximizing
the number of assignments, we know that increasing that set by 1 elements makes
it forbidden


The soft clauses for Max Sat solver (B) will be of the form

~b_{i,j}
where there are corresponding hard clauses
b_{i,j} ==> (a_{i,j} == A_{i,j}) [Note that this is a 2-clause since A_{i,j} is constant]
The b_{i,j} variables will determine the subset of indices that are active.
"""

from typing import Iterable, List, Tuple, Dict
from itertools import product, islice
from collections import Counter
import numpy as np
from pysat.formula import CNF, IDPool, WCNF
from pysat.solvers import Solver
from pysat.card import CardEnc, EncType
from pysat.examples.optux import OptUx
from pysat.examples.musx import MUSX
from .lex import lex_compare, Comparator, standard_lex
from .logic import MODEL
from .logic import set_equal, set_and
from .bdd import not_equal
from .symmbreak import double_lex, snake_lex


CLAUSE = List[int]
FORMULA = List[CLAUSE]
CONFLICT = Tuple[int,...]

class Minimizer:
    """
    A class to find the minimum set of forbidden coordinates of the A matrix
    which will still yield a counterexample.
    """

    
