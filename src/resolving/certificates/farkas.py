""" Try to use LP to give a certificate for UNSAT """

from itertools import product, chain, combinations
from typing import List, Tuple
import cvxpy as cp
import numpy as np

def lp_model(num: int, mdim: int, conflicts: List[Tuple[int,...]]):
    """
      Set up quadratic program for m by n detecting matrix
      using the list of conflicts.

      If c is a conflict vector (column vector), and X is is
      the m by n matrix of unknowns, we must have

      (X c)^T (X c) = c^T X^T X c = Tr(c c^T X^T X) >= 1

      Make up variables for the entries of X^T X, as 0/1 variables
      and take the LP relaxation.
    """

    def big_index(ind: int, jind: int):
        return ind * dim + jind
    xvars = cp.Variable((mdim, dim), name = 'x')
    xxvars = cp.Variable((mdim * dim, mdim * dim), name = 'xx')
    objective = cp.sum(xvars)
    limitsx = chain(*([xvars[_] >==0, xvars[_] <=1]
        for _ in product(range(mdim), range(dim))))
    limitsxx = chain(*([xxvars[_] >==0, xxvars[_] <=1]
        for _ in product(range(mdim * dim), range(mdim * dim))))
    symxx = (xxvars[_] == xxvars[_[1],_[0]]
        for _ in combinations(range(mdim * dim), 2))
    idents = (xxvars[big_index(*_), big_index(*_)] == xvars[_]
        for _ in product(range(mdim), range(dim)))
    # xx[(i,j),(i',j')] = x[i,j] & x[i',j']
    # 
    links = chain(*([
        xxvars[big_index(*_[0]), big_index(*_[1])] <=
        xvars[_[0]],
        xxvars[big_index(*_[0]), big_index(*_[1])] <= 
        xvars[_[1]],
        xxvars[big_index(*_[0]), big_index(*_[1])] >= 
        xvars[_[0]] + xvars[_[1]] - 1]
        for _ in combinations(product(range(mdim),
                                      range(dim)), 2)))
    constraints = list(chain(limitsx, limitsxx, links,
        symxx,idents))
    # Now for the conflicts
    # Make X^T X
    xxmat = {(ind, jind): cp.sum([xxvars[big_index(ind, kind),
        big_index(kind, jind)] for kind in range(mdim)])
        for ind, jind in product(range(dim), repeat=2)}
    for conf in conflicts:
        # Column vector
        cvec = np.array(conf)
        cmat = np.outer(cvec, cvec)
        constraints.append(cp.sum([
            cvec[ind, kind] * xxmat[ind, kind]
            for ind, kind in product(range(dim), repeat=2)]) >= 1)
    # This is it!
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True) # I hope that its Infeasible
