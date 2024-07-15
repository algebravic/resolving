"""
  Use the MIP package to check the status of a putative balanced resolving matrix.
"""
import numpy as np
from mip import Model, xsum, maximize, BINARY, OptimizationStatus

def check_resolving(amat: np.ndarray) -> bool:
    """
      Input: A: a putative 0/1 balanced resolving matrix.
      Output: T/F according the MIP model.
    """
    
    mdim, ndim = amat.shape
    mod = Model(name = "resolving", solver_name = "CBC")
    xvar = [mod.add_var(var_type=BINARY) for _ in range(ndim)]
    yvar = [mod.add_var(var_type=BINARY) for _ in range(ndim)]
    exclude = [_[0] + _[1] <= 1 for _ in zip(xvar, yvar)]
    equals = [xsum(xvar) == xsum(yvar)]
    nozero = [xsum(xvar) >= 1]
    amat_x =[xsum(_[0] * _[1] for _ in zip(amat[ind], xvar)) for ind in range(mdim)]
    amat_y =[xsum(_[0] * _[1] for _ in zip(amat[ind], yvar)) for ind in range(mdim)]
    no_detect = [_[0] == _[1] for _ in zip(amat_x, amat_y)]
    mod.objective = maximize(xsum(xvar))
    for constr in exclude + equals + nozero + no_detect:
        mod += constr
    #mod.verbose = 1
    mod.optimize()
    print(f"Status = {mod.status}")
    return mod.status in [OptimizationStatus.INFEASIBLE,
                          OptimizationStatus.INT_INFEASIBLE]
