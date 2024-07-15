'''
Enable SMT solvers for PySMT

Boolector Logics:  QF_ABV, QF_AUFBV, QF_BV and QF_UFBV

From pySMT docs:

from pysmt.shortcuts import Symbol, get_env, Solver
from pysmt.logics import QF_UFLRA

name = "mathsat-smtlib" # Note: The API version is called 'msat'

# Path to the solver. The solver needs to take the smtlib file from
# stdin. This might require creating a tiny shell script to set the
# solver options.
path = ["/tmp/mathsat"]
logics = [QF_UFLRA,]    # List of the supported logics

# Add the solver to the environment
env = get_env()
env.factory.add_generic_solver(name, path, logics)

# The solver name of the SMT-LIB solver can be now used anywhere
# where pySMT would accept an API solver name
with Solver(name=name, logic="QF_UFLRA") as s:
  print(s.is_sat(Symbol("x"))) # True
'''

from pysmt.shortcuts import Symbol, get_env, Solver
from pysmt.logics import  QF_ABV, QF_AUFBV, QF_BV, QF_UFBV, BV

def enable_boolector():
    """
    Tell PySMT about boolector.
    """
    name = "boolector"
    path = ["/opt/homebrew/bin/boolector"]
    logics = [QF_ABV, QF_AUFBV, QF_BV, QF_UFBV, BV]
    env = get_env()
    env.factory.add_generic_solver(name, path, logics)
