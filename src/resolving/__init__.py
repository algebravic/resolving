"""
Calculate the metric dimension of the hypercube

For general graphs the functions
resolve_hypercube_maxsat, resolve_hypercube_sat and resolve_hypercube_hitman
model the problem as a hitting set problem.

For the special family of graphs, the hypercube, ping_pong
models it as an 'exists/forall' problem which overcomes the
exponential dependence on the parameter n.

Finally, hypercube_model and inverse_hypercube_model
construct a QDIMACS model as a QBF problem.
"""
from .resolve import resolve_hypercube_maxsat, resolve_hypercube_hitman
from .resolve import resolve_hypercube_sat
from .pingpong import ping_pong
from .hypercube_qbf import hypercube_model, inverse_hypercube_model
__all__ = ['resolve_hypercube_maxsat',
           'resolve_hypercube_sat',
           'resolve_hypercube_hitman',
           'ping_pong',
           'hypercube_model',
           'inverse_hypercube_model'
           ]
