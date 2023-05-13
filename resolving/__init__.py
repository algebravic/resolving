"""
Calculate the metric dimension of the hypercube
"""
from .resolve import resolve_hypercube_maxsat, resolve_hypercube_hitman
from .resolve import resolve_hypercube_sat
from .pingpong import ping_pong
__all__ = ['resolve_hypercube_maxsat',
           'resolve_hypercube_sat',
           'resolve_hypercube_hitman',
           'ping_pong'
           ]
