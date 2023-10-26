"""
Generate Schreier-Sims clauses.

See 'Schreier-Sims Cuts meet Stable Set: Preserving problem structure
when handling symmetries' by Hojny, Pfetsch and Verschae

"""

from typing import Tuple, Iterable
from itertools import product
from sympy.combinatorics.permutations import Permutation 
from sympy.combinatorics.perm_groups import PermutationGroup

CYCLE = Tuple[int, ...]
PERM = Tuple[CYCLE, ...]

def double_lex_gens(dim: int, mdim: int) -> Iterable[PERM]:
    """
    We want one origin

    We use the Coxeter generators for S_n.
    """
    trans = {_: _[0] + mdim * _[1] for _ in product(range(mdim), range(dim))}
    yield from (tuple((trans[row, _], trans[row + 1, _]) for _ in range(dim))
                for row in range(mdim - 1))
    yield from (tuple((trans[_, col], trans[_, col + 1]) for _ in range(mdim))
                for col in range(dim - 1))

def double_lex_group(dim: int, mdim: int) -> PermutationGroup:
    """
    Use sympy to form the double lex group.
    """
    return PermutationGroup([Permutation(_, size = dim * mdim) for _ in double_lex_gens(dim, mdim)])

def schreier_sims_cuts(dim: int, mdim: int) -> Iterable[Tuple[int, int]]:
    """
    Return pairs from the Schreir-Sims table
    """
    trans = {1 + _[0] + mdim * _[1]:_ for _ in product(range(mdim), range(dim))}
    grp = double_lex_group(dim, mdim)
    for leader, followers in zip(grp.base, grp.basic_orbits):
        yield from ((trans[leader], trans[_])
                    for _ in followers if _ != leader)
