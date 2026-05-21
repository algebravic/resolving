"""
  General lex implication symmetry break.
"""

from typing import Iterable, Tuple, List, Set, FrozenSet, Any, Callable
from typing import Hashable, Dict
from functools import partial
from itertools import chain, product
from pysat.formula import IDPool

ELEMENT = Tuple[Hashable, Hashable]
IMPL = List[ELEMENT]
LEX_IMPL = Iterable[IMPL]
CLAUSE = List[int]
PAIR = Tuple[int, int]

def encode_implications(pool: IDPool,
                        avar: Dict[PAIR, int],
                        impls: LEX_IMPL,
                        tseytin: str = 'T') -> Iterable[CLAUSE]:
    """
      Encode the lex implications using the given prefixes.
      Inputs:
      pool: A pysat IDPool
      avar: a dictionary indexed by pairs yielding boolean variables
      impls: an Iterable of implications
        Each implication is a tuple of pairs of coordinates
      tseytin: option prefix string for tseytin variables to
        use in the IDPool:

      Output:

        This is a generator that produces clauses implementing
        the implications.
    """
    tseytins = set()
    for impl in impls:
        
        clause = []
        for lft, rgt in impl[: -1]:
            left = avar[lft]
            right = avar[rgt]
            key = tuple(sorted((lft, rgt)))
            tseytins.add(key)
            # tst <==> left == right
            clause.append(- pool.id((tseytin, key)))
        lft0, rgt0 = impl[-1]
        yield clause + [-avar[lft0], avar[rgt0]]

    for key in tseytins:
        lft, rgt = key
        left = avar[lft]
        right = avar[rgt]
        tst = pool.id((tseytin, key))
        yield from [[tst, left, right],
                    [-tst, -left, right],
                    [-tst, left, -right],
                    [tst, -left, -right]]
                

def _tuple_it(arg: Any) -> Any:

    if isinstance(arg, list):
        return tuple(map(_tuple_it, arg))
    else:
        return arg
