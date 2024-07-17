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
    """

    for impl in impls:

        clause = []
        for lft, rgt in impl[: -1]:
            left = avar[lft]
            right = avar[rgt]
            key = (tseytin, tuple(sorted((lft, rgt))))
            gen_clauses = key not in pool.obj2id
            tst = pool.id(key)
            # tst <==> left == right
            if gen_clauses:
                yield from [[tst, left, right],
                            [-tst, -left, right],
                            [-tst, left, -right],
                            [tst, -left, -right]]
            clause.append(-tst)
        lft0, rgt0 = impl[-1]
        yield clause + [-avar[lft0], avar[rgt0]]

def _tuple_it(arg: Any) -> Any:

    if isinstance(arg, list):
        return tuple(map(_tuple_it, arg))
    else:
        return arg
