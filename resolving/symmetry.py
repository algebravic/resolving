"""
Original symmetry breakers
"""
from pysat.formula import IDPool
from typing import Iterable, List, Tuple
from itertools import product, chain

def compatible_pairs(num: int, wgt: int, wgt2: int) -> Iterable[Tuple[int, ...]]:
    """Generate compatible pairs.

    Parameters
    ----------
    num: the dimension of the hypercube
    wgt, wgt2: The Hamming weights of the pair elements

    Yields
    ------
       A sequence of compatible pairs.

    The first required element will be wgt * (1,) + (num - wgt) * (0,)
    The second will be i * (0,) + wgt2 * (1,) + (num - i - j) * (0,)
    where 0 <= i <= wgt - wgt2 and 1 <= wgt2 <= min(wgt, num - i).
    However, if wgt == wgt2 we must have i >= 1.
    """
    if wgt2 > wgt:
        return

    yield from (ind * (0,) + wgt2 * (1,) + (num - ind - wgt2) * (0,)
                for ind in range(
                    int(wgt == wgt2), min(wgt, num - wgt2) + 1))

def compatible(num: int, wgt: int) -> Iterable[Tuple[int, ...]]:
    """
    Generate all compatible pairs.
    """
    yield from chain(*(compatible_pairs(num, wgt, wgt2)
                       for wgt2 in range(1, wgt + 1)))



def symmetry_breaking_clauses(num: int,
                              degree: int,
                              pool: IDPool) -> Iterable[List[int]]:
    """
    Generate symmetry breaking clauses for the Hypercube.

    PARAMETERS:
    num: the dimension of the hypercube
    degree: the depth of the symmetry breaking tree
    pool: The IDPool being used for this SAT instance

    YIELDS:
    A sequence of clauses

    Note that is S is a resolving set and sigma is an automorphism of G
    then sigma(S) is resolving.
    The automorphism group of H_n is generated by GF(2)^n (via addition)
    and S_n (via permuting coordinates).

    Without loss of generality, we many assume that 0^n is in the resolving
    set.  Namely choose any element, z, in the resolving set and apply the
    automorphism which adds z.

    Now a second element in the resolving set may have its coordinates
    permuted arbitrarily.  If that element has weight w, we may assume
    that the first w bits are 1 and the remaining are 0.  In fact, we
    may assume that the second element has weight >= any other element.

    Let w be a third element in the resolving set.  For each possible z
    which is a second element, we are allowed to permute all of the
    positions of the 1's and 0's independently.  Thus, without loss
    of generality, we may assume that all of the 1's are in
    consecutive positions.  We make this conditional on z appearing.

    If x is in a resolving set, S, we may, replace x with its complement x'
    since d(u,x) = n - d(u,x').  Thus, we may assume that the weight of
    all the elements in a resolving set is <= n/2.
    """
    zero = num * (0,)
    nhalf = num // 2
    for elt in product(range(2), repeat = num):
        if sum(elt) > nhalf:
            yield [-pool.id(('x', elt))]
    if degree > 0:
        # By XORing the all 0 word is in the resolving set
        yield [pool.id(('x', zero))]
    if degree > 1:
        print("Degree 2 constraints")
        # A word of maximum weight is in the resolving set
        second = {_ * (1,) + (num - _) * (0,) for _ in range(1, nhalf + 1)}
        yield [pool.id(('x', _)) for _ in second]
        # Now forbid anything of big weight
        for ind in range(1, nhalf + 1):
            base = pool.id(('x', ind * (1,) + (num - ind) * (0,)))
            for elt in product(range(2), repeat=num):
                if (sum(elt) > ind) and (sum(elt) <= nhalf):
                    yield [-base, -pool.id(('x', elt))]
    if degree > 2:
        print("Degree 3 constraints")
        for wgt in range(1, nhalf + 1):
            base = pool.id(('x', wgt * (1,) + (num - wgt) * (0,)))
            # The second element as jind 0's at the beginning and
            # kind 1's following.
            # Since ind is the max weight of a nonzero we have kind <= ind
            # We either have jind < ind and num - kind >= jind
            # In which case it can straddle the first element.
            # Or jind = ind.
            yield [-base] + [pool.id(('x', _))
                             for _ in compatible(num, wgt)]
            # Make all the rest no bigger weight than the third
            for wgt2 in range(1, wgt + 1):
                for elt in compatible_pairs(num, wgt, wgt2):
                    ante = [-base, - pool.id(('x', elt))]
                    for tst in product(range(2), repeat=num):
                        if (sum(tst) > wgt2) and (sum(elt) <= nhalf):
                            yield ante + [-pool.id(('x', tst))]
