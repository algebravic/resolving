"""Routines for dealing with integer partitions."""
from typing import Tuple, Iterable, Dict
from operator import __mul__
from functools import cache, reduce
from itertools import product
from collections import Counter

PARTITION = Tuple[int, ...]

def partitions_sub(num: int, bnd: int) -> Iterable[PARTITION]:
    """
    All partitions of num with parts >= bnd.

    :param num: the degree of partition
    :type num: int
    :param bnd: a lower bound for the size of the parts
    :type bnd: int
    :returns: All partitions satisfying the constraint
    :rtype: Iterable[PARTITION]
    """
    if num == 0:
        yield tuple()
    elif num == bnd:
        yield (bnd,)
    elif num > bnd:
        # Must have num >= 2 * part
        # Or part <= num // 2
        for part in range(bnd, num + 1):
            for elt in partitions_sub(num - part, part):
                yield (part,) + elt

def partitions(num: int) -> Iterable[PARTITION]:
    """Unrestricted partitions"""
    yield from partitions_sub(num, 1)

def multiplicity(part: PARTITION) -> Dict[int, int]:
    """Multiplicity representation of a partition"""
    return Counter(part)

@cache
def gcd(num: int, mnum: int) -> int:
    """
    Greatest common divisor, binary method
    """
    res = 1
    while num % 2 == 0 and mnum % 2 == 0:
        num //= 2
        mnum //=2
        res *= 2
    while num % 2 == 0:
        num //= 2
    while mnum % 2 == 0:
        mnum //= 2
    if min(mnum, num) == 1:
        return res
    while True:
        # Both are odd
        if mnum < num:
            mnum, num = num, mnum
        mnum -= num
        if mnum == 0:
            return res * num
        while mnum % 2 == 0:
            mnum //= 2

@cache
def factorial(num: int) -> int:
    """
    factorial
    """
    return reduce(__mul__, range(2,num + 1),1)

def _contrib(part: PARTITION) -> int:
    """
    Cycle contribition
    """
    return (factorial(sum(part)) //
            reduce(__mul__, (factorial(mult) *
                             key ** mult
                             for key, mult
                             in multiplicity(part).items()),1))
def _test(part) -> bool:
    """
    Test multiplicity.
    """
    siz = sum(part)
    tst = sum((key * mult for key, mult in multiplicity(part).items()))
    return siz == tst

def harrison(num: int, mnum: int) -> int:
    """
    Harrison's formula for the number of m by n
    boolean matrices up to row/column permutation.
    """
    val = 0
    for parta, partb in product(partitions(num),
                                partitions(mnum)):
        expon = sum((multa * multb * gcd(keya, keyb)
                     for keya, multa in multiplicity(parta).items()
                     for keyb, multb in multiplicity(partb).items()))
        val += _contrib(parta) * _contrib(partb) * (2 ** expon)
    return val // (factorial(num) * factorial(mnum))

@cache
def combinations(num: int, mnum) -> int:
    """
    Binomal coefficient
    """
    if num < 0 or mnum > num:
        return 0
    return (factorial(num) //
            (factorial(mnum) * factorial(num - mnum)))

def balanced(num: int) -> int:
    """
    The number of 0/1/-1 vectors summing to 0.
    """
    return sum((2 ** ind * combinations(num, 2 * ind)
                * combinations(2 * ind, ind)
                for ind in range(num // 2)))
def excluded(num: int, mnum: int) -> int:
    """
    Return the number of 0/+1/-1 vectors whose sum is 0
    whose dot product with a 0/1 vector of weight m is 0.

    If the test vector has m 1's, there must be a balanced
    set of +/- 1 in that region.  Since the whole vector is
    balanced, the remaining nonzeros must also be balanced.
    If there are 2j nonzeros in the tested vector, and 2k
    nonzeros in that vector underneath the 1's in the test
    vector, we must have k <= j, 2*k <= m, and
    2*(j-k) <= n-m.  The latter holds if and only if
    2*k >= 2*j - (n-m), or k >= j - floor((n-m)/2)
    """
    return sum((combinations(mnum, 2*kind)*
                combinations(2*kind, kind)*
                combinations(num - mnum, 2*(jind - kind))*
                combinations(2*(jind - kind), jind - kind)
                for jind in range(num // 2)
                for kind in range(
                        max(0,jind - (num-mnum) // 2),
                        min(mnum // 2, jind) + 1)))
