"""
Given a 0/1 m x n matrix, normalize it under row/column full permutations.

We assume that all rows are non-zero and distinct.

We do this inductively row by row:

Having chosen the first k rows, we divide all of the other rows into
'regions': a region is a maximal subset of consecutive rows so that all columns
are equal.

1) We then gather all of the new rows whose weight is the smallest weight >=
than the weight of the k-th row.

2) For each of the the new rows we map them into a 'v-vector': one coordinate for
each region, whose value is the number of 1's in that region.

3) Choose the next row as being the row with lexicographically smallest v-vector, and
make that the k-th row. Then permute (implicitly) so that all of the 1's in a region
are pushed to the left.

"""
from typing import List, Tuple, Hashable, Iterable, Dict
from functools import partial
from itertools import pairwise
from collections import defaultdict
import numpy as np

INTS = (int, np.int8, np.int16, np.int32, np.int64)
PERM = List[int]

def mcollect(arg: Iterable[Hashable], start: int = 0) -> Dict[Hashable, List[int]]:
    """
    Create a dict from a sequence of Hashables, remembering the position in the list
    """

    res = defaultdict(list)
    for ind, elt in enumerate(arg, start = start):
        res[elt].append(ind)
    return res

def dict_diff(dct: Dict, other: Dict) -> Dict:

    for key in other.keys():
        del dct[key]

def _validate(arr: np.array) -> bool:

    return (isinstance(arr, np.ndarray)
            and arr.dtype in INTS
            and len(arr.shape) == 2
            and ((arr == 0) | (arr == 1)).all()
            # distinct rows
            and len(set(map(tuple, arr.tolist()))) == arr.shape[0]
            # no all zero row
            and not ((arr == 0).all(axis=1).any()))
def v_vector(boundary: Tuple[int, ...], row: Tuple[int, ...]) -> Tuple[int, ...]:

    return tuple((sum(row[_[0]:_[1]]) for _ in pairwise(boundary)))

def new_boundary(boundary: Tuple[int, ...],
                 cand: Tuple[int, ...]) -> Tuple[Tuple[int, ...],
                                                 Tuple[int,...],
                                                 List[int]]:

    # Given a v-vector and the boundary that produced it
    # return a normalized row vector from the v-vector and the new boundary

    if not (isinstance(cand, tuple)
            and all(((isinstance(_, INTS) and _ == 0 or _ == 1)
                     for _ in cand))):
            raise ValueError(f"Malformed candidate {cand}")

    perm = []
    nbound = [0] # always start with 0
    nrow = []
    # We need to update the permutation
    for (start, end) in pairwise(boundary):
        chunk = cand[start: end]
        vval = sum(chunk)
        if vval > 0:
            nbound.append(nbound[-1] + vval) # next boundary
        if nbound[-1] < end:
            nbound.append(end)
        # How do we permute this segment to get all 1's to the left
        # Map all 1's to the beginning, at put everything else at the end
        places = set(range(start, end))
        left = [_[0] for _ in enumerate(chunk, start = start) if _[1] == 1]
        perm += left + list(sorted(places.difference(left)))
        nrow += vval * [1] + (end - start - vval) * [0]
    return tuple(nbound), tuple(nrow), perm

def normalize_arr(arr: np.ndarray,
                  verbose: int = 0) -> Tuple[np.ndarray, PERM, PERM]:
    """
    Normalize the input array as the returned value.
    If the array is not a 0/1 2-d array with nonzero
    distinct rows, return None.
    """
    if _validate(arr):
        mval, nval = arr.shape
        boundary = (0, nval)
        rows = mcollect(map(tuple, arr.tolist()))
        new_arr = []
        cperm = list(range(nval))
        # In case rows is empty - this shouldn't happen
        rperm = []

        while rows:
            nwgt = min(map(sum, rows.keys()))
            cands = {_[0]: _[1] for _ in rows.items() if sum(_[0]) == nwgt}
            if verbose > 0:
                print(f"min_weight = {nwgt}, cands = {cands}")
            dict_diff(rows, cands)
            # start with the identity, and wait for the final composition
            perm = list(range(nval))
            while cands:
                mcand = min(cands.keys(), key=partial(v_vector, boundary))
                if verbose > 0:
                    print(f"boundary = {boundary}, vvec = {v_vector(boundary, mcand)}, perm={cperm}")
                rperm.append(cands[mcand][0]) # the list should be exactly length 1
                # We enforce distinct rows!
                del cands[mcand]
                # Normalize mcand
                # perm is the permutation preserving existing regions
                # we must compose the global perm with it.
                boundary, ncand, perm = new_boundary(boundary, mcand)
                if len(ncand) != nval:
                    raise ValueError(f"len({ncand}) != {nval}")
                # Now fix permute the remaining elements of cands
                cands = {tuple((_[0][ind] for ind in perm)): _[1] for _ in cands.items()}
                cperm = [cperm[_] for _ in perm]
                new_arr.append(ncand)
            rows = {tuple((_[0][ind] for ind in perm)): _[1] for _ in rows.items()}
        if verbose > 0:
            print(f"new_arr = {new_arr}")
            print(f"rperm = {rperm}, cperm = {cperm}")
        return np.array(new_arr, dtype=np.int8), rperm, cperm
    else:
        raise ValueError("malformed input")
