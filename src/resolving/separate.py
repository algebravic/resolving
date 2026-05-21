"""
Produced the partitions by column
"""
from typing import Iterable, List, Set
import numpy as np

def partitions_by_column(mat: np.ndarray) -> List[Set[int]]:
    """
    Produce the partition of the column indices,
    where each part has a constant value for the column.
    """

    _, ndim = mat.shape
    parts = dict()
    for ind in range(ndim):
        val = tuple(mat[:, ind].tolist())
        if val not in parts:
            parts[val] = set()
        parts[val].add(ind)
    return list(parts.values())
