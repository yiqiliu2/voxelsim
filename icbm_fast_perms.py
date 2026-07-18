#!/usr/bin/env python3
"""
Fast edit-distance-bounded permutation generation for operator reordering.

When exploring alternative execution orders for a sequence of DNN operators,
the full factorial space (n!) is prohibitively large.  This module provides
pruned permutation generators that only emit orderings where no element
moves farther than *max_edit_dist* positions from its original index.

Three generators are offered, each with different trade-offs:

* **edit_dist_permutations** -- generates all index permutations within the
  given edit-distance bound.  May contain duplicates when the input list
  has repeated elements.
* **reduced_edit_dist_permutations** -- deduplicates by tracking the
  *data* content (not just index order), returning only unique orderings.
* **permute_top** -- unrestricted (full factorial) permutation generator,
  useful as a baseline or for very short sequences.
"""

from copy import deepcopy


def edit_dist_permutations(a: list, max_edit_dist: int):
    """Generate all permutations of *a* where no element moves more than
    *max_edit_dist* positions from its original index.

    Uses a recursive swap-based approach with early pruning: after each swap,
    if either displaced element exceeds the distance bound the branch is
    abandoned immediately.

    Args:
        a:             A list of indices (typically ``list(range(n))``).
        max_edit_dist: Maximum allowed displacement for any single element.

    Returns:
        A list of permutations (each a list of ints).
    """
    def edit_dist_helper(a: list, l: int, r: int, results: list = None):
        if results is None:
            results = []
        if l == r:
            results.append(deepcopy(a))
        else:
            for i in range(l, r):
                a[l], a[i] = a[i], a[l]
                # Prune: check that neither swapped element exceeds the bound
                if not (abs(a[i] - i) > max_edit_dist or abs(a[l] - l) > max_edit_dist):
                    edit_dist_helper(a, l+1, r, results)
                a[i], a[l] = a[l], a[i]  # backtrack

    results = []
    edit_dist_helper(a, 0, len(a), results)
    return results

def reduced_edit_dist_permutations(data: list, a: list, max_edit_dist: int):
    """Like ``edit_dist_permutations`` but deduplicates on the *data* content.

    When *data* contains repeated elements, many distinct index permutations
    yield the same data ordering.  This function tracks seen data strings in
    a set and skips branches that would produce an already-discovered ordering,
    significantly reducing the output size.

    Args:
        data:          The actual operator list (may contain duplicates).
        a:             Index list (typically ``list(range(len(data)))``).
        max_edit_dist: Maximum allowed displacement per element.

    Returns:
        A list of unique index orderings (each a list of ints).
    """
    def edit_dist_helper(data: list, a: list, l: int, r: int, results: set, orders: set):
        if l == r:
            results.add(str(data))
            orders.add(str(a))
        else:
            for i in range(l, r):
                a[l], a[i] = a[i], a[l]
                data[l], data[i] = data[i], data[l]
                # Skip if this data arrangement was already found or distance violated
                if str(data) not in results and not (abs(a[i] - i) > max_edit_dist or abs(a[l] - l) > max_edit_dist):
                    edit_dist_helper(data, a, l+1, r, results, orders)
                a[l], a[i] = a[i], a[l]       # backtrack indices
                data[i], data[l] = data[l], data[i]  # backtrack data

    results, orders = set(), set()
    edit_dist_helper(data, a, 0, len(a), results, orders)
    # Deserialise the string representations back to integer lists
    orders = [[int(i) for i in s.strip('[]').split(',')] for s in orders]
    return orders


def permute_top(a: list):
    """Generate all permutations of *a* (unrestricted, full factorial).

    Provided as a baseline reference; for realistic operator counts prefer
    ``edit_dist_permutations`` to keep the search space tractable.

    Args:
        a: A list to permute.

    Returns:
        A list of all permutations (each a deep copy of the permuted list).
    """
    def permute(a: list, l: int, r: int, results: list = None):
        if results is None:
            results = []
        if l == r:
            results.append(deepcopy(a))
        else:
            for i in range(l, r):
                a[l], a[i] = a[i], a[l]
                permute(a, l+1, r, results)
                a[l], a[i] = a[i], a[l]

    results = []
    permute(a, 0, len(a), results)
    return results


if __name__=='__main__':
    data = [0, 0, 2]
    orders = reduced_edit_dist_permutations(data, list(range(len(data))), 3)
    print(len(orders))
    print(orders)
