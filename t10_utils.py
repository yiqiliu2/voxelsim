"""
Small utility functions shared across the 3D-stack simulator.

Currently provides:

* **dict_update** -- deep (recursive) dictionary merge.
* **IPU_Mk2_cycle_to_ms** -- convert Graphcore IPU Mk2 clock cycles to
  milliseconds for verification against published IPU numbers.
"""

from typing import Dict
import collections.abc


def dict_update(d: Dict, u: Dict) -> Dict:
    """Recursively merge dictionary *u* into *d* (in-place) and return *d*.

    For keys whose values are themselves mappings in both *d* and *u*, the
    merge recurses into the sub-dictionaries rather than overwriting.
    Leaf (non-mapping) values in *u* always overwrite those in *d*.

    Args:
        d: Target dictionary (modified in-place).
        u: Source dictionary whose entries are merged into *d*.

    Returns:
        The updated dictionary *d*.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v) # type: ignore
        else:
            d[k] = v
    return d


def IPU_Mk2_cycle_to_ms(cycles):
    """Convert Graphcore IPU Mk2 clock cycles to milliseconds.

    The IPU Mk2 runs at 1.325 GHz, so 1 ms = 1,325,000 cycles.

    Args:
        cycles: Number of clock cycles (int or float).

    Returns:
        Equivalent time in milliseconds.
    """
    return cycles / 1.325e6
