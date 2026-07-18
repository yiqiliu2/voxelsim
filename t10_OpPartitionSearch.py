#!/usr/bin/env python3
"""
Spatial and temporal partition search trees for operator-to-core mapping.

This module provides tree-based search structures that enumerate valid ways to
partition DNN operators across multiple processing cores.  Two complementary
search classes are offered:

* **OpSpatialPartitionSearch** -- explores scalar partition factors whose
  product must not exceed the total number of cores.  Each tree level
  corresponds to one tensor dimension, and nodes at that level hold the
  partition factor chosen for that dimension.

* **OpTemporalPartitionSearch** -- explores vector-valued partition factors
  (one entry per replica dimension) whose element-wise product must stay
  within per-dimension replica budgets.

Both classes build an explicit N-ary search tree iteratively, prune
infeasible branches via configurable filter functions, and then collect all
leaf-node configurations as candidate partition plans.
"""

from functools import lru_cache
from typing import List, Optional, Tuple
import numpy as np

class OpSpatialPartitionSearch:
    """Search tree for spatial partition factors that map an operator onto cores.

    Each path from root to leaf represents a complete partition configuration --
    a list of integer factors (one per tensor dimension) whose product equals
    the number of cores assigned to the operator.  The tree is pruned during
    construction so that only factor combinations satisfying the filter
    predicates and the core-count constraint are retained.

    Attributes:
        root:  Sentinel root node (value=1, no semantic meaning).
        depth: Number of tensor dimensions to partition (tree depth).
        dim_size_TH: Lower-bound utilisation threshold (fraction of cores
            that should be busy).
        tot_dim_size: Upper bound for the partition factor at each depth
            level (one entry per dimension).
        num_core: Total number of available cores.
    """

    class Node:
        """A single node in the spatial partition search tree.

        Attributes:
            value:     Partition factor chosen at this tree level.
            parent:    Reference to the parent node (``None`` for the root).
            agg_value: Running product of partition factors from the root
                       down to (and including) this node.
            children:  Child nodes at the next tree level.
        """

        def __init__(self, value: int = 1, parent = None, children = None):
            self.value: int = value
            self.parent = parent
            if self.parent == None:
                self.agg_value: int = self.value
            else:
                self.agg_value: int = self.parent.agg_value * self.value
            if children is None:
                self.children: List[OpSpatialPartitionSearch.Node] = []
            else:
                self.children = children
        
        def isRoot(self) -> bool:
            """Return True if this node is the tree root."""
            return self.parent is None

        def isLeaf(self) -> bool:
            """Return True if this node has no children."""
            return len(self.children) == 0

        def getPathToRoot(self) -> List[int]:
            """Return the list of partition factors from this node up to the root."""
            path = []
            cur_node = self
            while cur_node is not None:
                path.append(cur_node.value)
                cur_node = cur_node.parent
            return path

        def getConfig(self) -> List[int]:
            """Return the partition configuration (root-to-leaf order, root excluded)."""
            return list(reversed(self.getPathToRoot()[:-1]))

    def __init__(self, depth: int = 7, tot_dim_size: List = [], filter_func_high = None, filter_func_low = None, dim_size_TH: float = 0.9, num_core: int = 0):
        self.root = OpSpatialPartitionSearch.Node()
        self.depth: int = depth
        self.dim_size_TH: float = dim_size_TH
        self.tot_dim_size: List[int] = tot_dim_size
        self.num_core: int = num_core
        if filter_func_high is None:
            # filter_func(cur_node_value, parent_node, tot_dim_size) -> bool
            self.filter_func_high = self.filter_by_tot_dim_size_high
        else:
            self.filter_func_high = filter_func_high
        if filter_func_low is None:
            # filter_func(cur_node_value, parent_node, tot_dim_size) -> bool
            self.filter_func_low = self.filter_by_tot_dim_size_low
        else:
            self.filter_func_low = filter_func_low

    def generateSpatialSearchTreeIterative(self):
        """Build the spatial search tree level-by-level (breadth-first).

        At each depth *d* the method iterates over all nodes produced at the
        previous level and tries every partition factor from a computed lower
        bound up to ``tot_dim_size[d]``.  The lower bound is derived from the
        utilisation threshold so that the product of all chosen factors across
        remaining dimensions can still reach ``num_core * dim_size_TH``.
        Branches whose aggregate product already exceeds ``num_core`` are
        pruned by ``filter_func_high``.
        """
        # Pre-compute suffix products to avoid repeated np.prod in inner loop
        _suffix_prods = [int(np.prod(self.tot_dim_size[d + 1:])) for d in range(self.depth)]
        cur_level_nodes = [self.root]
        cur_depth = 0
        while cur_depth < self.depth:
            next_level_nodes = []
            max_remaining_dim_size = _suffix_prods[cur_depth]
            dim_upper = self.tot_dim_size[cur_depth]
            # Minimum factor that still allows the final product to meet the threshold
            threshold_factor = self.num_core * self.dim_size_TH / max_remaining_dim_size
            for node in cur_level_nodes:
                min_start = int(max(1, min(int(threshold_factor / node.agg_value), dim_upper)))
                for i in range(min_start, dim_upper + 1):
                    if self.filter_func_high(i, node, cur_depth) == False:
                        break
                    new_node = OpSpatialPartitionSearch.Node(value = i, parent = node)
                    node.children.append(new_node)
                    next_level_nodes.append(new_node)
            cur_level_nodes = next_level_nodes
            cur_depth += 1

    def generateSearchTree(self):
        """Public entry point -- delegates to the iterative builder."""
        self.generateSpatialSearchTreeIterative()

    def printSearchTree(self):
        """Pretty-print the tree level by level, separating sibling groups."""
        cur_level_nodes = [self.root]
        while len(cur_level_nodes) > 0:
            next_level_nodes = []
            cur_parent = cur_level_nodes[0].parent
            for node in cur_level_nodes:
                if node.parent != cur_parent:
                    print("; ")
                    cur_parent = node.parent
                print(node.value, end = " ")
                next_level_nodes.extend(node.children)
            print()
            cur_level_nodes = next_level_nodes
    
    def num_leaf_nodes(self, filter_func = None) -> int:
        """Count leaf nodes that satisfy *filter_func* (default: all leaves).

        The *filter_func* callback also serves as a visitor -- callers can
        use it to collect configurations while counting (see ``get_all_configs``).
        """
        if filter_func is None:
            filter_func = lambda x: True
        def num_leaf_nodes_helper(cur_node: OpSpatialPartitionSearch.Node) -> int:
            if cur_node.isLeaf() and filter_func(cur_node):
                return 1
            else:
                return sum([num_leaf_nodes_helper(child) for child in cur_node.children])
        return num_leaf_nodes_helper(self.root)

    def filter_by_tot_dim_size_high(self, cur_node_value, parent_node: Node, cur_depth) -> bool:
        """Prune when the aggregate partition product exceeds the core count."""
        return cur_node_value * parent_node.agg_value <= self.num_core

    def filter_by_tot_dim_size_low(self, cur_node_value, parent_node: Node, cur_depth) -> bool:
        """Default low-bound filter (no-op -- always returns True)."""
        return True

    def filter_config_by_min_dim_size(self, node) -> bool:
        """Default leaf-node filter (no-op -- always returns True)."""
        return True

    def get_all_configs(self, filter_func = None) -> List[List[int]]:
        """Return every valid partition configuration as a list of factor lists.

        Args:
            filter_func: Optional predicate applied to each leaf node.
                Defaults to ``filter_config_by_min_dim_size``.

        Returns:
            A list of configurations, where each configuration is a list of
            integer partition factors (one per tensor dimension).
        """
        if filter_func is None:
            filter_func = self.filter_config_by_min_dim_size
        spatial_configs = []
        def get_config_from_leaf_node(node):
            if filter_func(node):
                spatial_configs.append(node.getConfig())
                return True
            else:
                return False
        self.num_leaf_nodes(get_config_from_leaf_node)
        return spatial_configs

def build_spatial_search_tree(depth: int = 7, tot_dim_size: List[int] = [], filter_func_high = None, filter_func_low = None,
                              dim_size_TH: float = 0.9, num_core: int = 0) -> Tuple[float, OpSpatialPartitionSearch]:
    """Convenience wrapper: build a spatial search tree and time the construction.

    Args:
        depth:            Number of tensor dimensions (tree levels).
        tot_dim_size:     Maximum partition factor per dimension.
        filter_func_high: Upper-bound pruning predicate (optional).
        filter_func_low:  Lower-bound pruning predicate (optional).
        dim_size_TH:      Core-utilisation threshold in [0, 1].
        num_core:         Total available cores.

    Returns:
        A tuple of (build_time_seconds, search_tree).
    """
    import time
    start = time.perf_counter()
    search_tree = OpSpatialPartitionSearch(
        depth, tot_dim_size, filter_func_high, filter_func_low, dim_size_TH, num_core
    )
    search_tree.generateSearchTree()
    end = time.perf_counter()

    search_time = end - start
    print("Time to build spatial search tree:", search_time, "seconds; Threshold:", dim_size_TH, flush=True)

    return search_time, search_tree

class OpTemporalPartitionSearch:
    """Search tree for temporal (multi-dimensional) partition factors.

    Unlike ``OpSpatialPartitionSearch``, each node stores a *vector* of
    integers (one per replica dimension).  The aggregate value at any node
    is the element-wise product of all factors along the path from the root.
    A branch is pruned whenever any component of the aggregate exceeds the
    corresponding entry in ``num_replicas``.

    Attributes:
        root:         Sentinel root node (empty value list).
        depth:        Number of tree levels (tensor dimensions).
        search_space: Per-level list of candidate factor vectors.
        num_replicas: Per-dimension upper bound on the aggregate product.
    """

    class Node:
        """A single node in the temporal partition search tree.

        Attributes:
            value:     Vector of partition factors at this level.
            parent:    Parent node reference (``None`` for root).
            agg_value: Element-wise running product from root to this node.
            children:  Child nodes at the next level.
        """

        def __init__(self, value: Optional[List[int]] = None, parent = None, children = None):
            if value is None:
                self.value: List[int] = []
            else:
                self.value: List[int] = value
            self.parent = parent
            if self.parent == None:
                self.agg_value: List[int] = self.value
            else:
                self.agg_value: List[int] = [x * y for x, y in zip(self.parent.agg_value, self.value)]
            if children is None:
                self.children: List[OpTemporalPartitionSearch.Node] = []
            else:
                self.children = children
            self._config: Optional[List[List[int]]] = None  # memoized result of getConfig()

        def isRoot(self) -> bool:
            """Return True if this node is the tree root."""
            return self.parent is None

        def isLeaf(self) -> bool:
            """Return True if this node has no children."""
            return len(self.children) == 0

        def getPathToRoot(self) -> List[List[int]]:
            """Return factor vectors from this node up to the root."""
            path: List[List[int]] = []
            cur_node = self
            while cur_node is not None:
                path.append(cur_node.value)
                cur_node = cur_node.parent
            return path

        def getConfig(self) -> List[List[int]]:
            """Return the partition config (root-to-leaf order, root excluded). Memoized."""
            if self._config is None:
                self._config = list(reversed(self.getPathToRoot()[:-1]))
            return self._config

    def __init__(self, depth: int = 7, search_space: Optional[List[List[List[int]]]] = None, num_replicas: Optional[List[int]] = None, filter_func = None):
        self.root = OpTemporalPartitionSearch.Node()
        self.depth: int = depth
        if search_space is None:
            self.search_space: List[List[List[int]]] = [[] for _ in range(depth)]
        else:
            self.search_space: List[List[List[int]]] = search_space
        self.search_space.sort()
        if num_replicas is None:
            self.num_replicas: List[int] = []
        else:
            self.num_replicas: List[int] = num_replicas
        if filter_func is None:
            # filter_func(cur_node_value, parent_node) -> bool
            self.filter_func = self.filter_by_dim_size

    def generateSearchTreeIterative(self):
        """Build the temporal search tree level-by-level (breadth-first).

        For each depth, every candidate factor vector from ``search_space[depth]``
        is tested against the filter predicate.  Passing candidates become
        children of the current-level nodes.
        """
        cur_level_nodes = [self.root]
        cur_depth = 0
        while cur_depth < self.depth:
            next_level_nodes = []
            for node in cur_level_nodes:
                for i in self.search_space[cur_depth]:
                    if self.filter_func(i, node) == False:
                        continue
                    new_node = OpTemporalPartitionSearch.Node(value = i, parent = node)
                    node.children.append(new_node)
                    next_level_nodes.append(new_node)
            cur_level_nodes = next_level_nodes
            cur_depth += 1

    def generateSearchTree(self):
        """Public entry point -- delegates to the iterative builder."""
        self.generateSearchTreeIterative()

    def printSearchTree(self):
        """Pretty-print the tree level by level, separating sibling groups."""
        print("#replicas: ", self.num_replicas)
        cur_level_nodes = [self.root]
        while len(cur_level_nodes) > 0:
            next_level_nodes = []
            cur_parent = cur_level_nodes[0].parent
            for node in cur_level_nodes:
                if node.parent != cur_parent:
                    print(";", end = " ")
                    cur_parent = node.parent
                print(node.value, end = " ")
                next_level_nodes.extend(node.children)
            print()
            cur_level_nodes = next_level_nodes
    
    def num_leaf_nodes(self, filter_func = None) -> int:
        """Count (and optionally visit) leaf nodes satisfying *filter_func*."""
        if filter_func is None:
            filter_func = lambda x: True
        def num_leaf_nodes_helper(cur_node: OpTemporalPartitionSearch.Node) -> int:
            if cur_node.isLeaf() and filter_func(cur_node):
                return 1
            else:
                return sum([num_leaf_nodes_helper(child) for child in cur_node.children])
        return num_leaf_nodes_helper(self.root)

    def filter_by_dim_size(self, cur_node_value: List[int], parent_node: Node) -> bool:
        """Prune when any component of the aggregate product exceeds its replica limit."""
        if parent_node.isRoot():
            return True

        for x, y, z in zip(cur_node_value, parent_node.agg_value, self.num_replicas):
            if x * y > z:
                return False

        return True

    def get_all_configs(self, filter_func = None) -> List[List[List[int]]]:
        """Return every valid temporal partition configuration.

        Returns:
            A list of configurations; each configuration is a list of
            factor vectors (one per tensor dimension).
        """
        if filter_func is None:
            filter_func = lambda x: True
        configs = []
        def get_config_from_leaf_node(node: OpTemporalPartitionSearch.Node):
            if filter_func(node):
                configs.append(node.getConfig())
                return True
            else:
                return False
        self.num_leaf_nodes(get_config_from_leaf_node)
        return configs

if __name__ == "__main__":
    op_partition_search = OpTemporalPartitionSearch(7, [[[1,1]],[[1,2]],[[2,1]],[[2,2]]], [2,2])
    op_partition_search.generateSearchTree()
    # op_partition_search.printSearchTree()
    configs = op_partition_search.get_all_configs()

    print("num configs: ", len(configs))

    # print spatial configs line by line
    for config in configs:
        print(config)

