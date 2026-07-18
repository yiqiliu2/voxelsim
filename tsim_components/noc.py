"""Network-on-Chip (NoC) communication model for inter-core data movement.

This module estimates the cycle cost of inter-core communication over various
NoC topologies (2-D mesh, 2-D torus, 3-D mesh, 3-D torus, all-to-all, and
user-defined custom graphs).  It supports three data-movement primitives that
arise during distributed tensor computations:

  * **Broadcast** -- one core sends a tensor partition to multiple peers.
  * **Reduce**    -- partial results from multiple cores are accumulated at a
                     single destination, with link-contention modelling.
  * **Shift**     -- neighbouring cores exchange data along a ring dimension
                     (e.g. for systolic-style dataflow).

Two estimation modes are available:
  1. *Exact* -- constructs shortest paths (Dijkstra / Floyd-Warshall) and
     performs contention analysis on every link.
  2. *Approximate* -- assigns tensor dimensions to network dimensions by a
     greedy heuristic and computes a closed-form cycle estimate.

Typical usage::

    noc = NoC(bandwidth_bytepc=32, topology=Topo.TORUS, nodes=list(range(256)))
    bc, shift, red = noc.get_total_cycles_from_expression(
        tensor_sizes, temporal_replicas, spatial_replicas, shift_info)
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from enum import Enum
from math import ceil, inf, sqrt, isqrt
import heapq
from itertools import permutations
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------

USE_APPROX_FOR_SEQ_MAPPING = False

# Per-topology fixed overhead (pipeline fill / handshake) added once per
# transfer group.  Units: cycles.
MESH_INIT_CYCLES = 20
TORUS_INIT_CYCLES = 20
ALL_INIT_CYCLES = 100
CUSTOM_INIT_CYCLES = 50  # TODO: consider making this an input parameter

class Topo(Enum):
    """Supported NoC topology types.

    MESH    -- 2-D mesh (no wrap-around links).
    TORUS   -- 2-D torus (wrap-around links on both axes).
    ALL     -- Fully-connected (all-to-all, 1-hop between any pair).
    CUSTOM  -- User-supplied adjacency matrix.
    MESH3D  -- 3-D mesh built from the most-cubic factorisation of *N*.
    TORUS3D -- 3-D torus built from the most-cubic factorisation of *N*.
    """
    MESH = 1
    TORUS = 2
    ALL = 3
    CUSTOM = 4
    MESH3D = 5
    TORUS3D = 6

def find_closest_triplet(N):
    """Find the factorisation of *N* into three factors (a, b, c) that is
    closest to cubic (i.e. minimises ``max(a,b,c) - min(a,b,c)``).

    Used by MESH3D / TORUS3D topologies to determine the 3-D grid dimensions.

    Returns:
        Tuple[int, int, int]: Sorted triplet ``(a, b, c)`` with ``a*b*c == N``.
    """
    min_range = float('inf')
    best_triplet = (1, 1, N)

    max_a = int(round(N ** (1/3))) + 1
    for a in range(1, max_a * 2):
        if N % a != 0:
            continue
        M = N // a
        max_b = int(isqrt(M)) + 1
        for b in range(1, max_b * 2):
            if M % b != 0:
                continue
            c = M // b
            triplet = sorted([a, b, c])
            range_val = triplet[2] - triplet[0]
            if range_val < min_range:
                min_range = range_val
                best_triplet = tuple(triplet)
    return best_triplet

class NoC:
    """Network-on-Chip model for estimating inter-core communication cycles.

    The model builds an adjacency matrix for the chosen topology, lazily
    computes shortest paths via Dijkstra, and exposes methods that translate
    a high-level tensor-partitioning description (broadcast / reduce / shift)
    into an estimated cycle count.

    Attributes:
        bandwidth_bytepc (float): Link bandwidth in bytes per cycle.
        topology (Topo): Active topology enum value.
        is_spatial (bool): True for mesh / torus variants (multi-hop);
            False for all-to-all / custom (single-hop abstraction).
        interconnect_graph (list[list[int]]): NxN adjacency matrix
            (1 = direct link, 0 = no link).
        dist (list[list[float]]): Lazily-populated shortest-distance matrix.
        prev (list[list[int]]): Lazily-populated predecessor matrix for path
            reconstruction.
        exact_topo (bool): False when the topology is unknown / unsupported
            and only approximate estimation is available.
    """

    def __init__(self,
                 bandwidth_bytepc: float,
                 topology: Topo,
                 nodes: list,
                 interconnect_graph: list = None,
                 use_sram: bool = False
                 ) -> None:
        """Initialise the NoC and build the adjacency matrix.

        Args:
            bandwidth_bytepc: Link bandwidth in bytes per cycle.
            topology: Desired topology (``Topo`` enum or its int value).
            nodes: List of node identifiers (typically ``range(num_cores)``).
            interconnect_graph: Required only for ``Topo.CUSTOM``; an NxN
                adjacency matrix where entry ``[i][j] == 1`` means a direct
                link exists between nodes *i* and *j*.
            use_sram: If True, broadcast transfers use a fixed SRAM bandwidth
                (3.57 bytes/cycle) instead of ``bandwidth_bytepc``.
        """
        self.bandwidth_bytepc = bandwidth_bytepc
        self.topology = Topo(topology)
        self.use_sram = use_sram
        self.num_cores = len(nodes)

        # Mesh and torus variants are "spatial" (multi-hop); all-to-all is not.
        self.is_spatial = self.topology in (
            Topo.MESH, Topo.TORUS, Topo.MESH3D, Topo.TORUS3D
        )

        # --- Determine 2-D grid dimensions (n x m) for mesh/torus ----------
        # Try a perfect square first, then find the closest factor pair by
        # searching downward from sqrt(N).
        n = None
        m = None
        sqrt_num = int(len(nodes)**0.5)
        if sqrt_num*sqrt_num == len(nodes):
            n = sqrt_num
            m = sqrt_num

        a = sqrt_num
        while a > 0:
            if len(nodes) % a == 0:
                n = a
                m = len(nodes) // a
                break
            a -= 1

        # --- Build adjacency matrix per topology ---------------------------
        self.exact_topo = True
        if self.topology == Topo.MESH:
            # 2-D mesh: connect each node to its right and down neighbours
            # (no wrap-around).  Self-loops are marked with 1 for path init.
            interconnect_graph = [[0]*len(nodes) for node in nodes]

            for i in range(n):
                for j in range(m):
                    cur = i*m + j
                    down = (cur + m) if i < (n-1) else inf
                    right = (cur + 1) if j < (m-1) else inf

                    if cur < len(nodes):
                        interconnect_graph[cur][cur] = 1

                        if down < len(nodes):
                            interconnect_graph[cur][down] = 1
                            interconnect_graph[down][cur] = 1

                        if right < len(nodes):
                            interconnect_graph[cur][right] = 1
                            interconnect_graph[right][cur] = 1

        elif self.topology == Topo.TORUS:
            # 2-D torus: like mesh but with wrap-around links on both axes.
            interconnect_graph = [[0]*len(nodes) for node in nodes]

            for i in range(n):
                for j in range(m):
                    cur = i*m + j
                    # Modular arithmetic gives wrap-around neighbours
                    down = ((i+1)%n)*m + j
                    right = i*m + ((j + 1) % m)

                    interconnect_graph[cur][cur] = 1

                    interconnect_graph[cur][down] = 1
                    interconnect_graph[down][cur] = 1

                    interconnect_graph[cur][right] = 1
        elif self.topology == Topo.ALL:
            # Fully-connected: every pair of nodes has a direct link.
            interconnect_graph = [[1]*len(nodes) for node in nodes]
        elif self.topology == Topo.MESH3D:
            # 3-D mesh: factored into (n, m, p) dimensions as cubically as
            # possible.  Only in-bounds neighbours are connected (no wrap).
            interconnect_graph = [[0]*len(nodes) for node in nodes]
            n, m, p = find_closest_triplet(len(nodes))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        cur = i*m*p + j*p + k
                        right = ((i+1)%n)*m*p + j*p + k
                        down = i*m*p + ((j+1)%m)*p + k
                        back = i*m*p + j*p + ((k+1)%p)

                        assert cur < len(nodes)
                        interconnect_graph[cur][cur] = 1

                        if down < len(nodes):
                            interconnect_graph[cur][down] = 1
                            interconnect_graph[down][cur] = 1

                        if right < len(nodes):
                            interconnect_graph[cur][right] = 1
                            interconnect_graph[right][cur] = 1

                        if back < len(nodes):
                            interconnect_graph[cur][back] = 1
                            interconnect_graph[back][cur] = 1
                    
        elif self.topology == Topo.TORUS3D:
            # 3-D torus: like 3-D mesh but every axis has wrap-around links.
            interconnect_graph = [[0]*len(nodes) for node in nodes]
            n, m, p = find_closest_triplet(len(nodes))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        cur = i*m*p + j*p + k
                        right = ((i+1)%n)*m*p + j*p + k
                        down = i*m*p + ((j+1)%m)*p + k
                        back = i*m*p + j*p + ((k+1)%p)

                        interconnect_graph[cur][cur] = 1

                        interconnect_graph[cur][down] = 1
                        interconnect_graph[down][cur] = 1

                        interconnect_graph[cur][right] = 1
                        interconnect_graph[right][cur] = 1

                        interconnect_graph[cur][back] = 1
                        interconnect_graph[back][cur] = 1

        elif self.topology == Topo.CUSTOM:
            assert interconnect_graph != None, "Must provide graph for custom NoC topology"
        else:
            self.exact_topo = False

        self.interconnect_graph = interconnect_graph

        # --- Initialise shortest-path tables (lazily computed via Dijkstra) -
        # ``dist[u][v]`` holds the shortest hop-distance from u to v (inf if
        # not yet computed).  ``prev[u][v]`` holds the predecessor of v on the
        # shortest path from u, enabling path reconstruction.
        if self.exact_topo:
            self.prev = [[n for _ in nodes] for n in nodes]

            # Convert adjacency (1/0) to initial distances (1/inf).
            self.dist = [[inf if x == 0 else x for x in row] for row in interconnect_graph]
            for i in range(len(self.dist)):
                self.dist[i][i] = 0

            self.nodes = nodes

    def dijkstra(self, u) -> None:
        """Run Dijkstra's shortest path from node u, updating dist and prev tables."""
        # Skip if all distances from u are already computed
        if all(d != inf for d in self.dist[u]):
            return

        n = len(self.nodes)
        visited = [False] * n
        # Initialize heap with (distance, node) pairs
        heap = [(self.dist[u][i], i) for i in range(n)]
        heapq.heapify(heap)

        while heap:
            d, v = heapq.heappop(heap)
            if visited[v]:
                continue
            visited[v] = True
            for i, conn in enumerate(self.interconnect_graph[v]):
                if conn == 1 and not visited[i]:
                    new_dist = self.dist[u][v] + 1
                    if new_dist < self.dist[u][i]:
                        self.dist[u][i] = new_dist
                        self.prev[u][i] = v
                        heapq.heappush(heap, (new_dist, i))

    def num_cycle_of_access(self, num_elems: int,
                            num_transfers: int,
                            num_bytes_per_elem: int,
                            num_hops: int,
                            is_shift: bool = False,
                            use_sram: bool = False) -> float:
        """Estimate the total NoC cycles for a set of identical transfers.

        The cost model is::

            total = init_overhead + transfer_cycles

        where ``transfer_cycles = num_transfers * num_hops * payload_bytes /
        bandwidth``.  For torus topologies, non-shift (broadcast/reduce)
        traffic benefits from a 2x bandwidth factor because data can travel
        in both directions around the ring simultaneously.

        Args:
            num_elems: Number of elements per transfer.
            num_transfers: How many independent transfers are performed.
            num_bytes_per_elem: Byte width of each element (e.g. 2 for fp16).
            num_hops: Hop distance each transfer must traverse.
            is_shift: True for shift operations (no bi-directional torus
                advantage).
            use_sram: If True, use fixed SRAM bandwidth (3.57 B/cycle).

        Returns:
            Estimated total cycle count (float).
        """
        if num_transfers == 0:
            return 0

        # Topology-dependent initialisation overhead (pipeline fill, etc.)
        if self.is_spatial == False:
            assert num_hops == 1, "NoC is not spatial, so number of hops should be 1"
            init_cycles = ALL_INIT_CYCLES * num_transfers
        elif self.topology == Topo.MESH or self.topology == Topo.MESH3D:
            init_cycles = MESH_INIT_CYCLES * num_transfers
        elif self.topology == Topo.TORUS or self.topology == Topo.TORUS3D:
            init_cycles = TORUS_INIT_CYCLES * num_transfers
        else:
            assert False, "Unsupported topology. If you're defining a non-standard network, use Topo.CUSTOM and get_exact_cycles_from_expression()"

        # Data-movement cycles: total_bytes / effective_bandwidth
        num_bytes = num_elems * num_bytes_per_elem
        # if use_sram:
        #     transfer_cycles = num_transfers * num_hops * num_bytes / 3.57
        # else:
        transfer_cycles = num_transfers * num_hops * num_bytes / self.bandwidth_bytepc

        # Torus bi-directional advantage: non-shift traffic can use both ring
        # directions, effectively doubling bandwidth (halving cycles).
        if self.is_spatial:
            if self.topology == Topo.TORUS or self.topology == Topo.TORUS3D:
                if is_shift == False:
                    transfer_cycles /= 2
                else:
                    transfer_cycles /= 1.1

        return init_cycles + transfer_cycles

    def get_switch_area(self) -> int:
        '''
        Returns the area of a single switch in the NoC in square mm.
        Reference switch has 10-bit input and outputs.
        TODO: scale control logic and datapath logic size separately instead of scaling everything linearly.
        '''
        num_wires = self.bandwidth_bytepc * 8 # One wire for each bit
        reference_num_wires = 10
        reference_switch_area_sq_um = 22545 # https://ieeexplore.ieee.org/document/8358473
        area_scaling_factor_45_to_7 = 8.041 # DeepScaleTool
        noc_switch_area_sq_um = reference_switch_area_sq_um /area_scaling_factor_45_to_7 * num_wires / reference_num_wires 
        noc_switch_area_sq_mm = noc_switch_area_sq_um / (1000 ** 2)
        return noc_switch_area_sq_mm

    def get_hops(self, u, v) -> list:
        """Return the shortest-path distance and the full node path from *u* to *v*.

        Lazily triggers ``dijkstra(u)`` if shortest paths from *u* have not
        been computed yet.

        Returns:
            Tuple[int, list[int]]: ``(distance, [u, ..., v])``  where
            *distance* is the hop count and the list is the ordered sequence
            of nodes on the shortest path.
        """
        self.dijkstra(u)

        d = self.dist[u][v]
        # Reconstruct path by following predecessor pointers backwards.
        path = [v]
        while u != v:
            v = self.prev[u][v]
            path.append(v)
        path = path[::-1]
        return d, path

    def get_total_cycles_from_expression(self,
                                         tensor_sizes: List[int],
                                         temporal_var_replicas: List[int],
                                         spatial_var_replicas: List[int],
                                         shift_info: tuple,
                                         num_bytes_per_elem: int = 2,
                                         spmd_compiler: bool = False,
                                         seq_noc: bool = False,
                                         ) -> Tuple[int, int, int]:
        """High-level entry point: estimate broadcast, shift, and reduce cycles.

        Delegates to :meth:`get_approx_cycles_from_expression` and
        optionally re-attributes shift cost when ``spmd_compiler`` is True
        (models a compiler that cannot overlap shift with compute, folding
        the shift penalty into the broadcast and reduce phases instead).

        Args:
            tensor_sizes: Element count for each tensor (output first, then
                inputs).
            temporal_var_replicas: Temporal replication factor per tensor.
            spatial_var_replicas: Spatial replication factor per tensor.
            shift_info: Tuple describing shift dimensions / iteration counts.
            num_bytes_per_elem: Bytes per element (default 2 for fp16).
            spmd_compiler: If True, shift cost is folded into broadcast/reduce.
            seq_noc: If True, uses a pessimistic (sequential) mapping heuristic.

        Returns:
            Tuple[int, int, int]: ``(broadcast_cycles, shift_cycles,
            reduce_cycles)``.
        """
        all_broadcast_cycles, shift_cycles, reduce_cycles = self.get_approx_cycles_from_expression(tensor_sizes, temporal_var_replicas, spatial_var_replicas, shift_info, num_bytes_per_elem, bad_mapping=seq_noc)

        if spmd_compiler:
            # Shift cost cannot be hidden: add it to both broadcast and reduce.
            return all_broadcast_cycles+shift_cycles, 0, reduce_cycles+shift_cycles
        else:
            return all_broadcast_cycles, shift_cycles, reduce_cycles

    def get_exact_cycles_from_expression(self,
                                         tensor_sizes: List[int],
                                         temporal_var_replicas: List[int],
                                         spatial_var_replicas: List[int],
                                         shift_info: tuple,
                                         num_bytes_per_elem: int = 2) -> Tuple[int, int, int]:
        """Cycle estimation with explicit shortest-path and contention analysis.

        Unlike the approximate variant, this method reconstructs full paths
        via :meth:`get_hops`, pads them to equal length, then sums per-link
        byte volumes to identify the bottleneck link (maximum contention).

        Returns:
            Tuple[int, int, int]: ``(broadcast_cycles, shift_cycles,
            reduce_cycles)``.
        """
        if self.topology == Topo.MESH or self.topology == Topo.MESH3D:
            init_cycles = MESH_INIT_CYCLES
        elif self.topology == Topo.TORUS or self.topology == Topo.TORUS3D:
            init_cycles = TORUS_INIT_CYCLES
        elif self.topology == Topo.ALL:
            init_cycles = ALL_INIT_CYCLES
        elif self.topology == Topo.CUSTOM:
            init_cycles = CUSTOM_INIT_CYCLES

        total_broadcast_cycles = 0
        total_reduce_cycles = 0
        for i, (temporal_var_replica, spatial_var_replica, tensor_size) in enumerate(
                zip(temporal_var_replicas, spatial_var_replicas, tensor_sizes)):
            num_copies = ceil(spatial_var_replica / temporal_var_replica)
            num_shares = int(temporal_var_replica)

            # Assume a contiguous mapping: core indices 0..num_shares-1.
            mappings = list(range(num_shares))

            # --- (1) Broadcast: input tensors (i != 0) are sent to all copies -
            if i != 0:
                # Worst-case broadcast hops: max over all shares and copies.
                broadcast_hops = max([max([self.get_hops(mappings[s], mappings[s]+num_shares*c)[0]
                                           for s in range(num_shares)]) for c in range(num_copies)])
                if broadcast_hops > 0: # this is to make sure init_cycles isn't added needlessly
                    total_broadcast_cycles += init_cycles + broadcast_hops*num_bytes_per_elem / self.bandwidth_bytepc

            # --- (2) Reduce: output tensor (i == 0) accumulates partial sums --
            if i == 0:
                reduce_paths = [(self.get_hops(mappings[i], mappings[i] + len(mappings)*n)[1]) for n in range(num_copies)]
                max_hops = max([len(path) for path in reduce_paths])
                reduce_paths = [[path[i] if i < len(path) else path[-1] for i in range(max_hops)]
                                for path in reduce_paths]

                link_contention = [[0 for _ in self.nodes] for _ in self.nodes]
                for path in reduce_paths:
                    visited_nodes = {path[0]}
                    for i in range(max_hops-1):
                        u, v = path[i], path[i+1]

                        if u == v:
                            continue

                        if v in visited_nodes:
                            continue

                        visited_nodes.add(v)

                        # Accumulate bytes on both directions of the link (symmetric).
                        link_contention[u][v] += (num_bytes_per_elem*tensor_size) // temporal_var_replica // spatial_var_replica
                        link_contention[v][u] += (num_bytes_per_elem*tensor_size) // temporal_var_replica // spatial_var_replica

                # Convert byte volumes to cycle counts; bottleneck link dominates.
                link_contention = [[c / self.bandwidth_bytepc for c in V] for V in link_contention]
                reduce_cycles = max([max(V) for V in link_contention])

                # unlike broadcast, reduce can have contention because data is all different (to start)
                if reduce_cycles > 0:
                    total_reduce_cycles += init_cycles + reduce_cycles

        total_shift_size, shifted_dim, shifted_iter, shifted_vars, list_of_list_of_size_iter = shift_info

        # Sort shift dimensions by total shifted tensor size (largest first)
        # so the greedy mapper assigns the biggest tensor to the shortest
        # network dimension first, minimising total data movement.
        shifted_sizes = [sum([tensor_sizes[i] for i in vars]) for vars in shifted_vars]
        order = np.argsort(-np.array(shifted_sizes))

        # --- (3) Shift: ring-style data exchanges along loop dimensions ----
        total_shift_cycles = 0
        iter_count = 1

        for o in order:
            shift_cores = list(range(shifted_iter[o] + 1)) # TODO this is definitely wrong

            best_hops = 0
            best_path = shift_cores[:]  # Shallow copy is sufficient for list of ints

            # want to minimize total path? try all permutations of shift_cores?
            for perm in permutations(shift_cores):
                total_hops = 0
                for i in range(len(perm)-1):
                    hops, shift_path = self.get_hops(shift_cores[i], shift_cores[i+1])
                    total_hops += hops

                if total_hops < best_hops:
                    best_hops = total_hops
                    best_path = list(perm)  # Convert tuple to list, no need for deepcopy
                break # We only try one permutation for now

            # Build list of per-shift hop paths for contention modelling.
            shift_list = []

            for i in range(len(best_path)-1):
                hops, shift_path = self.get_hops(shift_cores[i], shift_cores[i+1])
                shift_list.append(shift_path)

            # Pad all paths to equal length for uniform iteration.
            max_hops = max([len(path) for path in shift_list])
            shift_list = [[path[i] if i < len(path) else path[-1] for i in range(max_hops)]
                          for path in shift_list]

            # Accumulate data volume on each link across all shift paths.
            link_contention = [[0 for _ in self.nodes] for _ in self.nodes]
            for path in shift_list:
                visited_nodes = {path[0]} # path[0] is the src node
                for i in range(max_hops-1):
                    u, v = path[i], path[i+1]

                    if u == v:
                        continue

                    if v in visited_nodes:
                        continue

                    visited_nodes.add(v)

                    link_contention[u][v] += shifted_sizes[o]
                    link_contention[v][u] += shifted_sizes[o]

            # Convert link data volumes (elements) to cycles.
            link_contention = [[c*num_bytes_per_elem / self.bandwidth_bytepc for c in V] for V in link_contention]

            # The bottleneck link (max contention) determines shift cost.
            iter_count *= shifted_iter[o]
            shift_cycles = max([max(V) for V in link_contention])*iter_count

            total_shift_cycles += init_cycles + shift_cycles

        return int(total_broadcast_cycles), int(total_shift_cycles), int(total_reduce_cycles)

    def get_approx_cycles_from_expression(self,
                                         tensor_sizes: List[int],
                                         temporal_var_replicas: List[int],
                                         spatial_var_replicas: List[int],
                                         shift_info: tuple,
                                         num_bytes_per_elem: int = 2,
                                         bad_mapping: bool = False) -> Tuple[int, int, int]:
        """Approximate cycle estimation using a greedy dimension-mapping heuristic.

        Instead of constructing explicit paths, this method models the NoC as
        having independent dimensions (2 for 2-D topologies, 3 for 3-D).  It
        greedily assigns the tensor with the largest traffic volume to the
        network dimension with the fewest accumulated hops, then computes
        cycles via :meth:`num_cycle_of_access`.

        Args:
            tensor_sizes: Element count per tensor (output first, then inputs).
            temporal_var_replicas: Temporal replication factor per tensor.
            spatial_var_replicas: Spatial replication factor per tensor.
            shift_info: Tuple ``(total_shift_size, shifted_dim, shifted_iter,
                shifted_vars, list_of_list_of_size_iter)``.
            num_bytes_per_elem: Bytes per element (default 2 for fp16).
            bad_mapping: If True, reduces available mapping dimensions by one
                to simulate a sub-optimal placement.

        Returns:
            Tuple[int, int, int]: ``(broadcast_cycles, shift_cycles,
            reduce_cycles)``.
        """
        # Initialise per-dimension accumulated hop counts.
        # 2-D topologies get 2 dimensions; 3-D topologies get 3.
        num_hops_list = [1, 1]
        if self.topology == Topo.MESH3D or self.topology == Topo.TORUS3D:
            num_hops_list = [1,1,1]
        if bad_mapping:
            # Pessimistic: remove one dimension, forcing all traffic onto fewer
            # axes (simulates a poor placement decision).
            num_hops_list = num_hops_list[:-1]

        reduce_broadcast_num_elems_list = []
        reduce_broadcast_num_transfers_list = []
        reduce_broadcast_total_elems_list = []
        # For each tensor, compute the per-transfer element count, number of
        # transfers, and total elements moved (used to rank tensors for
        # dimension assignment).
        for (i, (temporal_var_replica, spatial_var_replica, size)) in enumerate(zip(temporal_var_replicas,
                                                                                    spatial_var_replicas,
                                                                                    tensor_sizes)):
            replica_ratio = temporal_var_replica
            assert replica_ratio > 0
            elems_per_transfer = size / replica_ratio
            num_transfers = replica_ratio - 1
            reduce_broadcast_num_elems_list.append(elems_per_transfer)
            reduce_broadcast_num_transfers_list.append(num_transfers)
            reduce_broadcast_total_elems_list.append(elems_per_transfer * num_transfers)

        # Greedy dimension assignment for spatial topologies:
        # Repeatedly pick the tensor with the most remaining traffic and map
        # it onto the network dimension with the fewest accumulated hops.
        # After assignment, that dimension's hop count grows by the number of
        # nodes the tensor spans (num_transfers + 1).
        reduce_broadcast_num_hops_list = [1]*len(reduce_broadcast_num_elems_list)
        if self.is_spatial:
            while sum(reduce_broadcast_total_elems_list):
                max_tensor_idx = max(range(len(reduce_broadcast_total_elems_list)), key=reduce_broadcast_total_elems_list.__getitem__)
                min_dim_idx = min(range(len(num_hops_list)), key=num_hops_list.__getitem__)
                reduce_broadcast_num_hops_list[max_tensor_idx] = num_hops_list[min_dim_idx]
                num_hops_list[min_dim_idx] *= (reduce_broadcast_num_transfers_list[max_tensor_idx]+1)
                reduce_broadcast_total_elems_list[max_tensor_idx] = 0

        num_hops_list[0] = min(num_hops_list[0], int(sqrt(num_hops_list[0]))*2)

        # Index 0 is the output tensor -> reduce traffic.
        reduce_cycles = self.num_cycle_of_access(reduce_broadcast_num_elems_list[0],
                                                reduce_broadcast_num_transfers_list[0],
                                                num_bytes_per_elem,
                                                reduce_broadcast_num_hops_list[0])
        # Indices 1.. are input tensors -> broadcast traffic.
        all_broadcast_cycles = 0
        for num_elems, num_transfers, num_hops in zip(reduce_broadcast_num_elems_list[1:],
                                                    reduce_broadcast_num_transfers_list[1:],
                                                    reduce_broadcast_num_hops_list[1:]):
            all_broadcast_cycles += self.num_cycle_of_access(num_elems,
                                                            num_transfers,
                                                            num_bytes_per_elem,
                                                            num_hops,
                                                            use_sram=self.use_sram)
        
        total_shift_size, shifted_dim, shifted_iter, shifted_vars, list_of_list_of_size_iter = shift_info
        shift_cycles = 0
        if total_shift_size != 0:
            for list_of_size_iter in list_of_list_of_size_iter:
                max_shift_cycles = 0
                for size, num_shifts, num_iters in list_of_size_iter:
                    num_hops = 1
                    if self.is_spatial:
                        min_dim_idx = min(range(len(num_hops_list)), key=num_hops_list.__getitem__)
                        num_hops = num_hops_list[min_dim_idx]
                        num_hops = min(num_hops, int(sqrt(self.num_cores)) * 2)
                        num_hops_list[min_dim_idx] *= (num_shifts+1)
                    cur_shift_cycles = self.num_cycle_of_access(size,
                                                                num_shifts*num_iters,
                                                                num_bytes_per_elem,
                                                                num_hops,
                                                                is_shift=True)
                    if self.is_spatial:
                        max_shift_cycles = max(max_shift_cycles, cur_shift_cycles)
                    else:
                        max_shift_cycles += cur_shift_cycles
                shift_cycles += max_shift_cycles

        return int(all_broadcast_cycles), int(shift_cycles), int(reduce_cycles)

if __name__ == "__main__":
    # Quick smoke test: 10-node 3-D torus with a simple two-tensor workload.
    N = 10

    noc_bw = 1
    noc_topo = Topo.TORUS3D
    nodes = list(range(N))

    noc = NoC(noc_bw, noc_topo, nodes)

    tensor_sizes = [10, 20]
    temporal_var_replicas = [1, 1]
    spatial_var_replicas = [1, 1]

    shift_info = (
        177880,            # total_shift_size
        [0, 6],            # shifted_dim
        [1, 4],            # shifted_iter
        [[0, 1], [2]],    # shifted_vars
        [],                # list_of_list_of_size_iter
    )

    noc.get_total_cycles_from_expression(
        tensor_sizes, temporal_var_replicas, spatial_var_replicas,
        shift_info, num_bytes_per_elem=2,
    )

