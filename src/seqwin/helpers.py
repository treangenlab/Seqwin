"""
Helpers
=======

Helper functions and dtypes for `kmers.py`. 

Dependencies:
-------------
- numpy
- numba
- networkx
- .graph
- .utils
- .config

Functions:
----------
- get_edges
- merge_weighted_edges
- sort_by_hash
- agg_by_hash
- get_subgraphs
- filter_kmers
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import logging
from random import Random
from heapq import heappush, heappop

logger = logging.getLogger(__name__)

import numpy as np
import networkx as nx
from numpy.typing import NDArray
from numba import njit

from .utils import log_and_raise
from .config import NODE_P


def get_subgraphs(
    graph: nx.Graph, 
    penalty_th: float, 
    min_nodes: int, 
    max_nodes: int | None, 
    rng: Random
) -> tuple[
    tuple[frozenset[np.uint64], ...], 
    frozenset[np.uint64]
]:
    """Find disjoint (no shared node) subgraphs whose average node-penalty ≤ `penalty_th` and size within `size_th`. 
    1. Remove low-weight edges and isolated nodes from `graph`. 
    2. Find nodes with penalty ≤ `penalty_th` as seeds of subgraphs. 
    3. Greedy seed-expansion with breadth first search (BFS), where the neighboring node with the lowest penalty is 
        selected in each iteration. 

    A heap frontier (nodes to be visited in BFS) is used to accelerate the expansion process. 
    The heap is implemented with the built-in Python `heapq` module, which is a min-heap. 
    E.g., when tuples of `(penalty, node)` are pushed to the heap, it will always pop the tuple with the smallest `penalty` first. 
    This is faster than calling `min()` everytime to fetch the node with the lowest penalty. 
    When tested on the Salmonella dataset (576 genomes, no edge filtering), this implementation is more than 3x faster than the naive one. 
    However, the performance gain becomes less significant when more low-weight edges are removed. 

    Args:
        graph (nx.Graph): See `KmerGraph.graph`. 
        penalty_th (float): See `Config` in `config.py`. 
        min_nodes (int): See `Config` in `config.py`. 
        max_nodes (int | None): See `Config` in `config.py`. 
        rng (random.Random): See `RunState` in `config.py`. 

    Returns:
        tuple: A tuple containing
            1. tuple[frozenset[np.uint64], ...]: See `KmerGraph.subgraphs`. 
            2. frozenset[np.uint64]: Union of k-mer hash values in all subgraphs. 
    """
    # a dict mapping node to penalty for faster lookup
    node_penalty: dict[int, float] = dict(
        # sort nodes for reproducibility (nodes order decides seeds order)
        sorted(graph.nodes(data=NODE_P))
    )

    # collect all seed nodes and shuffle
    # use <=, otherwise there will be no seed when penalty_th = 0
    seeds = list(n for n, p in node_penalty.items() if p <= penalty_th)
    rng.shuffle(seeds)
    logger.info(f' - Expanding subgraphs from {len(seeds)} seed nodes (penalty<={penalty_th:.5f})...')

    used: set[int] = set() # nodes already assigned to a subgraph
    subgraphs: list[set[int]] = list() # list of subgraphs to return

    for s in seeds:
        if s in used:
            continue

        # initialize the subgraph (sg)
        sg = {s}
        sum_penalty = node_penalty[s]

        #---------- subgraph expansion (the naive way) ----------#
        # # add initial neighbors to frontier
        # frontier = set(graph.neighbors(s)) - used - sg

        # # expand the subgraph by adding the node in the frontier with the lowest penalty
        # while frontier and len(sg) < max_size:
        #     node = min(frontier, key=lambda n: (node_penalty[n], n))

        #     # whether to accept this node
        #     new_sum_penalty = sum_penalty + node_penalty[node]
        #     if new_sum_penalty / (len(sg)+1) <= penalty_th:
        #         # accept
        #         sg.add(node)
        #         sum_penalty = new_sum_penalty

        #         # bring in its neighbors
        #         frontier |= (set(graph.neighbors(node)) - used - sg)

        #     # whether accepted or not, never reconsider this node
        #     frontier.remove(node)
        #---------- subgraph expansion (the naive way) ----------#

        #---------- subgraph expansion (heap frontier) ----------#
        # min‐heap of (penalty, node)
        frontier_heap: list[tuple[float, int]] = list()
        # a set synced with frontier_heap
        # for faster membership checking, also guarantees every node in the heap is unique
        frontier_set: set[int] = set()

        # add initial neighbors to frontier
        for nbr in graph.neighbors(s):
            if (nbr not in used) and (nbr not in sg):
                heappush(frontier_heap, (node_penalty[nbr], nbr))
                frontier_set.add(nbr)

        # expand the subgraph by adding the node in the frontier with the lowest penalty
        while frontier_heap and ((max_nodes is None) or (len(sg) < max_nodes)):
            penalty, node = heappop(frontier_heap)
            # keep frontier_heap and frontier_set consistent (might not be necessary but to be safe)
            if node not in frontier_set:
                continue

            # whether to accept this node
            new_sum_penalty = sum_penalty + penalty
            # use <=, otherwise there will be no new node when penalty_th = 0
            if new_sum_penalty / (len(sg)+1) <= penalty_th:
                # accept
                sg.add(node)
                sum_penalty = new_sum_penalty

                # bring in its neighbors
                for nbr in graph.neighbors(node):
                    if (nbr not in used) and (nbr not in sg) and (nbr not in frontier_set):
                        heappush(frontier_heap, (node_penalty[nbr], nbr))
                        frontier_set.add(nbr)

            # whether accepted or not, never reconsider this node
            frontier_set.remove(node)
        #---------- subgraph expansion (heap frontier) ----------#

        # keep or discard the subgraph
        if len(sg) >= min_nodes:
            subgraphs.append(sg)
            used |= sg

    if len(subgraphs) > 0:
        logger.info(f' - Found {len(subgraphs)} low-penalty subgraphs')
    else:
        log_and_raise(
            RuntimeError, 
            ('No low-penalty subgraph was found. '
            'Try decrease --stringency, or increase --penalty-th (penalty threshold, check log for the calculated value).')
        )

    # due to the greedy nature of node expansion, subgraphs created first are usually larger
    # by shuffling the subgraphs, we can get a more balanced distribution of sizes in downstream multiprocessing
    rng.shuffle(subgraphs)

    return tuple(frozenset(sg) for sg in subgraphs), frozenset(used)


@njit(nogil=True, cache=True)
def _copy_blocks(arr_old: NDArray, arr_new: NDArray, starts: NDArray, sizes: NDArray) -> None:
    """Copy contiguous blocks from an old array to a new array. 

    Args:
        arr_old (NDArray): The old array. 
        arr_new (NDArray): The new array. 
        starts (NDArray): Start position of each block in the old array. 
        sizes (NDArray): Size of each block. 
    """
    ptr_new = 0
    for i in range(len(starts)):
        # iterate through each block
        ptr_old = starts[i]
        s = sizes[i]
        arr_new[ptr_new : ptr_new + s] = arr_old[ptr_old : ptr_old + s]
        ptr_new += s


def filter_kmers(
    kmers: NDArray, idx: NDArray, nodes: NDArray, used: frozenset[int]
) -> tuple[NDArray, NDArray, NDArray]:
    """
    1. Remove k-mers (`kmers`, `idx` and `nodes`) not included in `used`. 
    2. Update 'start' and 'stop' in nodes. 

    Args:
        kmers (NDArray): See `KmerGraph.kmers` (already sorted by 'hash'). 
        idx (NDArray): See `KmerGraph.idx`. 
        nodes (NDArray): See `KmerGraph.nodes` (sorted by 'hash'). 
        used (frozenset[int]): Output of `get_subgraphs()`. 

    Returns:
        tuple: A tuple containing
            1. NDArray: See `KmerGraph.kmers`. 
            2. NDArray: See `KmerGraph.idx`. 
            3. NDArray: See `KmerGraph.nodes`. 
    """
    logger.info(' - Removing k-mers not included in any of the subgraphs...')
    hashes = nodes['hash']

    # select used nodes
    used_arr = np.fromiter(used, dtype=hashes.dtype, count=len(used))
    used_arr.sort()
    nodes = nodes[
        # used_arr is sorted to preserve node order
        np.searchsorted(hashes, used_arr)
    ]
    starts_old = nodes['start'].copy()

    # update start and stop in nodes
    sizes = nodes['stop'] - nodes['start'] # group size
    starts = np.empty(len(nodes), dtype=hashes.dtype)
    starts[0] = 0
    if len(nodes) > 1:
        starts[1:] = np.cumsum(sizes)[:-1]
    nodes['start'] = starts
    nodes['stop'] = starts + sizes

    # copy used k-mers
    total = nodes['stop'][-1]
    kmers_new = np.empty(total, dtype=kmers.dtype)
    _copy_blocks(kmers, kmers_new, starts_old, sizes)
    idx_new = np.empty(total, dtype=idx.dtype)
    _copy_blocks(idx, idx_new, starts_old, sizes)

    logger.info(f' - {len(kmers_new)} k-mers left')
    return kmers_new, idx_new, nodes
