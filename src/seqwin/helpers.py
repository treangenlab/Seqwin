"""
Helpers
=======

Helper functions and dtypes for `kmers.py`. 

Dependencies:
-------------
- numpy
- numba
- networkx
- .minimizer
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
from numba import types, typed, prange, njit, from_dtype, get_num_threads

from .minimizer import KMER_DTYPE
from .utils import log_and_raise
from .config import NODE_P

# dtypes for numpy
_HASH_NP_DT = KMER_DTYPE['hash'] # k-mer hash values
NODE_DTYPE = np.dtype([ # k-mer nodes
    ('hash', _HASH_NP_DT), 
    ('n_tar', KMER_DTYPE['assembly_idx']), 
    ('n_neg', KMER_DTYPE['assembly_idx']), 
    ('penalty', np.float64), 
    # store start/stop indices in the sorted KmerGraph.kmers, for faster access of each k-mer group
    ('start', _HASH_NP_DT), 
    ('stop', _HASH_NP_DT), 
])

# dtypes for numba
_HASH_NB_DT = from_dtype(_HASH_NP_DT) # k-mer hash values
_EDGE_NB_DT = types.UniTuple(_HASH_NB_DT, 2) # graph edges
_WEIGHT_NB_DT = types.Tuple((_HASH_NB_DT, types.intp)) # graph weight: (w, last_seen_assembly_idx)
HASH_ARR_NB_DT = types.Array(_HASH_NB_DT, 1, 'A') # array of hashes; 'A': accepts arbitrary/strided arrays


@njit(nogil=True, cache=True)
def get_edges(hashes) -> tuple[NDArray, NDArray]:
    """Get weighted edges and isolated nodes. 

    Args:
        hashes (List[List[Array]]): K-mer hash values from all assemblies. 
    
    Returns:
        tuple: A tuple containing
            1. NDArray: A 3-column array of unique edges and their weights. 
            2. NDArray: Isolated nodes. 
    """
    # a counter for edges and weight
    edge_w = typed.Dict.empty(
        # all type variables must be defined outside this function
        key_type=_EDGE_NB_DT, 
        value_type=_WEIGHT_NB_DT # a tuple of (w, last_seen_assembly_idx)
    )
    # isolated k-mers found in short sequence records
    # there is no numba typed set, just use set()
    isolates = set()

    for assembly_idx, hashes_assembly in enumerate(hashes):

        for hashes_record in hashes_assembly:
            n = len(hashes_record)
            if n == 1:
                # isolate nodes cannot be added to graph since they don't have any edge
                # they might be included in another path of another record, but that doesn't matter
                isolates.add(hashes_record[0])
            else:
                # get edges of the current sequence record
                for i in range(n - 1):
                    h1 = hashes_record[i]
                    h2 = hashes_record[i + 1]
                    # canonical order of edge nodes (undirected graph)
                    if h1 > h2:
                        h1, h2 = h2, h1
                    e = (h1, h2)

                    # check if edge exists in previous assemblies
                    if e in edge_w:
                        w, last_seen = edge_w[e]
                        # only increment if we haven't seen this edge in the current assembly yet
                        if last_seen != assembly_idx:
                            edge_w[e] = (w + _HASH_NB_DT(1), assembly_idx)
                    else:
                        # first time seeing this edge
                        edge_w[e] = (_HASH_NB_DT(1), assembly_idx)

    # prepare output edges and weights as a 3-column np array
    n_edges = len(edge_w)
    edges = np.empty((n_edges, 3), dtype=_HASH_NP_DT)
    i = 0
    for e, w in edge_w.items():
        edges[i, 0] = e[0]
        edges[i, 1] = e[1]
        edges[i, 2] = w[0]
        i += 1

    # convert isolates to a np array
    n_isolates = len(isolates)
    isolates_arr = np.empty(n_isolates, dtype=_HASH_NP_DT)
    i = 0
    for h in isolates:
        isolates_arr[i] = h
        i += 1

    return edges, isolates_arr


@njit(nogil=True, parallel=True)
def merge_weighted_edges(edges: NDArray) -> NDArray:
    """Add weights of the same edges. 
    1. Parallel LSD radix sort (similar to `sort_by_hash()`). 
    2. Aggregation. 

    Args:
        edges (NDArray): A 3-column array of weighted edges (u, v, w). 
    
    Returns:
        NDArray: Unique edges with sum of weights. 
    """
    n = edges.shape[0]
    n_cpu = get_num_threads()

    buf = np.empty_like(edges)
    counts = np.empty((n_cpu, 65536), dtype=np.int64)
    chunk_size = (n + n_cpu - 1) // n_cpu

    # LSD radix sort: sort secondary key (2nd col) then primary key (1st col)
    for col in (1, 0):
        for shift in range(0, 64, 16):
            counts[:] = 0

            # --- parallel count ---
            for t in prange(n_cpu):
                start = t * chunk_size
                end = min(start + chunk_size, n)
                for i in range(start, end):
                    key = (edges[i, col] >> shift) & 0xFFFF
                    counts[t, key] += 1

            # --- global prefix sum (serial) ---
            current = 0
            for key in range(65536):
                for t in range(n_cpu):
                    c = counts[t, key]
                    counts[t, key] = current
                    current += c

            # --- parallel scatter (move) ---
            for t in prange(n_cpu):
                start = t * chunk_size
                end = min(start + chunk_size, n)
                if start < end:
                    local_offsets = counts[t]

                    for i in range(start, end):
                        key = (edges[i, col] >> shift) & 0xFFFF
                        pos = local_offsets[key]

                        # move the entire row
                        buf[pos, 0] = edges[i, 0]
                        buf[pos, 1] = edges[i, 1]
                        buf[pos, 2] = edges[i, 2]
                        local_offsets[key] += 1

            # swap buffers
            edges, buf = buf, edges

    # sequential aggregation
    buf = np.empty_like(edges) # make buf point to virtual memory again
    idx = 0
    u = edges[0, 0]
    v = edges[0, 1]
    w = edges[0, 2]
    for i in range(1, n):
        if edges[i, 0] != u or edges[i, 1] != v:
            buf[idx, 0] = u
            buf[idx, 1] = v
            buf[idx, 2] = w
            idx += 1

            u = edges[i, 0]
            v = edges[i, 1]
            w = edges[i, 2]
        else:
            w += edges[i, 2]

    buf[idx, 0] = u
    buf[idx, 1] = v
    buf[idx, 2] = w

    return buf[:idx + 1]


@njit(nogil=True, parallel=True) # cannot be cached
def sort_by_hash(kmers: NDArray) -> NDArray:
    """Sort `kmers` by 'hash' in-place in a stable manner, and return the sorted indices. 
    Use LSD (Least Significant Digit) radix sort. 
    - Indices have the same dtype as `hash`. 
    - A buffer is needed (same shape and dtype as the indices), so memory usage should be around `kmers + idx + buffer`. 
    - Note that this function is hard coded for the k-mer dtype below. 
    ```python
    np.dtype([
        ('hash', np.uint64), 
        ('pos', np.uint32), 
        ('record_idx', np.uint16), 
        ('assembly_idx', np.uint16), 
        ('is_target', np.bool_)
    ])
    ```

    Args:
        kmers (NDArray): See `KmerGraph.kmers`. 

    Returns:
        NDArray: See `KmerGraph.idx`. 
    """
    n = kmers.size
    n_cpu = get_num_threads()

    # create k-mer indices
    idx = np.arange(n, dtype=_HASH_NP_DT) # use the same dtype as hash
    # create a buffer, also use the same dtype as hash
    buf = np.empty_like(idx)

    # 1. sort idx by 'hash' using a stable algorithm (LSD radix sort on 64-bit keys)
    # process 16 bits at a time (4 passes for 64-bit keys: 0-15, 16-31, 32-47, 48-63)
    # allocate thread-local count array
    counts = np.empty((n_cpu, 65536), dtype=np.int64)
    # chunk size for each thread
    chunk_size = (n + n_cpu - 1) // n_cpu
    for shift in range(0, 64, 16):
        counts[:] = 0

        # --- parallel count ---
        for t in prange(n_cpu):
            start = t * chunk_size
            end = min(start + chunk_size, n)
            if start < end:
                for i in range(start, end):
                    # extract 16-bit key
                    key = (kmers[idx[i]]['hash'] >> shift) & 0xFFFF
                    counts[t, key] += 1

        # --- global prefix sum (serial) ---
        # we need to calculate where each thread should start writing each key
        # convert counts to per-thread starting offsets in-place
        current = 0
        for key in range(65536):
            for t in range(n_cpu):
                c = counts[t, key]
                counts[t, key] = current # global offset
                current += c

        # --- parallel scatter (move) ---
        # use the calculated offsets to place indices into the correct sorted position
        for t in prange(n_cpu):
            start = t * chunk_size
            end = min(start + chunk_size, n)
            if start < end:
                # cache local offsets to avoid repeated global array access
                local_offsets = counts[t]

                for i in range(start, end):
                    val_idx = idx[i]
                    key = (kmers[val_idx]['hash'] >> shift) & 0xFFFF

                    pos = local_offsets[key]
                    buf[pos] = val_idx
                    local_offsets[key] += 1 # increment for the next item with same key

        # swap buffers
        idx, buf = buf, idx

    # 2. reorder kmers in-place using the sorted indices 
    # use buf as the buffer for column-wise gather (lower peak mem than a full copy of kmers)
    # hash (same dtype as buf)
    for i in prange(n):
        buf[i] = kmers[idx[i]]['hash']
    for i in prange(n):
        kmers[i]['hash'] = buf[i]

    # pack pos (uint32), record_idx (uint16) and assembly_idx (uint16) into one uint64
    for i in prange(n):
        j = idx[i]
        buf[i] = (
            np.uint64(kmers[j]['pos']) | 
            (np.uint64(kmers[j]['record_idx']) << 32) | 
            (np.uint64(kmers[j]['assembly_idx']) << 48)
        )
    for i in prange(n):
        v = buf[i]
        kmers[i]['pos'] = np.uint32(v)
        kmers[i]['record_idx'] = np.uint16(v >> 32)
        kmers[i]['assembly_idx'] = np.uint16(v >> 48)

    # is_target (single bit)
    buf_u8 = buf.view(np.uint8) # reduce memory traffic
    for i in prange(n):
        buf_u8[i] = kmers[idx[i]]['is_target']
    for i in prange(n):
        kmers[i]['is_target'] = (buf_u8[i] != 0)

    return idx


@njit(nogil=True, cache=True)
def agg_by_hash(hashes: NDArray, assembly_idx: NDArray, is_target: NDArray) -> NDArray:
    """Count the number of target/non-target assemblies for each unique hash value. 

    Args:
        hashes (NDArray): Field of `KmerGraph.kmers`. 
        assembly_idx (NDArray): Field of `KmerGraph.kmers`. 
        is_target (NDArray): Field of `KmerGraph.kmers`. 

    Returns:
        NDArray: See `KmerGraph.nodes`. 
    """
    n = hashes.size
    # pre-allocate output array (never larger than n)
    nodes = np.empty(n, dtype=NODE_DTYPE) # define dtype outside the numba function

    node_i = 0
    i = 0
    while i < n:
        curr_hash = hashes[i]
        start = i # start index of the current hash group

        n_tar = n_neg = 0
        # walk all rows having the same hash
        while i < n and hashes[i] == curr_hash:
            curr_a = assembly_idx[i]
            curr_t = is_target[i]

            # skip any further duplicates in the same assembly
            j = i + 1
            while (
                j < n 
                and hashes[j] == curr_hash
                and assembly_idx[j] == curr_a
            ):
                j += 1

            # count the current assembly
            if curr_t:
                n_tar += 1
            else:
                n_neg += 1
            i = j # advance by the duplicates

        # set individual values in numba, no fancy indexing
        nodes[node_i]['hash'] = curr_hash
        nodes[node_i]['n_tar'] = n_tar
        nodes[node_i]['n_neg'] = n_neg
        nodes[node_i]['penalty'] = .0 # placeholder for penalty
        nodes[node_i]['start'] = start
        nodes[node_i]['stop'] = i # stop index of the current hash group
        node_i += 1

    # trim the over-allocated buffers
    return nodes[:node_i]


def get_subgraphs(
    graph: nx.Graph, penalty_th: float, min_nodes: int, max_nodes: int | None, rng: Random
) -> tuple[tuple[frozenset[int], ...], frozenset[int]]:
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
            1. tuple[frozenset[int], ...]: See `KmerGraph.subgraphs`. 
            2. frozenset[int]: Union of k-mer hash values in all subgraphs. 
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
    logger.info(f' - Expanding subgraphs from {len(seeds)} seed nodes (penalty≤{penalty_th:.5f})...')

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
        log_and_raise(RuntimeError, 'No low-penalty subgraph was found. Try increase penalty threshold')

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
