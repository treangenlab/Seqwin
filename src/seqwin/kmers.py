"""
K-mer Graph
===========

A core module of Seqwin. Build a k-mer graph from all input assemblies and extract low-penalty subgraphs.

Dependencies:
-------------
- numpy
- networkx
- .graph
- .assemblies
- .utils
- .config

Classes:
--------
- FilteredGraph

Functions:
----------
- build_graph
- filter_graph
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import logging
from random import Random
from time import time
from heapq import heappush, heappop

logger = logging.getLogger(__name__)

import numpy as np
import networkx as nx
from numpy.typing import NDArray

from .graph import KmerGraph, _get_penalty, _filter_kmers
from .assemblies import Assemblies
from .utils import print_time_delta, log_and_raise
from .config import Config, RunState, HAS_MASH, WORKINGDIR, EDGE_W, NODE_P


class FilteredGraph(KmerGraph):
    r"""The filtered minimizer graph class.

    Attributes:
        kmers (NDArray[np.void]): Only includes k-mers with hashes in `subgraphs`.
        nodes (NDArray[np.void]): Only includes nodes with hashes in `subgraphs`.
        edges (NDArray[np.void]): Low-weight edges are filtered.
        record_offsets (NDArray[np.uint32]): Inherited from `KmerGraph.record_offsets`.
        record_ids (NDArray[np.str\_]): Inherited from `KmerGraph.record_ids`.
        nx_graph (nx.Graph): The NetworkX graph instance built from filtered edges.
        subgraphs (tuple[frozenset[np.uint64], ...]): Low-penalty subgraphs. Each subgraph is a set of k-mer hash values.
    """
    __slots__ = ('nx_graph', 'subgraphs')
    nx_graph: nx.Graph
    subgraphs: tuple[frozenset[np.uint64], ...]

    def __init__(
        self,
        kmers: NDArray[np.void],
        nodes: NDArray[np.void],
        edges: NDArray[np.void],
        record_offsets: NDArray[np.uint32],
        record_ids: NDArray[np.str_],
        nx_graph: nx.Graph,
        subgraphs: tuple[frozenset[np.uint64], ...]
    ) -> None:
        """Initialized a filtered minimizer graph from computed graph data.
        """
        self.kmers = kmers
        self.nodes = nodes
        self.edges = edges
        self.record_offsets = record_offsets
        self.record_ids = record_ids
        self.nx_graph = nx_graph
        self.subgraphs = subgraphs


def _get_filtered_graph(
    graph: KmerGraph, penalty_th: float, edge_weight_th: float, min_nodes: int, max_nodes: int | None, rng: Random
) -> FilteredGraph:
    """
    1. Remove low-weight edges and isolated nodes.
    2. Create the graph instance and extract low-penalty subgraphs.
    3. Remove k-mers not included in any of the subgraphs.

    Args:
        penalty_th (float): See `RunState` in `config.py`.
        edge_weight_th (float): See `RunState` in `config.py`.
        min_nodes (int): See `RunState` in `config.py`.
        max_nodes (int | None): See `RunState` in `config.py`.
        rng (random.Random): See `RunState` in `config.py`.

    Returns:
        FilteredGraph: The filtered graph.
    """
    logger.info('Extracting low-penalty subgraphs from the k-mer graph...')
    tik = time()

    if max_nodes is None:
        logger.warning(f' - Upper limit of subgraph size is not set. Lower limit is set to {min_nodes}')
    else:
        logger.info(f' - Subgraph size limit is set to [{min_nodes}, {max_nodes}]')

    # remove low-weight edges and isolated nodes
    nodes, edges, nx_graph = _filter_edges_and_nodes(graph.nodes, graph.edges, edge_weight_th)

    # get low-penalty subgraphs
    subgraphs, used_hashes = _get_subgraphs(nx_graph, penalty_th, min_nodes, max_nodes, rng)

    # remove unused k-mers
    logger.info(' - Removing k-mers not included in any of the subgraphs...')
    kmers, nodes = _filter_kmers(graph.kmers, nodes, used_hashes)
    logger.info(f' - {len(kmers)} k-mers left')

    print_time_delta(time()-tik)
    return FilteredGraph(
        kmers=kmers,
        nodes=nodes,
        edges=edges,
        record_offsets=graph.record_offsets,
        record_ids=graph.record_ids,
        nx_graph=nx_graph,
        subgraphs=subgraphs
    )


def _filter_edges_and_nodes(
    nodes: NDArray, edges: NDArray, edge_weight_th: float
) -> tuple[NDArray, NDArray, nx.Graph]:
    """Remove low-weight edges and isolated nodes, and create the graph instance.

    Args:
        nodes (NDArray): See `KmerGraph.nodes`.
        edges (NDArray): See `KmerGraph.edges`.
        edge_weight_th (float): See `RunState` in `config.py`.

    Returns:
        tuple: A tuple containing
            1. NDArray: Filtered nodes.
            2. NDArray: Filtered edges.
            3. nx.Graph: See `KmerGraph.graph`.
    """
    logger.info(' - Filtering graph edges and nodes...')
    n_nodes, n_edges = len(nodes), len(edges)

    # remove low-weight edges
    th = np.uintp(edge_weight_th) # for faster comparison
    edges = edges[edges['weight'] > th]
    edge_values = edges.view(np.uint64).reshape(-1, 3)
    logger.info(f' - Removed {n_edges - len(edges)} edges with weight<{edge_weight_th:.3f}, {len(edges)} edges left')

    # remove isolated nodes
    nodes_to_keep = np.unique(edge_values[:, :2])
    nodes = nodes[
        np.searchsorted(nodes['hash'], nodes_to_keep)
    ]
    logger.info(f' - Removed {n_nodes - len(nodes)} isolated nodes, {len(nodes)} nodes left')

    logger.info(' - Building graph...')
    nx_graph = nx.Graph()
    nx_graph.add_weighted_edges_from(edge_values, weight=EDGE_W)
    nx.set_node_attributes(
        nx_graph,
        values=dict(zip(nodes['hash'], nodes['penalty'])),
        name=NODE_P
    )

    return nodes, edges, nx_graph


def _get_subgraphs(
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
    This is faster than calling `min()` every time to fetch the node with the lowest penalty.
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
            'Try decrease --stringency, or increase --penalty-th (penalty threshold, check log for the calculated value)')
        )

    # due to the greedy nature of node expansion, subgraphs created first are usually larger
    # by shuffling the subgraphs, we can get a more balanced distribution of sizes in downstream multiprocessing
    rng.shuffle(subgraphs)

    return tuple(frozenset(sg) for sg in subgraphs), frozenset(used)


def _expected_frac(jaccard_mtx: NDArray) -> np.floating:
    """Calculate the expected fraction from a matrix of pairwise Jaccard indices.
    Here fraction means: for a k-mer `h` in a group of genomes (k-mer sets), the fraction of sets in
    a second group that also contain `h`. `jaccard_mtx` is the pairwise Jaccard indices between sets
    in the two groups. Note that this could be a self comparison (two groups are the same).
    - `E(frac) = mean(2J / (1+J))`, where `J` is the Jaccard matrix.
    - This expectation ≥ mean of the Jaccard matrix, (`2J / (1+J) ≥ J, 0 ≤ J ≤ 1`).
    """
    return np.mean(2 * jaccard_mtx / (1 + jaccard_mtx))


def build_graph(assemblies: Assemblies, config: Config) -> KmerGraph:
    """
    Args:
        assemblies (Assemblies): See `Assemblies` in `assemblies.py`.
        config (Config): See `Config` in `config.py`.

    Returns:
        KmerGraph: The unscored minimizer graph.
    """
    logger.info(f'Building minimizer graph from {len(assemblies)} assemblies...')
    if config.low_memory:
        logger.warning(' - Low-memory mode is enabled; graph construction may take longer.')
    tik = time()

    graph = KmerGraph(
        assembly_paths=assemblies.paths,
        kmerlen=config.kmerlen,
        windowsize=config.windowsize,
        n_cpu=config.n_cpu,
        low_memory=config.low_memory
    )

    logger.info(f' - Found {len(graph.kmers)} minimizers')
    logger.info(f' - Found {len(graph.nodes)} nodes (unique minimizers)')
    logger.info(f' - Found {len(graph.edges)} weighted edges')

    print_time_delta(time()-tik)
    return graph


def filter_graph(
    graph: KmerGraph, assemblies: Assemblies, config: Config, state: RunState
) -> tuple[FilteredGraph, NDArray | None]:
    """
    1. Calculate node penalty scores. This will populate empty fields in `KmerGraph.nodes` in place.
    2. Calculate filtering thresholds.
    3. Filter the graph and extract low-penalty subgraphs.

    Args:
        assemblies (Assemblies): See `Assemblies` in `assemblies.py`.
        config (Config): See `Config` in `config.py`.
        state (RunState): See `RunState` in `config.py`.

    Returns:
        tuple: A tuple containing
            1. FilteredGraph: The filtered graph.
            2. NDArray | None: A matrix of Jaccard indices of all assembly pairs.
    """
    overwrite = config.overwrite
    kmerlen = config.kmerlen
    windowsize = config.windowsize
    penalty_th = config.penalty_th
    run_mash = config.run_mash
    stringency = config.stringency
    min_len = config.min_len
    max_len = config.max_len
    penalty_th_cap = config.penalty_th_cap
    edge_w_th_mul = config.edge_w_th_mul
    min_nodes_floor = config.min_nodes_floor
    max_nodes_cap = config.max_nodes_cap
    sketchsize = config.sketchsize
    n_cpu = config.n_cpu

    working_dir = state.working_dir
    rng = state.rng
    n_tar = state.n_tar
    n_neg = state.n_neg

    # fill in nodes['n_tar'], nodes['n_neg'] and nodes['penalty']
    nodes = graph.nodes
    _get_penalty(
        kmers=graph.kmers,
        nodes=nodes,
        record_offsets=graph.record_offsets,
        is_targets=assemblies.is_targets,
        n_cpu=n_cpu
    )

    # calculate filter params
    # 1. calculate penalty threshold
    if penalty_th is None:
        logger.info(f'Calculating penalty threshold...')
        tik = time()

        # we only care about the presence & absence of k-mers in target assemblies
        if run_mash and HAS_MASH:
            jaccard = assemblies.mash(
                kmerlen=kmerlen,
                sketchsize=sketchsize,
                out_path=working_dir / WORKINGDIR.mash,
                overwrite=overwrite,
                n_cpu=n_cpu
            )
            target_idx = np.flatnonzero(assemblies.is_targets)
            non_target_idx = np.flatnonzero(~assemblies.is_targets)
            e_absence_tar = 1 - _expected_frac(jaccard[np.ix_(target_idx, target_idx)])
            e_presence_neg = _expected_frac(jaccard[np.ix_(non_target_idx, target_idx)])
        else:
            if run_mash:
                logger.error('Mash is not installed. Falling back to minimizer sketches.')
            # calculate expected fractions directly from minimizer sketches
            # for tar absence or neg presence, k-mer weights should always be nodes['n_tar'] (how many target assemblies have this k-mer)
            frac_tar = nodes['n_tar'] / n_tar
            e_absence_tar = 1 - np.sum(frac_tar * nodes['n_tar']) / np.sum(nodes['n_tar'])
            frac_neg = nodes['n_neg'] / n_neg
            e_presence_neg = np.sum(frac_neg * nodes['n_tar']) / np.sum(nodes['n_tar'])
            jaccard = None

        logger.info(f' - expected k-mer absence in targets: {e_absence_tar:.5f}')
        logger.info(f' - expected k-mer presence in non-targets: {e_presence_neg:.5f}')

        penalty_th_mul = 1 - stringency / 10
        penalty_th = penalty_th_mul * (e_absence_tar * e_presence_neg)**0.5 # geometric mean
        logger.info(f' - calculated penalty threshold: {penalty_th:.5f}')

        if penalty_th > penalty_th_cap:
            penalty_th = penalty_th_cap
            logger.warning(f' - calculated penalty threshold is too large (capped at {penalty_th})')

        print_time_delta(time()-tik)
    else:
        logger.warning(f'Penalty threshold is provided (--penalty-th), skip auto estimation')
        jaccard = None

    # 2. calculate edge weight threshold
    # consider N as the number of assemblies that include a certain k-mer
    # since we want k-mers with penalty lower than penalty_th
    # based on the definition of penalty, N ≥ (1 - penalty_th) * n_tar
    # so edge weight threshold is calculated based on the lower bound of N, times a multiplier
    edge_weight_th = edge_w_th_mul * (1 - penalty_th) * n_tar

    # 3. calculate size range of subgraphs
    gap_len = (windowsize + 1) // 2 # average length of gap between minimizers
    min_nodes = max(min_nodes_floor, min_len // gap_len + 1)
    if max_len is None:
        max_nodes = max_nodes_cap
    else:
        max_nodes = max_len // gap_len + 1

    graph = _get_filtered_graph(
        graph=graph,
        penalty_th=penalty_th,
        edge_weight_th=edge_weight_th,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        rng=rng
    )

    state.penalty_th = penalty_th
    state.edge_weight_th = edge_weight_th
    state.min_nodes = min_nodes
    state.max_nodes = max_nodes
    return graph, jaccard
