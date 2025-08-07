"""
K-mer Graph
===========

A core module of Seqwin. Create an instance for k-mers of all input genome assemblies, 
including the weighted k-mer graph and low-penalty subgraphs. 

Dependencies:
-------------
- numpy
- numba
- networkx
- pandas (optional)
- scipy (optional)
- .assemblies
- .minimizer
- .graph
- .utils
- .config

Classes:
--------
- KmerGraph

Functions:
----------
- get_kmers
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import logging
from random import Random
from itertools import repeat, chain
from time import time
from heapq import heappush, heappop
from multiprocessing.shared_memory import SharedMemory

logger = logging.getLogger(__name__)

import numpy as np
import numba as nb
import networkx as nx
try:
    # try import dependencies for distance calculation
    import pandas as pd
    from scipy.sparse import coo_matrix
    _HAS_DIST_DEPS = True
except ImportError:
    _HAS_DIST_DEPS = False

from .assemblies import Assemblies
from .minimizer import indexlr_py, KMER_DTYPE
from .utils import StartMethod, SharedArr, print_time_delta, log_and_raise, mp_wrapper, \
    get_chunks, concat_to_shm, concat_from_shm
from .config import Config, RunState, WORKINGDIR, EDGE_W, NODE_P

_CLUSTER_DTYPE = np.dtype([ # dtype for each k-mer cluster
    ('hash', KMER_DTYPE['hash']), 
    ('n_tar', KMER_DTYPE['assembly_idx']), 
    ('n_neg', KMER_DTYPE['assembly_idx'])
])


class KmerGraph(object):
    """
    1. Create a weighted k-mer graph, and calculate node penalty scores. 
    2. (Optional) Calculate `Mash distances<https://mash.readthedocs.io/en/latest/distances.html>`__ for each assembly pair. 
    3. Extract low-penalty subgraphs from the k-mer graph with `self.filter()`. 
    
    Attributes:
        kmers (np.ndarray): A 1-D Numpy array of k-mers from all assemblies, with dtype `KMER_DTYPE` in `minimizer.py`. 
        graph (nx.Graph): A weighted, undirected graph of k-mers. 
            Edge weight is the number of assemblies where the two k-mers are adjacent. 
        clusters (np.ndarray): A 1-D Numpy array of k-mer clusters, with fields ['hash', 'n_tar', 'n_neg']. 
        cnt_mtx (np.ndarray | None): A matrix of the number of shared k-mers between each assembly pair. 
            Calculated when`get_dist=True`. 
        dist_mtx (np.ndarray | None): A matrix of the Mash distance between each assembly pair. 
            Calculated when `get_dist=True`. 
        subgraphs (tuple[tuple[frozenset[int], ...] | None): Low-penalty subgraphs. Each subgraph is a set of k-mer hash values. 
            Generated with `self.filter()`. 
        idx (np.ndarray | None): The original k-mer indices (k-mers with consecutive indices are adjacent in the genome). 
            Generated with `self.filter()`. 
        _filtered_flag (bool): True if `self.filter()` is called. 
    """
    __slots__ = (
        'kmers', 'graph', 'clusters', 'cnt_mtx', 'dist_mtx', 'subgraphs', 'idx', '_filtered_flag'
    )
    kmers: np.ndarray
    graph: nx.Graph
    clusters: np.ndarray
    cnt_mtx: np.ndarray | None
    dist_mtx: np.ndarray | None
    subgraphs: tuple[frozenset[int], ...] | None
    idx: np.ndarray | None
    _filtered_flag: bool

    def __init__(self, assemblies: Assemblies, kmerlen: int, windowsize: int, get_dist: bool, n_cpu: int) -> None:
        """
        1. Create a weighted, undirected graph of k-mers. 
        2. Calculate node penalty scores. 
        3. Calculate assembly distances if `get_dist=True`. 

        Args:
            assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
            kmerlen (int): See `Config` in `config.py`. 
            windowsize (int): See `Config` in `config.py`. 
            get_dist (bool): See `Config` in `config.py`. 
            n_cpu (int): See `Config` in `config.py`. 
        """
        # merge k-mers from all assemblies and create the graph
        kmers, graph = KmerGraph.__get_graph(assemblies, kmerlen, windowsize, n_cpu)

        # calculate penalty scores by clustering k-mers
        clusters, penalty, cnt_mtx = KmerGraph.__get_penalty(kmers, assemblies, get_dist)

        # sanity check
        if set(graph.nodes) != set(clusters['hash']):
            log_and_raise(ValueError, 'Inconsistent graph nodes and k-mer clusters.')

        # add penalties to networkx graph nodes
        nx.set_node_attributes(
            graph, 
            values=dict(zip(clusters['hash'], penalty)), 
            name=NODE_P
        )

        # calculate assembly distances
        if cnt_mtx is not None:
            dist_mtx = KmerGraph.__get_dist(cnt_mtx, kmerlen)
        else:
            dist_mtx = None

        self.kmers = kmers
        self.graph = graph
        self.clusters = clusters
        self.cnt_mtx = cnt_mtx
        self.dist_mtx = dist_mtx
        self.subgraphs = None
        self.idx = None
        self._filtered_flag = False

    @staticmethod
    def __get_graph(assemblies: Assemblies, kmerlen: int, windowsize: int, n_cpu: int) -> tuple[np.ndarray, nx.Graph]:
        """
        1. Merge k-mers from all assemblies into a single DataFrame. 
        2. Create a weighted, undirected graph of k-mers; edge weight is the number of assemblies where the two k-mers are adjacent. 
        3. Add record IDs as a column in assemblies. 

        Args:
            assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
            kmerlen (bool): See `Config` in `config.py`. 
            windowsize (bool): See `Config` in `config.py`. 
            get_dist (bool): See `Config` in `config.py`. 
            n_cpu (int): See `Config` in `config.py`. 
        
        Returns:
            tuple: A tuple containing
                1. np.ndarray: See `KmerGraph.kmers`. 
                2. nx.Graph: See `KmerGraph.graph`. 
        """
        n_assemblies = len(assemblies)
        logger.info(f'Creating a weighted, undirected minimizer graph from {n_assemblies} assemblies...')
        tik = time()

        # merge k-mers from all assembies
        if n_cpu <= 1:
            # create the graph with a single thread
            kmers, edges, isolates, record_ids = _get_graph(assemblies, kmerlen, windowsize, return_shm=False)
        else:
            # to make mp work, the method/function must be static and should not start with double underscores (single is fine)
            # difference between single & double underscores: https://docs.python.org/3/tutorial/classes.html#private-variables
            logger.info(f' - Parallelizing across {n_cpu} processes (~{n_assemblies//n_cpu} assemblies per process)...')
            graph_args = zip(
                get_chunks(assemblies, n_cpu), 
                repeat(kmerlen, n_cpu), 
                repeat(windowsize, n_cpu)
            )
            kmers, edges, isolates, record_ids = mp_wrapper(
                _get_graph, graph_args, n_cpu, unpack_output=True, 
                start_method=StartMethod.forkserver # must use spawn/forkserver for the indexlr python wrapper
            )
            # merge outputs from multiple processes
            logger.info(' - Merging from all processes...')
            kmers = concat_from_shm(kmers)
            edges = concat_from_shm(edges)
            edges = _merge_weighted_edges(edges)
            isolates = set.union(*isolates)
            record_ids = list(chain.from_iterable(record_ids))

        # convert to a networkx graph
        graph = nx.Graph()
        graph.add_weighted_edges_from(edges, weight=EDGE_W)
        # add isolated nodes to graph, so that the number of nodes is the same as the number of k-mer clusters
        graph.add_nodes_from(isolates)

        logger.info(f' - {len(graph)} nodes and {len(graph.edges)} edges from {len(kmers)} k-mers')
        logger.info(f' - {len(isolates)} nodes are isolates')
        print_time_delta(time()-tik)

        assemblies.record_ids = record_ids
        return kmers, graph

    @staticmethod
    def __get_penalty(
        kmers: np.ndarray, assemblies: Assemblies, get_dist: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Calculate node penalties by clustering k-mers with the same hash value. 
        Ideally, this can be done during graph creation. But in python, it is faster to do this with pandas groupby. 
        Assembly distance can also be calculated in this step (`get_dist=True`), using a slower clustering function `_cluster_kmers_dist()`. 

        Args:
            kmers (np.ndarray): See `KmerGraph.kmers`. 
            assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
            get_dist (bool): See `Config` in `config.py`. 
        
        Returns:
            tuple: A tuple containing
                1. np.ndarray: See `KmerGraph.clusters`. 
                2. np.ndarray: Penalty of each k-mer cluster. 
                3. np.ndarray | None: See `KmerGraph.cnt_mtx`. 
        """
        logger.info(f'Calculating penalty score for each k-mer node...')
        tik = time()

        n_assemblies = len(assemblies)
        n_tar = sum(assemblies.is_target)
        n_neg = n_assemblies - n_tar

        if get_dist and (not _HAS_DIST_DEPS):
            logger.error(' - Pandas and SciPy are not installed, skip assembly distance calculation')
            get_dist = False

        # choose which clustering function to use
        if get_dist:
            logger.warning(' - Assembly distance calculation is turned on, extra time and memory needed')
            clusters, cnt_mtx = KmerGraph.__cluster_dist(kmers, n_assemblies)
        else:
            clusters = KmerGraph.__cluster(kmers)
            cnt_mtx = None

        # calculate penalty for each cluster
        penalty = _penalty_func(
            clusters['n_tar'] / n_tar, 
            clusters['n_neg'] / n_neg
        )

        print_time_delta(time()-tik)
        return clusters, penalty, cnt_mtx

    @staticmethod
    def __cluster_dist(kmers: np.ndarray, n_assemblies: int) -> tuple[np.ndarray, np.ndarray]:
        """
        1. Cluster k-mers by their hash values. 
        2. Count the number of target/non-target assemblies each cluster. 
        3. Count the number of shared k-mers between all assembly pairs. 

        Args:
            kmers (np.ndarray): See `KmerGraph.kmers`. 
            n_assemblies (int): Number of assemblies. 

        Returns:
            tuple: A tuple containing
                1. np.ndarray: See `KmerGraph.clusters`. 
                2. np.ndarray: See `KmerGraph.cnt_mtx`. 
        """
        logger.info(' - Clustering k-mers across all assemblies...')
        # drop duplicate hash values for each assembly
        clusters = pd.DataFrame(kmers, copy=False)[
            ['hash', 'assembly_idx', 'is_target']
        ].drop_duplicates(keep='first', inplace=False)

        # cluster by hash and count the number of target / total assemblies in each cluster
        clusters = clusters.groupby(
            by='hash', as_index=False, sort=False
        ).agg(
            assembly_idx=pd.NamedAgg(column='assembly_idx', aggfunc=frozenset), 
            n_tar=pd.NamedAgg(column='is_target', aggfunc='sum'), 
            total=pd.NamedAgg(column='is_target', aggfunc='size'), 
        )
        assembly_idx = clusters.pop('assembly_idx')

        # formatting
        clusters['n_neg'] = clusters.pop('total') - clusters['n_tar']
        # convert to structured array
        clusters = clusters.to_records(
            index=False, 
            # column_dtypes must be a dict
            column_dtypes={name: _CLUSTER_DTYPE[name] for name in _CLUSTER_DTYPE.names}
        )
        clusters = clusters.view(np.ndarray)

        logger.info(' - Counting shared k-mers between all assembly pairs...')
        # calculate cnt_mtx
        # 1. build a membership matrix (use a sparse matrix to save memory)
        # each row represents a k-mer cluster, each column represents an assembly
        # if the i'th k-mer exists in the j'th assembly, then M[i, j] = 1; else M[i, j] = 0
        kmer_idx = list(chain.from_iterable( # row indexes
            repeat(i, len(s)) for i, s in enumerate(assembly_idx)
        ))
        assembly_idx = list(chain.from_iterable(assembly_idx)) # col indexes
        M = coo_matrix( # the membership matrix
            ( # construct from three arrays: data entries, row indexes, col indexes
                np.ones(len(kmer_idx)), (kmer_idx, assembly_idx)
            ), 
            shape=( # N_kmers, N_assemblies
                len(clusters), n_assemblies
            )
        ).tocsr() # convert to sparse

        # 2. calculate the number of shared k-mers by matrix multiplication
        cnt_mtx: coo_matrix = M.T @ M

        return clusters, cnt_mtx.toarray()

    @staticmethod
    def __cluster(kmers: np.ndarray) -> np.ndarray:
        """
        1. Cluster (sort) k-mers by their hash values. 
        2. Count the number of target/non-target assemblies each cluster. 

        Args:
            kmers (np.ndarray): See `KmerGraph.kmers`. 

        Returns:
            np.ndarray: See `KmerGraph.clusters`. 
        """
        logger.info(' - Sorting k-mers by hash values...')
        # sort to place k-mers with the same hash together
        order = np.lexsort((
            kmers['assembly_idx'], 
            kmers['is_target'], 
            kmers['hash'] # highest-priority key
        ))

        logger.info(' - Aggregating k-mer clusters...')
        # aggregation with numba
        return _agg_sorted(
            kmers['hash'][order], 
            kmers['is_target'][order], 
            kmers['assembly_idx'][order]
        )

    @staticmethod
    def __get_dist(cnt_mtx: np.ndarray, kmerlen: int) -> np.ndarray:
        """Calculate distances between each assembly pair based on the number of shared k-mers, 
        using the distance formula used by `Mash<https://mash.readthedocs.io/en/latest/distances.html>`__. 

        Args:
            cnt_mtx (np.ndarray): See `KmerGraph.cnt_mtx`. 
            kmerlen (int): See `Config` in `config.py`. 
        
        Returns:
            np.ndarray: See `KmerGraph.dist_mtx`. 
        """
        logger.info(f'Calculating distances between each assembly pair...')
        tik = time()

        # get the number of unique k-mers in each assembly
        nunique_kmer = np.diagonal(cnt_mtx)

        # calculate jaccard and distance
        dist_mtx = np.ones(cnt_mtx.shape)
        for i, n1 in enumerate(nunique_kmer):
            for j, n2 in enumerate(nunique_kmer[i+1:], start=i+1):
                c = cnt_mtx[i, j] # number of shared k-mers
                jaccard = c / (n1 + n2 - c)
                if jaccard > 0:
                    dist_mtx[i, j] = (-1/kmerlen) * np.log(2*jaccard/(1+jaccard))

        dist_mtx = dist_mtx - np.tril(dist_mtx) # remove all the ones in the lower triangle
        dist_mtx = dist_mtx + dist_mtx.T # make it symmetric

        print_time_delta(time()-tik)
        return dist_mtx

    def filter(
        self, penalty_th: float, edge_weight_th: float, min_nodes: int, max_nodes: int | None, rng: Random
    ) -> None:
        """Filter graph edges and nodes. 
        1. Remove low-weight edges and isolated nodes. 
        2. Extract low-penalty subgraphs. 
        3. Remove k-mers not included in any of the subgraphs. 

        Args:
            penalty_th (float): See `RunState` in `config.py`. 
            edge_weight_th (float): See `RunState` in `config.py`. 
            min_nodes (int): See `RunState` in `config.py`. 
            max_nodes (int | None): See `RunState` in `config.py`. 
            rng (random.Random): See `RunState` in `config.py`. 
        """
        kmers = self.kmers
        graph = self.graph
        filtered_flag = self._filtered_flag

        if filtered_flag:
            logger.error(f'K-mers are already filtered, cannot filter again.')
            return None

        logger.info('Extracting low-penalty subgraphs from the k-mer graph...')
        tik = time()

        if max_nodes is None:
            logger.warning(f' - Upper limit of subgraph size is not set. Lower limit is set to {min_nodes}')
        else:
            logger.info(f' - Subgraph size limit is set to [{min_nodes}, {max_nodes}]')

        # remove low-weight edges and isolated nodes
        graph = KmerGraph.__filter_graph(graph, edge_weight_th)

        # get low-penalty subgraphs
        subgraphs, used = KmerGraph.__get_subgraphs(graph, penalty_th, min_nodes, max_nodes, rng)

        # remove unused k-mers
        kmers, idx = KmerGraph.__filter_kmers(kmers, used)

        print_time_delta(time()-tik)
        self.kmers = kmers
        self.graph = graph
        self.subgraphs = subgraphs
        self.idx = idx
        self._filtered_flag = True

    @staticmethod
    def __filter_graph(graph: nx.Graph, edge_weight_th: float) -> nx.Graph:
        """Remove low-weight edges and isolated nodes from the graph. 
        
        Args:
            graph (nx.Graph): See `KmerGraph.graph`. 
            edge_weight_th (float): See `RunState` in `config.py`. 
        
        Returns:
            nx.Graph: The filtered graph. 
        """
        logger.info(' - Filtering graph edges and nodes...')

        # remove low-weight edges
        low_weight_edges = list((u, v) for u, v, w in graph.edges(data=EDGE_W) if w < edge_weight_th)
        graph.remove_edges_from(low_weight_edges)
        logger.info(f' - Removed {len(low_weight_edges)} edges with weight<{edge_weight_th:.3f}, {len(graph.edges)} edges left')

        # remove isolated nodes
        isolates = set(nx.isolates(graph)) # NOTE: might still be useful
        graph.remove_nodes_from(isolates)
        logger.info(f' - Removed {len(isolates)} isolated nodes from graph, {len(graph)} nodes left')

        return graph

    @staticmethod
    def __get_subgraphs(
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
                1. tuple[tuple[frozenset[int], ...]: See `KmerGraph.subgraphs`. 
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
        logger.info(f' - Found {len(subgraphs)} low-penalty subgraphs')

        # due to the greedy nature of node expansion, subgraphs created first are usually larger
        # by shuffling the subgraphs, we can get a more balanced distribution of sizes in downstream multiprocessing
        rng.shuffle(subgraphs)

        return tuple(frozenset(sg) for sg in subgraphs), frozenset(used)

    @staticmethod
    def __filter_kmers(kmers: np.ndarray, used: frozenset[int]) -> tuple[np.ndarray, np.ndarray]:
        """Remove k-mers not included in `used`. 

        Args:
            kmers (pd.DataFrame): See `KmerGraph.kmers`. 
            used (frozenset[int]): Output of `KmerGraph.__get_subgraphs()`. 
        
        Returns:
            tuple: A tuple containing
                1. np.ndarray: See `KmerGraph.kmers`. 
                2. np.ndarray: See `KmerGraph.idx`. 
        """
        logger.info(' - Removing k-mers not included in any of the subgraphs...')

        n_kmers = len(kmers) # save total before filtering
        used = np.fromiter(used, dtype=KMER_DTYPE['hash'], count=len(used))

        mask = np.isin(kmers['hash'], used)
        kmers = kmers[mask]
        # keep the original k-mer indices, which holds k-mer adjacency info
        idx = np.arange(n_kmers, dtype=KMER_DTYPE['hash'])[mask]

        logger.info(f' - {len(kmers)} k-mers left')
        return kmers, idx

    def filter_strict(self, penalty_th: float, edge_weight_th: float) -> None:
        """Filter graph edges and nodes. 
        1. Remove nodes with penalty > `penalty_th`. 
        2. Remove low-weight edges and isolated nodes. 
        3. Extract connected components (subgraphs). 
        4. Remove k-mers not included in any of the subgraphs. 

        Args:
            penalty_th (float): See `Config` in `config.py`. 
            edge_weight_th (float): See `RunState` in `config.py`. 
        """
        kmers = self.kmers
        graph = self.graph
        filtered_flag = self._filtered_flag

        if filtered_flag:
            logger.error(f'K-mers are already filtered, cannot filter again.')
            return None

        logger.info('Extracting low-penalty subgraphs from the k-mer graph...')
        tik = time()

        # remove high penalty nodes
        # use greater than (>), otherwise everything wiil be bad when penalty_th = 0
        bad_nodes: set[int] = set(n for n, p in graph.nodes(data=NODE_P) if p > penalty_th)
        # remove from graph
        graph.remove_nodes_from(bad_nodes)
        logger.info(f' - Removed {len(bad_nodes)} nodes with penalty>{penalty_th:.5f}')

        # remove low-weight edges and isolated nodes
        graph = KmerGraph.__filter_graph(graph, edge_weight_th)

        # get connected components (subgraphs)
        logger.info(f' - Finding connected components...')
        subgraphs = tuple(frozenset(sg) for sg in nx.connected_components(graph))
        used = frozenset.union(*subgraphs)
        logger.info(f' - Found {len(subgraphs)} components (low-penalty subgraphs)')

        # remove unused k-mers
        kmers, idx = KmerGraph.__filter_kmers(kmers, used)

        print_time_delta(time()-tik)
        self.kmers = kmers
        self.graph = graph
        self.subgraphs = subgraphs
        self.idx = idx
        self._filtered_flag = True


def _stack_to_shm(edges: np.ndarray, weights: np.ndarray) -> SharedArr:
    """Merge edges and weights into a single 3-column array. 

    Args:
        edges (np.ndarray): A 2-column array of edges. 
        weights (np.ndarray): Edge weights with the same length of `edges`. 
    
    Returns:
        SharedArr: The merged array attached to a SharedMemory instance. 
    """
    dtype = edges.dtype
    weights = weights.astype(dtype, copy=False)

    # create shared memory
    n_edges = edges.shape[0]
    shm = SharedMemory(create=True, size=n_edges * 3 * dtype.itemsize)
    arr_view = np.ndarray((n_edges, 3), dtype=dtype, buffer=shm.buf)

    # copy data into shared memory
    arr_view[:, :2] = edges
    arr_view[:, 2] = weights

    edges = SharedArr(shm.name, arr_view.shape, dtype)
    shm.close()
    return edges


def _get_graph(
    assemblies: Assemblies, kmerlen: int, windowsize: int, return_shm: bool=True
) -> tuple[SharedArr | np.ndarray, SharedArr | np.ndarray, set[int], list[tuple[str]]]:
    """Merge k-mers of multiple assemblies into a single pandas DataFrame, and create a weighted, undirected graph. 
    - For multiprocessing, this function cannot be a method of `KmerGraph`. 
    - Numpy arrays are returned as shared memory blocks for multiprocessing. 

    Args:
        assemblies (Assemblies): See `Assemblies` in `assemblies.py` (could be a slice of the DataFrame). 
        kmerlen (int): See `Config` in `config.py`. 
        windowsize (int): See `Config` in `config.py`. 
        return_shm (bool, optional): If True, return the kmers and edges arrays using shared memory blocks; else return regular Numpy arrays. 
    
    Returns:
        tuple: A tuple containing
            1. SharedArr | np.ndarray: Merged k-mers. Return a `SharedArr` instance if `return_shm` is True; else return a Numpy array. 
            2. SharedArr | np.ndarray: The weighted edges in a 3-column Numpy array: the first two columns are edges and the third column is weights. 
                Return a `SharedArr` instance if `return_shm` is True; else return a Numpy array. 
            3. set[int]: Isolated k-mers. Some short sequence records might only have one k-mer, leaving isolated graph nodes. 
            4. list[tuple[str]]: Record IDs of each assembly. 
    """
    #---------- generate k-mers ----------#
    kmers: list[list[np.ndarray]] = list()
    record_ids: list[tuple[str]] = list() # record ids of each assembly

    n_edges = 0 # total number of edges (not unique)
    for idx, assembly in assemblies.iterrows():
        # get the minimizer sketch of the current assembly
        kmers_assembly, ids = indexlr_py(
            assembly.path, kmerlen, windowsize, idx, assembly.is_target
        )
        n_edges += sum(len(kmers_record) - 1 for kmers_record in kmers_assembly)
        kmers.append(kmers_assembly)
        record_ids.append(ids)
    #---------- generate k-mers ----------#

    #---------- generate graph edges ----------#
    # pre-allocate an array for edges from all assemblies (instead of concat later)
    # this might be larger than needed since only unique edges in each assembly are kept
    edges = np.empty((n_edges, 2), dtype=KMER_DTYPE['hash'])
    # isolated k-mers found in short sequence records
    isolates = list()

    start = 0 # position in edges
    for kmers_assembly in kmers:
        # get unique edges of the current assembly
        edges_assembly = list()
        for kmers_record in kmers_assembly:
            hashes = kmers_record['hash']

            if len(hashes) == 1:
                # isolate nodes cannot be added to graph since they don't have any edge
                # they might be included in another path of another record, but that doesn't matter
                isolates.append(hashes[0])

            elif len(hashes) > 1:
                # get edges of the current sequence record
                u, v = hashes[:-1], hashes[1:]
                edges_record = np.empty((hashes.size - 1, 2), dtype=hashes.dtype)
                # canonical order of edge nodes (undirected graph)
                np.minimum(u, v, out=edges_record[:, 0])
                np.maximum(u, v, out=edges_record[:, 1])
                edges_assembly.append(edges_record)

        edges_assembly = np.concatenate(edges_assembly, axis=0)
        edges_assembly = np.unique(edges_assembly, axis=0)

        stop = start + edges_assembly.shape[0]
        edges[start:stop] = edges_assembly
        start = stop
    edges.resize((start, 2), refcheck=False) # shrink in-place
    isolates = set(isolates)
    #---------- generate graph edges ----------#

    # calculate edge weights
    # somehow np.unique returns a view
    edges, weights = np.unique(edges, axis=0, return_counts=True)

    # concat arrays and return
    kmers = list(chain.from_iterable(kmers)) # flatten kmers before concat
    if return_shm:
        kmers = concat_to_shm(kmers)
        edges = _stack_to_shm(edges, weights)
    else:
        kmers = np.concatenate(kmers)
        # must convert to the same dtype, otherwise both converted to float64
        edges = np.column_stack((
            edges, weights.astype(edges.dtype, copy=False)
        ))

    return kmers, edges, isolates, record_ids


def _merge_weighted_edges(edges: np.ndarray):
    """Add the weights of the same edges. 

    Args:
        edges (np.ndarray): Concatenated outputs of `_get_graph()`. 
    
    Returns:
        np.ndarray: Unique edges with total weights. 
    """
    # idx maps edges to unique_edges
    unique_edges, idx = np.unique(edges[:, :2], axis=0, return_inverse=True)
    # calculate the weight of each unique edge; weights are sorted by idx
    weights = np.bincount(idx, weights=edges[:, 2])
    edges = np.column_stack((
        unique_edges, weights.astype(edges.dtype, copy=False)
    ))
    return edges


@nb.njit(nogil=True)
def _agg_sorted(hashes: np.ndarray, is_tar: np.ndarray, assembly_idx: np.ndarray) -> np.ndarray:
    """Numba function for `KmerGraph.__cluster()`. 
    """
    n = hashes.size
    # pre-allocate outputs (never larger than n)
    out = np.empty(n, dtype=_CLUSTER_DTYPE)

    out_i = 0
    i = 0
    while i < n:
        curr_hash = hashes[i]

        n_tar = n_neg = 0
        # walk all rows having the same hash
        while i < n and hashes[i] == curr_hash:
            curr_t = is_tar[i]
            curr_a = assembly_idx[i]

            # skip any further duplicates in the same assembly
            j = i + 1
            while (
                j < n and hashes[j] == curr_hash
                #and is_tar[j] == curr_t
                and assembly_idx[j] == curr_a
            ):
                j += 1

            # count the current assembly
            if curr_t:
                n_tar += 1
            else:
                n_neg += 1
            i = j # advance by the duplicates

        out[out_i]['hash'] = curr_hash
        out[out_i]['n_tar']  = n_tar
        out[out_i]['n_neg']  = n_neg
        out_i += 1

    # trim the over-allocated buffers
    return out[:out_i]


def _penalty_func(frac_tar: np.ndarray | float, frac_neg: np.ndarray | float) -> np.ndarray | float:
    """The penalty function. 
    """
    return ((1 - frac_tar)**2 + frac_neg**2)**0.5


def get_kmers(
    assemblies: Assemblies, config: Config, state: RunState
) -> tuple[KmerGraph, np.ndarray | None]:
    """Get clustered k-mers from all assemblies, create a k-mer graph and find low-penalty subgraphs. 

    Args:
        assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
        config (Config): See `Config` in `config.py`. 
        state (RunState): See `RunState` in `config.py`. 

    Returns:
        Returns:
        tuple: A tuple containing
            1. KmerGraph: The KmerGraph instance. 
            2. np.ndarray | None: A matrix of Jaccard indexes of all assembly pairs. 
    """
    overwrite = config.overwrite
    kmerlen = config.kmerlen
    windowsize = config.windowsize
    penalty_th = config.penalty_th
    min_len = config.min_len
    max_len = config.max_len
    no_filter = config.no_filter
    penalty_th_cap = config.penalty_th_cap
    edge_w_th_mul = config.edge_w_th_mul
    min_nodes_floor = config.min_nodes_floor
    max_nodes_cap = config.max_nodes_cap
    sketchsize = config.sketchsize
    get_dist = config.get_dist
    n_cpu = config.n_cpu

    working_dir = state.working_dir
    rng = state.rng
    n_tar = state.n_tar
    n_neg = state.n_neg

    kmers = KmerGraph(assemblies, kmerlen, windowsize, get_dist, n_cpu)

    if no_filter:
        # skip kmers.filter(), debug only
        return kmers, None

    # calculate filter params
    # 1. calculate penalty threshold
    if penalty_th is None:
        logger.info(f'Calculating penalty threshold...')
        tik = time()

        jaccard = assemblies.mash(
            kmerlen=kmerlen, 
            sketchsize=sketchsize, 
            out_path=working_dir / WORKINGDIR.mash, 
            overwrite=overwrite, 
            n_cpu=n_cpu
        )

        # average jaccard index of all target-target pairs
        j_tar = (np.sum(jaccard[:n_tar, :n_tar]) - n_tar) / (n_tar * (n_tar - 1))
        logger.info(f' - average Jaccard within targets: {j_tar:.5f}')
        # average jaccard index of all target-(non-target) pairs
        j_neg = np.sum(jaccard[n_tar:, :n_tar]) / (n_tar * n_neg)
        logger.info(f' - average Jaccard between targets and non-targets: {j_neg:.5f}')
        # choose the more stringent threshold
        penalty_th = min(1 - j_tar, j_neg)
        if penalty_th < penalty_th_cap:
            logger.info(f' - penalty threshold set as {penalty_th:.5f} (calculated with Jaccard)')
        else:
            penalty_th = penalty_th_cap
            logger.warning(f' - penalty threshold is capped at {penalty_th} (check Jaccard)')

        print_time_delta(time()-tik)
    else:
        logger.warning(f'Penalty threshold is provided (--penalty-th), skip running Mash')
        jaccard = None

    # 2. calculate edge weight threshold
    # consider N as the number of assemblies that include a certain k-mer
    # since we want k-mers with penalty lower than penalty_th
    # based on the definition of penalty, N ≥ (1 - penalty_th) * n_tar
    # so edge weight threshold is calculated based on the lower bound of N, times a multiplier
    edge_weight_th = edge_w_th_mul * (1 - penalty_th) * n_tar

    # 3. calculate size range of subgraphs
    gap_len = windowsize // 2 # average length of gap between minimizers
    min_nodes = max(min_nodes_floor, min_len // gap_len + 1)
    if max_len is None:
        max_nodes = max_nodes_cap
    else:
        max_nodes = max_len // gap_len + 1

    # extract low-penalty subgraphs and remove unused k-mers
    kmers.filter(penalty_th, edge_weight_th, min_nodes, max_nodes, rng)
    #kmers.filter_strict(penalty_th, edge_weight_th)

    state.penalty_th = penalty_th
    state.edge_weight_th = edge_weight_th
    state.min_nodes = min_nodes
    state.max_nodes = max_nodes
    return kmers, jaccard
