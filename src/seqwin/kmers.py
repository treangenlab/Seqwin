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
- .helpers
- .minimizer
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

logger = logging.getLogger(__name__)

import numpy as np
import networkx as nx
from numba import types, typed
try:
    # try import dependencies for distance calculation
    import pandas as pd
    from scipy.sparse import coo_matrix
    _HAS_DIST_DEPS = True
except ImportError:
    _HAS_DIST_DEPS = False

from .assemblies import Assemblies
from .helpers import get_edges, stack_to_shm, merge_weighted_edges, sort_by_hash, agg_by_hash, \
    get_subgraphs, filter_kmers, HASH_ARR_NB_DT, CLUSTER_DTYPE
from .minimizer import indexlr_py
from .utils import StartMethod, SharedArr, print_time_delta, log_and_raise, mp_wrapper, \
    get_chunks, concat_to_shm, concat_from_shm
from .config import Config, RunState, WORKINGDIR, EDGE_W, NODE_P


class KmerGraph(object):
    """
    1. Create a weighted k-mer graph, and calculate node penalty scores. 
    2. (Optional) Calculate `Mash distances<https://mash.readthedocs.io/en/latest/distances.html>`__ for each assembly pair. 
    3. Extract low-penalty subgraphs from the k-mer graph with `self.filter()`. 
    
    Attributes:
        kmers (np.ndarray): A 1-D Numpy array of k-mers from all assemblies, with dtype `KMER_DTYPE` defined in `minimizer.py`. 
        idx (np.ndarray | None): The original indices when k-mers are generated (k-mers with consecutive indices are adjacent in the genome). 
        graph (nx.Graph): A weighted, undirected graph of k-mers. 
            Edge weight is the number of assemblies where the two k-mers are adjacent. 
        clusters (np.ndarray): A 1-D Numpy array of k-mer clusters, with fields ['hash', 'n_tar', 'n_neg', 'penalty']. 
        cnt_mtx (np.ndarray | None): A matrix of the number of shared k-mers between each assembly pair. 
            Calculated when`get_dist=True`. 
        dist_mtx (np.ndarray | None): A matrix of the Mash distance between each assembly pair. 
            Calculated when `get_dist=True`. 
        subgraphs (tuple[tuple[frozenset[int], ...] | None): Low-penalty subgraphs. Each subgraph is a set of k-mer hash values. 
            Generated with `self.filter()`. 
    """
    __slots__ = (
        'kmers', 'idx', 'graph', 'clusters', 'cnt_mtx', 'dist_mtx', 'subgraphs', '_filtered_flag'
    )
    kmers: np.ndarray
    idx: np.ndarray
    graph: nx.Graph
    clusters: np.ndarray
    cnt_mtx: np.ndarray | None
    dist_mtx: np.ndarray | None
    subgraphs: tuple[frozenset[int], ...] | None
    _filtered_flag: bool # True if `self.filter()` is called. 

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
        # k-mers in the array have the same order as they appear in the genomes, 
        # and their indices are just their positions in the array (0, 1, 2, ...)
        kmers, graph = KmerGraph.__get_graph(assemblies, kmerlen, windowsize, n_cpu)

        # calculate penalty scores by clustering k-mers
        # k-mers are now sorted by hash values, idx is the original indices
        kmers, idx, clusters, cnt_mtx = KmerGraph.__get_penalty(kmers, assemblies, get_dist)

        # sanity check
        if set(graph.nodes) != set(clusters['hash']):
            log_and_raise(ValueError, 'Inconsistent graph nodes and k-mer clusters.')

        # add penalties to networkx graph nodes
        nx.set_node_attributes(
            graph, 
            values=dict(zip(clusters['hash'], clusters['penalty'])), 
            name=NODE_P
        )

        # calculate assembly distances
        if cnt_mtx is not None:
            dist_mtx = KmerGraph.__get_dist(cnt_mtx, kmerlen)
        else:
            dist_mtx = None

        self.kmers = kmers
        self.idx = idx
        self.graph = graph
        self.clusters = clusters
        self.cnt_mtx = cnt_mtx
        self.dist_mtx = dist_mtx
        self.subgraphs = None
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
            edges = merge_weighted_edges(edges)
            isolates = np.unique(np.concatenate(isolates))
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Calculate node penalty scores by 
        1. sorting k-mers by hash values, 
        2. aggregating k-mers with the same hash value to calculate penalty. 
        - Ideally, this can be done during graph creation. But in python, it is faster to do this in bulk after all k-mers are merged. 
        Plus, sorting k-mers here can also make removing unused k-mers faster in `KmerGraph.__filter_kmers()`. 
        - Assembly distance can also be calculated in this step (`get_dist=True`), using a slower clustering function `KmerGraph.__cluster_dist()`. 

        Args:
            kmers (np.ndarray): See `KmerGraph.kmers`. 
            assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
            get_dist (bool): See `Config` in `config.py`. 
        
        Returns:
            tuple: A tuple containing
                1. np.ndarray: See `KmerGraph.kmers`. 
                2. np.ndarray: See `KmerGraph.idx`. 
                3. np.ndarray: See `KmerGraph.clusters`. 
                4. np.ndarray | None: See `KmerGraph.cnt_mtx`. 
        """
        logger.info(f'Calculating penalty score for each k-mer node...')
        tik = time()

        n_assemblies = len(assemblies)
        n_tar = sum(assemblies.is_target)
        n_neg = n_assemblies - n_tar

        logger.info(' - Sorting k-mers by hash values...')
        # stable sort in-place (preserve assembly order), and return the sorted indices
        # here we don't sort with the built-in numpy functions, because argsort (or lexsort) 
        # creates a C contiguous copy of the strided kmers['hash'], and it uses int64 for indices; 
        # reordering a struct array with kmers[:] = kmer[idx] also needs a large buffer; 
        # all these would increase peak RAM by a lot. 
        # to solve this, we could 
        # 1) structure kmers as a tuple of contiguous arrays (instead of a structured array), and use numpy sorting; 
        # 2) write a custom sorting function that works on a strided array. 
        # option 1) involves rewriting a lot of functions, and numpy still uses in64 for indexing
        # option 2) is easier to implement and equally fast
        idx = sort_by_hash(kmers)

        if get_dist and (not _HAS_DIST_DEPS):
            logger.error(' - Pandas and SciPy are not installed, skip assembly distance calculation')
            get_dist = False

        # choose which clustering function to use
        if get_dist:
            logger.warning(' - Assembly distance calculation is turned on, extra time and memory needed')
            clusters, cnt_mtx = KmerGraph.__cluster_dist(kmers, n_assemblies)
        else:
            logger.info(' - Aggregating k-mer clusters...')
            # better pass individual fields to a numba function, since hashes = kmers['hash'] is not supported
            clusters = agg_by_hash(
                kmers['hash'], 
                kmers['assembly_idx'], 
                kmers['is_target']
            )
            cnt_mtx = None

        # calculate penalty for each cluster
        clusters['penalty'] = _frac_to_penalty(
            clusters['n_tar'] / n_tar, 
            clusters['n_neg'] / n_neg
        )

        print_time_delta(time()-tik)
        return kmers, idx, clusters, cnt_mtx

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
        clusters['penalty'] = .0 # placeholder for penalty
        # convert to structured array
        clusters = clusters.to_records(
            index=False, 
            # column_dtypes must be a dict
            column_dtypes={name: CLUSTER_DTYPE[name] for name in CLUSTER_DTYPE.names}
        )
        clusters = clusters.view(np.ndarray)

        logger.info(' - Counting shared k-mers between all assembly pairs...')
        # calculate cnt_mtx
        # 1. build a membership matrix (use a sparse matrix to save memory)
        # each row represents a k-mer cluster, each column represents an assembly
        # if the i'th k-mer exists in the j'th assembly, then M[i, j] = 1; else M[i, j] = 0
        kmer_idx = list(chain.from_iterable( # row indices
            repeat(i, len(s)) for i, s in enumerate(assembly_idx)
        ))
        assembly_idx = list(chain.from_iterable(assembly_idx)) # col indices
        M = coo_matrix( # the membership matrix
            ( # construct from three arrays: data entries, row indices, col indices
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
        idx = self.idx
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
        subgraphs, used = get_subgraphs(graph, penalty_th, min_nodes, max_nodes, rng)

        # remove unused k-mers
        kmers, idx = filter_kmers(kmers, idx, used)

        print_time_delta(time()-tik)
        self.kmers = kmers
        self.idx = idx
        self.graph = graph
        self.subgraphs = subgraphs
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
        idx = self.idx
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
        kmers, idx = KmerGraph.__filter_kmers(kmers, idx, used)

        print_time_delta(time()-tik)
        self.kmers = kmers
        self.idx = idx
        self.graph = graph
        self.subgraphs = subgraphs
        self._filtered_flag = True


def _get_graph(
    assemblies: Assemblies, kmerlen: int, windowsize: int, return_shm: bool=True
) -> tuple[SharedArr | np.ndarray, SharedArr | np.ndarray, np.ndarray, list[tuple[str]]]:
    """
    1. Merge k-mers from multiple assemblies into a 1-D structured Numpy array. 
    2. Generate weighted, undirected edges for the k-mer graph. 
    
    Numpy arrays are returned as shared memory blocks for multiprocessing. 

    Args:
        assemblies (Assemblies): See `Assemblies` in `assemblies.py` (could be a slice of the DataFrame). 
        kmerlen (int): See `Config` in `config.py`. 
        windowsize (int): See `Config` in `config.py`. 
        return_shm (bool, optional): If True, return the kmers and edges arrays using shared memory blocks; else return regular Numpy arrays. 
    
    Returns:
        tuple: A tuple containing
            1. SharedArr | np.ndarray: Merged k-mers. Return a `SharedArr` instance if `return_shm` is True; else return a Numpy array. 
                dtype is defined in `minimizer.py` (`KMER_DTYPE`). 
            2. SharedArr | np.ndarray: The weighted edges in a 3-column Numpy array: the first two columns are edges and the third column is weights. 
                Return a `SharedArr` instance if `return_shm` is True; else return a Numpy array. 
            3. np.ndarray: Isolated k-mer nodes. Some short sequence records might only have one k-mer, leaving isolated graph nodes. 
            4. list[tuple[str]]: Record IDs of each assembly. 
    """
    #---------- generate k-mers ----------#
    kmers: list[list[np.ndarray]] = list()
    record_ids: list[tuple[str]] = list() # record ids of each assembly

    for idx, assembly in assemblies.iterrows():
        # get the minimizer sketch of the current assembly
        kmers_assembly, ids = indexlr_py(
            assembly.path, kmerlen, windowsize, idx, assembly.is_target
        )
        kmers.append(kmers_assembly)
        record_ids.append(ids)
    #---------- generate k-mers ----------#

    #---------- generate graph edges ----------#
    # define input types of the numba function
    # this is a List[List[Array]]
    hashes = typed.List.empty_list(types.ListType(HASH_ARR_NB_DT))

    for kmers_assembly in kmers:
        # this is a List[Array]
        hashes_assembly = typed.List.empty_list(HASH_ARR_NB_DT)

        for kmers_record in kmers_assembly:
            # this does not create a copy, kmers_record['hash'] returns a strided view
            # must be specified in the numba dtype ('A')
            hashes_assembly.append(kmers_record['hash'])

        hashes.append(hashes_assembly)

    # get weighted edges and isolated nodes
    edges, weights, isolates = get_edges(hashes)
    #---------- generate graph edges ----------#

    # concat arrays and return
    kmers = list(chain.from_iterable(kmers)) # flatten kmers before concat
    if return_shm:
        kmers = concat_to_shm(kmers)
        edges = stack_to_shm(edges, weights)
    else:
        kmers = np.concatenate(kmers)
        # must convert to the same dtype, otherwise both converted to float64
        edges = np.column_stack((
            edges, weights.astype(edges.dtype, copy=False)
        ))

    return kmers, edges, isolates, record_ids


def _expected_frac(jaccard_mtx: np.ndarray):
    """Calculate the expected fraction from a matrix of pairwise Jaccard indices. 
    Here fraction means: for a k-mer `h` in a group of genomes (k-mer sets), the fraction of sets in 
    a second group that also contain `h`. `jaccard_mtx` is the pairwise Jaccard indices between sets 
    in the two groups. Note that this could be a self comparison (two groups are the same). 
    - `E(frac) = mean(2J / (1+J))`, where `J` is the Jaccard matrix. 
    - This expectation ≥ mean of the Jaccard matrix, (`2J / (1+J) ≥ J, 0 ≤ J ≤ 1`). 
    """
    return np.mean(2 * jaccard_mtx / (1 + jaccard_mtx))


def _frac_to_penalty(frac_tar: np.ndarray | float, frac_neg: np.ndarray | float) -> np.ndarray | float:
    """The penalty formula (Euclidean / L2 norm of the two fractions). 
    - When `frac_tar` and `frac_neg` are expectations, this formula doesn't give the expected penalty, since it's non-linear. 
    - However, since it is convex, the returned value ≤ the true expectation (Jensen’s inequality). 
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
        tuple: A tuple containing
            1. KmerGraph: The KmerGraph instance. 
            2. np.ndarray | None: A matrix of Jaccard indices of all assembly pairs. 
    """
    overwrite = config.overwrite
    kmerlen = config.kmerlen
    windowsize = config.windowsize
    penalty_th = config.penalty_th
    stringency = config.stringency
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

        e_absence_tar = 1 - _expected_frac(jaccard[:n_tar, :n_tar])
        logger.info(f' - expected k-mer absence in targets: {e_absence_tar:.5f}')

        e_presence_neg = _expected_frac(jaccard[n_tar:, :n_tar])
        logger.info(f' - expected k-mer presence in non-targets: {e_presence_neg:.5f}')

        stringent_e = min(e_absence_tar, e_presence_neg)
        penalty_th_mul = 1 - stringency / 10
        penalty_th = penalty_th_mul * stringent_e
        logger.info(f' - calculated penalty threshold: {penalty_th:.5f} ({penalty_th_mul} * {stringent_e:.5f})')

        if penalty_th > penalty_th_cap:
            penalty_th = penalty_th_cap
            logger.warning(f' - calculated penalty threshold is too large (capped at {penalty_th})')

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
