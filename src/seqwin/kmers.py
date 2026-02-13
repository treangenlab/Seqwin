"""
K-mer Graph
===========

A core module of Seqwin. Create an k-mer graph from k-mers of all input genome assemblies. 

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
from numpy.typing import NDArray
from numba import types, typed, set_num_threads
try:
    # try import dependencies for distance calculation
    import pandas as pd
    from scipy.sparse import coo_matrix
    _HAS_DIST_DEPS = True
except ImportError:
    _HAS_DIST_DEPS = False

from .assemblies import Assemblies
from .helpers import get_edges, merge_weighted_edges, sort_by_hash, agg_by_hash, \
    get_subgraphs, filter_kmers, HASH_ARR_NB_DT, NODE_DTYPE
from .minimizer import indexlr_py
from .utils import StartMethod, SharedArr, print_time_delta, log_and_raise, mp_wrapper, \
    get_chunks, concat_to_shm, concat_from_shm
from .config import Config, RunState, WORKINGDIR, EDGE_W, NODE_P


class KmerGraph(object):
    """
    1. Create a weighted, undirected k-mer graph, and calculate node penalty scores. 
    2. (Optional) Calculate `Mash distances<https://mash.readthedocs.io/en/latest/distances.html>`__ for each assembly pair. 
    3. Extract low-penalty subgraphs from the k-mer graph with `self.filter()`. 
    
    Attributes:
        kmers (NDArray): A 1-D Numpy array of k-mers from all assemblies, with dtype `KMER_DTYPE` defined in `minimizer.py`. 
        idx (NDArray | None): The original indices when k-mers are generated (k-mers with consecutive indices are adjacent in the genome). 
        nodes (NDArray): A 1-D Numpy structured array of k-mer nodes, with fields ['hash', 'n_tar', 'n_neg', 'penalty']. 
        edges (NDArray): A 3-column Numpy array of weighted edges (u, v, w). 
            Edge weight is the number of assemblies where the two k-mers are adjacent. 
        graph (nx.Graph): The graph instance built from filtered nodes and edges. 
        cnt_mtx (NDArray | None): A matrix of the number of shared k-mers between each assembly pair. 
            Calculated when`get_dist=True`. 
        dist_mtx (NDArray | None): A matrix of the Mash distance between each assembly pair. 
            Calculated when `get_dist=True`. 
        subgraphs (tuple[frozenset[int], ...] | None): Low-penalty subgraphs. Each subgraph is a set of k-mer hash values. 
            Generated with `self.filter()`. 
    """
    __slots__ = (
        'kmers', 'idx', 'nodes', 'edges', 'graph', 'cnt_mtx', 'dist_mtx', 'subgraphs', '_filtered_flag'
    )
    kmers: NDArray
    idx: NDArray
    nodes: NDArray
    edges: NDArray
    graph: nx.Graph
    cnt_mtx: NDArray | None
    dist_mtx: NDArray | None
    subgraphs: tuple[frozenset[int], ...] | None
    _filtered_flag: bool # True if `self.filter()` is called

    def __init__(self, assemblies: Assemblies, kmerlen: int, windowsize: int, get_dist: bool, n_cpu: int) -> None:
        """
        1. Generate weighted edges and k-mer nodes. 
        2. Calculate node penalty scores. 
        3. Calculate assembly distances if `get_dist=True`. 

        Args:
            assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
            kmerlen (int): See `Config` in `config.py`. 
            windowsize (int): See `Config` in `config.py`. 
            get_dist (bool): See `Config` in `config.py`. 
            n_cpu (int): See `Config` in `config.py`. 
        """
        # collect k-mers from all assemblies and generate weighted edges
        # k-mers in the array have the same order as they appear in the genomes, 
        # and their indices are just their positions in the array (0, 1, 2, ...)
        kmers, edges = KmerGraph.__get_edges(assemblies, kmerlen, windowsize, n_cpu)

        # generate k-mer nodes and calculate penalty scores by grouping kmers
        # kmers is sorted in-place by hash values, idx is the sorted original indices
        idx, nodes, cnt_mtx = KmerGraph.__get_nodes(kmers, assemblies, get_dist)

        # calculate assembly distances
        if cnt_mtx is not None:
            dist_mtx = KmerGraph.__get_dist(cnt_mtx, kmerlen)
        else:
            dist_mtx = None

        self.kmers = kmers
        self.idx = idx
        self.nodes = nodes
        self.edges = edges
        self.graph = None # create graph after filtering nodes and edges
        self.cnt_mtx = cnt_mtx
        self.dist_mtx = dist_mtx
        self.subgraphs = None
        self._filtered_flag = False

    @staticmethod
    def __get_edges(assemblies: Assemblies, kmerlen: int, windowsize: int, n_cpu: int) -> tuple[NDArray, NDArray]:
        """
        1. Collect k-mers from all assemblies. 
        2. Generate weighted, undirected edges. 
        3. Add record IDs as a column in `assemblies`. 

        Args:
            assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
            kmerlen (bool): See `Config` in `config.py`. 
            windowsize (bool): See `Config` in `config.py`. 
            n_cpu (int): See `Config` in `config.py`. 
        
        Returns:
            tuple: A tuple containing
                1. NDArray: See `KmerGraph.kmers`. 
                2. NDArray: See `KmerGraph.edges`. 
        """
        n_assemblies = len(assemblies)
        logger.info(f'Generating minimizer sketches from {n_assemblies} assemblies...')
        tik = time()

        # collect k-mers from all assembies
        if n_cpu <= 1:
            kmers, edges, record_ids = _get_edges(assemblies, kmerlen, windowsize, return_shm=False)
        else:
            # to make mp work, the method/function must be static and should not start with double underscores (single is fine)
            # difference between single & double underscores: https://docs.python.org/3/tutorial/classes.html#private-variables
            logger.info(f' - Parallelizing across {n_cpu} processes (~{n_assemblies//n_cpu} assemblies per process)...')
            graph_args = zip(
                get_chunks(assemblies, n_cpu), 
                repeat(kmerlen, n_cpu), 
                repeat(windowsize, n_cpu)
            )
            kmers, edges, record_ids = mp_wrapper(
                _get_edges, graph_args, n_cpu, unpack_output=True, 
                start_method=StartMethod.forkserver # must use spawn/forkserver for the indexlr python wrapper
            )
            # merge outputs from multiple processes
            logger.info(' - Merging from all processes...')
            kmers = concat_from_shm(kmers, n_cpu)
            edges = concat_from_shm(edges, n_cpu)
            edges = merge_weighted_edges(edges)
            record_ids = list(chain.from_iterable(record_ids))

        logger.info(f' - {len(edges)} weighted edges from {len(kmers)} k-mers')
        assemblies.record_ids = record_ids
        print_time_delta(time()-tik)

        return kmers, edges

    @staticmethod
    def __get_nodes(
        kmers: NDArray, assemblies: Assemblies, get_dist: bool
    ) -> tuple[NDArray, NDArray, NDArray | None]:
        """Generate k-mer nodes and calculate penalty scores. 
        1. Sort k-mers by hash values (in-place). 
        2. Aggregate each k-mer group to calculate penalty. 
        - Sorting k-mers here can also make removing unused k-mers faster (see `filter_kmers()` in `helpers.py`). 
        - Assembly distance can also be calculated in this step (`get_dist=True`), using a slower aggregation function `KmerGraph.__agg_dist()`. 

        Args:
            kmers (NDArray): See `KmerGraph.kmers`. 
            assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
            get_dist (bool): See `Config` in `config.py`. 
        
        Returns:
            tuple: A tuple containing
                1. NDArray: See `KmerGraph.idx`. 
                2. NDArray: See `KmerGraph.nodes`. 
                3. NDArray | None: See `KmerGraph.cnt_mtx`. 
        """
        logger.info(f'Generating k-mer nodes and penalty scores...')
        tik = time()

        n_assemblies = len(assemblies)
        n_tar = sum(assemblies.is_target)
        n_neg = n_assemblies - n_tar

        logger.info(' - Sorting k-mers by hash values...')
        # stable sort in-place (preserve assembly order), and return the sorted indices
        # here we don't sort with the built-in numpy functions, because argsort (or lexsort) 
        # creates a c-contiguous copy of the strided kmers['hash'], and it uses int64 for indices; 
        # reordering a struct array with kmers[:] = kmer[idx] also needs a large buffer; 
        # all these would increase peak RAM by a lot. 
        # to solve this, we could 
        # 1) structure kmers as a tuple of contiguous arrays (f-contiguous), and use numpy sorting; 
        # 2) write a custom sorting function that works on a strided array. 
        # option 1) involves rewriting a lot of functions, and numpy still uses in64 for indexing
        # option 2) is easier to implement and equally fast
        idx = sort_by_hash(kmers)

        if get_dist and (not _HAS_DIST_DEPS):
            logger.error(' - Pandas and SciPy are not installed, skip assembly distance calculation')
            get_dist = False

        # choose which aggregation function to use
        if get_dist:
            logger.warning(' - Assembly distance calculation is turned on, extra time and memory needed')
            nodes, cnt_mtx = KmerGraph.__agg_dist(kmers, n_assemblies)
        else:
            logger.info(' - Aggregating k-mer groups...')
            # better pass individual fields to a numba function, since hashes = kmers['hash'] is not supported
            nodes = agg_by_hash(
                kmers['hash'], 
                kmers['assembly_idx'], 
                kmers['is_target']
            )
            cnt_mtx = None

        # calculate penalty for each node
        nodes['penalty'] = _frac_to_penalty(
            nodes['n_tar'] / n_tar, 
            nodes['n_neg'] / n_neg
        )

        logger.info(f' - {len(nodes)} k-mer nodes')
        print_time_delta(time()-tik)
        return idx, nodes, cnt_mtx

    @staticmethod
    def __agg_dist(kmers: NDArray, n_assemblies: int) -> tuple[NDArray, NDArray]:
        """
        1. Group k-mers by their hash values. 
        2. Count the number of target/non-target assemblies in each k-mer group. 
        3. Count the number of shared k-mers between all assembly pairs. 

        Args:
            kmers (NDArray): See `KmerGraph.kmers`. 
            n_assemblies (int): Number of assemblies. 

        Returns:
            tuple: A tuple containing
                1. NDArray: See `KmerGraph.nodes`. 
                2. NDArray: See `KmerGraph.cnt_mtx`. 
        """
        logger.info(' - Grouping k-mers across all assemblies...')
        # drop duplicate hash values for each assembly
        nodes = pd.DataFrame(kmers, copy=False)[
            ['hash', 'assembly_idx', 'is_target']
        ].drop_duplicates(keep='first', inplace=False)

        # group by hash and count the number of target / total assemblies in each group
        nodes = nodes.groupby(
            by='hash', as_index=False, sort=False
        ).agg(
            # we also need the assembly indices in each group for distance calculation
            assembly_idx=pd.NamedAgg(column='assembly_idx', aggfunc=frozenset), 
            n_tar=pd.NamedAgg(column='is_target', aggfunc='sum'), 
            total=pd.NamedAgg(column='is_target', aggfunc='size'), 
        )
        assembly_idx = nodes.pop('assembly_idx')

        # formatting
        nodes['n_neg'] = nodes.pop('total') - nodes['n_tar']
        nodes['penalty'] = .0 # placeholder for penalty
        # convert to structured array
        nodes = nodes.to_records(
            index=False, 
            # column_dtypes must be a dict
            column_dtypes={name: NODE_DTYPE[name] for name in NODE_DTYPE.names}
        )
        nodes = nodes.view(np.ndarray)

        logger.info(' - Counting shared k-mers between all assembly pairs...')
        # calculate cnt_mtx
        # 1. build a membership matrix (use a sparse matrix to save memory)
        # each row represents a k-mer group, each column represents an assembly
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
                len(nodes), n_assemblies
            ), 
            dtype=np.uint32 # same as KMER_DTYPE['pos']
        ).tocsr() # convert to sparse

        # 2. calculate the number of shared k-mers by matrix multiplication
        cnt_mtx: coo_matrix = M.T @ M

        return nodes, cnt_mtx.toarray()

    @staticmethod
    def __get_dist(cnt_mtx: NDArray, kmerlen: int) -> NDArray:
        """Calculate distances between each assembly pair based on the number of shared k-mers, 
        using the distance formula used by `Mash<https://mash.readthedocs.io/en/latest/distances.html>`__. 

        Args:
            cnt_mtx (NDArray): See `KmerGraph.cnt_mtx`. 
            kmerlen (int): See `Config` in `config.py`. 
        
        Returns:
            NDArray: See `KmerGraph.dist_mtx`. 
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
        2. Create the graph instance and extract low-penalty subgraphs. 
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
        nodes = self.nodes
        edges = self.edges
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
        nodes, edges, graph = KmerGraph.__filter_graph(nodes, edges, edge_weight_th)

        # get low-penalty subgraphs
        subgraphs, used = get_subgraphs(graph, penalty_th, min_nodes, max_nodes, rng)

        # remove unused k-mers
        kmers, idx = filter_kmers(kmers, idx, used)

        print_time_delta(time()-tik)
        self.kmers = kmers
        self.idx = idx
        self.nodes = nodes
        self.edges = edges
        self.graph = graph
        self.subgraphs = subgraphs
        self._filtered_flag = True

    @staticmethod
    def __filter_graph(nodes: NDArray, edges: NDArray, edge_weight_th: float) -> tuple[NDArray, NDArray, nx.Graph]:
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
        n_nodes, n_edges = len(nodes),len(edges)

        # remove low-weight edges
        th = np.uint64(edge_weight_th) # for faster comparison
        edges = edges[edges[:, 2] > th]
        logger.info(f' - Removed {n_edges - len(edges)} edges with weight<{edge_weight_th:.3f}, {len(edges)} edges left')

        # # remove broken edges where nodes were removed; required for filter_strict()
        # n_edges = len(edges)
        # hashes = np.ascontiguousarray(nodes['hash']) # already sorted
        # # np.isin() should be slower, since the 'sort' algo will be used ('table' would require huge RAM for hashes)
        # idx_u = np.searchsorted(hashes, edges[:, 0])
        # idx_v = np.searchsorted(hashes, edges[:, 1])
        # # if u is not in hashes, np.searchsorted will return len(hashes) as its index
        # idx_u[idx_u == len(hashes)] = 0 # here we can assign any value < len(hashes)
        # idx_v[idx_v == len(hashes)] = 0
        # mask_u = hashes[idx_u] == edges[:, 0]
        # mask_v = hashes[idx_v] == edges[:, 1]
        # edges = edges[mask_u & mask_v]
        # logger.info(f' - Removed {n_edges - len(edges)} broken edges, {len(edges)} edges left')

        # remove isolated nodes
        nodes_to_keep = np.unique(edges[:, :2])
        nodes = nodes[
            np.isin(nodes['hash'], nodes_to_keep, assume_unique=True)
        ]
        logger.info(f' - Removed {n_nodes - len(nodes)} isolated nodes, {len(nodes)} nodes left')

        logger.info(' - Building graph...')
        graph = nx.Graph()
        graph.add_weighted_edges_from(edges, weight=EDGE_W)
        nx.set_node_attributes(
            graph, 
            values=dict(zip(nodes['hash'], nodes['penalty'])), 
            name=NODE_P
        )

        return nodes, edges, graph

    def filter_strict(self, penalty_th: float, edge_weight_th: float) -> None:
        """Filter graph edges and nodes. 
        1. Remove nodes with penalty > `penalty_th`. 
        2. Remove low-weight edges and isolated nodes. 
        3. Create the graph instance and extract connected components (subgraphs). 
        4. Remove k-mers not included in any of the subgraphs. 

        Args:
            penalty_th (float): See `Config` in `config.py`. 
            edge_weight_th (float): See `RunState` in `config.py`. 
        """
        kmers = self.kmers
        idx = self.idx
        nodes = self.nodes
        edges = self.edges
        filtered_flag = self._filtered_flag

        if filtered_flag:
            logger.error(f'K-mers are already filtered, cannot filter again.')
            return None

        logger.info('Extracting low-penalty subgraphs from the k-mer graph...')
        tik = time()

        # remove high penalty nodes
        # use <=, otherwise nothing will left when penalty_th = 0
        n_nodes = len(nodes)
        nodes = nodes[nodes['penalty'] <= penalty_th]
        logger.info(f' - Removed {n_nodes - len(nodes)} nodes with penalty>{penalty_th:.5f}')

        # remove low-weight edges and isolated nodes
        nodes, edges, graph = KmerGraph.__filter_graph(nodes, edges, edge_weight_th)

        # get connected components (subgraphs)
        logger.info(f' - Finding connected components...')
        subgraphs = tuple(frozenset(sg) for sg in nx.connected_components(graph))
        used = frozenset.union(*subgraphs)
        if len(subgraphs) > 0:
            logger.info(f' - Found {len(subgraphs)} components (low-penalty subgraphs)')
        else:
            log_and_raise(RuntimeError, 'No connected component was found. Try increase penalty threshold')

        # remove unused k-mers
        kmers, idx = filter_kmers(kmers, idx, used)

        print_time_delta(time()-tik)
        self.kmers = kmers
        self.idx = idx
        self.nodes = nodes
        self.edges = edges
        self.graph = graph
        self.subgraphs = subgraphs
        self._filtered_flag = True


def _get_edges(
    assemblies: Assemblies, 
    kmerlen: int, 
    windowsize: int, 
    return_shm: bool=True
) -> tuple[
    SharedArr | NDArray, 
    SharedArr | NDArray, 
    list[tuple[str, ...]]
]:
    """Worker function for `KmerGraph.__get_edges()`. If `return_shm` is True, 
    Numpy arrays are returned as shared memory blocks for multiprocessing. 

    Args:
        assemblies (Assemblies): See `Assemblies` in `assemblies.py` (could be a slice of the DataFrame). 
        kmerlen (int): See `Config` in `config.py`. 
        windowsize (int): See `Config` in `config.py`. 
        return_shm (bool, optional): If True, return the kmers and edges arrays using shared memory blocks; else return regular Numpy arrays. 
    
    Returns:
        tuple: A tuple containing
            1. SharedArr | NDArray: K-mers. Return a `SharedArr` instance if `return_shm` is True; else return a Numpy array. 
                dtype is defined in `minimizer.py` (`KMER_DTYPE`). 
            2. SharedArr | NDArray: Weighted edges in a 3-column Numpy array: the first two columns are edges and the third column is weights. 
                Return a `SharedArr` instance if `return_shm` is True; else return a Numpy array. 
            3. list[tuple[str, ...]]: Record IDs of each assembly. 
    """
    #---------- generate k-mers ----------#
    kmers = list()
    record_ids = list() # record ids of each assembly

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
    edges, isolates = get_edges(hashes)
    #---------- generate graph edges ----------#

    # concat arrays and return
    kmers = list(chain.from_iterable(kmers)) # flatten kmers before concat
    if return_shm:
        kmers = concat_to_shm(kmers)
        edges = concat_to_shm((edges,)) # just send to shared mem
    else:
        kmers = np.concatenate(kmers)

    return kmers, edges, record_ids


def _expected_frac(jaccard_mtx: NDArray) -> np.floating:
    """Calculate the expected fraction from a matrix of pairwise Jaccard indices. 
    Here fraction means: for a k-mer `h` in a group of genomes (k-mer sets), the fraction of sets in 
    a second group that also contain `h`. `jaccard_mtx` is the pairwise Jaccard indices between sets 
    in the two groups. Note that this could be a self comparison (two groups are the same). 
    - `E(frac) = mean(2J / (1+J))`, where `J` is the Jaccard matrix. 
    - This expectation ≥ mean of the Jaccard matrix, (`2J / (1+J) ≥ J, 0 ≤ J ≤ 1`). 
    """
    return np.mean(2 * jaccard_mtx / (1 + jaccard_mtx))


def _frac_to_penalty(frac_tar: NDArray | float, frac_neg: NDArray | float) -> NDArray | float:
    """The penalty formula (Euclidean / L2 norm of the two fractions). 
    - When `frac_tar` and `frac_neg` are expectations, this formula doesn't give the expected penalty, since it's non-linear. 
    - However, since it is convex, the returned value ≤ the true expectation (Jensen’s inequality). 
    """
    return ((1 - frac_tar)**2 + frac_neg**2)**0.5


def get_kmers(
    assemblies: Assemblies, config: Config, state: RunState
) -> tuple[KmerGraph, NDArray | None]:
    """Create a `KmerGraph` instance, calculate filtering thresholds and run the `filter()` method to generate low-penalty subgraphs. 

    Args:
        assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
        config (Config): See `Config` in `config.py`. 
        state (RunState): See `RunState` in `config.py`. 

    Returns:
        tuple: A tuple containing
            1. KmerGraph: The KmerGraph instance. 
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

    set_num_threads(n_cpu) # limit the number of threads to use for numba

    kmers = KmerGraph(assemblies, kmerlen, windowsize, get_dist, n_cpu)

    if no_filter:
        # skip kmers.filter(), debug only
        return kmers, None

    # calculate filter params
    # 1. calculate penalty threshold
    if penalty_th is None:
        logger.info(f'Calculating penalty threshold...')
        tik = time()

        # we only care about the presence & absence of k-mers in target assemblies
        if run_mash:
            jaccard = assemblies.mash(
                kmerlen=kmerlen, 
                sketchsize=sketchsize, 
                out_path=working_dir / WORKINGDIR.mash, 
                overwrite=overwrite, 
                n_cpu=n_cpu
            )
            e_absence_tar = 1 - _expected_frac(jaccard[:n_tar, :n_tar])
            e_presence_neg = _expected_frac(jaccard[n_tar:, :n_tar])
        else:
            # calculate expected fractions directly from minimizer sketches
            # for tar absence or neg presence, k-mer weights should always be nodes['n_tar'] (how many target assemblies have this k-mer)
            nodes = kmers.nodes
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

    # extract low-penalty subgraphs and remove unused k-mers
    kmers.filter(penalty_th, edge_weight_th, min_nodes, max_nodes, rng)
    #kmers.filter_strict(penalty_th, edge_weight_th)

    state.penalty_th = penalty_th
    state.edge_weight_th = edge_weight_th
    state.min_nodes = min_nodes
    state.max_nodes = max_nodes
    return kmers, jaccard
