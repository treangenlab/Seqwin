"""
K-mer Graph
===========

A core module of Seqwin. Build a k-mer graph from all input assemblies and extract low-penalty subgraphs.

Dependencies:
-------------
- numpy
- .graph
- .assemblies
- .utils
- .config

Functions:
----------
- build_graph
- filter_graph
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import logging
from time import time

logger = logging.getLogger(__name__)

from .graph import KmerGraph, FilteredGraph, Signature, _filter_native
from .assemblies import Assemblies
from .utils import print_time_delta
from .config import Config, RunState, HAS_MASH, WORKINGDIR


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
) -> tuple[FilteredGraph, list[Signature]]:
    """Filter a minimizer graph and find low-penalty subgraphs.
    """
    logger.info('Filtering minimizer graph...')
    tik = time()
    jaccard = None
    if config.penalty_th is None and config.run_mash:
        if HAS_MASH:
            jaccard = assemblies.mash(
                kmerlen=config.kmerlen,
                sketchsize=config.sketchsize,
                out_path=state.working_dir / WORKINGDIR.mash,
                overwrite=config.overwrite,
                n_cpu=config.n_cpu
            )
        else:
            logger.error('Mash is not installed. Falling back to minimizer sketches.')

    filtered, signatures = _filter_native(
        kmers=graph.kmers,
        nodes=graph.nodes,
        edges=graph.edges,
        record_offsets=graph.record_offsets,
        assembly_paths=list(map(str, assemblies.paths)),
        is_targets=assemblies.is_targets,
        jaccard=jaccard,
        kmerlen=config.kmerlen,
        windowsize=config.windowsize,
        penalty_th=config.penalty_th,
        stringency=config.stringency,
        min_len=config.min_len,
        max_len=config.max_len,
        penalty_th_cap=config.penalty_th_cap,
        edge_w_th_mul=config.edge_w_th_mul,
        min_nodes_floor=config.min_nodes_floor,
        max_nodes_cap=config.max_nodes_cap,
        consec_kmer_mul=config.consec_kmer_mul,
        n_cpu=config.n_cpu
    )

    print_time_delta(time() - tik)
    state.jaccard = jaccard
    return filtered, signatures
