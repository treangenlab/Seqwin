"""
Core
====

Seqwin entry point.

Classes:
--------
- Seqwin

Functions:
----------
- run
- load
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import logging, pickle
from time import time
from pathlib import Path

logger = logging.getLogger(__name__)

from .assemblies import Assemblies, get_assemblies
from .graph import KmerGraph, FilteredGraph, Signature, _filter_native
from .evaluation import SignatureMetrics, process_signatures
from .utils import print_time_delta, overwrite_warning, overwrite_error, mkdir, file_to_write
from .config import Config, RunState, config_logger, HAS_MASH, WORKINGDIR


def _build_graph(assemblies: Assemblies, config: Config) -> KmerGraph:
    """Build the raw (unscored) minimizer graph.
    """
    logger.info(f'Building minimizer graph from {len(assemblies)} assemblies...')
    if config.low_memory:
        logger.warning(' - Low-memory mode is enabled; graph construction may take longer')
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


def _filter_graph(
    graph: KmerGraph, assemblies: Assemblies, config: Config, state: RunState
) -> tuple[FilteredGraph, list[Signature]]:
    """Filter the minimizer graph and extract signatures from low-penalty subgraphs.
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


class Seqwin(object):
    """Seqwin run instance.

    Attributes:
        config (Config): See `Config` in `config.py`.
        state (RunState): See `RunState` in `config.py`.
        assemblies (Assemblies): See `Assemblies` in `assemblies.py`.
        filtered (FilteredGraph | None): Filtered minimizer graph.
        signatures (tuple[Signature] | None): Extracted signatures.
        metrics (tuple[SignatureMetrics] | None): Evaluation metrics parallel to `signatures`.
    """
    __slots__ = ('config', 'state', 'assemblies', 'filtered', 'signatures', 'metrics')
    config: Config
    state: RunState
    assemblies: Assemblies
    filtered: FilteredGraph | None
    signatures: tuple[Signature, ...] | None
    metrics: tuple[SignatureMetrics, ...] | None

    def __init__(self, config: Config) -> None:
        """Initiate a Seqwin run instance.
        1. Create a working directory.
        2. Initialize the logger.
        3. Save config to JSON.
        4. Load all assemblies.

        Args:
            config (Config): See `Config` in `config.py`.
        """
        prefix = config.prefix
        title = config.title
        overwrite = config.overwrite
        n_cpu = config.n_cpu
        version = config.version

        # create working dir, or overwrite the existing one
        working_dir = prefix / title
        try:
            # prefix is validated in config.py
            working_dir.mkdir(parents=False, exist_ok=False)
            logger.info(f'Created output directory {working_dir}')
        except FileExistsError:
            # if working_dir exist, it should be a directory
            if working_dir.is_file():
                raise NotADirectoryError(f'Cannot create {working_dir}, since it already exists as a file') from None
            elif overwrite:
                overwrite_warning(working_dir)
            else:
                overwrite_error(working_dir)

        # log to file, must happen after working_dir is created
        config_logger(working_dir / WORKINGDIR.log, logging.INFO)

        logger.info(f'Running Seqwin v{version}')
        if n_cpu == 1:
            logger.warning('Using only one CPU thread, longer running time is expected')

        # save configs
        config_path = working_dir / WORKINGDIR.config
        file_to_write(config_path, overwrite)
        config_path.write_text(config.model_dump_json(indent=4))
        logger.info(f'Run configurations saved as {config_path}')

        # initiate run states
        state = RunState(working_dir=working_dir)

        # load assemblies
        assemblies = get_assemblies(config, state)

        self.config = config
        self.state = state
        self.assemblies = assemblies
        self.filtered = None
        self.signatures = None
        self.metrics = None

    def run(self) -> None:
        """Build and filter the k-mer graph, then extract candidate markers.
        """
        config = self.config
        state = self.state
        assemblies = self.assemblies

        overwrite = config.overwrite
        save_graph = config.save_graph
        working_dir = state.working_dir

        graph = _build_graph(assemblies, config)
        if save_graph:
            graph_path = working_dir / WORKINGDIR.graph
            mkdir(graph_path, overwrite=overwrite, verbose=overwrite)
            graph.save(graph_path)
            logger.info(f'Raw minimizer graph is saved as {graph_path}')

        filtered, signatures = _filter_graph(graph, assemblies, config, state)
        signatures, metrics = process_signatures(
            signatures, filtered, assemblies, graph, config, state
        )

        self.filtered = filtered
        self.signatures = signatures
        self.metrics = metrics

        # save run instance
        # results_path = working_dir / WORKINGDIR.results
        # file_to_write(results_path, overwrite)
        # results_path.write_bytes(pickle.dumps(self))
        # logger.info(f'Run instance (includes all run data) saved as {results_path}')


def run(config: Config) -> Seqwin:
    """Run Seqwin.

    Args:
        config (Config): See `Config` in `config.py`.

    Returns:
        Seqwin: The Seqwin run instance.
    """
    seqwin = Seqwin(config)
    if not config.download_only:
        seqwin.run()
    return seqwin


def load(path: str | Path) -> Seqwin:
    """Load a Seqwin run instance from file.

    Args:
        path (str | Path): Path to the Seqwin run snapshot (`results.seqwin`).

    Returns:
        Seqwin: The Seqwin run instance.
    """
    if isinstance(path, str):
        path = Path(path)
    return pickle.loads(path.read_bytes())
