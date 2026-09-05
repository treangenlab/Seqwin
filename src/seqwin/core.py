"""
Core
====

Seqwin entry point.

Dependencies:
-------------
- numpy
- .assemblies
- .graph
- .markers
- .utils
- .config

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
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
from numpy.typing import NDArray

from .assemblies import Assemblies, get_assemblies
from .kmers import FilteredGraph, build_graph, filter_graph
from .markers import ConnectedKmers, get_markers
from .utils import overwrite_warning, overwrite_error, mkdir, file_to_write
from .config import Config, RunState, config_logger, WORKINGDIR


class Seqwin(object):
    """Seqwin run instance.

    Attributes:
        config (Config): See `Config` in `config.py`.
        state (RunState): See `RunState` in `config.py`.
        assemblies (Assemblies): See `Assemblies` in `assemblies.py`.
        graph (FilteredGraph | None): See `FilteredGraph` in `kmers.py`. Generated with `self.run()`.
        jaccard (NDArray[np.float64] | None): Pairwise assembly Jaccard matrix. Generated with `self.run()`.
        markers (list[ConnectedKmers] | None): See `ConnectedKmers` in `markers.py`. Generated with `self.run()`.
    """
    __slots__ = ('config', 'state', 'assemblies', 'graph', 'jaccard', 'markers')
    config: Config
    state: RunState
    assemblies: Assemblies
    graph: FilteredGraph | None
    jaccard: NDArray[np.float64] | None
    markers: list[ConnectedKmers] | None

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
        self.graph = None
        self.jaccard = None
        self.markers = None

    def run(self) -> None:
        """Build and filter the k-mer graph, then extract candidate markers.
        """
        config = self.config
        state = self.state
        assemblies = self.assemblies

        overwrite = config.overwrite
        save_graph = config.save_graph
        working_dir = state.working_dir

        graph = build_graph(assemblies, config)
        if save_graph:
            graph_path = working_dir / WORKINGDIR.graph
            mkdir(graph_path, overwrite=overwrite, verbose=overwrite)
            graph.save(graph_path)
            logger.info(f'Raw minimizer graph is saved as {graph_path}')

        graph, jaccard = filter_graph(graph, assemblies, config, state)
        markers = get_markers(graph, assemblies, config, state)

        self.graph = graph
        self.jaccard = jaccard
        self.markers = markers

        # save run instance
        results_path = working_dir / WORKINGDIR.results
        file_to_write(results_path, overwrite)
        results_path.write_bytes(pickle.dumps(self))
        logger.info(f'Run instance (includes all run data) saved as {results_path}')


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
