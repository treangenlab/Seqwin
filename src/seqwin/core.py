"""
Core
====

Seqwin entry point. 

Dependencies
------------
- pandas
- .assemblies
- .kmers
- .markers
- .utils
- .config

Classes:
--------
- Seqwin

Functions:
----------
- run
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import logging, pickle
from random import Random

logger = logging.getLogger(__name__)

import pandas as pd

from .assemblies import Assemblies, get_assemblies
from .kmers import KmerGraph, get_kmers
from .markers import ConnectedKmers, get_markers
from .utils import overwrite_warning, overwrite_error, file_to_write
from .config import Config, RunState, config_logger, WORKINGDIR


class Seqwin(object):
    """Seqwin run instance. 

    Attributes:
        config (Config): See `Config` in `config.py`. 
        state (RunState): See `RunState` in `config.py`. 
        assemblies (Assemblies): See `Assemblies` in `assemblies.py`. 
        kmers (KmerGraph | None): See `KmerGraph` in `kmers.py`. Generated with `self.run()`. 
        mash (pd.DataFrame | None): Tabular output of `mash dist`. Generated with `self.run()`. 
        markers (list[ConnectedKmers] | None): See `ConnectedKmers` in `markers.py`. Generated with `self.run()`. 
    """
    __slots__ = ('config', 'state', 'assemblies', 'kmers', 'mash', 'markers')
    config: Config
    state: RunState
    assemblies: Assemblies
    kmers: KmerGraph | None
    mash: pd.DataFrame | None
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
        seed = config.seed
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

        # save configs
        config_path = working_dir / WORKINGDIR.config
        file_to_write(config_path, overwrite)
        config_path.write_text(config.model_dump_json(indent=4))
        logger.info(f'Run configurations saved as {config_path}')

        # initiate run states
        state = RunState(working_dir=working_dir, rng=Random(seed))

        # load assemblies
        assemblies = get_assemblies(config, state)

        self.config = config
        self.state = state
        self.assemblies = assemblies
        self.kmers = None
        self.mash = None
        self.markers = None

    def run(self) -> None:
        """Build the k-mer graph and extract candidate markers. 
        """
        config = self.config
        state = self.state
        assemblies = self.assemblies

        overwrite = config.overwrite
        working_dir = state.working_dir

        kmers, jaccard = get_kmers(assemblies, config, state)

        if kmers.subgraphs is None:
            # if config.no_filter is True (debug only)
            markers = None
        else:
            markers = get_markers(kmers, assemblies, config, state)

        self.kmers = kmers
        self.mash = jaccard
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
