"""
Configurations
==============

Seqwin run configurations. Including user/dev configs and internal configs. 

Dependencies:
-------------
- pydantic
- .ncbi
- ._version

Classes:
--------
- Config
- RunState
- WorkingDir
- BlastConfig

Functions:
----------
- config_logger

Attributes:
-----------
- WORKINGDIR (WorkingDir)
- BLASTCONFIG (BlastConfig)
- NODE_P (str)
- CONSEC_KMER_TH (int)
- LEN_TH_MUL (float)
- NO_BLAST_DIV (float)
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import sys, logging

_LOG_FMT = '%(asctime)s | %(levelname)-8s | %(message)s'
_LOG_DATEFMT = '%Y-%m-%d %H:%M:%S'

# init root logger
logging.basicConfig(
    format=_LOG_FMT, 
    datefmt=_LOG_DATEFMT, 
    level=logging.INFO, 
    stream=sys.stdout, 
)

from pathlib import Path
from random import Random
from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping

from pydantic import BaseModel, Field, field_validator, model_validator

from .ncbi import Level, Source, Task
from ._version import __version__


class Config(BaseModel):
    """Seqwin configurations. 

    Attributes:
        tar_taxa (list[str] | None): Target NCBI taxonomy name(s) / ID(s). Must be exact matches. [None]
        neg_taxa (list[str] | None): Non-target NCBI taxonomy name(s) / ID(s). Must be exact matches. [None]
        tar_paths (Path | None): A text file of paths to target assemblies in FASTA format (gzip supported), 
            with one path per line. [None]
        neg_paths (Path | None): A text file of paths to non-target assemblies in FASTA format (gzip supported), 
            with one path per line. [None]
        
        prefix (Path): Path prefix for the output directory. [cwd]
        title (str): Name of the output directory. ['seqwin-out']
        overwrite (bool): If True, overwrite existing output files. [False]
        
        kmerlen (int): K-mer length. [21]
        windowsize (int): Window size for minimizer sketch. [200]
        penalty_th (float | None): Node penalty threshold, ranging between [0, 1]. 
            None to compute with Jaccard indices (capped by `penalty_th_cap`). [None]
        run_mash (bool): If True, use MinHash sketches (Mash) to estimate penalty_th; else use minimizer sketches (faster but might be biased). [True]
        stringency (int): If `penalty_th` is None (computed with Jaccard), multiply the computed penalty threshold with `(1 - x/10)`. [5]
        min_len (int): Min length of output markers. [200]
        max_len (int | None): Max length of output markers (estimated). None for no explicit limit (capped by `max_nodes_cap`). [None]
        run_blast (bool): If True, BLAST check representative sequences. [True]
        blast_neg_only (bool): If True, only include non-target assemblies in the BLAST database (less sensitive but faster). [False]
        seed (int): Random seed for reproducibility. [42]
        
        no_filter (bool): If True, skip filtering k-mers (debug only). [False]
        penalty_th_cap (float): If `penalty_th` is None (computed with Jaccard), penalty threshold cannot be higher than this value. [0.2]
        edge_w_th_mul (float): Multiplier for determining the threshold for low-weight edges. [0.3]
        min_nodes_floor (int): Lowest possible value for `min_nodes` (see `RunState`), regardless of `min_len`. [3]
        max_nodes_cap (int | None): If `max_len` is None, `max_nodes` (see `RunState`) cannot be higher than this value. None for no limit. [100]
        
        sketchsize (int): Sketch size for Mash (MinHash) sketch. [1000]
        get_dist (bool): If True, calculate assembly distance with minimizer sketches (slow). [False]
        
        n_cpu (int): Number of threads to use. [1]
        
        level (Level): NCBI download option. Limit to genomes ≥ this assembly level ('contig' < 'scaffold' < 'chromosome' < 'complete'). ['contig']
        source (Source): NCBI download option. Genome source ('genbank' or 'refseq'). ['genbank']
        annotated (bool): NCBI download option. If True, limit to GenBank (submitter) or RefSeq annotated genomes, 
            based on the selection of source. [False]
        exclude_mag (bool): NCBI download option. If True, exclude metagenome-assembled genomes (MAGs). [False]
        gzip (bool): NCBI download option. If True, download genome sequences in gzip format. [True]
        download_only (bool): If True, only download genome sequences without running Seqwin. [True]
        
        version (str): Seqwin version. 
    """
    # Inputs
    tar_taxa: list[str] | None = None
    neg_taxa: list[str] | None = None
    tar_paths: Path | None = None
    neg_paths: Path | None = None

    # Outputs
    prefix: Path = Field(default_factory=Path.cwd) # pass a func for dynamic default
    title: str = 'seqwin-out'
    overwrite: bool = False

    # Marker options
    kmerlen: int = 21
    windowsize: int = 200
    penalty_th: float | None = None
    run_mash: bool = True
    stringency: int = 5
    min_len: int = 200
    max_len: int | None = None
    run_blast: bool = True
    blast_neg_only: bool = False # NOTE: need to fix when this is turned on
    seed: int = 42

    # Graph filtering options (not included in CLI)
    no_filter: bool = False
    penalty_th_cap: float = 0.2
    edge_w_th_mul: float = 0.3
    min_nodes_floor: int = 3
    max_nodes_cap: int | None = 100

    # Mash and assembly distance estimation (not included in CLI)
    sketchsize: int = 1000
    get_dist: bool = False

    # Performance
    n_cpu: int = 1

    # NCBI download options
    level: Level = Level.contig
    source: Source = Source.genbank
    annotated: bool = False
    exclude_mag: bool = False
    gzip: bool = True
    download_only: bool = False

    version: str = __version__

    # resolve all input paths and make sure they exist
    @field_validator('tar_paths', 'neg_paths', 'prefix', mode='before')
    @classmethod
    def _resolve_paths(cls, v):
        if v is None:
            return v
        p = Path(v).expanduser()
        return p.resolve(strict=True)

    @model_validator(mode='after')
    def _check_inputs(self) -> 'Config':
        if not self.download_only:
            if (self.tar_paths is None) and (self.tar_taxa is None):
                raise ValueError('You must provide either tar_paths or tar_taxa')
            elif (self.neg_paths is None) and (self.neg_taxa is None):
                raise ValueError('You must provide either neg_paths or neg_taxa')
        if (self.penalty_th is not None) and (self.penalty_th < 0 or self.penalty_th > 1):
            raise ValueError('penalty_th must be between [0, 1]')
        if self.stringency < 0 or self.stringency > 10:
            raise ValueError('stringency must be between [0, 10]')
        if (self.max_len is not None) and (self.max_len < self.min_len):
            raise ValueError('max_len must be greater than min_len')
        return self

    model_config = {
        # similar to @dataclass(slots=True, frozen=True)
        'frozen': True, 
        'slots': True, 
        'validate_default': True
    }


@dataclass(slots=True)
class RunState:
    """Runtime variables. 

    Attributes:
        working_dir (Path): Working directory, defined by prefix and title. 
        rng (random.Random): Built-in `random.Random` instance for reproducibility. 
        n_tar (int | None): Number of target assemblies. 
        n_neg (int | None): Number of non-target assemblies. 
        penalty_th (float | None): Node penalty threshold (user input or auto-computed). 
        edge_weight_th (float | None): Graph edge weight threshold. 
        min_nodes (int | None): Min number of nodes for a low-penalty subgraph. 
        max_nodes (int | None): Max number of nodes for a low-penalty subgraph. 
        blastdb (Path | None): Path to the BLAST database inside the working directory. 
    """
    working_dir: Path
    rng: Random
    n_tar: int | None = None
    n_neg: int | None = None
    penalty_th: float | None = None
    edge_weight_th: float | None = None
    min_nodes: int | None = None
    max_nodes: int | None = None
    blastdb: Path | None = None


@dataclass(slots=True, frozen=True)
class WorkingDir:
    """Files and directories under the working directory. 

    Attributes:
        log (str): Seqwin log file. ['seqwin.log']
        config (str): The `Config` instance saved as JSON. ['config.json']
        assemblies_dir (str): Directory for downloaded assemblies. ['assemblies']
        assemblies_csv (str): The `Assemblies` instance (`assemblies.py`) saved as CSV. ['assemblies.csv']
        mash (str): Mash sketch of all assemblies (.msh will be added by Mash). ['sketches']
        blast_dir (str): Directory for the BLAST database. ['blastdb']
        blast_log (str): Console output of the makeblastdb command, inside `blast_dir`. ['makeblastdb.log']
        markers_fasta (str): Sequences and coordinates of candidate markers. ['markers.fasta']
        markers_csv (str): Metrics of candidate markers. ['markers.csv']
        results (str): The `Seqwin` instance (`core.py`) saved as pickle. ['results.seqwin']
    """
    log: str = 'seqwin.log'
    config: str = 'config.json'
    assemblies_dir: str = 'assemblies'
    assemblies_csv: str = 'assemblies.csv'
    mash: str = 'sketches'
    blast_dir: str = 'blastdb'
    blast_log: str = 'makeblastdb.log'
    markers_fasta: str = 'markers.fasta'
    markers_csv: str = 'markers.csv'
    results: str = 'results.seqwin'


@dataclass(slots=True, frozen=True)
class BlastConfig:
    """Settings for the BLAST commands `makeblastdb` and `blastn`. 

    Attributes:
        title_neg_only (str): DB title when created from non-target assemblies only. ['neg-only']
        title_all (str): DB title when created from all assemblies. ['all']
        queue_size (int): Max queue size when streaming FASTA files to stdin of `makeblastdb`, to limit memory usage. [50]
        bool2str (Mapping[bool, str]): Mapping of bool to string. [True -> 'y', False -> 'n']
        str2bool (Mapping[str, bool]): Reversed mapping of bool2str for parsing BLAST outputs. ['y' -> True, 'n' -> False]
        header_sep (str): Separator used in FASTA headers. Should pick a rare char (cannot be '$', BLAST treats it as a special char). ['@']
        task (str): Presets for BLAST parameters ('blastn', 'blastn-short', 'megablast'). ['blastn']
        columns (tuple[str]): Columns to be included in the BLAST TSV output. See `ncbi.py` for more information. 
        batch_size (int): # Number of query sequences in a single BLAST run. [1000]
    """
    title_neg_only: str = 'neg-only'
    title_all: str = 'all'
    queue_size: int = 50
    bool2str: Mapping[bool, str] = field( # a dict cannot be used as the default value in a dataclass
        # MappingProxyType is not hashable until python 3.12, so have to use default_factory for compatibility
        default_factory=lambda: MappingProxyType({True: 'y', False: 'n'})
    )
    str2bool: Mapping[str, bool] = field(
        default_factory=lambda: MappingProxyType({'y': True, 'n': False})
    )
    header_sep: str = '@'
    task: Task = Task.blastn
    columns: tuple[str] = (
        'qseqid', 
        'sseqid', 
        'nident', 
        'mismatch', 
        'gaps', 
        'qstart', 
        'qend', 
        'sstart', 
        'send', 
        'evalue', 
        'bitscore', 
        'sseq'
    )
    batch_size: int = 1000


def config_logger(file: Path, level: int) -> None:
    """Add a file handler and set logging level for the root logger. 

    Args:
        file (Path): Path to the log file. 
        level (int): Logging level (e.g., `logging.INFO`). 
    """
    # use the same format
    logFormatter = logging.Formatter(
        fmt=_LOG_FMT, 
        datefmt=_LOG_DATEFMT, 
        style='%'
    )
    fileHandler = logging.FileHandler(file, mode='a')
    fileHandler.setFormatter(logFormatter)

    logger = logging.getLogger()
    logger.addHandler(fileHandler)
    logger.setLevel(level)


# freeze dataclasses
WORKINGDIR = WorkingDir()
BLASTCONFIG = BlastConfig()

EDGE_W: str = 'w' # Key for edge weight, used in networkx graphs. ['w']
NODE_P: str = 'p' # Key for node penalty, used in networkx graphs. ['p']
CONSEC_KMER_TH: int = 2 # Max index difference between consecutive k-mers. [2]
LEN_TH_MUL: float = 1.5 # Multiplier for candidate sequence length threshold, deprecated. [1.5]
NO_BLAST_DIV: float = 0.5 # Assumed divergence when there is no BLAST hit. [0.5]
