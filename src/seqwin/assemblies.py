"""
Assemblies
==========

Create an instance for all input genome assemblies.

Dependencies:
-------------
- numpy
- pandas
- blast
- .ncbi
- .mash
- .utils
- .config

Classes:
--------
- Assemblies

Functions:
----------
- get_assemblies
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import gzip, logging, subprocess
import multiprocessing as mp
from pathlib import Path
from time import time
from collections import deque
from collections.abc import Sequence

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .ncbi import download_taxon
from .mash import sketch, get_jaccard
from .utils import print_time_delta, log_and_raise, mkdir, file_to_write, \
    mp_wrapper, get_dups, load_paths_txt, load_fasta, GZIP_EXT
from .config import Config, RunState, WORKINGDIR, BLASTCONFIG

_FASTA_EXT = (
    '.fna', '.fasta', '.fna.gz', '.fasta.gz',
    '.fa', '.fas', '.fa.gz', '.fas.gz'
)


class Assemblies:
    r"""Ordered collection of input genome assemblies.

    Attributes:
        paths (tuple[Path, ...]): Assembly paths in global assembly-index order.
        is_targets (NDArray[np.bool\_]): True for target assemblies.
    """
    __slots__ = ('paths', 'is_targets')
    paths: tuple[Path, ...]
    is_targets: NDArray[np.bool_]

    def __init__(self, paths: Sequence[Path], is_targets: Sequence[bool]) -> None:
        """Package assembly paths and their target statuses.

        Args:
            paths (Sequence[Path]): A list of assembly paths.
            is_targets (Sequence[bool]): True for target assemblies.
        """
        self.paths = tuple(paths)
        self.is_targets = np.asarray(is_targets, dtype=np.bool_, order='C')
        if self.is_targets.ndim != 1:
            raise ValueError(f'is_targets must be one-dimensional, got shape {self.is_targets.shape}')
        if len(self.paths) != len(self.is_targets):
            raise ValueError('len(paths) must equal len(is_targets)')

    def __len__(self) -> int:
        """Return the number of assemblies."""
        return len(self.paths)

    def mash(self, kmerlen: int, sketchsize: int, out_path: Path, overwrite: bool, n_cpu: int) -> NDArray[np.floating]:
        """Calculate the Jaccard indices of all assembly pairs with Mash.

        Args:
            kmerlen (int): K-mer length for `mash sketch`.
            sketchsize (int): Sketch size for `mash sketch`.
            out_path (Path): Output path for the Mash sketch file (.msh).
            overwrite (bool): If True, overwrite the existing Mash sketch file.
            n_cpu (int): Number of processes to run in parallel.

        Returns:
            NDArray[np.floating]: A matrix of Jaccard indices of all assembly pairs.
        """
        mash_sketch = sketch(
            self.paths,
            kmerlen=kmerlen,
            sketchsize=sketchsize,
            out_path=out_path,
            overwrite=overwrite,
            n_cpu=n_cpu
        )
        return np.fromiter(
            get_jaccard(mash_sketch, n_cpu=n_cpu), dtype=float
        ).reshape(len(self), len(self))

    def fetch_seq(self, loc: pd.DataFrame, n_cpu: int) -> pd.Series:
        """Fetch the actual sequences for a DataFrame of assembly locations.
        - Fetching the sequence of each location one by one is slow, since it needs layers of indices to
        access the actual sequence (assembly, record, start and stop).
        - To solve this, rows from the same assembly are grouped together,
        and different groups are fetched in parallel.

        Args:
            loc (pd.DataFrame): Assembly locations. Row indices are kept in the returned Series, but the
                ordering might be different. To make sure the returned Series has the same order as `loc`,
                row indices should be sorted with `ascending=True`.
                Required columns: ['assembly_idx', 'record_idx', 'start', 'stop'].
            n_cpu (int): Number of processes to run in parallel.

        Returns:
            pd.Series: A sequence is fetched for each row in `loc`. indices are sorted with `ascending=True`.
        """
        if loc.empty:
            return pd.Series(index=loc.index, dtype=object)

        groups = loc.groupby(
            by='assembly_idx', sort=False
        )[['record_idx', 'start', 'stop']]
        n_groups = groups.ngroups
        logger.info(f' - {n_groups} assemblies to be loaded')

        # fetch the actual sequences by start and stop in the source sequences
        fetch_seq_args = (
            (group, self.paths[assembly_idx])
            for assembly_idx, group in groups
        )
        all_seq = pd.concat(
            mp_wrapper(_fetch_seq, fetch_seq_args, min(n_cpu, n_groups), n_jobs=n_groups),
            axis=0
        )
        # sort the returned sequences by the original ordering (before groupby)
        all_seq.sort_index(ascending=True, inplace=True)
        return all_seq

    def makeblastdb(self, prefix: Path, neg_only: bool, overwrite: bool, n_cpu: int) -> Path:
        """Create a BLAST database for all (or non-target) assemblies. Use native Python streaming and multiprocessing.
        - Note: macOS (x64 or ARM) has a hard-wired pipe buffer size of 64kB (vs. 1MB on Linux), so `makeblastdb` will
        be a lot slower on a Mac when the input is streamed to `stdin`. While on Linux the difference is negligible
        due to the larger buffer size.

        Args:
            prefix (Path): Output directory of the BLAST database.
            neg_only (bool): If True, create the BLAST database on non-target assemblies only.
            overwrite (bool): If True, overwrite prefix if it already exists.
            n_cpu (int): Number of processes to run in parallel.

        Returns:
            Path: Path to the BLAST database.
        """
        # NOTE: when the size of the blastdb changes, the evalue of a specific hit also changes.
        # Since the evalue threshold for a blast task is set, this hit might not be included when the blastdb gets larger
        if neg_only:
            logger.info('Creating a BLAST database of non-target assemblies (less sensitive but faster)...')
            assembly_indices = np.flatnonzero(~self.is_targets)
            title = BLASTCONFIG.title_neg_only
        else:
            logger.info('Creating a BLAST database of all assemblies...')
            assembly_indices = np.arange(len(self))
            title = BLASTCONFIG.title_all
        tik = time()

        # create a folder for BLAST
        mkdir(prefix, overwrite)
        blastdb = prefix / title

        # create a process for makeblastdb
        makeblastdb_args = [
            'makeblastdb',
            '-title', title,
            '-dbtype', 'nucl',
            '-out', blastdb
        ]
        proc = subprocess.Popen(
            makeblastdb_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=False # use bytes
        )

        try:
            # keep only one prepared assembly per worker in memory and consume results in submission order
            jobs = iter(assembly_indices)
            n_workers = min(n_cpu, len(assembly_indices))
            if n_workers:
                with mp.Pool(processes=n_workers) as pool:
                    pending = deque()
                    for assembly_idx in jobs:
                        pending.append(pool.apply_async(
                            _prepare_fasta,
                            args=(
                                self.paths[assembly_idx],
                                assembly_idx,
                                self.is_targets[assembly_idx]
                            )
                        ))
                        if len(pending) == n_workers:
                            break

                    while pending:
                        content = pending.popleft().get()
                        proc.stdin.write(content)
                        del content
                        try:
                            assembly_idx = next(jobs)
                        except StopIteration:
                            continue
                        pending.append(pool.apply_async(
                            _prepare_fasta,
                            args=(
                                self.paths[assembly_idx],
                                assembly_idx,
                                self.is_targets[assembly_idx]
                            )
                        ))

            proc.stdin.flush() # empty stdin but don't close the process
            stdout, stderr = proc.communicate()
            stdout, stderr = stdout.decode(), stderr.decode()
        except BaseException:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.communicate()
            except BaseException:
                pass # preserve the original preprocessing or pipe exception
            raise

        # save command, stdout and stderr
        blast_log = prefix / WORKINGDIR.blast_log
        blast_log.write_text('\n'.join((
            str(makeblastdb_args),
            stdout,
            stderr
        )))
        if proc.returncode != 0:
            log_and_raise(RuntimeError, msg=f'Failed to create the BLAST database. For details, please check {blast_log}')

        logger.info(f' - BLAST database created: {blastdb}')
        print_time_delta(time()-tik)
        return blastdb


def _prepare_fasta(path: Path, assembly_idx: int, is_target: bool) -> bytes:
    """Add assembly index and target status to all headers in an assembly FASTA file.

    Args:
        path (Path): Path to the assembly FASTA file.
        assembly_idx (int): Assembly index.
        is_target (bool): True for target assemblies.
    Returns:
        bytes: Complete FASTA content with modified headers.
    """
    # read file content as bytes
    if path.suffix == GZIP_EXT:
        content = gzip.decompress(
            path.read_bytes()
        )
    else:
        content = path.read_bytes()

    # string to be inserted into fasta header
    mod_str = f'>{assembly_idx}{BLASTCONFIG.header_sep}{BLASTCONFIG.bool2str[is_target]}{BLASTCONFIG.header_sep}'.encode()

    # modify header lines
    content = content.replace(b'\n>', b'\n' + mod_str) # faster than re.sub()
    if content.startswith(b'>'):
        content = mod_str + content[1:]
    return content


def _fetch_seq(loc: pd.DataFrame, src_fasta: Path) -> pd.Series:
    """Fetch sequences from a source FASTA file, based on their record id, start and stop coordinates.

    Args:
        loc (pd.DataFrame): A group of sequences in the same assembly, with three columns: 'record_idx', 'start' and 'stop'.
        src_fasta (Path): Path to the assembly FASTA file.

    Returns:
        pd.Series: Fetched sequences with the same index of `loc`.
    """
    src_seq = load_fasta(src_fasta)
    # NOTE: assume all forward strand
    return pd.Series(
        (src_seq[record_idx][start:stop] for record_idx, start, stop in loc.itertuples(index=False, name=None)),
        index=loc.index
    )


def _get_paths_dl(taxa_list: list[str], prefix: Path, config: Config) -> list[Path]:
    """Download assembly files for each taxon, and return the file paths.

    Args:
        taxa_list (list[str]): See `tar_taxa` and `neg_taxa` in `Config` in `config.py`.
        prefix (Path): Download prefix.
        config (Config): See `Config` in `config.py`.

    Returns:
        list[Path]: Paths to assembly files.
    """
    paths = list()
    # download genome assemblies under each taxon
    for taxon in taxa_list:
        download_paths = download_taxon(
            taxon=taxon,
            prefix=prefix,
            level=config.level,
            source=config.source,
            annotated=config.annotated,
            exclude_mag=config.exclude_mag,
            gzip=config.gzip,
            api_key=config.api_key.get_secret_value() if config.api_key is not None else None,
            overwrite=config.overwrite,
            n_cpu=config.n_cpu
        )
        if download_paths is not None:
            paths.extend(download_paths)
    return paths


def _get_paths_txt(paths_txt: Path) -> list[Path]:
    """Load assembly paths from a text file.

    Args:
        paths_txt (Path): See `tar_paths` and `neg_paths` in `Config` in `config.py`.

    Returns:
        list[Path]: Paths to assembly files.
    """
    paths = load_paths_txt(paths_txt)
    logger.info(f'Found {len(paths)} assemblies from {paths_txt}')
    return paths


def _get_paths_dir(input_dir: Path) -> list[Path]:
    """Load assembly paths from a directory (non-recursive).

    Args:
        input_dir (Path): See `tar_dir` and `neg_dir` in `Config` in `config.py`.

    Returns:
        list[Path]: Paths to assembly files.
    """
    paths = list()

    for p in sorted(input_dir.iterdir(), key=lambda x: x.name):
        if p.is_dir():
            logger.warning(f'- Skipped subdirectory {p}')
            continue
        if p.is_file():
            if p.name.lower().endswith(_FASTA_EXT):
                paths.append(p.resolve(strict=True))
            else:
                logger.warning(f'- Skipped unsupported file {p}')

    logger.info(f'Found {len(paths)} assemblies from {input_dir}')
    return paths


def _download(config: Config, working_dir: Path) -> tuple[list[Path], list[Path]]:
    """Download assemblies and return file paths. Return empty lists if nothing to download.

    Args:
        config (Config): See `Config` in `config.py`.
        working_dir (Path): See `RunState` in `config.py`.

    Returns:
        tuple: A tuple containing
            1. list[Path]: Paths to downloaded target assemblies.
            2. list[Path]: Paths to downloaded non-target assemblies.
    """
    tar_taxa = config.tar_taxa
    neg_taxa = config.neg_taxa
    tar_taxa = list() if tar_taxa is None else tar_taxa
    neg_taxa = list() if neg_taxa is None else neg_taxa

    tar_paths, neg_paths = list(), list()

    if tar_taxa or neg_taxa:
        # check if all taxa are unique
        all_taxa = tar_taxa + neg_taxa
        if len(all_taxa) != len(set(all_taxa)):
            dup_taxa = '\n'.join(
                map(str, get_dups(all_taxa))
            ) # for python <=3.10, '\n' can't be included in a f-string
            log_and_raise(RuntimeError, f"Duplicated taxa:\n{dup_taxa}")

        # create a dir to download assemblies under each taxon
        assemblies_prefix = working_dir / WORKINGDIR.assemblies_dir
        if assemblies_prefix.exists():
            logger.warning(f'Existing assemblies directory is found, genome packages might be reused: {assemblies_prefix}')
        else:
            assemblies_prefix.mkdir()

        # download assemblies to assemblies_prefix
        if tar_taxa:
            tar_paths = _get_paths_dl(tar_taxa, assemblies_prefix, config)
        if neg_taxa:
            neg_paths = _get_paths_dl(neg_taxa, assemblies_prefix, config)

    return tar_paths, neg_paths


def get_assemblies(config: Config, state: RunState) -> Assemblies:
    """Load assembly paths and package them in an Assemblies instance.
    If taxonomy names are provided, download the genome FASTA files from NCBI.

    Args:
        config (Config): See `Config` in `config.py`.
        state (RunState): See `RunState` in `config.py`.

    Returns:
        Assemblies: The Assemblies instance.
    """
    tar_paths_txt = config.tar_paths
    neg_paths_txt = config.neg_paths
    tar_dir = config.tar_dir
    neg_dir = config.neg_dir
    overwrite = config.overwrite
    download_only = config.download_only

    working_dir = state.working_dir

    # download assemblies and get file paths (return empty lists if nothing to download)
    tar_paths, neg_paths = _download(config, working_dir)

    if not download_only:
        # load assemblies from txt files
        if tar_paths_txt is not None:
            tar_paths.extend(_get_paths_txt(tar_paths_txt))
        if neg_paths_txt is not None:
            neg_paths.extend(_get_paths_txt(neg_paths_txt))
        if tar_dir is not None:
            tar_paths.extend(_get_paths_dir(tar_dir))
        if neg_dir is not None:
            neg_paths.extend(_get_paths_dir(neg_dir))

        if not tar_paths:
            log_and_raise(RuntimeError, msg='No target assembly found')
        if not neg_paths:
            log_and_raise(RuntimeError, msg='No non-target assembly found')

        # check if all paths are unique
        all_paths = tar_paths + neg_paths
        if len(all_paths) != len(set(all_paths)):
            dup_paths = '\n'.join(
                map(str, get_dups(all_paths))
            ) # for python <=3.10, '\n' can't be included in a f-string
            log_and_raise(RuntimeError, f"Duplicated assembly file paths:\n{dup_paths}")

    # package all assemblies
    paths = tar_paths + neg_paths
    is_targets = [True] * len(tar_paths) + [False] * len(neg_paths)
    assemblies = Assemblies(paths, is_targets)
    n_tar, n_neg = len(tar_paths), len(neg_paths)
    logger.info(f'Loaded {n_tar} target assemblies and {n_neg} non-target assemblies, {len(assemblies)} in total.')

    # save assemblies as csv
    assemblies_path = working_dir / WORKINGDIR.assemblies_csv
    file_to_write(assemblies_path, overwrite)
    pd.DataFrame({
        'path': assemblies.paths,
        'is_target': assemblies.is_targets,
    }).to_csv(assemblies_path, index=True)
    logger.info(f'Assembly indices and paths saved as {assemblies_path}')

    # load assembly sequences
    # NOTE: loading sequences in advance will slow everything else (maybe too much RAM)

    state.n_tar, state.n_neg = n_tar, n_neg
    return assemblies
