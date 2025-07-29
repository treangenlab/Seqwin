"""
Assemblies
==========

Create an instance for all input genome assemblies. 

Dependencies:
-------------
- pandas
- blast
- .ncbi
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

import re, gzip, shutil, logging, subprocess
import multiprocessing as mp
from pathlib import Path
from io import BufferedWriter
from time import time
from queue import Empty
from collections.abc import Callable

logger = logging.getLogger(__name__)

import pandas as pd

from .ncbi import download_taxon
from .mash import sketch, dist
from .utils import print_time_delta, log_and_raise, mkdir, file_to_write, \
    mp_wrapper, get_dups, load_paths_txt, load_fasta, GZIP_EXT
from .config import Config, RunState, WORKINGDIR, BLASTCONFIG

if shutil.which('makeblastdb') is None:
    raise ImportError('BLAST+ is not installed (`makeblastdb` is not found in your PATH).')


class Assemblies(pd.DataFrame):
    """Package all input genome assemblies as a pandas DataFrame. 

    Attributes:
        path (pd.Series[Path]): Assembly paths. 
        is_target (pd.Series[bool]): True for target assemblies. 
        record_ids (pd.Series[tuple[str]]): Record IDs of each assembly. 
    """
    def __init__(self, tar_paths: list[Path], neg_paths: list[Path]) -> None:
        """Package all input genome assemblies as a pandas DataFrame. 

        Args:
            tar_paths (list[Path]): A list of paths to target assemblies. 
            neg_paths (list[Path]): A list of paths to non-target assemblies. 
        """
        data = dict(
            path=tar_paths + neg_paths, 
            is_target=[True]*len(tar_paths) + [False]*len(neg_paths), 
            record_ids=None
        )
        super().__init__(data)
    
    def mash(self, kmerlen: int, sketchsize: int, out_path: Path, overwrite: bool, n_cpu: int) -> pd.DataFrame:
        mash_sketch = sketch(
            self.path.tolist(), 
            kmerlen=kmerlen, 
            sketchsize=sketchsize, 
            out_path=out_path, 
            overwrite=overwrite, 
            n_cpu=n_cpu
        )
        return dist(mash_sketch, n_cpu=n_cpu)

    def fetch_seq(self, loc: pd.DataFrame, n_cpu: int=1) -> pd.Series:
        """Fetch the actual sequences for a DataFrame of assembly locations. 
        - Fetching the sequence of each location one by one is slow, since it needs layers of indexes to 
        access the actual sequence (assembly, record, start and stop). 
        - To solve this, rows from the same assembly are grouped together, 
        and different groups are fetched in parallel. 

        Args:
            loc (pd.DataFrame): Assembly locations. Row indexes are kept in the returned Series, but the 
                ordering might be different. To make sure the returned Series has the same order as `loc`, 
                row indexes should be sorted with `ascending=True`. 
                Required columns: ['assembly_idx', 'record_idx', 'start', 'stop']. 
            assemblies (Assemblies): Includes data of all assemblies. 
            n_cpu (int, optional): Number of processes to run in parallel. [1]
        
        Returns:
            pd.Series: A sequence is fetched for each row in `loc`. Indexes are sorted with `ascending=True`. 
        """
        # group sequences by assembly_idx
        loc: dict[int, pd.DataFrame] = dict(tuple(
            loc.groupby(
                by='assembly_idx', sort=False
            )[['record_idx', 'start', 'stop']]
        ))
        logger.info(f' - {len(loc)} assemblies to be loaded')

        # fetch the source sequence for each group of sequences
        assemblies_paths = self.path
        all_src_paths = (assemblies_paths.loc[assembly_idx] for assembly_idx in loc)

        # fetch the actual sequences by start and stop in the source sequences
        fetch_seq_args = zip(
            loc.values(), 
            all_src_paths
        )
        all_seq: pd.Series = pd.concat(
            mp_wrapper(_fetch_seq, fetch_seq_args, n_cpu, n_jobs=len(loc)), 
            axis=0
        )
        # sort the returned sequences by the original ordering (before groupby)
        all_seq.sort_index(ascending=True, inplace=True)
        return all_seq

    def makeblastdb(self, prefix: Path, neg_only: bool, overwrite: bool, n_cpu: int) -> Path:
        """Create a BLAST database for all or non-target assemblies. Use native Python streaming and multiprocessing. 
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
            df = self[self.is_target == False]
            title = BLASTCONFIG.title_neg_only
        else:
            logger.info('Creating a BLAST database of all assemblies (more sensitive but slower)...')
            df = self
            title = BLASTCONFIG.title_all
        tik = time()

        # create a folder for BLAST
        mkdir(prefix, overwrite)
        blastdb = prefix / title

        # load fasta files and stream to stdin to save memory
        with mp.Manager() as manager:
            # only Manager().Queue() can be shared between processes (e.g, when used with Pool())
            # so that different processes can put items in the same queue. 
            queue = manager.Queue(maxsize=BLASTCONFIG.queue_size+n_cpu) # set queue size to limit memory usage
            queue_idx = range(len(df)) # queue index must start from 0 (for df.index, this is not True when neg_only is True)

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

            # process fasta files in parallel and add file contents to queue
            pool = mp.Pool(processes=n_cpu)
            for args in zip(df.path, df.index, df.is_target, queue_idx):
                pool.apply_async(_add_fasta_to_queue, args=(*args, queue))
            pool.close() # no more task to be added to pool

            # dequeue and stream to stdin
            _stream_to_stdin(queue, len(df), proc.stdin)

            # wait for everything to finish
            pool.join()
            stdout, stderr = proc.communicate()
            stdout, stderr = stdout.decode(), stderr.decode()

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
    
    def __makeblastdb_cmd(self, title: str, prefix: Path, neg_only: bool, overwrite: bool, n_cpu: int) -> Path:
        """Create a BLAST database for all or non-target assemblies. Use streaming with subprocess.run(shell=True) to save memory. 
        `sd` is needed for modifying FASTA content, and `parallel` is needed for multiprocessing. 

        Dependencies:
        - sd (https://github.com/chmln/sd, https://anaconda.org/conda-forge/sd)
        - parallel (https://www.gnu.org/software/parallel/, https://anaconda.org/conda-forge/parallel)

        Args:
            title (str): Name of the BLAST database. 
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
            df = self[self.is_target == False]
            title += BLASTCONFIG.neg_only
        else:
            logger.info('Creating a BLAST database of all assemblies (more sensitive but slower)...')
            df = self
        tik = time()

        # create a folder for BLAST
        mkdir(prefix, overwrite)
        blastdb = prefix / title

        # make a tsv of assembly path, assembly index and is_target (input for GNU parallel)
        assemblies_tsv = '\n'.join(
            f'{p}\t{i}\t{BLASTCONFIG.bool2str[b]}' 
            for p, i, b in zip(df.path, df.index, df.is_target)
        )
        
        # GNU parallel command to modify assembly fasta headers (>record_id to >idx@is_target@record_id)
        # {1}=assembly path, {2}=assembly index, {3}=is_target
        # use zcat if the fasta file is gzipped, else use cat; then modify the headers with sd
        # since this command uses shell syntax ([[ ... ]], &&, ||, and |), shell=True is necessary in subprocess
        modify_fasta = '[[ {1} == *.gz ]] && zcat "{1}" || cat "{1}" | sd "^>" ">{2}%s{3}%s"' \
            % (BLASTCONFIG.header_sep, BLASTCONFIG.header_sep)
        
        # command to create the blast db
        makeblastdb = f'makeblastdb -title {title} -dbtype nucl -out {blastdb}'
        
        # load assembly files in paralle, and use pipes to save memory (shell=True)
        # tried to use native python pipes with shell=False but failed
        proc = subprocess.run(
            rf"parallel -j {n_cpu} --colsep '\t' '{modify_fasta}' | {makeblastdb}", 
            input=assemblies_tsv, 
            shell=True, capture_output=True, text=True
        )

        # save the full command, stdout and stderr
        blast_log = prefix / WORKINGDIR.blast_log
        blast_log.write_text('\n'.join((
            proc.args, 
            proc.stdout, 
            proc.stderr, 
            assemblies_tsv
        )))
        if proc.returncode != 0:
            log_and_raise(RuntimeError, msg=f'Failed to create the BLAST database. For details, please check {blast_log}')
        
        logger.info(f' - BLAST database created: {blastdb}')
        print_time_delta(time()-tik)
        return blastdb


def _add_fasta_to_queue(path: Path, assembly_idx: int, is_target: bool, queue_idx: int, queue: mp.Queue) -> None:
    """
    1. Add assembly index and is_target into the header lines of an assembly FASTA file. 
    2. Put the modified FASTA file content, as well as its queue_idx in a queue. 

    The use of queue_idx is to make sure that items in the queue can be fetched in order (in `_stream_to_stdin()`), 
    so that the behavior is exactly the same as reading all FASTA files into memory using a single thread. 

    Args:
        path (Path): Path to the assembly FASTA file. 
        assembly_idx (int): Assembly index. 
        is_target (bool): True for target assemblies. 
        queue_idx (int): Used to determine the order of assemblies in the queue. Must start from 0. 
        queue (mp.Queue): queue_idx and modified FASTA file content are put into this queue. 
            Use `mp.Manager().Queue()` to share this queue across different processes (e.g., when this function is called by `mp.Pool()`). 
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

    # modify header lines and put modified content in queue
    # set re.MULTILINE to search for '>' at the beginning of each line
    content = re.sub(pattern=rb'^>', repl=mod_str, string=content, flags=re.MULTILINE)
    queue.put((queue_idx, content))


def _stream_to_stdin(queue: mp.Queue, n_items: int, proc_stdin: BufferedWriter) -> None:
    """Get items from an indexed queue, and write them to the stdin of a process by the order of their indexes. 
    
    Args:
        indexed_queue (mp.Queue): Each queue item should be a tuple of (idx, data). 
        n_items (int): Total number of items in the queue. 
        proc_stdin (io.BufferedWriter): Standard input of a process (e.g., `subprocess.Popen().stdin`). 
    """
    next_idx = 0
    buffer: dict[int, bytes] = dict()

    while next_idx < n_items:
        try:
            idx, data = queue.get()
            buffer[idx] = data

            while next_idx in buffer:
                proc_stdin.write(
                    buffer.pop(next_idx)
                )
                next_idx += 1
        except Empty:
            continue

    proc_stdin.flush() # empty stdin but don't close the process


def _load_seq(
    paths: list[Path], 
    parser: Callable[[Path], tuple[dict[str, str], int]]=load_fasta, 
    n_cpu: int=1
) -> list[dict[str, str]]:
    """Load assemblies sequences from files. 

    Args:
        paths (list[Path]): A list of paths to the assembly files. 
        parser (Callable, optional): The function to parse the assembly files. It should return a dict of record id -> 
            sequence (upper case), and the total number of bases in the assembly. Supported functions: `parsers.load_fasta()`, 
            `parsers.load_genbank()`. [load_fasta]
        n_cpu (int, optional): Number of processes to run in parallel. [1]
    
    Returns:
        list[dict[str,str]]: Each dict has record IDs as keys and record sequences as values. 
    """
    # load assembly files in paralelle
    n_assemblies = len(paths)
    logger.info(f' - Loading {n_assemblies} assembly files...')
    all_seq, all_len = mp_wrapper(
        parser, paths, n_cpu, 
        starmap=False, unpack_output=True
    )
    total_len = sum(all_len)
    logger.info(f' - Average assembly size: {total_len/n_assemblies:.0f} bp; total: {total_len} bp')
    return all_seq


def _fetch_seq(loc: pd.DataFrame, src_fasta: Path) -> pd.Series:
    """Fetch sequences from a source FASTA file, based on their record id, start and stop coordinates. 

    Args:
        loc (pd.DataFrame): A group of sequences in the same assembly. Required columns: 'record_idx', 'start' and 'stop'. 
        src_fasta (Path): Path to the assembly FASTA file. 

    Returns:
        pd.Series: Fetched sequences with the same index of `loc`. 
    """
    src_seq = load_fasta(src_fasta)
    # NOTE: assume all forward strand
    return loc.apply(
        lambda row: src_seq[row['record_idx']][row['start']:row['stop']], 
        axis=1
    )


def _get_paths_dl(taxa_list: list[str], prefix: Path, config: Config) -> list[Path]:
    """Download assembly files for each taxon, and return the file paths. 

    Args:
        taxa_list (list[str]): See `tar_taxa` and `neg_taxa` in `Config` in `config.py`. 
        prefix (Path): Download prefix. 
        config (Config): See `Config` in `config.py`. 

    Returns:
        list[str]: Paths to assembly files. 
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
            n_cpu=config.n_cpu
        )
        if download_paths is None:
            log_and_raise(
                RuntimeError, 
                f'Unsuccessful download of taxon {taxon}. \
                Try removing this taxon and re-running Seqwin with the same "--prefix" and "--title". \
                Downloaded taxon packages will be reused.'
            )
        else:
            paths.extend(download_paths)
    return paths


def _get_paths_txt(paths_txt: Path) -> list[Path]:
    """Load assembly paths from txt. 

    Args:
        paths_txt (Path): See `tar_paths` and `neg_paths` in `Config` in `config.py`. 

    Returns:
        list[Path]: Paths to assembly files. 
    """
    paths = load_paths_txt(paths_txt)
    logger.info(f'Found {len(paths)} assemblies from {paths_txt}')
    return paths


def _download(config: Config, working_dir: Path) -> tuple[list[Path], list[Path]]:
    """Download assemblies and return file paths. Return empty lists if nothing to download. 
    
    Args:
        config (Config): See `Seqwin` in `main.py`. 
        working_dir (Path): See `RunState` in `config.py`. 
    
    Returns:
        tuple: A tuple containing
            1. tar_paths (list[Path]): Paths to downloaded target assemblies. 
            1. neg_paths (list[Path]): Paths to downloaded non-target assemblies. 
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
    """Load assembly paths and build a BLAST database. 
    
    Args:
        config (Config): See `Seqwin` in `main.py`. 
        state (RunState): See `Seqwin` in `main.py`. 
    
    Returns:
        Assemblies: The Assemblies instance. 
    """
    tar_paths_txt = config.tar_paths
    neg_paths_txt = config.neg_paths
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

        if not tar_paths:
            log_and_raise(RuntimeError, msg='No target assembly found.')
        if not neg_paths:
            log_and_raise(RuntimeError, msg='No non-target assembly found.')

        # check if all paths are unique
        all_paths = tar_paths + neg_paths
        if len(all_paths) != len(set(all_paths)):
            dup_paths = '\n'.join(
                map(str, get_dups(all_paths))
            ) # for python <=3.10, '\n' can't be included in a f-string
            log_and_raise(RuntimeError, f"Duplicated assembly file paths:\n{dup_paths}")

    # package all assemblies
    assemblies = Assemblies(tar_paths, neg_paths)
    n_tar, n_neg = len(tar_paths), len(neg_paths)
    logger.info(f'Loaded {n_tar} target assemblies and {n_neg} non-target assemblies, {len(assemblies)} in total.')

    # save assemblies as csv
    assemblies_path = working_dir / WORKINGDIR.assemblies_csv
    file_to_write(assemblies_path, overwrite)
    assemblies.to_csv(assemblies_path, columns=('path', 'is_target'), index=True)
    logger.info(f'Assembly indexes and paths saved as {assemblies_path}')

    # load assembly sequences
    # NOTE: loading sequences in advance will slow everything else (maybe too much RAM)

    state.n_tar, state.n_neg = n_tar, n_neg
    return assemblies
