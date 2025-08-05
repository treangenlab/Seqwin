"""
Utilities
=========

Dependencies:
-------------
- numpy
- biopython (optional)

Functions:
----------
- print_time_delta
- log_and_raise
- overwrite_warning
- overwrite_error
- mkdir
- file_to_write
- list_dir
- run_cmd
- mp_wrapper
- get_chunks
- get_dups
- revcomp
- most_common
- most_common_weighted
- load_paths_txt
- load_fasta
- load_genbank

Attributes:
-----------
- GZIP_EXT (str)
- BASE_COMP (str.maketrans)
- StartMethod (str, Enum)
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import gzip, shutil, logging, datetime, subprocess, multiprocessing
from pathlib import Path
from time import time
from enum import Enum
from io import StringIO
from typing import Literal
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from collections import Counter
from collections.abc import Callable, Iterable, Generator, Sequence, Hashable

logger = logging.getLogger(__name__)

import numpy as np
from numpy.typing import DTypeLike
try:
    from Bio import SeqIO
    _HAS_BIO = True
except ImportError:
    _HAS_BIO = False

GZIP_EXT = '.gz'
BASE_COMP = str.maketrans('ATCGatcg', 'TAGCtagc') # Translation table for complement DNA bases


class StartMethod(str, Enum):
    """Start methods for multiprocessing. 
    """
    spawn = 'spawn'
    fork = 'fork'
    forkserver = 'forkserver'


@dataclass(slots=True, frozen=True)
class SharedArr:
    """Class for a Numpy array attached to a SharedMemory instance. 

    Attributes:
        name (str): Name of the SharedMemory instance. 
        shape (tuple): Shape of the Numpy array. 
        dtype (DTypeLike): dtype of the Numpy array. 
    """
    name: str
    shape: tuple[int]
    dtype: DTypeLike


def print_time_delta(seconds: float) -> None:
    """Print time in seconds. 
    """
    logger.info(f' - Finished in {datetime.timedelta(seconds=seconds)}')


def log_and_raise(exception: Exception=Exception, msg: str | None=None, from_none: bool=False) -> None:
    """Log and raise an error. 
    """
    logger.error(msg)
    if from_none:
        raise exception(msg)
    else:
        raise exception(msg) from None


def overwrite_warning(path: Path) -> None:
    """Log overwrite warning. 
    """
    logger.warning(f'File/directory already exists, content is overwritten (overwriting is turned on): {path}')


def overwrite_error(path: Path) -> None:
    """Raise FileExistsError. 
    """
    log_and_raise(FileExistsError, f'File/directory already exists, and overwriting is turned off: {path}', from_none=True)


def mkdir(path: Path, overwrite: bool=False, verbose=False) -> None:
    """Create a directory. If the directory exists, remove it or raise an error. 

    Args:
        path (Path): Path of the directory to be created. 
        overwrite (bool, optional): If True and the dir already exists, delete the existing dir and create a new empty one. [False]
        verbose (bool, optional): If True, print a warning message when content is overwritten. [False]
    """
    try:
        path.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        if path.is_file():
            log_and_raise(NotADirectoryError, f'Cannot create directory, since it already exists as a file: {path}')
        elif overwrite:
            if verbose:
                overwrite_warning(path)
            shutil.rmtree(path)
            path.mkdir()
        else:
            overwrite_error(path)


def file_to_write(path: Path, overwrite: bool=False, verbose=False) -> None:
    """Prepare to write a file. If the file exists, remove it or raise an error. 

    Args:
        path (Path): Path of the file to write. 
        overwrite (bool, optional): If True and the file already exists, delete the existing file. [False]
        verbose (bool, optional): If True, print a warning message when content is overwritten. [False]
    """
    if path.is_file():
        if overwrite:
            if verbose:
                overwrite_warning(path)
            path.unlink()
        else:
            overwrite_error(path)
    elif path.is_dir():
        log_and_raise(IsADirectoryError, f'Expected a file, but a directory is found: {path}')


def list_dir(path: Path=Path.cwd(), mode: Literal['a', 'd', 'f']='a') -> list[Path]:
    """Lists all subdirectories and/or files under a directory. 

    Args:
        path (Path, optional): Path of the directory to list. ['./']
        mode (str, optional): 'd' to list subdirectories, 'f' to list files, 'a' to list all. ['a']

    Returns:
        list: A list of subdirectories and/or files. 
    """
    # sanity check
    if not path.is_dir():
        log_and_raise(NotADirectoryError, f'Not a directory: {path}')
    
    entries = path.iterdir()
    if mode == 'd':
        return list(p for p in entries if p.is_dir())
    elif mode == 'f':
        return list(p for p in entries if p.is_file())
    elif mode == 'a':
        return list(entries)
    else:
        log_and_raise(ValueError, f'Invalid mode for list_dir: {mode}')


def run_cmd(*args, stdin: str | None=None, raise_error: bool=True) -> subprocess.CompletedProcess[str]:
    """Run a command using subprocess.run(). Example usage: run_cmd('ls', '-a'). 

    Args:
        *args (str): Command arguments. Must be strings. 
        stdin (str, optional): Standard input. [None]
        raise_error (bool, optional): If True, raise an error if the command did not run successfully. [True]

    Returns:
        CompletedProcess: command outputs, including stdout and stderr. 
    """
    for a in args:
        if not isinstance(a, (str, Path)):
            log_and_raise(TypeError, 'Only str or Path are accepted as command line arguments')
    cmd_out = subprocess.run(args, input=stdin, capture_output=True, text=True)
    # capture error message
    if raise_error and cmd_out.returncode != 0:
        log_and_raise(Exception, cmd_out.stderr)
    return cmd_out


def mp_wrapper(
    func: Callable, 
    all_args: Iterable, 
    n_cpu: int=1, 
    text: str | None=None, 
    starmap: bool=True, 
    unpack_output: bool=False, 
    n_jobs: int | None=None, 
    start_method: StartMethod | None=None
) -> list:
    """Wrapper for multiprocessing.Pool(). 

    Args:
        func (Callable): Function for multiprocessing. 
        all_args (Iterable): Iterable of function arguments/parameters. 
        n_cpu (int, optional): Number of processes to run in parallel [1]
        text (str | None, optional): Message to be printed when multiprocessing starts. [None]
        starmap (bool, optional): Use pool.starmap if True (func takes multiple arguments); 
            use pool.map if False (func takes only one argument). [True]
        unpack_output (bool, optional): If func has multiple output, return multiple lists instead of a single list of tuples. [False]
        n_jobs (int | None, optional): Number of elements in `all_args`. 
            Helps determine the `chunksize` option for `pool.map` and `pool.starmap`. None to let Python decide. [None]
        start_method (str | None, optional): Set the start methods for multiprocessing ('fork', 'spawn', 'forkserver'). 
            None to use the default method. [None]

    Returns:
        list: A list of func outputs, in the same order as all_args. 
    """
    tik = time()
    if text:
        logger.info(f'{text} (threads={n_cpu})')
    
    if n_cpu == 1:
        if starmap:
            # func_out = [func(*args) for args in tqdm(all_args, ascii=' >')]
            func_out = list(func(*args) for args in all_args)
        else:
            func_out = list(func(args) for args in all_args)
    elif n_cpu > 1:
        # calculate chunksize (the default python way when len(args) can be determined)
        if n_jobs is not None:
            chunksize, extra = divmod(n_jobs, 4 * n_cpu)
            if extra:
                chunksize += 1
        else:
            chunksize = None

        with multiprocessing.get_context(method=start_method).Pool(processes=n_cpu) as pool:
            if starmap:
                func_out = pool.starmap(func, all_args, chunksize=chunksize)
            else:
                func_out = pool.map(func, all_args, chunksize=chunksize)
    else:
        log_and_raise(ValueError, 'n_cpu should be an positive integer')
    
    if text:
        print_time_delta(time()-tik)

    if unpack_output:
        return list(zip(*func_out))
    else:
        return func_out


def get_chunks(ls: Sequence, n: int=1) -> Generator[Sequence, None, None]:
    """Yield n roughly same size chunks of from a list. 

    Args:
        ls (Sequence): A list or list-alike. 
        n (int): Number of chunks. [1]

    Yields:
        Sequence: Chunks of ls. 
    """
    l = len(ls)
    size, remainder = divmod(l, n)
    stop = 0
    for i in range(n):
        start = stop
        if i < remainder:
            stop = start + size + 1
            yield ls[start: stop]
        else:
            stop = start + size
            yield ls[start: stop]


def get_dups(iterable: Iterable[Hashable]) -> set:
    """Returns a set of duplicated element(s) in an iterable. All elements should be Hashable. 
    """
    seen = set()
    duplicates = list()
    for i in iterable:
        if i in seen:
            duplicates.append(i)
        else:
            seen.add(i)
    return set(duplicates)


def concat_to_shm(arrays: Sequence[np.ndarray]) -> SharedArr:
    """Concat Numpy arrays along the first dimension into a shared memory block. 
    Each individual array is deleted during this process to save memory. 

    Args:
        arrays (Sequence[np.ndarray]): Arrays to be concatenated. 
    
    Returns:
        SharedArr: The concatenated array attached to a SharedMemory instance. 
    """
    if not arrays:
        log_and_raise(ValueError, 'No array is provided')
    dtype = arrays[0].dtype
    trailing_shape = arrays[0].shape[1:]

    # validation
    for arr in arrays:
        if arr.dtype != dtype:
            log_and_raise(TypeError, 'Arrays must have the same dtype')
        elif arr.shape[1:] != trailing_shape:
            log_and_raise(ValueError, 'Arrays must match on dimensions 1...N')

    # create shared memory
    n0 = sum(arr.shape[0] for arr in arrays)
    out_shape = (n0, *trailing_shape)
    out_shm = SharedMemory(
        create=True, 
        size=int(np.prod(out_shape, dtype=np.int64) * dtype.itemsize)
    )
    try:
        out_view = np.ndarray(out_shape, dtype=dtype, buffer=out_shm.buf)

        # copy each array into its slot in out_shm
        start = 0 # position in out_view
        for arr in arrays:
            stop = start + arr.shape[0]
            out_view[start:stop] = arr
            start = stop
            try:
                arr.resize((0,), refcheck=False) # release memory
            except:
                pass
    finally:
        out_shm.close()
    return SharedArr(out_shm.name, out_shape, dtype)


def concat_from_shm(arrays: Sequence[SharedArr]) -> np.ndarray:
    """Concat SharedArr instances along the first dimension into a Numpy array. 
    Each SharedArr is unlinked during this process to save memory. 

    Args:
        arrays (Sequence[SharedArr]): Arrays to be concatenated. 
    
    Returns:
        np.ndarray: The concatenated Numpy array. 
    """
    if not arrays:
        log_and_raise(ValueError, 'No array is provided')
    dtype = arrays[0].dtype
    trailing_shape = arrays[0].shape[1:]

    # validation
    for arr in arrays:
        if arr.dtype != dtype:
            log_and_raise(TypeError, 'Arrays must have the same dtype')
        elif arr.shape[1:] != trailing_shape:
            log_and_raise(ValueError, 'Arrays must match on dimensions 1...N')

    # pre-allocate an numpy array
    n0 = sum(arr.shape[0] for arr in arrays)
    out = np.empty(
        (n0, *trailing_shape), 
        dtype=dtype
    )

    # copy each array into its slot in out
    start = 0 # position in out
    for arr in arrays:
        shm = SharedMemory(name=arr.name)
        try:
            arr_view = np.ndarray(arr.shape, dtype=dtype, buffer=shm.buf)
            stop = start + arr_view.shape[0]
            out[start:stop] = arr_view
            start = stop
        finally:
            shm.close()
            shm.unlink()

    return out


def revcomp(seq: str) -> str:
    """Returns the reverse complement of a DNA sequence. 
    """
    return seq.translate(BASE_COMP)[::-1]


def most_common(iterable: Iterable[Hashable]):
    """Returns the most common element in an Iterable, weighted by element length. 
    All elements should be Hashable. 
    """
    return Counter(iterable).most_common(1)[0][0]


def most_common_weighted(iterable: Iterable):
    """Returns the most common element in an Iterable, weighted by element length. 
    Each element should be Hashable and Sized (e.g., tuple or str). 
    """
    c = Counter(iterable)
    return max(c, key=lambda k: len(k)*c[k])


def load_paths_txt(paths_txt: Path) -> list[Path]:
    """Load file paths from a text file, with one path per line. 

    Args:
        paths_txt (Path): A text file with one path per line. 

    Returns:
        list[str]: A list of valid and resolved file paths. 
    """
    paths_list = list()
    for path in paths_txt.read_text().splitlines():
        path = Path(path.strip())
        if path.is_file():
            paths_list.append(path.resolve(strict=True))
        elif path.is_dir():
            logger.error(f' - This is a directory, skipped: {path}')
        else:
            logger.error(f' - File not found, skipped: {path}')
    return paths_list


def load_fasta(path: Path) -> tuple[str]:
    """Parse an assembly file in FASTA format, and get the sequences and IDs of all records. 
    Gzip files are supported (file name should end with .gz). 

    Args:
        path (Path): Path to the FASTA file. If the file is gzipped, the extension should be .gz. 

    Returns:
        tuple[str]: Sequences of FASTA records, in the same order as they appear in the file. 
    """
    # read file content
    if path.suffix == GZIP_EXT:
        content = gzip.decompress(
            path.read_bytes()
        ).decode()
    else:
        content = path.read_text()

    if content[0] != '>':
        log_and_raise(ValueError, f"FASTA file must start with '>', in: {path}")

    all_record: list[str] = list() # record id -> record sequence (upper case)
    all_id = list()
    for record in content.split('>')[1:]: # skip the first empty string
        header_pos = record.find('\n')
        record_id = record[:header_pos].split(' ')[0]
        if header_pos == -1:
            # in case record_id is too long
            logger.warning(f' - {record_id[:30]} has no sequence, in: {path}')
            seq = ''
        else:
            seq = record[header_pos:].replace('\n', '').upper()
        all_record.append(seq)
        all_id.append(record_id)

    # check duplicate record ID
    if len(all_id) != len(set(all_id)):
        logger.warning(f' - Duplicate record ID(s) {get_dups(all_id)}, in: {path}')
    return tuple(all_record)


if _HAS_BIO:
    def load_genbank(path: Path) -> tuple[dict[str, str], int]:
        """Parse an assembly file in GenBank format, and get the sequences and IDs of all records. 
        Gzip files are supported (file name should end with .gz). 

        Args:
            path (Path): Path to the GenBank file. If the file is gzipped, the extension should be .gz. 

        Returns:
            tuple: A tuple containing
                1. dict[str, str]: A dictionary with record ID as key and record sequence as value. 
                2. int: Sum of the length of all sequence records. 
        """
        if path.suffix == GZIP_EXT:
            with gzip.open(path, 'rb') as f:
                handle = StringIO(f.read().decode())
        else:
            handle = path

        all_record: dict[str, str] = dict() # record id -> record sequence (upper case)
        all_id = list()
        total_len = 0
        # for NCBI complete assemblies, first record is the chromosome
        for record in SeqIO.parse(handle, 'genbank'):
            record_id = str(record.id)
            seq = str(record.seq).upper()
            all_record[record_id] = seq
            all_id.append(record_id)
            total_len += len(seq)
        
        # check duplicate record ID
        if len(all_id) != len(set(all_id)):
            logger.warning(f' - Duplicate record ID(s) {get_dups(all_id)}, in: {path}')
        return all_record, total_len
else:
    def load_genbank(path) -> None:
        log_and_raise(
            ImportError, 
            'Biopython is needed for parsing GenBank files', 
            from_none=True
        )