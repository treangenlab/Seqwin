"""
Utilities
=========
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import sys, gzip, shutil, logging, datetime, subprocess, shlex, multiprocessing
from pathlib import Path
from time import time
from enum import Enum
from typing import Literal
from collections.abc import Callable, Iterable, Generator, Sequence, Hashable

logger = logging.getLogger(__name__)

GZIP_EXT = '.gz'
BASE_COMP = str.maketrans('ATCGatcg', 'TAGCtagc') # Translation table for complement DNA bases

class StartMethod(str, Enum):
    """Start methods for multiprocessing.
    """
    spawn = 'spawn'
    fork = 'fork'
    forkserver = 'forkserver'

_START_METHOD = (
    StartMethod.spawn if sys.platform == 'win32' else StartMethod.fork
)


def print_time_delta(seconds: float) -> None:
    """Print time in seconds.
    """
    logger.info(f' - Finished in {datetime.timedelta(seconds=seconds)}')


def log_and_raise(
    exception: type[Exception] = Exception,
    msg: str = '',
    from_none: bool = False,
    from_e: BaseException | None = None
) -> None:
    """Log and raise an error.

    Args:
        exception (type[Exception], optional): Exception type. [Exception]
        msg (str, optional): Message to be logged and printed. ['']
        from_none (bool, optional): If True, run `raise ... from None`. [False]
        from_e (BaseException | None, optional): If provided, run `raise ... from ...`. [None]
    """
    logger.critical((msg or exception.__name__))

    if from_none and from_e is not None:
        raise ValueError('Use only one of from_none or from_e')

    if from_none:
        raise exception(msg) from None
    if from_e is not None:
        raise exception(msg) from from_e
    else:
        raise exception(msg)


def overwrite_warning(path: Path) -> None:
    """Log overwrite warning.
    """
    logger.warning(f'File/directory already exists, content is overwritten (overwriting is turned on): {path}')


def overwrite_error(path: Path) -> None:
    """Raise FileExistsError.
    """
    log_and_raise(FileExistsError, f'File/directory already exists, and overwriting is turned off: {path}', from_none=True)


def read_text(path: Path) -> str:
    """Read a UTF-8 text file with universal newline normalization.
    """
    with open(path, 'r', encoding='utf-8', newline=None) as f:
        return f.read()


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


def list_dir(
    path: Path = Path.cwd(),
    mode: Literal['a', 'd', 'f'] = 'a'
) -> list[Path]:
    """List all subdirectories and/or files under a directory.

    Args:
        path (Path, optional): Path of the directory to list. ['./']
        mode (str, optional): 'd' to list subdirectories, 'f' to list files, 'a' to list all. ['a']

    Returns:
        list: A list of sub-directories and/or files, sorted by names.
    """
    # sanity check
    if not path.is_dir():
        log_and_raise(NotADirectoryError, f'Not a directory: {path}')

    if mode == 'd':
        entries = (p for p in path.iterdir() if p.is_dir())
    elif mode == 'f':
        entries = (p for p in path.iterdir() if p.is_file())
    elif mode == 'a':
        entries = path.iterdir()
    else:
        log_and_raise(ValueError, f'Invalid mode for list_dir: {mode}')

    return sorted(entries, key=lambda p: p.name)


def run_cmd(
    *args: str | Path,
    stdin: str | None = None,
    raise_error: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run a command using subprocess.run(). Example usage: run_cmd('ls', '-a').

    Args:
        *args (str | Path): Command arguments. Must be strings or paths.
        stdin (str, optional): Standard input. [None]
        raise_error (bool, optional): If True, raise an error if the command did not run successfully. [True]

    Returns:
        CompletedProcess: command outputs, including stdout and stderr.
    """
    for a in args:
        if not isinstance(a, (str, Path)):
            log_and_raise(TypeError, 'Only str or Path are accepted as command line arguments')
    try:
        return subprocess.run(args, input=stdin, capture_output=True, text=True, check=raise_error)
    except subprocess.CalledProcessError as e:
        msg = (
            'Subprocess failed\n'
            f'cmd: {shlex.join(str(c) for c in e.cmd)}\n'
            f'exit code: {e.returncode}\n'
            f'stderr:\n{(e.stderr or "").strip()}'
        )
        log_and_raise(RuntimeError, msg, from_e=e)


def mp_wrapper(
    func: Callable,
    all_args: Iterable,
    n_cpu: int=1,
    text: str | None=None,
    starmap: bool=True,
    unpack_output: bool=False,
    n_jobs: int | None=None,
    start_method: StartMethod | None=_START_METHOD
) -> list:
    """Wrapper for multiprocessing.Pool().

    Args:
        func (Callable): Function for multiprocessing.
        all_args (Iterable): Iterable of function arguments/parameters.
        n_cpu (int, optional): Number of processes to run in parallel. [1]
        text (str | None, optional): Message to be printed when multiprocessing starts. [None]
        starmap (bool, optional): Use pool.starmap if True (func takes multiple arguments);
            use pool.map if False (func takes only one argument). [True]
        unpack_output (bool, optional): If func has multiple output, return multiple lists instead of a single list of tuples. [False]
        n_jobs (int | None, optional): Number of elements in `all_args`.
            Helps determine the `chunksize` option for `pool.map` and `pool.starmap`. None to let Python decide. [None]
        start_method (str | None, optional): Set the start method for multiprocessing ('fork', 'spawn', 'forkserver').
            By default, 'spawn' is used for Windows and 'fork' for other systems. Use None to let Python decide.

    Returns:
        list: A list of func outputs, in the same order as all_args.
    """
    tik = time()
    if text:
        logger.info(f'{text} (processes={n_cpu})')

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

    if unpack_output and func_out:
        return list(zip(*func_out, strict=True))
    else:
        return func_out


def get_chunks(ls: Sequence, n: int=1) -> Generator[Sequence, None, None]:
    """Yield `n` roughly same size chunks of from a list.

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


def revcomp(seq: str) -> str:
    """Return the reverse complement of a DNA sequence.
    """
    return seq.translate(BASE_COMP)[::-1]


def load_paths_txt(paths_txt: Path) -> list[Path]:
    """Load file paths from a text file, with one path per line.
    Relative paths are resolved relative to the directory containing `paths_txt`.

    Args:
        paths_txt (Path): A text file with one path per line.

    Returns:
        list[Path]: A list of valid and resolved file paths.
    """
    paths_txt = paths_txt.resolve(strict=True)
    base_dir = paths_txt.parent

    paths_list = list()
    for path in paths_txt.read_text().splitlines():
        path = path.strip()
        if not path:
            continue

        path = Path(path)
        if not path.is_absolute():
            path = base_dir / path

        if path.is_file():
            paths_list.append(path.resolve(strict=True))
        elif path.is_dir():
            logger.error(f' - This is a directory, skipped: {path}')
        else:
            logger.error(f' - File not found, skipped: {path}')

    return paths_list


def load_fasta(path: Path) -> tuple[str, ...]:
    """Parse an assembly file in FASTA format and return its sequences.
    Gzip files are supported (file name should end with .gz).

    Args:
        path (Path): Path to the FASTA file. If the file is gzipped, the extension should be .gz.

    Returns:
        tuple[str, ...]: Sequences of FASTA records (upper case), in the same order as they appear in the file.
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
